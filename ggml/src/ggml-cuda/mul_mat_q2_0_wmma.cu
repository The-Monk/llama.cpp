// See mul_mat_q2_0_wmma.cuh for the full rationale.
//
// Two kernels, deliberately mirroring mul_mat_iu4.cu's structure:
//   1. k_quantize_act_iu4_q2 -- identical online plain-RTN per-32-block
//      symmetric int4 activation quantizer to mul_mat_iu4.cu's
//      k_quantize_act_iu4 (duplicated here rather than shared across TUs --
//      that file's copy has internal linkage and this is a small, self
//      contained experimental path).
//   2. k_mul_mat_q2_0_wmma -- the GEMV. Same warp-per-16x16-tile structure as
//      k_mul_mat_iu4, except the B (weight) operand is unpacked ON THE FLY
//      from block_q2_0's 2-bit ternary codes {0,1,2} -> signed int4 {-1,0,+1}
//      instead of being read from a native block_iu4. A Q2_0 block spans 128
//      elements (QK2_0) with ONE scale; the WMMA K-chunk is QK_IU4=32, so
//      each Q2_0 block covers exactly 4 WMMA chunks (subc = c % 4), all
//      sharing the same block scale (bq2->d).
//
// Toolchain note (same as mul_mat_iu4.cu): the host launcher below must NOT
// be wrapped in `#if defined(RDNA4)` -- that macro is device-pass-only.
#include "mul_mat_q2_0_wmma.cuh"

#include "mma.cuh"
#include <cstring>
#include <thread>
#include <chrono>

using namespace ggml_cuda_mma;

namespace ggml_cuda_mul_mat_q2_0_wmma_detail {

static __global__ void k_quantize_act_iu4_q2(
        const float * __restrict__ x, block_iu4 * __restrict__ y,
        const int64_t n_blocks_k, const int64_t row_stride_floats) {
    const int64_t c   = blockIdx.x; // which 32-elem block along K
    const int64_t m   = blockIdx.y; // which token/row
    const int     tid = threadIdx.x; // 0..31, element index within the block

    __shared__ float sh_val[32];
    __shared__ float sh_scale;
    __shared__ int   sh_q[32];

    const float v = x[m * row_stride_floats + c * 32 + tid];
    sh_val[tid] = fabsf(v);
    __syncthreads();

    if (tid == 0) {
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            amax = fmaxf(amax, sh_val[i]);
        }
        const float d = amax / 7.0f;
        sh_scale = d;
        y[m * n_blocks_k + c].d = __float2half(d);
    }
    __syncthreads();

    const float d  = sh_scale;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q = (int) rintf(v * id);
    q = q < -8 ? -8 : (q > 7 ? 7 : q);
    sh_q[tid] = q;
    __syncthreads();

    if ((tid & 1) == 0) {
        const int q0 = sh_q[tid];
        const int q1 = sh_q[tid + 1];
        y[m * n_blocks_k + c].qs[tid / 2] = (uint8_t) ((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }
}

// Unpack 8 consecutive Q2_0 qs bytes (32 chunk-local ternary codes, 4/byte at
// 2 bits each -- byte b holds codes for chunk-local k=[4b, 4b+3], low bits
// first) into the 4 physical int32 words the validated iu4_w4a4.cu/
// mul_mat_iu4.cu packing convention expects: word j covers logical
// k=[8j,8j+7], byte b of that word packs nlo=k(8j+2b) in the low nibble,
// nhi=k(8j+2b+1) in the high nibble. Ternary code c in {0,1,2} -> signed
// symbol c-1 in {-1,0,+1}, an EXACT (lossless) int4 representation.
static __device__ __forceinline__ void unpack_q2_0_chunk_to_iu4_words(const uint8_t * __restrict__ qs8, int (&w)[4]) {
    int vals[32];
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const int byte = qs8[b];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int code = (byte >> (e * 2)) & 0x3;
            vals[4*b + e] = code - 1;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t word = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int      k_lo = 8*j + 2*b;
            const int      k_hi = 8*j + 2*b + 1;
            const uint32_t nlo  = (uint32_t) (vals[k_lo] & 0xF);
            const uint32_t nhi  = (uint32_t) (vals[k_hi] & 0xF);
            word |= (nlo | (nhi << 4)) << (8*b);
        }
        w[j] = (int) word;
    }
}

__launch_bounds__(32, 1)
static __global__ void k_mul_mat_q2_0_wmma(
        const char * __restrict__ vweight, const block_iu4 * __restrict__ act, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t nb01_q2_0, const int64_t n_blocks_k,
        const int64_t dst_row_stride_floats) {
    const int64_t n0  = (int64_t) blockIdx.x * 16;
    const int64_t m0  = (int64_t) blockIdx.y * 16;
    const int     tid = threadIdx.x;

    __shared__ int   sh_A[16][4];
    __shared__ int   sh_B[16][4];
    __shared__ float sh_da[16];
    __shared__ float sh_dw[16];

    float acc[8];
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        acc[l] = 0.0f;
    }

    for (int64_t c = 0; c < n_blocks_k; ++c) {
        if (tid < 16) {
            const int     row = tid;
            const int64_t m   = m0 + row;
            if (m < M) {
                const block_iu4 & blk = act[m * n_blocks_k + c];
                int w[4];
                memcpy(&w[0], blk.qs + 0,  4);
                memcpy(&w[1], blk.qs + 4,  4);
                memcpy(&w[2], blk.qs + 8,  4);
                memcpy(&w[3], blk.qs + 12, 4);
                sh_A[row][0] = w[0]; sh_A[row][1] = w[1]; sh_A[row][2] = w[2]; sh_A[row][3] = w[3];
                sh_da[row] = __half2float(blk.d);
            } else {
                sh_A[row][0] = sh_A[row][1] = sh_A[row][2] = sh_A[row][3] = 0;
                sh_da[row] = 0.0f;
            }
        } else {
            const int row = tid - 16;
            const int64_t n = n0 + row;
            if (n < N) {
                const int64_t       q2blk   = c / 4;
                const int            subc    = (int) (c % 4);
                const block_q2_0 * bq2      = (const block_q2_0 *) (vweight + n * nb01_q2_0) + q2blk;
                const uint8_t     * qs8      = bq2->qs + subc * 8;
                int w[4];
                unpack_q2_0_chunk_to_iu4_words(qs8, w);
                sh_B[row][0] = w[0]; sh_B[row][1] = w[1]; sh_B[row][2] = w[2]; sh_B[row][3] = w[3];
                sh_dw[row] = __half2float(bq2->d);
            } else {
                sh_B[row][0] = sh_B[row][1] = sh_B[row][2] = sh_B[row][3] = 0;
                sh_dw[row] = 0.0f;
            }
        }
        __syncthreads();

        tile<16, 4, int> A;
        tile<16, 4, int> B;
        load_generic(A, &sh_A[0][0], 4);
        load_generic(B, &sh_B[0][0], 4);

        // Same DATA_LAYOUT_J_MAJOR readback fix mul_mat_iu4.cu documents:
        // the WMMA accumulator's physical layout is the transpose of the
        // A/B input layout on RDNA4.
        tile<16, 16, int, DATA_LAYOUT_J_MAJOR> D;
#pragma unroll
        for (int l = 0; l < D.ne; ++l) {
            D.x[l] = 0;
        }

        mma_iu4(D, A, B);

#pragma unroll
        for (int l = 0; l < D.ne; ++l) {
            const int i = D.get_i(l);
            const int j = D.get_j(l);
            acc[l] += (float) D.x[l] * sh_da[i] * sh_dw[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int     i = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>::get_i(l);
        const int     j = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>::get_j(l);
        const int64_t m = m0 + i;
        const int64_t n = n0 + j;
        if (m < M && n < N) {
            dst[m * dst_row_stride_floats + n] = acc[l];
        }
    }
}

} // namespace ggml_cuda_mul_mat_q2_0_wmma_detail

bool ggml_cuda_q2_0_wmma_decode_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_Q2_0) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0] || src0->ne[0] % QK_IU4 != 0) {
        return false;
    }
    // Kernel is correct for any M (same M-padding convention as mul_mat_iu4.cu).
    // 2026-07-17 (card 156): the M<=8 (verify-batch) test showed W4A4 LOSES 40-63% vs
    // dp4a -- because WMMA is a 16x16 MATRIX engine and at M<=8 you waste >=half the
    // tile (skinny GEMV on a matrix unit). That is a SHAPE mismatch, not a ternary
    // problem. The regime where native iu4 WMMA wins is PREFILL / large-M GEMM: the
    // isolated iu4 selftest measured 2.228x the int8 path at real-GEMM sizes, and our
    // own decode data trends toward crossover (gap -50%->-41% as M 1->8). So allow ALL
    // M here and test prefill (pp512) vs the current MMQ int8-activation path. Decode
    // (M<=8) will now also route here and be slow -- that is expected; the point of
    // this run is the PREFILL number. Revert the "allow all M" once prefill is measured.
    // (kept the RDNA4 + Q2_0 + F32 + K%QK_IU4 checks below intact)
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    return GGML_CUDA_CC_IS_RDNA4(cc);
}

using namespace ggml_cuda_mul_mat_q2_0_wmma_detail;

// --- W2A4 dot8 decode GEMV (T213) ------------------------------------------
// Identical OPERAND path to k_mul_mat_q2_0_wmma above -- same
// k_quantize_act_iu4_q2 int4 activations, same lossless
// unpack_q2_0_chunk_to_iu4_words 2-bit -> signed int4 weights -- but the
// compute is v_dot8_i32_iu4 instead of a 16x16 WMMA tile. At M=1 a WMMA tile
// discards 15/16 of its rows, which is why the WMMA decode intercept measures
// -55% vs dp4a; that was a verdict on the instruction, never on W2A4 operands.
//
// Why this should be faster than the shipped dp4a mmvq path: dp4a consumes
// int8 lanes, so 2-bit weights must expand 4x into registers and each thread
// drags a full 36B q8_1 activation chunk per 8B of weight (act:wt = 4.5).
// int4 lanes halve both -- 2x expansion, ~18B activations (act:wt = 2.3, the
// same ratio IU4 runs at). Measured act:wt vs efficiency across four formats
// (Q8_0 1.1 -> 100%, IU4 2.3 -> 87%, Q2_0 4.5 -> 68%, Q1_0 9.0 -> 46%)
// predicts this lands near IU4's ~87% of Q8_0-class, i.e. ~+34% decode.
// ACCURACY NOTE: weights stay EXACTLY ternary (the unpack is lossless); only
// the ACTIVATIONS drop q8_1 -> int4. That is the premise under test.
#define Q2_0_DOT8_BLOCK 64
#define Q2_0_DOT8_ROWS  3   // activation reuse, matching the dp4a template's rpb=3


// Bit-parallel 2-bit -> 4-bit spread: 2 Q2_0 bytes (8 ternary codes) become one
// int of 8 UNSIGNED int4 lanes, in 3 mask/shift/or steps (~9 instrs). Replaces
// unpack_q2_0_chunk_to_iu4_words' 32-element scalar loop (~176 ops per 8 bytes)
// in the GEMV inner loop -- that helper is fine amortised over a WMMA tile load
// but here it runs per chunk PER ROW, costing ~6x the ALU of the dp4a bit-spread
// this path is trying to beat.
//
// Codes stay UNSIGNED {0,1,2}; the ternary symbol s = c-1 is recovered with the
// same identity the dp4a path uses, dot(s,u) = dot(c,u) - sum(u), so no packed
// subtract (which would borrow across nibbles) is needed. sum(u) is per-CHUNK,
// so it is computed once and reused across all rows in the block.
static __device__ __forceinline__ uint32_t q2_spread8(uint32_t v16) {
    uint32_t v = v16 & 0xFFFFu;
    v = (v | (v <<  8)) & 0x00FF00FFu;   // 4 codes per 16-bit half
    v = (v | (v <<  4)) & 0x0F0F0F0Fu;   // 2 codes per byte
    v = (v | (v <<  2)) & 0x33333333u;   // 1 code per nibble
    return v;
}

static __global__ void k_mmvq_dot8_q2_0(
        const char * __restrict__ vweight, const block_iu4 * __restrict__ act,
        float * __restrict__ dst, int64_t n_blocks_k, int64_t nb01, int64_t N) {
    const int64_t row0 = (int64_t) blockIdx.x * Q2_0_DOT8_ROWS;
    const int     tid  = threadIdx.x;

    float partial[Q2_0_DOT8_ROWS];
#pragma unroll
    for (int r = 0; r < Q2_0_DOT8_ROWS; ++r) {
        partial[r] = 0.0f;
    }

    // ROW REUSE: load each int4 activation chunk ONCE and apply it to
    // Q2_0_DOT8_ROWS weight rows. This is the same lever rows_per_cuda_block
    // gives the dp4a mmvq template, and the rpb sweep (T212) showed it is what
    // actually drives effective bandwidth here: dp4a measured 55.5% at rpb=1
    // vs 62.8% at rpb=3. A one-row-per-block GEMV is an rpb=1 kernel and must
    // not be compared against the rpb=3 baseline.
    for (int64_t c = tid; c < n_blocks_k; c += blockDim.x) {
        uint32_t xw[4];
        memcpy(xw, act[c].qs, 16);
        const float da   = __half2float(act[c].d);
        const int64_t q2b = c >> 2;
        const int     sub = (int) (c & 3);

        // sum(u) for this chunk: one unsigned-weight dot against all-ones
        // nibbles. Per CHUNK, not per row, so it amortises over the block.
        int sum_u = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sum_u = __builtin_amdgcn_sudot8(false, (int) 0x11111111, true, (int) xw[i], sum_u, false);
        }
#pragma unroll
        for (int r = 0; r < Q2_0_DOT8_ROWS; ++r) {
            const int64_t row = row0 + r;
            if (row >= N) {
                continue;
            }
            const block_q2_0 * bq2 = (const block_q2_0 *) (vweight + row * nb01) + q2b;
            uint32_t q[2];
            memcpy(q, bq2->qs + sub * 8, 8);
            const uint32_t w0 = q2_spread8(q[0]);
            const uint32_t w1 = q2_spread8(q[0] >> 16);
            const uint32_t w2 = q2_spread8(q[1]);
            const uint32_t w3 = q2_spread8(q[1] >> 16);
            int sumi = 0;
            sumi = __builtin_amdgcn_sudot8(false, (int) w0, true, (int) xw[0], sumi, false);
            sumi = __builtin_amdgcn_sudot8(false, (int) w1, true, (int) xw[1], sumi, false);
            sumi = __builtin_amdgcn_sudot8(false, (int) w2, true, (int) xw[2], sumi, false);
            sumi = __builtin_amdgcn_sudot8(false, (int) w3, true, (int) xw[3], sumi, false);
            partial[r] += (float) (sumi - sum_u) * __half2float(bq2->d) * da;
        }
    }

    __shared__ float sdata[Q2_0_DOT8_ROWS][Q2_0_DOT8_BLOCK];
#pragma unroll
    for (int r = 0; r < Q2_0_DOT8_ROWS; ++r) {
        sdata[r][tid] = partial[r];
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int r = 0; r < Q2_0_DOT8_ROWS; ++r) {
                sdata[r][tid] += sdata[r][tid + s];
            }
        }
        __syncthreads();
    }
    if (tid < Q2_0_DOT8_ROWS && row0 + tid < N) {
        dst[row0 + tid] = sdata[tid][0];
    }
}

bool ggml_cuda_op_mul_mat_q2_0_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using namespace ggml_cuda_mul_mat_q2_0_wmma_detail;

    GGML_ASSERT(src0->type == GGML_TYPE_Q2_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(src0->ne[0] % QK_IU4 == 0);

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t n_blocks_k = K / QK_IU4;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_iu4> act_q(ctx.pool(), (size_t) (M * n_blocks_k));

    {
        const int64_t row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
        const dim3 grid(n_blocks_k, M, 1);
        const dim3 block(32, 1, 1);
        k_quantize_act_iu4_q2<<<grid, block, 0, stream>>>((const float *) src1->data, act_q.get(), n_blocks_k, row_stride_floats);
    }

    {
        const int64_t nb01 = src0->nb[1];
        const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
        const dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
        const dim3 block(32, 1, 1);
        k_mul_mat_q2_0_wmma<<<grid, block, 0, stream>>>(
                (const char *) src0->data, act_q.get(), (float *) dst->data,
                M, N, nb01, n_blocks_k, dst_row_stride_floats);
    }

    return true;
}


// --- W2A4 dot8 decode entry points (T213) ----------------------------------
bool ggml_cuda_q2_0_dot8_decode_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_Q2_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0] || src0->ne[0] % QK2_0 != 0) {
        return false;
    }
    if (src1->ne[1] != 1) {
        return false; // M=1 decode only -- prefill keeps the existing path
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return GGML_CUDA_CC_IS_RDNA4(cc);
}

bool ggml_cuda_op_mul_mat_q2_0_dot8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using namespace ggml_cuda_mul_mat_q2_0_wmma_detail;

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t n_blocks_k = K / QK_IU4;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<block_iu4> act_q(ctx.pool(), (size_t) n_blocks_k);

    {
        const int64_t row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
        const dim3 grid(n_blocks_k, 1, 1);
        const dim3 block(32, 1, 1);
        k_quantize_act_iu4_q2<<<grid, block, 0, stream>>>((const float *) src1->data, act_q.get(), n_blocks_k, row_stride_floats);
    }
    {
        const dim3 grid((unsigned) ((N + Q2_0_DOT8_ROWS - 1) / Q2_0_DOT8_ROWS), 1, 1);
        const dim3 block(Q2_0_DOT8_BLOCK, 1, 1);
        k_mmvq_dot8_q2_0<<<grid, block, 0, stream>>>(
                (const char *) src0->data, act_q.get(), (float *) dst->data, n_blocks_k, src0->nb[1], N);
    }
    return true;
}

// --- Concurrency PoC companion (see mul_mat_q2_0_wmma.cuh) -----------------
// Standalone throughput hammer, same shape as the dependency-free
// /tmp poc.cpp microbench used to establish the cross-process baseline, but
// running as a background thread INSIDE this process/HIP context so it
// shares streams/queue scheduling with whatever else this process (e.g.
// llama-bench doing real Q2_0 decode) is doing on the same device -- this is
// the mechanism T112's async draft/verify pipeline actually uses (separate
// hipStream_t, same process), not a separate-process test.
namespace ggml_cuda_wmma_concurrency_poc {

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

static __global__ void k_wmma_companion_poc(int * __restrict__ out, long iters) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
    int32x2_t a = {(int) threadIdx.x, (int) blockIdx.x};
    int32x2_t b = {(int) (threadIdx.x * 3), (int) (blockIdx.x * 5)};
    int32x8_t acc = {0,0,0,0,0,0,0,0};
#pragma unroll 1
    for (long i = 0; i < iters; ++i) {
        acc = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, acc, true);
        a[0] += acc[0] & 0xff;
    }
    if (threadIdx.x < 8) {
        out[blockIdx.x * 8 + threadIdx.x] = acc[threadIdx.x];
    }
#else
    GGML_UNUSED(out);
    GGML_UNUSED(iters);
#endif
}

static void companion_thread_fn(double seconds, long iters_per_launch) {
    hipStream_t s;
    if (hipStreamCreate(&s) != hipSuccess) {
        return;
    }
    int * d_out = nullptr;
    if (hipMalloc(&d_out, 512 * 8 * sizeof(int)) != hipSuccess) {
        hipStreamDestroy(s);
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    long launches = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        k_wmma_companion_poc<<<512, 32, 0, s>>>(d_out, iters_per_launch);
        hipStreamSynchronize(s);
        ++launches;
    }
    GGML_LOG_INFO("%s: WMMA concurrency-poc companion done, %ld launches over %.1fs\n", __func__, launches, seconds);
    hipStreamDestroy(s);
    hipFree(d_out);
}

} // namespace ggml_cuda_wmma_concurrency_poc

void ggml_cuda_start_wmma_concurrency_poc_companion(double seconds, long iters_per_launch) {
    using namespace ggml_cuda_wmma_concurrency_poc;
    std::thread t(companion_thread_fn, seconds, iters_per_launch);
    t.detach();
}
