// T165: route TERNARY (Q2_0) weights through the FP8 WMMA datapath instead
// of the native iu4 int4x int4->int32 datapath (mul_mat_iu4_mmq.cu).
//
// Thesis (coordinator directive, following the T162-T164 iu4 investigation):
// the iu4 path has two independent, ISA/diagnostic-confirmed root causes --
// RC1 (WMMA issue/dependency latency, ~12-14%) and RC2 (the int32->float
// rescale epilogue, ~13-15%). RC2 is an ARTIFACT of the INTEGER accumulator:
// every iu4 K-chunk produces an int32 partial dot product that must be
// explicitly CONVERTED to float before it can be rescaled and added to the
// running total. The RDNA4 fp8 e4m3 WMMA path (mma.cuh's
// `mma(tile<16,16,float>&, tile<16,8,int>&, tile<16,8,int>&)` overload, the
// SAME one the production F8E4M3 MMQ vec_dot uses -- see mmq.cuh's
// vec_dot_f8e4m3_f8e4m3_mma, GREEN in the card-155 audit) accumulates the
// true dot product NATIVELY in fp32 -- there is no int32 intermediate, so
// RC2's int->float CONVERSION instruction is structurally absent. Ternary
// {-1,0,+1} is EXACTLY representable in e4m3 (verified: torch
// float8_e4m3fn byte patterns 0x00/0x38/0xB8 for 0/+1/-1), so weights can be
// unpacked from the SAME Q2_0 GGUF bytes already in VRAM (no format change,
// no re-quantization, no accuracy re-gate needed beyond the one already run
// -- w4a4_gate_bonsai.py's fp8_e4m3 activation mode measured -0.09% vs the
// int8 floor, i.e. accuracy-free) via a trivial 4-entry LUT immediately
// before the WMMA, same as the existing int4 LUT unpack in
// mul_mat_iu4_mmq.cu's Q2_0Loader.
//
// Tradeoff (explicitly acknowledged, not hidden): the fp8 WMMA is
// `16x16x16` (needs TWO builtin calls to cover a K=32 chunk, matching
// `mma()`'s existing two-call implementation) vs iu4's single `16x16x32`
// call -- i.e. this path does NOT get iu4's 2x MAC-density-per-instruction.
// Per the T163 diagnostic (RC1 measured on the iu4 op itself), that density
// advantage never translated into a net win anyway (WMMA issue latency
// dominated regardless of MAC count), so trading it away to eliminate RC2
// entirely is the whole point of this experiment.
//
// Structure: DELIBERATELY mirrors mul_mat_iu4_mmq.cu's already-tuned MMQ-
// grade tiling (BM=BN=64, NWARPS=4 -- T163 Lever A's validated ILP width,
// double-buffered shared staging, the need_check template-bool pattern from
// T162) as closely as possible, changing ONLY what fp8 requires: tile shapes
// (`tile<16,8,int>` operands / `tile<16,16,float>` accumulator instead of
// iu4's `tile<16,4,int>` / `tile<16,16,int>`), the weight/activation LUT
// unpack, and the epilogue (no int->float cast). This is a from-scratch
// STANDALONE file, NOT a change to mmq.cuh's shared multi-type dispatch --
// zero risk to the production F8E4M3/Q8_0/etc. MMQ paths. Gate:
// GGML_HIP_Q2_0_FP8ROUTE_MMQ (opt-in, default OFF, same discipline as every
// other T162-T164 lever).
#include "mul_mat_q2_0_fp8route_mmq.cuh"

#include "mma.cuh"
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

using namespace ggml_cuda_mma;

namespace ggml_cuda_mul_mat_q2_0_fp8route_mmq_detail {

// One 32-wide (QK_IU4-matching) act/weight chunk packed as RAW fp8 e4m3
// bytes (1 byte/value, unlike iu4's 2-values/byte nibble packing) + one
// fp16 scale, matching what tile<16,8,int>'s load expects (8 ints = 32 bytes).
struct block_fp8_32 {
    ggml_half d;
    uint8_t   qs[32];
};
static_assert(sizeof(block_fp8_32) == sizeof(ggml_half) + 32, "wrong block_fp8_32 size/padding");

// Exact e4m3 byte patterns for ternary {-1,0,+1} (verified against torch's
// native float8_e4m3fn: -1.0->0xB8, 0.0->0x00, +1.0->0x38).
__device__ __forceinline__ uint8_t ternary_code_to_fp8_byte(int code /* 0,1,2 for -1,0,+1 */) {
    // Q2_0 2-bit code convention (matches unpack_q2_0_chunk_to_iu4_words in
    // mul_mat_iu4_mmq.cu): code-1 maps {0,1,2} -> {-1,0,+1}.
    switch (code) {
        case 0: return 0xB8; // -1.0
        case 2: return 0x38; // +1.0
        default: return 0x00; // 0.0 (code==1)
    }
}

// Unpacks 8 consecutive Q2_0 qs bytes (32 chunk-local ternary codes, 4/byte
// at 2 bits each -- SAME convention as unpack_q2_0_chunk_to_iu4_words) into
// 32 raw fp8 e4m3 bytes, packed as 8 int32 words for tile<16,8,int>.
__device__ __forceinline__ void unpack_q2_0_chunk_to_fp8_words(const uint8_t * __restrict__ qs8, int (&w)[8]) {
    uint8_t fp8[32];
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const int byte = qs8[b];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int code = (byte >> (e * 2)) & 0x3;
            fp8[4*b + e] = ternary_code_to_fp8_byte(code);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        uint32_t word;
        std::memcpy(&word, fp8 + 4*j, 4);
        w[j] = (int) word;
    }
}

// --- activation quantization (fp32 -> per-32-group fp8 e4m3) ---------------
static __global__ void k_quantize_act_fp8route_mmq(
        const float * __restrict__ x, block_fp8_32 * __restrict__ y,
        const int64_t n_blocks_k, const int64_t row_stride_floats) {
#if defined(RDNA4) // arch-guard: k_quantize_act_fp8route_mmq
    const int64_t c   = blockIdx.x; // which 32-elem block along K
    const int64_t m   = blockIdx.y; // which token/row
    const int     tid = threadIdx.x; // 0..31

    __shared__ float sh_val[32];
    __shared__ float sh_scale;

    const float v = x[m * row_stride_floats + c * 32 + tid];
    sh_val[tid] = fabsf(v);
    __syncthreads();

    if (tid == 0) {
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            amax = fmaxf(amax, sh_val[i]);
        }
        // 448.0 = e4m3fn max finite magnitude (matches quantize_mmq_f8e4m3's
        // own convention exactly -- see quantize.cu).
        const float d = amax > 0.0f ? amax / 448.0f : 0.0f;
        sh_scale = d;
        y[m * n_blocks_k + c].d = __float2half(d);
    }
    __syncthreads();

    const float d  = sh_scale;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const float qf = v * id;

    // Same hardware packed converter as the production quantize_mmq_f8e4m3
    // kernel (quantize.cu): V_CVT_PK_FP8_F32, 2 fp32 -> 2 e4m3 bytes/call.
    // Pair even/odd lanes via shuffle instead of float4 (this kernel is
    // one-value-per-lane, quantize_mmq_f8e4m3 is float4-per-lane -- same
    // instruction, different grouping).
    const float qf_pair = __shfl_xor_sync(0xFFFFFFFF, qf, 1, 32);
    if ((tid & 1) == 0) {
        const uint32_t packed = __builtin_amdgcn_cvt_pk_fp8_f32(qf, qf_pair, 0u, false);
        std::memcpy(y[m * n_blocks_k + c].qs + tid, &packed, 2);
    }
#else
    NO_DEVICE_CODE;
#endif // arch-guard
}

// --- the MMQ-grade fp8-route kernel -----------------------------------------
// Mirrors k_mul_mat_iu4_mmq's tiling/staging exactly (see mul_mat_iu4_mmq.cu
// for the detailed rationale of each design choice: need_check, warp-uniform
// dispatch via readfirstlane, NWARPS=4/BM=BN=64 ILP width). Only the tile
// shapes/WMMA call/epilogue differ (fp8 native fp32 accumulate, no int cast).
static constexpr int MMQ_FP8_BM     = 64;
static constexpr int MMQ_FP8_BN     = 64;
static constexpr int MMQ_FP8_NWARPS = 4;
static constexpr int MMQ_FP8_NTILES = (MMQ_FP8_BM / 16) * (MMQ_FP8_BN / 16);
static constexpr int MMQ_FP8_NTX    = MMQ_FP8_NTILES / MMQ_FP8_NWARPS;
static_assert(MMQ_FP8_BM + MMQ_FP8_BN <= MMQ_FP8_NWARPS * 32, "staging load assumes 1 thread/row");

template <int BM, int BN, int NWARPS, bool need_check>
__launch_bounds__(NWARPS * 32, 1)
static __global__ void k_mul_mat_q2_0_fp8route_mmq(
        const char * __restrict__ vweight, const block_fp8_32 * __restrict__ act, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t nb01, const int64_t n_blocks_k,
        const int64_t dst_row_stride_floats) {
#if defined(RDNA4) // arch-guard: k_mul_mat_q2_0_fp8route_mmq
    constexpr int NTILES_N = BN / 16;
    constexpr int NTX      = (BM / 16) * (BN / 16) / NWARPS;

    const int64_t n0      = (int64_t) blockIdx.x * BN;
    const int64_t m0      = (int64_t) blockIdx.y * BM;
    const int     lane    = threadIdx.x;
    const int     warp_id = threadIdx.y;
    const int     warp_id_u = __builtin_amdgcn_readfirstlane(warp_id);
    static_assert(BM % 32 == 0 && BN % 32 == 0, "warp-uniform dispatch assumes BM/BN are multiples of the wave size");
    constexpr int WARPS_A = BM / 32;
    constexpr int WARPS_B = BN / 32;

    __shared__ int   sh_A[2][BM][8];
    __shared__ float sh_da[2][BM];
    __shared__ int   sh_B[2][BN][8];
    __shared__ float sh_dw[2][BN];

    auto load_chunk = [&] (int64_t c, int buf) {
        if (warp_id_u < WARPS_A) {
            const int     row = warp_id_u * 32 + lane;
            const int64_t m   = m0 + row;
            if (!need_check || m < M) {
                const block_fp8_32 & blk = act[m * n_blocks_k + c];
                int w[8];
                std::memcpy(w, blk.qs, 32);
#pragma unroll
                for (int i = 0; i < 8; ++i) { sh_A[buf][row][i] = w[i]; }
                sh_da[buf][row] = __half2float(blk.d);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) { sh_A[buf][row][i] = 0; }
                sh_da[buf][row] = 0.0f;
            }
        } else if (warp_id_u < WARPS_A + WARPS_B) {
            const int     row = (warp_id_u - WARPS_A) * 32 + lane;
            const int64_t n   = n0 + row;
            if (!need_check || n < N) {
                const int64_t      q2blk = c / 4;
                const int           subc  = (int) (c % 4);
                const block_q2_0 * bq2   = (const block_q2_0 *) (vweight + n * nb01) + q2blk;
                int w[8];
                unpack_q2_0_chunk_to_fp8_words(bq2->qs + subc * 8, w);
#pragma unroll
                for (int i = 0; i < 8; ++i) { sh_B[buf][row][i] = w[i]; }
                sh_dw[buf][row] = __half2float(bq2->d);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) { sh_B[buf][row][i] = 0; }
                sh_dw[buf][row] = 0.0f;
            }
        }
    };

    float acc[NTX][8];
#pragma unroll
    for (int s = 0; s < NTX; ++s) {
#pragma unroll
        for (int l = 0; l < 8; ++l) { acc[s][l] = 0.0f; }
    }

    load_chunk(0, 0);
    __syncthreads();

    int cur = 0;
    for (int64_t c = 0; c < n_blocks_k; ++c) {
        const int nxt = cur ^ 1;
        if (c + 1 < n_blocks_k) {
            load_chunk(c + 1, nxt);
        }

#pragma unroll
        for (int s = 0; s < NTX; ++s) {
            const int t  = warp_id + s * NWARPS;
            const int mi = t / NTILES_N;
            const int ni = t % NTILES_N;

            tile<16, 8, int> A;
            tile<16, 8, int> B;
            load_generic(A, &sh_A[cur][mi * 16][0], 8);
            load_generic(B, &sh_B[cur][ni * 16][0], 8);

            tile<16, 16, float, DATA_LAYOUT_J_MAJOR> D;
#pragma unroll
            for (int l = 0; l < D.ne; ++l) { D.x[l] = 0.0f; }

            mma(D, A, B); // native fp32 accumulate, K=32 (2 internal WMMA calls) -- NO int32 intermediate (RC2 gone)

#pragma unroll
            for (int l = 0; l < D.ne; ++l) {
                const int i = D.get_i(l);
                const int j = D.get_j(l);
                // No (float) cast needed -- D.x[l] IS ALREADY float (this is
                // the whole point of the fp8 route: RC2's int->float
                // conversion instruction is structurally absent here).
                acc[s][l] += D.x[l] * sh_da[cur][mi * 16 + i] * sh_dw[cur][ni * 16 + j];
            }
        }

        __syncthreads();
        cur = nxt;
    }

#pragma unroll
    for (int s = 0; s < NTX; ++s) {
        const int t  = warp_id + s * NWARPS;
        const int mi = t / NTILES_N;
        const int ni = t % NTILES_N;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int     i = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>::get_i(l);
            const int     j = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>::get_j(l);
            const int64_t m = m0 + mi * 16 + i;
            const int64_t n = n0 + ni * 16 + j;
            if (m < M && n < N) {
                dst[m * dst_row_stride_floats + n] = acc[s][l];
            }
        }
    }
#else
    NO_DEVICE_CODE;
#endif // arch-guard
}

} // namespace ggml_cuda_mul_mat_q2_0_fp8route_mmq_detail

bool ggml_cuda_q2_0_fp8route_mmq_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_Q2_0) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0] || src0->ne[0] % 32 != 0) {
        return false;
    }
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    return GGML_CUDA_CC_IS_RDNA4(cc);
}

bool ggml_cuda_op_mul_mat_q2_0_fp8route_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using namespace ggml_cuda_mul_mat_q2_0_fp8route_mmq_detail;
    GGML_ASSERT(src0->type == GGML_TYPE_Q2_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t n_blocks_k = K / 32;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_fp8_32> act_q(ctx.pool(), (size_t) (M * n_blocks_k));
    {
        const int64_t row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
        const dim3 grid(n_blocks_k, M, 1);
        const dim3 block(32, 1, 1);
        k_quantize_act_fp8route_mmq<<<grid, block, 0, stream>>>((const float *) src1->data, act_q.get(), n_blocks_k, row_stride_floats);
    }

    {
        constexpr int BM = MMQ_FP8_BM;
        constexpr int BN = MMQ_FP8_BN;
        constexpr int NWARPS = MMQ_FP8_NWARPS;
        const int64_t nb01 = src0->nb[1];
        const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
        const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
        const dim3 block(32, NWARPS, 1);
        const bool need_check = (M % BM != 0) || (N % BN != 0);
        if (need_check) {
            k_mul_mat_q2_0_fp8route_mmq<BM, BN, NWARPS, true><<<grid, block, 0, stream>>>(
                    (const char *) src0->data, act_q.get(), (float *) dst->data,
                    M, N, nb01, n_blocks_k, dst_row_stride_floats);
        } else {
            k_mul_mat_q2_0_fp8route_mmq<BM, BN, NWARPS, false><<<grid, block, 0, stream>>>(
                    (const char *) src0->data, act_q.get(), (float *) dst->data,
                    M, N, nb01, n_blocks_k, dst_row_stride_floats);
        }
    }

    return true;
}
