// T89 driver-completeness run: see mul_mat_iu4.cuh for the full rationale.
//
// Two kernels:
//   1. k_quantize_act_iu4  -- online plain-RTN per-block (32-elem) symmetric
//      int4 quantization of the F32 activation tensor, amax/7 scale, packed
//      into the SAME block_iu4 nibble convention the weight quantizer
//      (ggml_quants.c: quantize_row_iu4_ref) and the validated WMMA operand
//      packing (iu4_w4a4.cu pack_row_i4) both use: byte b's low nibble =
//      element 2*b, high nibble = element 2*b+1.
//   2. k_mul_mat_iu4        -- the actual GEMM. One warp (32 threads) per
//      16x16 output tile, looping over K in 32-wide chunks. Each chunk: 16
//      threads cooperatively stage the activation operand into shared
//      memory, the other 16 stage the weight operand, both in the plain
//      16-row x 4-int32-word layout `ggml_cuda_mma::load_generic()` expects
//      (verbatim the same layout iu4_w4a4.cu's self-test builds on the
//      host), then `mma_iu4()` is called and the int32 accumulator is
//      descaled (weight-scale x activation-scale, BOTH genuinely per-block)
//      and added into a running float accumulator. This per-chunk
//      descale-then-add (rather than one descale after summing raw int32
//      over all K) is what makes genuine per-block scaling correct here.
//
// IMPORTANT toolchain note (found the hard way): do NOT wrap the host-side
// launcher function below in `#if defined(RDNA4)` the way iu4_w4a4.cu wraps
// its selftest. `RDNA4` (vendors/hip.h) is defined from `__GFX12__`, which
// clang's HIP frontend only predefines during the DEVICE compilation pass of
// a .cu translation unit -- the HOST pass (which is what actually emits the
// callable `ggml_cuda_op_mul_mat_iu4` symbol linked into libggml-hip.so)
// never sees it. Gating the host function itself on `#if defined(RDNA4)`
// silently compiles+links the `#else` stub branch as the ONLY visible
// symbol, no matter what GPU is attached -- confirmed by a live GGML_ABORT
// on real gfx1201 hardware. The correct pattern (used throughout mma.cuh's
// own `mma()`/`mma_iu4()` __device__ functions) is: keep the HOST wrapper
// unconditional, and gate only the __device__/__global__ code that truly
// only exists per compilation pass -- `mma_iu4()` already does this via its
// own internal `#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)` (which
// resolves correctly because it's evaluated while compiling __device__ code
// for a specific --offload-arch, not at file scope).
#include "mul_mat_iu4.cuh"

#include "mma.cuh"
#include <cstring>

using namespace ggml_cuda_mma;

static __global__ void k_quantize_act_iu4(
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

static __device__ __forceinline__ void load_iu4_words(const block_iu4 & blk, int (&w)[4]) {
    memcpy(&w[0], blk.qs + 0,  4);
    memcpy(&w[1], blk.qs + 4,  4);
    memcpy(&w[2], blk.qs + 8,  4);
    memcpy(&w[3], blk.qs + 12, 4);
}

__launch_bounds__(32, 1)
static __global__ void k_mul_mat_iu4(
        const char * __restrict__ vweight, const block_iu4 * __restrict__ act, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t nb01, const int64_t n_blocks_k,
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
                load_iu4_words(blk, w);
                sh_A[row][0] = w[0]; sh_A[row][1] = w[1]; sh_A[row][2] = w[2]; sh_A[row][3] = w[3];
                sh_da[row] = __half2float(blk.d);
            } else {
                sh_A[row][0] = sh_A[row][1] = sh_A[row][2] = sh_A[row][3] = 0;
                sh_da[row] = 0.0f;
            }
        } else {
            const int     row = tid - 16;
            const int64_t n   = n0 + row;
            if (n < N) {
                const block_iu4 * blk = (const block_iu4 *) (vweight + n * nb01 + c * (int64_t) sizeof(block_iu4));
                int w[4];
                load_iu4_words(*blk, w);
                sh_B[row][0] = w[0]; sh_B[row][1] = w[1]; sh_B[row][2] = w[2]; sh_B[row][3] = w[3];
                sh_dw[row] = __half2float(blk->d);
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

        // KNOWN BUG (EXPERIMENTAL / model-blocked path, not fixed here -- see
        // block_iu4 comment in ggml-common.h): this accumulator uses the
        // default DATA_LAYOUT_I_MAJOR, but 615f718ca (iu4_w4a4.cu selftest)
        // found the WMMA accumulator's physical VGPR layout on RDNA4 is the
        // TRANSPOSE of the A/B input layout, so get_i()/get_j() here read
        // (row, col) swapped -- only ever verified correct on the diagonal.
        // The fix there was `tile<16, 16, int, DATA_LAYOUT_J_MAJOR> D`; this
        // kernel was not ported to that fix. Output is therefore expected to
        // be silently wrong (garbage/RTN-at-best) off the diagonal until it
        // is.
        tile<16, 16, int> D;
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
        __syncthreads(); // shared staging buffers get overwritten next chunk
    }

#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int     i = tile<16, 16, int>::get_i(l);
        const int     j = tile<16, 16, int>::get_j(l);
        const int64_t m = m0 + i;
        const int64_t n = n0 + j;
        if (m < M && n < N) {
            dst[m * dst_row_stride_floats + n] = acc[l];
        }
    }
}

bool ggml_cuda_op_mul_mat_iu4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_IU4);
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
        k_quantize_act_iu4<<<grid, block, 0, stream>>>((const float *) src1->data, act_q.get(), n_blocks_k, row_stride_floats);
    }

    {
        const int64_t nb01 = src0->nb[1];
        const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
        const dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
        const dim3 block(32, 1, 1);
        k_mul_mat_iu4<<<grid, block, 0, stream>>>(
                (const char *) src0->data, act_q.get(), (float *) dst->data,
                M, N, nb01, n_blocks_k, dst_row_stride_floats);
    }

    return true;
}
