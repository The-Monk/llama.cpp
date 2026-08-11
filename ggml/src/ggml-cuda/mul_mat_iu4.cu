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
#include <random>
#include <vector>
#include <algorithm>

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

        // Fix ported from 615f718ca (iu4_w4a4.cu selftest): the WMMA
        // accumulator's physical VGPR layout on RDNA4 is the TRANSPOSE of
        // the A/B input layout, so reading it back through the default
        // DATA_LAYOUT_I_MAJOR tile gives get_i()/get_j() swapped relative
        // to the true (row, col) -- invisible on the diagonal, wrong
        // everywhere else. DATA_LAYOUT_J_MAJOR swaps get_i()<->get_j() back
        // to the correct orientation (same convention mmq.cuh's
        // vec_dot_q8_0_16_q8_1_mma already uses for an identical
        // tile<16,4,int> A/B shape). The final store loop below must use
        // the same data_layout tag for its get_i/get_j so it maps `l` back
        // to the same (i, j) that acc[l] was accumulated under.
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
        __syncthreads(); // shared staging buffers get overwritten next chunk
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

// Card 133 item 3: minimal synthetic correctness check for k_mul_mat_iu4
// itself (not just the mma_iu4() primitive, which iu4_w4a4.cu's selftest
// already covers). iu4 has no real GGUF producer model yet (rotation-free
// packed-int4 W4A4, see mul_mat_iu4.cuh), so this bypasses
// ggml_cuda_op_mul_mat_iu4()/k_quantize_act_iu4 and the ggml_tensor
// machinery entirely: it hand-packs both operands into block_iu4 with a
// fixed per-block scale of 1.0 (so the WMMA int32 accumulator IS the
// answer, no floating-point rounding anywhere) and launches k_mul_mat_iu4
// directly against an exact CPU int4 x int4 reference. Deliberately uses
// non-multiple-of-16 M/N (exercises the boundary clamp) and K spanning 2
// blocks (exercises the multi-chunk accumulation loop), same style as
// iu4_w4a4.cu's run_trial().
namespace ggml_cuda_mul_mat_iu4_selftest_detail {

static void pack_iu4_block(const int vals[QK_IU4], block_iu4 & blk) {
    blk.d = __float2half(1.0f); // fixed unit scale: accumulator == answer, no rounding
    for (int j = 0; j < 4; ++j) {
        uint32_t word = 0;
        for (int b = 0; b < 4; ++b) {
            const int      k_lo = 8*j + 2*b;
            const int      k_hi = 8*j + 2*b + 1;
            const uint32_t nlo  = (uint32_t) (vals[k_lo] & 0xF);
            const uint32_t nhi  = (uint32_t) (vals[k_hi] & 0xF);
            word |= (nlo | (nhi << 4)) << (8*b);
        }
        std::memcpy(blk.qs + 4*j, &word, 4);
    }
}

static bool run_trial(std::mt19937 & rng, int64_t M, int64_t N, int64_t K, long & max_abs_err_out) {
    std::uniform_int_distribution<int> valdist(-8, 7);
    const int64_t n_blocks_k = K / QK_IU4;

    std::vector<std::vector<int>> act_logical(M, std::vector<int>(K));
    std::vector<std::vector<int>> w_logical(N, std::vector<int>(K));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            act_logical[m][k] = valdist(rng);
        }
    }
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            w_logical[n][k] = valdist(rng);
        }
    }

    std::vector<block_iu4> act_blocks((size_t) (M * n_blocks_k));
    std::vector<block_iu4> w_blocks((size_t) (N * n_blocks_k));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t c = 0; c < n_blocks_k; ++c) {
            pack_iu4_block(&act_logical[m][c * QK_IU4], act_blocks[m * n_blocks_k + c]);
        }
    }
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < n_blocks_k; ++c) {
            pack_iu4_block(&w_logical[n][c * QK_IU4], w_blocks[n * n_blocks_k + c]);
        }
    }

    block_iu4 * d_act = nullptr;
    block_iu4 * d_w   = nullptr;
    float *     d_dst = nullptr;
    if (hipMalloc(&d_act, act_blocks.size() * sizeof(block_iu4)) != hipSuccess ||
        hipMalloc(&d_w,   w_blocks.size()   * sizeof(block_iu4)) != hipSuccess ||
        hipMalloc(&d_dst, (size_t) (M * N) * sizeof(float)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    CUDA_CHECK(hipMemcpy(d_act, act_blocks.data(), act_blocks.size() * sizeof(block_iu4), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_w,   w_blocks.data(),   w_blocks.size()   * sizeof(block_iu4), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(d_dst, 0, (size_t) (M * N) * sizeof(float)));

    const int64_t nb01                   = n_blocks_k * (int64_t) sizeof(block_iu4);
    const int64_t dst_row_stride_floats  = N;
    const dim3    grid((N + 15) / 16, (M + 15) / 16, 1);
    const dim3    block(32, 1, 1);
    k_mul_mat_iu4<<<grid, block, 0, 0>>>(
            (const char *) d_w, d_act, d_dst, M, N, nb01, n_blocks_k, dst_row_stride_floats);
    const hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        GGML_LOG_ERROR("%s: k_mul_mat_iu4 failed: %s\n", __func__, hipGetErrorString(err));
        (void) hipFree(d_act); (void) hipFree(d_w); (void) hipFree(d_dst);
        return false;
    }

    std::vector<float> out((size_t) (M * N));
    CUDA_CHECK(hipMemcpy(out.data(), d_dst, out.size() * sizeof(float), hipMemcpyDeviceToHost));
    (void) hipFree(d_act); (void) hipFree(d_w); (void) hipFree(d_dst);

    long max_abs = 0;
    long diag_mismatches = 0, offdiag_mismatches = 0;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            long ref = 0;
            for (int64_t k = 0; k < K; ++k) {
                ref += (long) act_logical[m][k] * (long) w_logical[n][k];
            }
            const long got = (long) out[m * N + n];
            const long diff = std::labs(got - ref);
            max_abs = std::max(max_abs, diff);
            if (diff != 0) {
                if (m == n) { diag_mismatches++; } else { offdiag_mismatches++; }
            }
        }
    }
    max_abs_err_out = max_abs;
    if (max_abs != 0) {
        GGML_LOG_INFO("%s: mismatch signature: %ld diagonal, %ld off-diagonal (of %ld total)\n",
                       __func__, diag_mismatches, offdiag_mismatches, (long) (M * N));
    }
    return max_abs == 0;
}

} // namespace ggml_cuda_mul_mat_iu4_selftest_detail

bool ggml_cuda_mul_mat_iu4_selftest() {
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        // Not RDNA4 -- k_mul_mat_iu4's mma_iu4() call is a NO_DEVICE_CODE
        // no-op off RDNA4, nothing meaningful to test. Vacuously true.
        return true;
    }

    using namespace ggml_cuda_mul_mat_iu4_selftest_detail;
    std::mt19937 rng(133133); // card 133
    bool all_pass = true;
    long worst = 0;
    const int n_trials = 20;
    // M, N deliberately not multiples of 16 (boundary clamp); K spans 2
    // QK_IU4 blocks (multi-chunk accumulation loop).
    for (int t = 0; t < n_trials; ++t) {
        long max_abs_err = 0;
        if (!run_trial(rng, /*M=*/24, /*N=*/40, /*K=*/64, max_abs_err)) {
            all_pass = false;
        }
        worst = std::max(worst, max_abs_err);
    }
    GGML_LOG_INFO("%s: k_mul_mat_iu4 real-model wrapper, %d random trials (M=24,N=40,K=64) -> %s (max_abs_err=%ld)\n",
                   __func__, n_trials, all_pass ? "PASS" : "FAIL", worst);
    return all_pass;
}
