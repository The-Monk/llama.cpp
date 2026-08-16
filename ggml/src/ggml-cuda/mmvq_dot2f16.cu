// Stage 25: v_dot2_f32_f16 dormant F16 decode GEMV. See mmvq_dot2f16.cuh for
// the full rationale/verdict -- this is a completeness/tuning-bench asset,
// NOT a production win (measured LOSS vs dp4a-int8 decode, Stage 19/22).
//
// Weight (GGML_TYPE_F16, src0) is read as raw ggml_fp16_t (uint16_t bit
// pattern) and reinterpreted to _Float16 via memcpy -- avoids any HIP
// __half/_Float16 ABI ambiguity. Activation (F32, src1) is cast to _Float16
// pairs on the fly (this path does not use mmvf.cu's existing half2-FMA
// machinery at all -- it's a fully separate, self-contained kernel, zero
// risk to the production F16 path).

#include "mmvq_dot2f16.cuh"

#include <cstring>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

typedef _Float16 __attribute__((ext_vector_type(2))) half2_ev;

__device__ __forceinline__ _Float16 f16_from_bits(uint16_t bits) {
    _Float16 v;
    memcpy(&v, &bits, 2);
    return v;
}

#define DOT2F16_BLOCK 64

// ---------------------------------------------------------------------
// Single-issue fdot2 GEMV.
// ---------------------------------------------------------------------
__global__ void k_mmvq_dot2f16_single(
        const char * __restrict__ vweight, const float * __restrict__ x,
        float * __restrict__ dst, int64_t K, int64_t row_stride_bytes) {
#if defined(RDNA3) || defined(RDNA4) // arch-guard: k_mmvq_dot2f16_single
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;
    const uint16_t * wrow = (const uint16_t *) (vweight + row * row_stride_bytes);

    float partial = 0.0f;
    for (int64_t i = (int64_t) tid * 2; i < K; i += (int64_t) blockDim.x * 2) {
        half2_ev w2, x2;
        w2.x = f16_from_bits(wrow[i]);
        w2.y = (i + 1 < K) ? f16_from_bits(wrow[i + 1]) : (_Float16) 0;
        x2.x = (_Float16) x[i];
        x2.y = (i + 1 < K) ? (_Float16) x[i + 1] : (_Float16) 0;
        partial = __builtin_amdgcn_fdot2(w2, x2, partial, false);
    }

    extern __shared__ float sdata_s[];
    sdata_s[tid] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata_s[tid] += sdata_s[tid + s];
        __syncthreads();
    }
    if (tid == 0) dst[row] = sdata_s[0];
#else
    NO_DEVICE_CODE;
#endif // arch-guard
}

// ---------------------------------------------------------------------
// Dual-issue VOPD DOT2ACC GEMV -- pinned-register inline asm, ported
// verbatim (register-naming pattern) from the Stage 22 PoC
// (~/int4-research/pocs/dot2acc-dualissue/bench.hip). Processes K in
// chunks of 8 elements (4 dot2 lanes -> 2 VOPD bundles) per iteration per
// thread; any remainder (<8 elements) falls back to single-issue fdot2 for
// correctness (K is not guaranteed to be a multiple of 8*blockDim.x*2).
// ---------------------------------------------------------------------
__device__ __forceinline__ void dual_dot2acc_pairA(float & acc0, float & acc1,
        uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile(
        "v_dual_dot2acc_f32_f16 %0, %2, %4 :: v_dual_dot2acc_f32_f16 %1, %3, %5\n\t"
        : "+{v10}"(acc0), "+{v11}"(acc1)
        : "{v12}"(a0), "{v13}"(a1), "{v14}"(b0), "{v15}"(b1)
    );
}
__device__ __forceinline__ void dual_dot2acc_pairB(float & acc0, float & acc1,
        uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile(
        "v_dual_dot2acc_f32_f16 %0, %2, %4 :: v_dual_dot2acc_f32_f16 %1, %3, %5\n\t"
        : "+{v20}"(acc0), "+{v21}"(acc1)
        : "{v22}"(a0), "{v23}"(a1), "{v24}"(b0), "{v25}"(b1)
    );
}

__device__ __forceinline__ uint32_t pack_half2_bits(_Float16 lo, _Float16 hi) {
    uint16_t blo, bhi;
    memcpy(&blo, &lo, 2);
    memcpy(&bhi, &hi, 2);
    return (uint32_t) blo | ((uint32_t) bhi << 16);
}

__global__ void k_mmvq_dot2f16_dual(
        const char * __restrict__ vweight, const float * __restrict__ x,
        float * __restrict__ dst, int64_t K, int64_t row_stride_bytes) {
#if defined(RDNA3) || defined(RDNA4) // arch-guard: k_mmvq_dot2f16_dual
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;
    const uint16_t * wrow = (const uint16_t *) (vweight + row * row_stride_bytes);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int64_t stride8 = (int64_t) blockDim.x * 8;
    int64_t i = (int64_t) tid * 8;
    for (; i + 8 <= K; i += stride8) {
        uint32_t a0 = pack_half2_bits(f16_from_bits(wrow[i+0]), f16_from_bits(wrow[i+1]));
        uint32_t a1 = pack_half2_bits(f16_from_bits(wrow[i+2]), f16_from_bits(wrow[i+3]));
        uint32_t a2 = pack_half2_bits(f16_from_bits(wrow[i+4]), f16_from_bits(wrow[i+5]));
        uint32_t a3 = pack_half2_bits(f16_from_bits(wrow[i+6]), f16_from_bits(wrow[i+7]));
        uint32_t b0 = pack_half2_bits((_Float16) x[i+0], (_Float16) x[i+1]);
        uint32_t b1 = pack_half2_bits((_Float16) x[i+2], (_Float16) x[i+3]);
        uint32_t b2 = pack_half2_bits((_Float16) x[i+4], (_Float16) x[i+5]);
        uint32_t b3 = pack_half2_bits((_Float16) x[i+6], (_Float16) x[i+7]);
        dual_dot2acc_pairA(acc0, acc1, a0, a1, b0, b1);
        dual_dot2acc_pairB(acc2, acc3, a2, a3, b2, b3);
    }
    float partial = acc0 + acc1 + acc2 + acc3;
    // scalar single-issue tail for any remainder (<8 elements for this thread's stride)
    for (; i < K; i += (int64_t) blockDim.x * 8) {
        const int64_t j_end = (i + 8 < K) ? (i + 8) : K;
        for (int64_t j = i; j < j_end; j += 2) {
            half2_ev w2, x2;
            w2.x = f16_from_bits(wrow[j]);
            w2.y = (j + 1 < K) ? f16_from_bits(wrow[j + 1]) : (_Float16) 0;
            x2.x = (_Float16) x[j];
            x2.y = (j + 1 < K) ? (_Float16) x[j + 1] : (_Float16) 0;
            partial = __builtin_amdgcn_fdot2(w2, x2, partial, false);
        }
    }

    extern __shared__ float sdata_d[];
    sdata_d[tid] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata_d[tid] += sdata_d[tid + s];
        __syncthreads();
    }
    if (tid == 0) dst[row] = sdata_d[0];
#else
    NO_DEVICE_CODE;
#endif // arch-guard
}

bool ggml_cuda_mmvq_dot2f16_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_F16) return false;
    if (src1->type != GGML_TYPE_F32) return false;
    if (dst->type  != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src1->ne[1] != 1) return false; // M=1 (decode) only -- M>1 stays on mmvf.cu
    if (src0->ne[0] != src1->ne[0]) return false;
    return true;
}

bool ggml_cuda_op_mul_mat_vec_dot2f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int mode) {
    GGML_ASSERT(ggml_cuda_mmvq_dot2f16_supports(src0, src1, dst) && "ggml_cuda_op_mul_mat_vec_dot2f16 does not support this tensor shape");
    GGML_ASSERT(mode == 1 || mode == 2);

    {
        static bool logged_once = false;
        if (!logged_once) {
            logged_once = true;
            GGML_LOG_INFO("%s: dormant fdot2 decode path ACTIVE, mode=%d (%s) -- tuning-knob only, measured loss vs dp4a\n",
                          __func__, mode, mode == 1 ? "single-issue" : "dual-issue VOPD");
        }
    }

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    cudaStream_t stream = ctx.stream();

    const dim3 grid((unsigned) N, 1, 1);
    const dim3 block(DOT2F16_BLOCK, 1, 1);
    const size_t shmem = DOT2F16_BLOCK * sizeof(float);
    if (mode == 1) {
        k_mmvq_dot2f16_single<<<grid, block, shmem, stream>>>(
                (const char *) src0->data, (const float *) src1->data, (float *) dst->data, K, src0->nb[1]);
    } else {
        k_mmvq_dot2f16_dual<<<grid, block, shmem, stream>>>(
                (const char *) src0->data, (const float *) src1->data, (float *) dst->data, K, src0->nb[1]);
    }
    return true;
}

// ---------------------------------------------------------------------
// Correctness self-test -- hand-packs F16 weight rows + an F32 activation
// vector, bypasses the ggml_tensor machinery, launches both kernels
// directly, compares against a CPU fp32 reference dot product.
// ---------------------------------------------------------------------
namespace ggml_cuda_mmvq_dot2f16_selftest_detail {

static bool run_trial(std::mt19937 & rng, int64_t N, int64_t K, int mode, double & max_rel_err_out) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> w_bits((size_t) (N * K));
    std::vector<float> w_f32((size_t) (N * K));
    std::vector<float> x(K);
    for (auto & v : x) {
        // Kernel casts activation to _Float16 before dotting -- reference
        // must use the SAME fp16-rounded value (isolates the dot
        // arithmetic from expected fp16-rounding noise, same reasoning as
        // the weight-side rounding a few lines below).
        _Float16 h = (_Float16) nd(rng);
        v = (float) h;
    }
    for (size_t i = 0; i < w_f32.size(); i++) {
        float v = nd(rng);
        w_f32[i] = v;
        _Float16 h = (_Float16) v;
        uint16_t bits; memcpy(&bits, &h, 2);
        w_bits[i] = bits;
        w_f32[i] = (float) h; // reference uses the SAME fp16-rounded value, isolates the dot arithmetic
    }

    uint16_t * d_w = nullptr;
    float * d_x = nullptr, * d_dst = nullptr;
    if (hipMalloc(&d_w, w_bits.size() * sizeof(uint16_t)) != hipSuccess ||
        hipMalloc(&d_x, K * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_dst, (size_t) N * sizeof(float)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    CUDA_CHECK(hipMemcpy(d_w, w_bits.data(), w_bits.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_x, x.data(), K * sizeof(float), hipMemcpyHostToDevice));

    const dim3 grid((unsigned) N, 1, 1);
    const dim3 block(DOT2F16_BLOCK, 1, 1);
    const size_t shmem = DOT2F16_BLOCK * sizeof(float);
    const int64_t row_stride_bytes = K * (int64_t) sizeof(uint16_t);
    if (mode == 1) {
        k_mmvq_dot2f16_single<<<grid, block, shmem, 0>>>((const char *) d_w, d_x, d_dst, K, row_stride_bytes);
    } else {
        k_mmvq_dot2f16_dual<<<grid, block, shmem, 0>>>((const char *) d_w, d_x, d_dst, K, row_stride_bytes);
    }
    const hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        GGML_LOG_ERROR("%s: kernel failed: %s\n", __func__, hipGetErrorString(err));
        (void) hipFree(d_w); (void) hipFree(d_x); (void) hipFree(d_dst);
        return false;
    }

    std::vector<float> out(N);
    CUDA_CHECK(hipMemcpy(out.data(), d_dst, out.size() * sizeof(float), hipMemcpyDeviceToHost));
    (void) hipFree(d_w); (void) hipFree(d_x); (void) hipFree(d_dst);

    double max_rel = 0;
    bool ok = true;
    for (int64_t n = 0; n < N; n++) {
        double ref = 0;
        for (int64_t k = 0; k < K; k++) ref += (double) w_f32[n*K+k] * (double) x[k];
        double rel = std::fabs((double) out[n] - ref) / std::max(1.0, std::fabs(ref));
        max_rel = std::max(max_rel, rel);
        if (rel > 1e-2) ok = false; // fp16-accumulate-chain rounding, not exactness
    }
    max_rel_err_out = max_rel;
    return ok;
}

} // namespace ggml_cuda_mmvq_dot2f16_selftest_detail

bool ggml_cuda_mul_mat_vec_dot2f16_selftest() {
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        return true; // vacuously true off RDNA4 -- fdot2/dual-issue asm are gfx1201-specific
    }

    using namespace ggml_cuda_mmvq_dot2f16_selftest_detail;
    std::mt19937 rng(25025); // Stage 25
    bool all_pass = true;
    double worst_single = 0, worst_dual = 0;
    const int n_trials = 10;
    for (int t = 0; t < n_trials; t++) {
        double e1 = 0, e2 = 0;
        int64_t K = (t % 2 == 0) ? 4096 : 4104; // exact multiple of 8 vs not -- exercises the dual-issue tail path
        if (!run_trial(rng, /*N=*/17, K, /*mode=*/1, e1)) all_pass = false;
        if (!run_trial(rng, /*N=*/17, K, /*mode=*/2, e2)) all_pass = false;
        worst_single = std::max(worst_single, e1);
        worst_dual = std::max(worst_dual, e2);
    }
    GGML_LOG_INFO("%s: k_mmvq_dot2f16_{single,dual}, %d random trials (N=17,K=4096/4104) -> %s (max_rel single=%.4f dual=%.4f)\n",
                   __func__, n_trials, all_pass ? "PASS" : "FAIL", worst_single, worst_dual);
    return all_pass;
}
