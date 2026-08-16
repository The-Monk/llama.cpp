// Stage 26: stochastic-rounding fp8 weight quantizer implementation.
// See quantize_fp8_sr.cuh for the full rationale.

#include "quantize_fp8_sr.cuh"
#include "ggml-quants.h"
#include "ggml-impl.h" // ggml_fp32_to_e4m3/e5m2, ggml_e4m3_to_fp32/e5m2_to_fp32 -- raw scalar RTN refs, no block scaling

#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

#define SR_QUANT_BLOCK 32

// gfx1201 lacks __builtin_amdgcn_prng_b32 (needs target feature
// "prng-inst", confirmed absent via a direct build failure -- the same
// gfx950/CDNA4-exclusive story as the MX cvt_scalef32_sr family found dead
// in Stage 18/25: the CK ecosystem's SR-quantizer idiom, mxf4_utils.hpp's
// `__builtin_amdgcn_prng_b32(__builtin_readcyclecounter()*(gid+1))`,
// assumes hardware this card doesn't have). Substitute a standard
// integer-hash mix (Murmur3-style finalizer) of the same
// cycle-counter-derived seed -- cvt_sr_fp8_f32/cvt_sr_bf8_f32 only need a
// well-distributed 32-bit dither value, not specifically a hardware PRNG
// output.
__device__ __forceinline__ uint32_t sr_seed_hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__global__ void k_quantize_weight_f8e4m3_sr(const float * __restrict__ x, block_f8e4m3 * __restrict__ y) {
#if defined(RDNA4) // arch-guard: k_quantize_weight_f8e4m3_sr
    const int64_t c   = blockIdx.x;
    const int     tid = threadIdx.x;

    __shared__ float sh_val[SR_QUANT_BLOCK];
    __shared__ float sh_scale;

    const float v = x[c * SR_QUANT_BLOCK + tid];
    sh_val[tid] = fabsf(v);
    __syncthreads();

    if (tid == 0) {
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < SR_QUANT_BLOCK; ++i) amax = fmaxf(amax, sh_val[i]);
        const float d = amax / 448.0f; // e4m3 max finite magnitude, same convention as quantize_row_f8e4m3_ref
        sh_scale = d;
        y[c].d = __float2half(d);
    }
    __syncthreads();

    const float d  = sh_scale;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const float scaled = v * id;

    const uint64_t gid = (uint64_t) c * SR_QUANT_BLOCK + tid;
    const uint32_t rng = sr_seed_hash((uint32_t)(__builtin_readcyclecounter() * (gid + 1)));
    const int packed = __builtin_amdgcn_cvt_sr_fp8_f32(scaled, rng, 0, 0);
    y[c].qs[tid] = (uint8_t)(packed & 0xFF);
#else
    NO_DEVICE_CODE;
#endif // arch-guard
}

__global__ void k_quantize_weight_f8e5m2_sr(const float * __restrict__ x, block_f8e5m2 * __restrict__ y) {
#if defined(RDNA4) // arch-guard: k_quantize_weight_f8e5m2_sr
    const int64_t c   = blockIdx.x;
    const int     tid = threadIdx.x;

    __shared__ float sh_val[SR_QUANT_BLOCK];
    __shared__ float sh_scale;

    const float v = x[c * SR_QUANT_BLOCK + tid];
    sh_val[tid] = fabsf(v);
    __syncthreads();

    if (tid == 0) {
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < SR_QUANT_BLOCK; ++i) amax = fmaxf(amax, sh_val[i]);
        const float d = amax / 57344.0f; // e5m2 max finite magnitude, same convention as quantize_row_f8e5m2_ref
        sh_scale = d;
        y[c].d = __float2half(d);
    }
    __syncthreads();

    const float d  = sh_scale;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const float scaled = v * id;

    const uint64_t gid = (uint64_t) c * SR_QUANT_BLOCK + tid;
    const uint32_t rng = sr_seed_hash((uint32_t)(__builtin_readcyclecounter() * (gid + 1)));
    const int packed = __builtin_amdgcn_cvt_sr_bf8_f32(scaled, rng, 0, 0);
    y[c].qs[tid] = (uint8_t)(packed & 0xFF);
#else
    NO_DEVICE_CODE;
#endif // arch-guard
}

void ggml_cuda_quantize_weight_f8e4m3_sr(ggml_backend_cuda_context & ctx, const float * d_src, void * d_dst, int64_t n) {
    GGML_ASSERT(n % SR_QUANT_BLOCK == 0);
    const int64_t nb = n / SR_QUANT_BLOCK;
    k_quantize_weight_f8e4m3_sr<<<(unsigned) nb, SR_QUANT_BLOCK, 0, ctx.stream()>>>(d_src, (block_f8e4m3 *) d_dst);
}

void ggml_cuda_quantize_weight_f8e5m2_sr(ggml_backend_cuda_context & ctx, const float * d_src, void * d_dst, int64_t n) {
    GGML_ASSERT(n % SR_QUANT_BLOCK == 0);
    const int64_t nb = n / SR_QUANT_BLOCK;
    k_quantize_weight_f8e5m2_sr<<<(unsigned) nb, SR_QUANT_BLOCK, 0, ctx.stream()>>>(d_src, (block_f8e5m2 *) d_dst);
}

// ---------------------------------------------------------------------
// The decisive bias/RMS test. Launches the raw kernels directly on the
// default stream (stream 0) -- same bypass-the-ggml_tensor-machinery style
// as every other selftest in this codebase (e.g.
// ggml_cuda_mul_mat_vec_iu4_selftest) -- no ggml_backend_cuda_context
// needed for a standalone measurement like this.
// ---------------------------------------------------------------------
namespace ggml_cuda_fp8_sr_quant_detail {

struct Stats {
    double bias_sum = 0.0, sq_sum = 0.0;
    int64_t n = 0;
    bool range_ok = true;
    void add(const std::vector<float> & y, const std::vector<float> & x, float max_mag) {
        for (size_t i = 0; i < y.size(); i++) {
            if (!std::isfinite(y[i]) || std::fabs(y[i]) > max_mag * 1.0001f) range_ok = false;
            double diff = (double) y[i] - (double) x[i];
            bias_sum += diff;
            sq_sum   += diff * diff;
        }
        n += (int64_t) y.size();
    }
    double mean_bias() const { return n ? bias_sum / (double) n : 0.0; }
    double rms() const { return n ? std::sqrt(sq_sum / (double) n) : 0.0; }
};

// Runs the SR/RTN bias-vs-error comparison for one fp8 format. Returns
// false only on a hard infra failure (hipMalloc/kernel launch); the
// bias/error verdict itself is reported via GGML_LOG_INFO regardless of
// "pass/fail" since this is a measurement, not a pass/fail correctness gate
// (the correctness gate is `range_ok`, checked and asserted separately).
template <typename BlockT, typename SrKernel, typename RtnRef, typename Dequant>
static bool run_format(const char * name, float max_mag, SrKernel sr_kernel, RtnRef rtn_ref, Dequant dequant) {
    const int64_t N       = 1 << 16; // 65536 elements/draw = 2048 blocks of 32
    const int     n_draws = 20;      // "many random draws" per the task -- ~1.3M samples aggregate

    float * d_x = nullptr;
    BlockT * d_y = nullptr;
    if (hipMalloc(&d_x, N * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_y, (N / SR_QUANT_BLOCK) * sizeof(BlockT)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        (void) hipFree(d_x); (void) hipFree(d_y);
        return false;
    }

    Stats sr_stats, rtn_stats;
    std::mt19937 rng(2026);
    // Weight-like magnitude distribution: real transformer weights are
    // roughly unit-scale-normalized per block by construction (that's what
    // the amax/max_mag scale does) -- a standard normal draw exercises the
    // full fractional-position range across the format's representable
    // grid the same way real weights would after per-block scaling, so the
    // bias measurement is scale-agnostic (the SAME conclusion holds
    // whatever the pre-scale magnitude was).
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float>  x(N), y_sr(N), y_rtn(N);
    std::vector<BlockT> h_sr(N / SR_QUANT_BLOCK), h_rtn(N / SR_QUANT_BLOCK);

    bool ok = true;
    for (int draw = 0; draw < n_draws; draw++) {
        for (auto & v : x) v = nd(rng);

        CUDA_CHECK(hipMemcpy(d_x, x.data(), N * sizeof(float), hipMemcpyHostToDevice));
        sr_kernel<<<(unsigned) (N / SR_QUANT_BLOCK), SR_QUANT_BLOCK, 0, 0>>>(d_x, d_y);
        const hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            GGML_LOG_ERROR("%s: SR kernel failed: %s\n", __func__, hipGetErrorString(err));
            ok = false;
            break;
        }
        CUDA_CHECK(hipMemcpy(h_sr.data(), d_y, h_sr.size() * sizeof(BlockT), hipMemcpyDeviceToHost));
        dequant(h_sr.data(), y_sr.data(), N);
        sr_stats.add(y_sr, x, max_mag);

        rtn_ref(x.data(), h_rtn.data(), N);
        dequant(h_rtn.data(), y_rtn.data(), N);
        rtn_stats.add(y_rtn, x, max_mag);
    }

    (void) hipFree(d_x);
    (void) hipFree(d_y);

    if (!ok) return false;

    GGML_LOG_INFO("%s (%s): SR   mean_bias=%+.6f  rms_err=%.6f  range_ok=%s\n",
                   __func__, name, sr_stats.mean_bias(), sr_stats.rms(), sr_stats.range_ok ? "yes" : "NO");
    GGML_LOG_INFO("%s (%s): RTN  mean_bias=%+.6f  rms_err=%.6f  range_ok=%s\n",
                   __func__, name, rtn_stats.mean_bias(), rtn_stats.rms(), rtn_stats.range_ok ? "yes" : "NO");
    GGML_LOG_INFO("%s (%s): |bias| ratio RTN/SR = %.2fx   rms ratio SR/RTN = %.3fx\n",
                   __func__, name,
                   sr_stats.mean_bias() != 0.0 ? std::fabs(rtn_stats.mean_bias() / sr_stats.mean_bias()) : 0.0,
                   rtn_stats.rms() != 0.0 ? sr_stats.rms() / rtn_stats.rms() : 0.0);

    if (!sr_stats.range_ok || !rtn_stats.range_ok) {
        GGML_LOG_ERROR("%s (%s): CORRECTNESS FAILURE -- out-of-range or non-finite dequantized value\n", __func__, name);
        return false;
    }
    return true;
}

// The Gaussian sweep above (broad, smooth, symmetric distribution -- a
// realistic weight-tensor proxy) turned out to show near-zero mean bias for
// BOTH SR and RTN (see the file-header discussion appended after the first
// measurement run): for a smoothly/symmetrically distributed source, a
// value's fractional position within its local grid cell is itself roughly
// uniform across many elements, so RTN's per-element rounding errors
// largely cancel in aggregate too -- the classic "RTN is biased" case is
// really about a *fixed or narrowly-clustered* value being quantized
// repeatedly (RTN rounds the SAME direction every single time; SR
// averages toward the truth). This second test isolates exactly that
// textbook scenario directly, bypassing per-block scaling entirely (raw
// scalar ggml_fp32_to_e4m3/e5m2 CPU refs, id==1 semantics on the SR
// kernel side) so the mechanism itself is demonstrated cleanly.
__global__ void k_sr_fp8_repeated(float val, int * __restrict__ out_byte, uint64_t salt) {
#if defined(RDNA4) // arch-guard: k_sr_fp8_repeated
    const uint32_t rng = sr_seed_hash((uint32_t)(__builtin_readcyclecounter() * (salt + 1)));
    const int packed = __builtin_amdgcn_cvt_sr_fp8_f32(val, rng, 0, 0);
    *out_byte = packed & 0xFF;
#else
    NO_DEVICE_CODE;
#endif // arch-guard
}
__global__ void k_sr_bf8_repeated(float val, int * __restrict__ out_byte, uint64_t salt) {
#if defined(RDNA4) // arch-guard: k_sr_bf8_repeated
    const uint32_t rng = sr_seed_hash((uint32_t)(__builtin_readcyclecounter() * (salt + 1)));
    const int packed = __builtin_amdgcn_cvt_sr_bf8_f32(val, rng, 0, 0);
    *out_byte = packed & 0xFF;
#else
    NO_DEVICE_CODE;
#endif // arch-guard
}

static bool run_repeated_constant_test(
        const char * name, float val,
        void (*sr_kernel)(float, int *, uint64_t),
        float (*to_fp32)(uint8_t)) {
    const int n_draws = 500;
    int * d_out = nullptr;
    if (hipMalloc(&d_out, sizeof(int)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    double sr_sum = 0.0;
    for (int i = 0; i < n_draws; i++) {
        sr_kernel<<<1, 1, 0, 0>>>(val, d_out, (uint64_t) i * 2654435761ull + 1);
        const hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            GGML_LOG_ERROR("%s: kernel failed: %s\n", __func__, hipGetErrorString(err));
            (void) hipFree(d_out);
            return false;
        }
        int byte = 0;
        CUDA_CHECK(hipMemcpy(&byte, d_out, sizeof(int), hipMemcpyDeviceToHost));
        sr_sum += (double) to_fp32((uint8_t) byte);
    }
    (void) hipFree(d_out);
    const double sr_mean = sr_sum / n_draws;
    const double sr_bias = sr_mean - (double) val;
    GGML_LOG_INFO("%s (%s, repeated-constant x=%.6f, %d SR draws): SR avg_bias=%+.6f\n",
                   __func__, name, val, n_draws, sr_bias);
    return true;
}

} // namespace ggml_cuda_fp8_sr_quant_detail

bool ggml_cuda_fp8_sr_quant_bias_test() {
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        return true; // vacuously true off RDNA4 -- cvt_sr_fp8_f32/cvt_sr_bf8_f32 are gfx1201-specific
    }

    using namespace ggml_cuda_fp8_sr_quant_detail;

    bool ok_e4m3 = run_format<block_f8e4m3>(
        "F8E4M3", 448.0f,
        k_quantize_weight_f8e4m3_sr,
        quantize_row_f8e4m3_ref,
        dequantize_row_f8e4m3);

    bool ok_e5m2 = run_format<block_f8e5m2>(
        "F8E5M2", 57344.0f,
        k_quantize_weight_f8e5m2_sr,
        quantize_row_f8e5m2_ref,
        dequantize_row_f8e5m2);

    // Repeated-constant textbook case (see comment at k_sr_fp8_repeated):
    // x chosen to sit closer to the LOWER of its two neighboring e4m3/e5m2
    // grid points, so RTN deterministically rounds the same direction
    // every time (a real, nonzero, reproducible bias), while SR's average
    // over many independent draws should converge toward x itself.
    const float x_e4m3 = 1.06f;  // e4m3 grid near 1.0 steps by 1/8=0.125 -> nearest points {1.0, 1.125}, closer to 1.0
    const float x_e5m2 = 1.10f;  // e5m2 grid near 1.0 steps by 1/4=0.25  -> nearest points {1.0, 1.25},  closer to 1.0
    const float rtn_e4m3 = ggml_e4m3_to_fp32(ggml_fp32_to_e4m3(x_e4m3));
    const float rtn_e5m2 = ggml_e5m2_to_fp32(ggml_fp32_to_e5m2(x_e5m2));
    GGML_LOG_INFO("%s: F8E4M3 repeated-constant x=%.6f: RTN result=%.6f RTN_bias=%+.6f (deterministic, every call)\n",
                   __func__, x_e4m3, rtn_e4m3, rtn_e4m3 - x_e4m3);
    bool ok_rep4 = run_repeated_constant_test("F8E4M3", x_e4m3, k_sr_fp8_repeated, ggml_e4m3_to_fp32);
    GGML_LOG_INFO("%s: F8E5M2 repeated-constant x=%.6f: RTN result=%.6f RTN_bias=%+.6f (deterministic, every call)\n",
                   __func__, x_e5m2, rtn_e5m2, rtn_e5m2 - x_e5m2);
    bool ok_rep5 = run_repeated_constant_test("F8E5M2", x_e5m2, k_sr_bf8_repeated, ggml_e5m2_to_fp32);

    return ok_e4m3 && ok_e5m2 && ok_rep4 && ok_rep5;
}
