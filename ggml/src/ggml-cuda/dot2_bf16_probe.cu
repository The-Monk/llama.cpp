// gfx1201 ISA-coverage follow-up (T186): V_DOT2_F32_BF16 driver-completeness
// self-test. See dot2_bf16_probe.cuh for the full rationale/provenance.
//
// Method: 32 independent lanes, each lane gets its own random bf16 pair
// {a0,a1}/{b0,b1} plus a random fp32 partial accumulator `c0`, and computes
// `__builtin_amdgcn_fdot2_f32_bf16({a0,a1},{b0,b1},c0,false)`. Reference is
// computed on the host from the SAME bf16-rounded values (ggml_compute_
// fp32_to_bf16/ggml_compute_bf16_to_fp32, ggml-impl.h -- the exact codec
// ggml's own BF16 type uses) as `c0 + bf16(a0)*bf16(b0) + bf16(a1)*bf16(b1)`,
// so any residual error is real hardware/precision behavior, not an
// approximation of the round-trip.
#include "dot2_bf16_probe.cuh"
#include "ggml-impl.h" // ggml_compute_fp32_to_bf16 / ggml_compute_bf16_to_fp32 (host)

#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

#if defined(GGML_USE_HIP)

namespace ggml_cuda_dot2_bf16 {

typedef short v2s __attribute__((ext_vector_type(2)));

static __global__ void k_dot2_bf16_perlane(const v2s * __restrict__ a, const v2s * __restrict__ b,
                                            const float * __restrict__ c, float * __restrict__ dout) {
#if defined(RDNA4)
    dout[threadIdx.x] = __builtin_amdgcn_fdot2_f32_bf16(a[threadIdx.x], b[threadIdx.x], c[threadIdx.x], false);
#else
    // Host pass / non-RDNA4 device pass: V_DOT2_F32_BF16 doesn't exist here.
    // Never actually launched off RDNA4 (ggml_cuda_dot2_bf16_selftest()
    // runtime-gates on cc first).
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    GGML_UNUSED(c);
    GGML_UNUSED(dout);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

static bool run_trial(std::mt19937 & rng, double * max_abs_err) {
    std::uniform_real_distribution<float> valdist(-8.0f, 8.0f);

    std::vector<v2s>   a_arg(32), b_arg(32);
    std::vector<float> c_arg(32);
    float a_ref[32][2], b_ref[32][2]; // host bf16-rounded values, for the reference sum
    float c_ref[32];

    for (int l = 0; l < 32; ++l) {
        const ggml_bf16_t a0 = ggml_compute_fp32_to_bf16(valdist(rng));
        const ggml_bf16_t a1 = ggml_compute_fp32_to_bf16(valdist(rng));
        const ggml_bf16_t b0 = ggml_compute_fp32_to_bf16(valdist(rng));
        const ggml_bf16_t b1 = ggml_compute_fp32_to_bf16(valdist(rng));
        const float c0 = valdist(rng);

        a_arg[l] = { (short) a0.bits, (short) a1.bits };
        b_arg[l] = { (short) b0.bits, (short) b1.bits };
        c_arg[l] = c0;

        a_ref[l][0] = ggml_compute_bf16_to_fp32(a0);
        a_ref[l][1] = ggml_compute_bf16_to_fp32(a1);
        b_ref[l][0] = ggml_compute_bf16_to_fp32(b0);
        b_ref[l][1] = ggml_compute_bf16_to_fp32(b1);
        c_ref[l] = c0;
    }

    v2s * d_a = nullptr; v2s * d_b = nullptr; float * d_c = nullptr; float * d_out = nullptr;
    if (hipMalloc(&d_a, 32*sizeof(v2s)) != hipSuccess || hipMalloc(&d_b, 32*sizeof(v2s)) != hipSuccess ||
        hipMalloc(&d_c, 32*sizeof(float)) != hipSuccess || hipMalloc(&d_out, 32*sizeof(float)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    CUDA_CHECK(hipMemcpy(d_a, a_arg.data(), 32*sizeof(v2s), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_b, b_arg.data(), 32*sizeof(v2s), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_c, c_arg.data(), 32*sizeof(float), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(k_dot2_bf16_perlane, dim3(1), dim3(32), 0, 0, d_a, d_b, d_c, d_out);
    const hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        GGML_LOG_ERROR("%s: dot2_bf16 kernel failed: %s\n", __func__, hipGetErrorString(err));
        (void) hipFree(d_a); (void) hipFree(d_b); (void) hipFree(d_c); (void) hipFree(d_out);
        return false;
    }

    std::vector<float> out(32);
    CUDA_CHECK(hipMemcpy(out.data(), d_out, 32*sizeof(float), hipMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int l = 0; l < 32; ++l) {
        const double ref = (double) c_ref[l] + (double) a_ref[l][0] * (double) b_ref[l][0]
                                              + (double) a_ref[l][1] * (double) b_ref[l][1];
        const double got = (double) out[l];
        max_err = std::max(max_err, std::fabs(got - ref));
    }
    *max_abs_err = max_err;

    (void) hipFree(d_a); (void) hipFree(d_b); (void) hipFree(d_c); (void) hipFree(d_out);
    // bf16 inputs (7-bit mantissa) x bf16 inputs, fp32 accumulate: allow a
    // small tolerance for the multiply's fp32 rounding, not a bf16-sized one.
    return max_err < 1e-3;
}

static bool selftest_impl(int n_trials) {
    std::mt19937 rng(1201);
    bool all_pass = true;
    double max_abs_over_all = 0.0;
    for (int t = 0; t < n_trials; ++t) {
        double max_err = 0.0;
        if (!run_trial(rng, &max_err)) {
            all_pass = false;
        }
        max_abs_over_all = std::max(max_abs_over_all, max_err);
    }
    GGML_LOG_INFO("%s: V_DOT2_F32_BF16, %d random trials (32 lanes/trial), max_abs_err=%.6g -> %s\n",
                   __func__, n_trials, max_abs_over_all, all_pass ? "PASS" : "FAIL");
    return all_pass;
}

} // namespace ggml_cuda_dot2_bf16

// Host-pass-compiled, RUNTIME-gated entry point (same discipline as
// iu4_w4a4.cu/mxfp8_selftest.cu -- see those files' header comments for the
// full T122 incident writeup on why the whole TU must NOT be gated on the
// device-pass-only `RDNA4` macro).
bool ggml_cuda_dot2_bf16_selftest() {
    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        // Not RDNA4 -- nothing to test, vacuously true. V_DOT2_F32_BF16 does
        // not exist on other archs.
        return true;
    }
    return ggml_cuda_dot2_bf16::selftest_impl(20);
}

#else // !defined(GGML_USE_HIP)

bool ggml_cuda_dot2_bf16_selftest() { return true; }

#endif // defined(GGML_USE_HIP)
