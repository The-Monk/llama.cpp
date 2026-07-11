// RDNA4 2:4-structured-sparse SWMMAC driver-completeness self-test.
// See swmmac24.cuh for the full rationale, doctrine citation, and per-lane layout
// derivation/provenance notes. This file implements the runtime correctness gate:
// build a REAL 2:4-structured-sparse A + dense B (random, exercising every kept-pair
// pattern), run the actual V_SWMMAC_I32_16X16X32_IU4 / V_SWMMAC_F32_16X16X32_FP8_FP8
// hardware instructions, and compare against a host dense reference. Deliberately NOT
// a performance benchmark (per T99, every 2:4 perf angle on real models was already
// killed) -- this only proves the driver's instruction-level support is correct.
#include "swmmac24.cuh"

// T122 fix: was gated `defined(GGML_USE_HIP) && defined(RDNA4)`. `RDNA4` is
// device-pass-only (never defined in HIP-clang's host pass, on any arch) --
// this file's HOST-callable `ggml_cuda_swmmac24_selftest()` entry always
// compiled to the vacuous `#else` stub, so `GGML_HIP_SWMMAC24_SELFTEST=1`
// never actually ran the kernels, on any hardware, ever. Fix: gate the TU on
// `GGML_USE_HIP` only; the two `__global__` kernels' actual hardware
// instructions are now internally `RDNA4`-gated (swmmac24.cuh), and the
// top-level entry point below does a RUNTIME compute-capability check
// (`GGML_CUDA_CC_IS_RDNA4`, common.cuh) before dispatching to them.
#if defined(GGML_USE_HIP)

#include <hip/hip_fp8.h>
#include <random>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>

namespace ggml_cuda_swmmac24 {

using fp8e4m3 = __hip_fp8_e4m3;
static inline uint8_t fp8_bits(float v) { fp8e4m3 f(v); return f.__x; }
static inline float   fp8_to_f32(uint8_t bits) { fp8e4m3 f; f.__x = bits; return (float) f; }

static void gen_2to4_pattern(std::mt19937 & rng, int idx_pairs[8][2]) {
    static const int choices[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    std::uniform_int_distribution<int> pick(0, 5);
    for (int g = 0; g < 8; ++g) {
        const int c = pick(rng);
        idx_pairs[g][0] = choices[c][0];
        idx_pairs[g][1] = choices[c][1];
    }
}

static bool test_iu4_swmmac_24(int n_trials) {
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> valdist(-8, 7);
    bool all_pass = true;
    long max_abs_over_all = 0;

    for (int trial = 0; trial < n_trials; ++trial) {
        int A_logical[16][32] = {{0}};
        int B_dense[32][16];
        int idx_pairs[16][8][2];

        for (int i = 0; i < 16; ++i) {
            gen_2to4_pattern(rng, idx_pairs[i]);
            for (int g = 0; g < 8; ++g) {
                A_logical[i][4*g + idx_pairs[i][g][0]] = valdist(rng);
                A_logical[i][4*g + idx_pairs[i][g][1]] = valdist(rng);
            }
        }
        for (int k = 0; k < 32; ++k)
            for (int n = 0; n < 16; ++n)
                B_dense[k][n] = valdist(rng);

        std::vector<std::vector<uint32_t>> aregs(32, std::vector<uint32_t>(1, 0));
        std::vector<std::vector<uint32_t>> idxregs(32, std::vector<uint32_t>(1, 0));
        for (int i = 0; i < 16; ++i) {
            for (int g = 0; g < 8; ++g) {
                const int cp0 = 2*g, cp1 = 2*g + 1;
                const int v0 = A_logical[i][4*g + idx_pairs[i][g][0]];
                const int v1 = A_logical[i][4*g + idx_pairs[i][g][1]];
                swmmac24_put_bits(aregs, swmmac24_a_loc(4, i, cp0), (uint32_t)(v0 & 0xF), 4);
                swmmac24_put_bits(aregs, swmmac24_a_loc(4, i, cp1), (uint32_t)(v1 & 0xF), 4);
                const Loc idxloc = swmmac24_a_loc(4, i, cp0);
                const int idxFirstBit = ((4*g >> 2) & 3) * 4;
                swmmac24_put_bits(idxregs, {idxloc.lane, 0, idxFirstBit + 0}, idx_pairs[i][g][0], 2);
                swmmac24_put_bits(idxregs, {idxloc.lane, 0, idxFirstBit + 2}, idx_pairs[i][g][1], 2);
            }
        }
        std::vector<std::vector<uint32_t>> bregs(32, std::vector<uint32_t>(2, 0));
        for (int k = 0; k < 32; ++k)
            for (int n = 0; n < 16; ++n)
                swmmac24_put_bits(bregs, swmmac24_b32_loc_4bit(k, n), (uint32_t)(B_dense[k][n] & 0xF), 4);

        std::vector<int> a_arg(32);
        std::vector<v2i> b_arg(32);
        std::vector<unsigned> idx_arg(32);
        for (int l = 0; l < 32; ++l) {
            a_arg[l]   = (int) aregs[l][0];
            b_arg[l]   = { (int) bregs[l][0], (int) bregs[l][1] };
            idx_arg[l] = idxregs[l][0];
        }

        int *d_a = nullptr; v2i *d_b = nullptr; unsigned *d_idx = nullptr; v8i *d_dout = nullptr;
        if (hipMalloc(&d_a, 32*sizeof(int)) != hipSuccess || hipMalloc(&d_b, 32*sizeof(v2i)) != hipSuccess ||
            hipMalloc(&d_idx, 32*sizeof(unsigned)) != hipSuccess || hipMalloc(&d_dout, 32*sizeof(v8i)) != hipSuccess) {
            GGML_LOG_ERROR("%s: iu4 self-test hipMalloc failed\n", __func__);
            return false;
        }
        CUDA_CHECK(hipMemcpy(d_a, a_arg.data(), 32*sizeof(int), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_b, b_arg.data(), 32*sizeof(v2i), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_idx, idx_arg.data(), 32*sizeof(unsigned), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(k_swmmac_iu4_24_perlane, dim3(1), dim3(32), 0, 0, d_a, d_b, d_idx, d_dout);
        const hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            GGML_LOG_ERROR("%s: iu4 SWMMAC kernel failed: %s\n", __func__, hipGetErrorString(err));
            CUDA_CHECK(hipFree(d_a)); CUDA_CHECK(hipFree(d_b)); CUDA_CHECK(hipFree(d_idx)); CUDA_CHECK(hipFree(d_dout));
            return false;
        }

        std::vector<v8i> dout(32);
        CUDA_CHECK(hipMemcpy(dout.data(), d_dout, 32*sizeof(v8i), hipMemcpyDeviceToHost));
        std::vector<std::vector<int32_t>> dregs(32, std::vector<int32_t>(8, 0));
        for (int l = 0; l < 32; ++l) {
            int tmp[8]; memcpy(tmp, &dout[l], sizeof(tmp));
            for (int r = 0; r < 8; ++r) dregs[l][r] = tmp[r];
        }

        long max_abs = 0;
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                long ref = 0;
                for (int k = 0; k < 32; ++k) ref += (long) A_logical[i][k] * B_dense[k][j];
                const Loc loc = swmmac24_d_loc(i, j);
                const int32_t got = dregs[loc.lane][loc.vgpr];
                max_abs = std::max(max_abs, (long) llabs((long long) got - ref));
            }
        }
        max_abs_over_all = std::max(max_abs_over_all, max_abs);
        if (max_abs != 0) all_pass = false;
        CUDA_CHECK(hipFree(d_a)); CUDA_CHECK(hipFree(d_b)); CUDA_CHECK(hipFree(d_idx)); CUDA_CHECK(hipFree(d_dout));
    }
    GGML_LOG_INFO("%s: V_SWMMAC_I32_16X16X32_IU4 2:4-sparse, %d random trials, max_abs_err=%ld -> %s\n",
                   __func__, n_trials, max_abs_over_all, all_pass ? "PASS" : "FAIL");
    return all_pass;
}

static bool test_fp8_swmmac_24(int n_trials) {
    std::mt19937 rng(5678);
    std::uniform_real_distribution<float> valdist(-3.0f, 3.0f);
    bool all_pass = true;
    double max_rel_over_all = 0;

    for (int trial = 0; trial < n_trials; ++trial) {
        float A_logical[16][32] = {{0}};
        float B_dense[32][16];
        int idx_pairs[16][8][2];

        for (int i = 0; i < 16; ++i) {
            gen_2to4_pattern(rng, idx_pairs[i]);
            for (int g = 0; g < 8; ++g) {
                A_logical[i][4*g + idx_pairs[i][g][0]] = valdist(rng);
                A_logical[i][4*g + idx_pairs[i][g][1]] = valdist(rng);
            }
        }
        for (int k = 0; k < 32; ++k)
            for (int n = 0; n < 16; ++n)
                B_dense[k][n] = valdist(rng);

        std::vector<std::vector<uint32_t>> aregs(32, std::vector<uint32_t>(2, 0));
        std::vector<std::vector<uint32_t>> idxregs(32, std::vector<uint32_t>(1, 0));
        for (int i = 0; i < 16; ++i) {
            for (int g = 0; g < 8; ++g) {
                const int cp0 = 2*g, cp1 = 2*g + 1;
                const float v0 = A_logical[i][4*g + idx_pairs[i][g][0]];
                const float v1 = A_logical[i][4*g + idx_pairs[i][g][1]];
                swmmac24_put_bits(aregs, swmmac24_a_loc(8, i, cp0), fp8_bits(v0), 8);
                swmmac24_put_bits(aregs, swmmac24_a_loc(8, i, cp1), fp8_bits(v1), 8);
                const Loc idxloc = swmmac24_a_loc(8, i, cp0);
                const int idxFirstBit = ((4*g >> 2) & 3) * 4;
                swmmac24_put_bits(idxregs, {idxloc.lane, 0, idxFirstBit + 0}, idx_pairs[i][g][0], 2);
                swmmac24_put_bits(idxregs, {idxloc.lane, 0, idxFirstBit + 2}, idx_pairs[i][g][1], 2);
            }
        }
        std::vector<std::vector<uint32_t>> bregs(32, std::vector<uint32_t>(4, 0));
        for (int k = 0; k < 32; ++k)
            for (int n = 0; n < 16; ++n)
                swmmac24_put_bits(bregs, swmmac24_b32_loc_8bit(k, n), fp8_bits(B_dense[k][n]), 8);

        std::vector<v2i> a_arg(32);
        std::vector<v4i> b_arg(32);
        std::vector<unsigned> idx_arg(32);
        for (int l = 0; l < 32; ++l) {
            a_arg[l]   = { (int) aregs[l][0], (int) aregs[l][1] };
            b_arg[l]   = { (int) bregs[l][0], (int) bregs[l][1], (int) bregs[l][2], (int) bregs[l][3] };
            idx_arg[l] = idxregs[l][0];
        }

        v2i *d_a = nullptr; v4i *d_b = nullptr; unsigned *d_idx = nullptr; v8f *d_dout = nullptr;
        if (hipMalloc(&d_a, 32*sizeof(v2i)) != hipSuccess || hipMalloc(&d_b, 32*sizeof(v4i)) != hipSuccess ||
            hipMalloc(&d_idx, 32*sizeof(unsigned)) != hipSuccess || hipMalloc(&d_dout, 32*sizeof(v8f)) != hipSuccess) {
            GGML_LOG_ERROR("%s: fp8 self-test hipMalloc failed\n", __func__);
            return false;
        }
        CUDA_CHECK(hipMemcpy(d_a, a_arg.data(), 32*sizeof(v2i), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_b, b_arg.data(), 32*sizeof(v4i), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_idx, idx_arg.data(), 32*sizeof(unsigned), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(k_swmmac_fp8_24_perlane, dim3(1), dim3(32), 0, 0, d_a, d_b, d_idx, d_dout);
        const hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            GGML_LOG_ERROR("%s: fp8 SWMMAC kernel failed: %s\n", __func__, hipGetErrorString(err));
            CUDA_CHECK(hipFree(d_a)); CUDA_CHECK(hipFree(d_b)); CUDA_CHECK(hipFree(d_idx)); CUDA_CHECK(hipFree(d_dout));
            return false;
        }

        std::vector<v8f> dout(32);
        CUDA_CHECK(hipMemcpy(dout.data(), d_dout, 32*sizeof(v8f), hipMemcpyDeviceToHost));

        double max_rel = 0;
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                float ref = 0;
                for (int k = 0; k < 32; ++k) {
                    const float av = fp8_to_f32(fp8_bits(A_logical[i][k])); // 0 for structurally-dropped k
                    const float bv = fp8_to_f32(fp8_bits(B_dense[k][j]));
                    ref += av * bv;
                }
                const Loc loc = swmmac24_d_loc(i, j);
                float tmp[8]; memcpy(tmp, &dout[loc.lane], sizeof(tmp));
                const float got = tmp[loc.vgpr];
                const double rel = fabs((double) got - (double) ref) / fmax(1.0, fabs((double) ref));
                max_rel = fmax(max_rel, rel);
            }
        }
        max_rel_over_all = fmax(max_rel_over_all, max_rel);
        if (max_rel > 0.01) all_pass = false;
        CUDA_CHECK(hipFree(d_a)); CUDA_CHECK(hipFree(d_b)); CUDA_CHECK(hipFree(d_idx)); CUDA_CHECK(hipFree(d_dout));
    }
    GGML_LOG_INFO("%s: V_SWMMAC_F32_16X16X32_FP8_FP8 2:4-sparse, %d random trials, max_rel_err=%.6f -> %s\n",
                   __func__, n_trials, max_rel_over_all, all_pass ? "PASS" : "FAIL");
    return all_pass;
}

} // namespace ggml_cuda_swmmac24

// Host-pass-compiled, RUNTIME-gated entry point (T122 fix -- see the header
// comment above). `ggml_cuda_get_device()`/`ggml_cuda_info()` are plain host
// functions -- safe to call unconditionally here since this function itself
// has no __device__ qualifier and is therefore only ever compiled for the
// host target.
bool ggml_cuda_swmmac24_selftest() {
    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        // Not RDNA4 -- nothing to test, vacuously true. The 2:4 SWMMAC
        // instructions this file targets do not exist on other archs (see
        // wiki/tech/phase2-lever-validation.md ISA real-vs-mirage table).
        return true;
    }
    GGML_LOG_INFO("%s: RDNA4 2:4-sparse SWMMAC driver-completeness self-test starting "
                   "(dormant capability, T99: not used by any model path -- correctness-only gate)\n", __func__);
    const bool ok_iu4 = ggml_cuda_swmmac24::test_iu4_swmmac_24(20);
    const bool ok_fp8 = ggml_cuda_swmmac24::test_fp8_swmmac_24(20);
    const bool ok = ok_iu4 && ok_fp8;
    GGML_LOG_INFO("%s: overall result: %s\n", __func__, ok ? "PASS" : "FAIL");
    return ok;
}

#else // !defined(GGML_USE_HIP)

bool ggml_cuda_swmmac24_selftest() {
    // Not a HIP build -- nothing to test, vacuously true. The 2:4 SWMMAC
    // instructions this file targets are AMD/RDNA4-only.
    return true;
}

#endif // defined(GGML_USE_HIP)
