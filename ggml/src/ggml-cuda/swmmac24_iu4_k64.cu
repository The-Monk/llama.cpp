// gfx1201 ISA-coverage follow-up (T186): V_SWMMAC_I32_16X16X64_IU4
// driver-completeness self-test. See swmmac24_iu4_k64.cuh for the full
// rationale, provenance, and per-lane layout derivation notes.
#include "swmmac24_iu4_k64.cuh"

#include <random>
#include <vector>
#include <cstring>
#include <algorithm>

#if defined(GGML_USE_HIP)

using namespace ggml_cuda_swmmac24; // v2i/v4i/v8i, Loc, swmmac24_put_bits, swmmac24_d_loc

namespace ggml_cuda_swmmac24_iu4_k64 {

// A-Matrix (compressed, physical storage), K64 form. row=0..15 (M), g=0..15
// (K64/4 groups), slot=0 (lo-survivor, low nibble) / 1 (hi-survivor, high
// nibble) -- see .cuh for the calculator-derived formula provenance.
static __host__ __device__ __forceinline__ Loc a_loc_k64(int row, int g, int slot) {
    const int lane_half = g / 8;
    const int lane       = 16 * lane_half + (row & 0xF);
    const int gpr         = (g / 4) % 2;
    const int byte         = g % 4;
    return { lane, gpr, byte * 8 + slot * 4 };
}

// Sparsity-index field, K64 form. Same (row, g) addressing as a_loc_k64;
// entry=0 selects the lo-survivor's 2-bit position field, entry=1 the
// hi-survivor's -- natural (unswapped) convention, matching this driver's
// own already-hardware-validated K32 form (see .cuh provenance note).
static __host__ __device__ __forceinline__ Loc idx_loc_k64(int row, int g, int entry) {
    const int lane_half = g / 8;
    const int lane       = 16 * lane_half + (row & 0xF);
    const int g_local     = g % 8;
    return { lane, 0, 4 * g_local + entry * 2 };
}

// B-Matrix (dense), K64 form. k=0..63 (K row), n=0..15 (N col).
static __host__ __device__ __forceinline__ Loc b_loc_k64(int k, int n) {
    const int lane_half = k / 32;
    const int lane       = 16 * lane_half + (n & 0xF);
    const int k_local     = k % 32;
    const int gpr           = k_local / 8;
    return { lane, gpr, (k_local % 8) * 4 };
}

static void gen_2to4_pattern_16(std::mt19937 & rng, int idx_pairs[16][2]) {
    static const int choices[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    std::uniform_int_distribution<int> pick(0, 5);
    for (int g = 0; g < 16; ++g) {
        const int c = pick(rng);
        idx_pairs[g][0] = choices[c][0]; // lo (found first in scan order)
        idx_pairs[g][1] = choices[c][1]; // hi (found second)
    }
}

static __global__ void k_swmmac_iu4_24_k64_perlane(const v2i * __restrict__ a, const v4i * __restrict__ b,
                                              const unsigned * __restrict__ idx, v8i * __restrict__ dout) {
#if defined(RDNA4)
    v8i c = {0,0,0,0,0,0,0,0};
    dout[threadIdx.x] = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(1, a[threadIdx.x], 1, b[threadIdx.x], c, idx[threadIdx.x], 0);
#else
    // Host pass / non-RDNA4 device pass: V_SWMMAC_I32_16X16X64_IU4 doesn't
    // exist here. Never actually launched off RDNA4
    // (ggml_cuda_swmmac24_iu4_k64_selftest() runtime-gates on cc first).
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    GGML_UNUSED(idx);
    GGML_UNUSED(dout);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

static bool test_iu4_swmmac_24_k64(int n_trials) {
    std::mt19937 rng(186186);
    std::uniform_int_distribution<int> valdist(-8, 7);
    bool all_pass = true;
    long max_abs_over_all = 0;

    for (int trial = 0; trial < n_trials; ++trial) {
        int A_logical[16][64] = {{0}};
        int B_dense[64][16];
        int idx_pairs[16][16][2];

        for (int i = 0; i < 16; ++i) {
            gen_2to4_pattern_16(rng, idx_pairs[i]);
            for (int g = 0; g < 16; ++g) {
                A_logical[i][4*g + idx_pairs[i][g][0]] = valdist(rng);
                A_logical[i][4*g + idx_pairs[i][g][1]] = valdist(rng);
            }
        }
        for (int k = 0; k < 64; ++k) {
            for (int n = 0; n < 16; ++n) {
                B_dense[k][n] = valdist(rng);
            }
        }

        std::vector<std::vector<uint32_t>> aregs(32, std::vector<uint32_t>(2, 0));
        std::vector<std::vector<uint32_t>> idxregs(32, std::vector<uint32_t>(1, 0));
        for (int i = 0; i < 16; ++i) {
            for (int g = 0; g < 16; ++g) {
                const int v0 = A_logical[i][4*g + idx_pairs[i][g][0]]; // lo's value
                const int v1 = A_logical[i][4*g + idx_pairs[i][g][1]]; // hi's value
                swmmac24_put_bits(aregs, a_loc_k64(i, g, 0), (uint32_t)(v0 & 0xF), 4);
                swmmac24_put_bits(aregs, a_loc_k64(i, g, 1), (uint32_t)(v1 & 0xF), 4);
                swmmac24_put_bits(idxregs, idx_loc_k64(i, g, 0), (unsigned) idx_pairs[i][g][0], 2);
                swmmac24_put_bits(idxregs, idx_loc_k64(i, g, 1), (unsigned) idx_pairs[i][g][1], 2);
            }
        }
        std::vector<std::vector<uint32_t>> bregs(32, std::vector<uint32_t>(4, 0));
        for (int k = 0; k < 64; ++k) {
            for (int n = 0; n < 16; ++n) {
                swmmac24_put_bits(bregs, b_loc_k64(k, n), (uint32_t)(B_dense[k][n] & 0xF), 4);
            }
        }

        std::vector<v2i> a_arg(32);
        std::vector<v4i> b_arg(32);
        std::vector<unsigned> idx_arg(32);
        for (int l = 0; l < 32; ++l) {
            a_arg[l]   = { (int) aregs[l][0], (int) aregs[l][1] };
            b_arg[l]   = { (int) bregs[l][0], (int) bregs[l][1], (int) bregs[l][2], (int) bregs[l][3] };
            idx_arg[l] = idxregs[l][0];
        }

        v2i *d_a = nullptr; v4i *d_b = nullptr; unsigned *d_idx = nullptr; v8i *d_dout = nullptr;
        if (hipMalloc(&d_a, 32*sizeof(v2i)) != hipSuccess || hipMalloc(&d_b, 32*sizeof(v4i)) != hipSuccess ||
            hipMalloc(&d_idx, 32*sizeof(unsigned)) != hipSuccess || hipMalloc(&d_dout, 32*sizeof(v8i)) != hipSuccess) {
            GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
            return false;
        }
        CUDA_CHECK(hipMemcpy(d_a, a_arg.data(), 32*sizeof(v2i), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_b, b_arg.data(), 32*sizeof(v4i), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_idx, idx_arg.data(), 32*sizeof(unsigned), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(k_swmmac_iu4_24_k64_perlane, dim3(1), dim3(32), 0, 0, d_a, d_b, d_idx, d_dout);
        const hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            GGML_LOG_ERROR("%s: K64 iu4 SWMMAC kernel failed: %s\n", __func__, hipGetErrorString(err));
            (void) hipFree(d_a); (void) hipFree(d_b); (void) hipFree(d_idx); (void) hipFree(d_dout);
            return false;
        }

        std::vector<v8i> dout(32);
        CUDA_CHECK(hipMemcpy(dout.data(), d_dout, 32*sizeof(v8i), hipMemcpyDeviceToHost));
        std::vector<std::vector<int32_t>> dregs(32, std::vector<int32_t>(8, 0));
        for (int l = 0; l < 32; ++l) {
            int tmp[8]; memcpy(tmp, &dout[l], sizeof(tmp));
            for (int r = 0; r < 8; ++r) { dregs[l][r] = tmp[r]; }
        }

        long max_abs = 0;
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                long ref = 0;
                for (int k = 0; k < 64; ++k) { ref += (long) A_logical[i][k] * B_dense[k][j]; }
                const Loc loc = swmmac24_d_loc(i, j);
                const int32_t got = dregs[loc.lane][loc.vgpr];
                max_abs = std::max(max_abs, (long) llabs((long long) got - ref));
            }
        }
        max_abs_over_all = std::max(max_abs_over_all, max_abs);
        if (max_abs != 0) { all_pass = false; }
        (void) hipFree(d_a); (void) hipFree(d_b); (void) hipFree(d_idx); (void) hipFree(d_dout);
    }
    GGML_LOG_INFO("%s: V_SWMMAC_I32_16X16X64_IU4 2:4-sparse (K64), %d random trials, max_abs_err=%ld -> %s\n",
                   __func__, n_trials, max_abs_over_all, all_pass ? "PASS" : "FAIL");
    return all_pass;
}

} // namespace ggml_cuda_swmmac24_iu4_k64

// Host-pass-compiled, RUNTIME-gated entry point (same discipline as
// swmmac24.cu/swmmac24_iu4_fixed.cu -- see swmmac24.cu's T122 header comment
// for the full incident writeup on why the whole TU must NOT be gated on the
// device-pass-only `RDNA4` macro).
bool ggml_cuda_swmmac24_iu4_k64_selftest() {
    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        // Not RDNA4 -- nothing to test, vacuously true. V_SWMMAC_I32_16X16X64_IU4
        // does not exist on other archs.
        return true;
    }
    return ggml_cuda_swmmac24_iu4_k64::test_iu4_swmmac_24_k64(20);
}

#else // !defined(GGML_USE_HIP)

bool ggml_cuda_swmmac24_iu4_k64_selftest() { return true; }

#endif // defined(GGML_USE_HIP)
