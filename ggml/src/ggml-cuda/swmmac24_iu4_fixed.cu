// T162 int4-2:4 pivot: standalone, ISOLATED self-test for the 2:4-sparse
// iu4 SWMMAC instruction with the CK-research-derived idx-encoding fix
// applied (~/int4-research/FINDINGS.md Stage 17d, 2026-07-21): the existing
// swmmac24.cu::test_iu4_swmmac_24 (K32 only) uses the NATURAL/unfixed idx
// encoding (field i*2+0 = lo-survivor's position, field i*2+1 =
// hi-survivor's position) -- CK's from-scratch hardware validation found
// this WRONG for the iu4 sparse form specifically (fp8/fp16/int8 sparse
// forms are fine with the natural encoding; iu4 is not). The fix: SWAP
// which survivor's position goes in which field, AND XOR each with 1:
//   field i*2+0 (natural home of lo's position) <- (hi_position XOR 1)
//   field i*2+1 (natural home of hi's position) <- (lo_position XOR 1)
// This file does NOT touch swmmac24.cu (avoid risking already-shipped,
// relied-upon code) -- it is a new, standalone, opt-in-only self-test that
// reuses swmmac24.cuh's header-only per-lane layout helpers unchanged.
#include "swmmac24.cuh"

#include <random>
#include <vector>
#include <cstring>

#if defined(GGML_USE_HIP)

using namespace ggml_cuda_swmmac24;

static void gen_2to4_pattern_fixed(std::mt19937 & rng, int idx_pairs[8][2]) {
    static const int choices[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    std::uniform_int_distribution<int> pick(0, 5);
    for (int g = 0; g < 8; ++g) {
        const int c = pick(rng);
        idx_pairs[g][0] = choices[c][0]; // lo (found first in scan order)
        idx_pairs[g][1] = choices[c][1]; // hi (found second)
    }
}

// K32 sparse iu4 SWMMAC, idx-encoding-FIXED. Mirrors
// swmmac24.cu::test_iu4_swmmac_24 exactly except for the idx field
// swap+XOR1 at the two marked lines below.
static bool test_iu4_swmmac_24_fixed_k32(int n_trials) {
    std::mt19937 rng(162162);
    std::uniform_int_distribution<int> valdist(-8, 7);
    bool all_pass = true;
    long max_abs_over_all = 0;

    for (int trial = 0; trial < n_trials; ++trial) {
        int A_logical[16][32] = {{0}};
        int B_dense[32][16];
        int idx_pairs[16][8][2];

        for (int i = 0; i < 16; ++i) {
            gen_2to4_pattern_fixed(rng, idx_pairs[i]);
            for (int g = 0; g < 8; ++g) {
                A_logical[i][4*g + idx_pairs[i][g][0]] = valdist(rng);
                A_logical[i][4*g + idx_pairs[i][g][1]] = valdist(rng);
            }
        }
        for (int k = 0; k < 32; ++k) {
            for (int n = 0; n < 16; ++n) {
                B_dense[k][n] = valdist(rng);
            }
        }

        std::vector<std::vector<uint32_t>> aregs(32, std::vector<uint32_t>(1, 0));
        std::vector<std::vector<uint32_t>> idxregs(32, std::vector<uint32_t>(1, 0));
        for (int i = 0; i < 16; ++i) {
            for (int g = 0; g < 8; ++g) {
                const int cp0 = 2*g, cp1 = 2*g + 1;
                const int v0 = A_logical[i][4*g + idx_pairs[i][g][0]]; // lo's value, stored at physical col cp0
                const int v1 = A_logical[i][4*g + idx_pairs[i][g][1]]; // hi's value, stored at physical col cp1
                swmmac24_put_bits(aregs, swmmac24_a_loc(4, i, cp0), (uint32_t)(v0 & 0xF), 4);
                swmmac24_put_bits(aregs, swmmac24_a_loc(4, i, cp1), (uint32_t)(v1 & 0xF), 4);
                const Loc idxloc = swmmac24_a_loc(4, i, cp0);
                const int idxFirstBit = ((4*g >> 2) & 3) * 4;
                // FIX (swap + XOR1): field 0 <- hi_pos^1, field 1 <- lo_pos^1
                // (natural/unfixed would be field0<-lo_pos, field1<-hi_pos)
                swmmac24_put_bits(idxregs, {idxloc.lane, 0, idxFirstBit + 0}, (unsigned)(idx_pairs[i][g][1] ^ 1), 2);
                swmmac24_put_bits(idxregs, {idxloc.lane, 0, idxFirstBit + 2}, (unsigned)(idx_pairs[i][g][0] ^ 1), 2);
            }
        }
        std::vector<std::vector<uint32_t>> bregs(32, std::vector<uint32_t>(2, 0));
        for (int k = 0; k < 32; ++k) {
            for (int n = 0; n < 16; ++n) {
                swmmac24_put_bits(bregs, swmmac24_b32_loc_4bit(k, n), (uint32_t)(B_dense[k][n] & 0xF), 4);
            }
        }

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
            GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
            return false;
        }
        CUDA_CHECK(hipMemcpy(d_a, a_arg.data(), 32*sizeof(int), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_b, b_arg.data(), 32*sizeof(v2i), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_idx, idx_arg.data(), 32*sizeof(unsigned), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(k_swmmac_iu4_24_perlane, dim3(1), dim3(32), 0, 0, d_a, d_b, d_idx, d_dout);
        const hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            GGML_LOG_ERROR("%s: iu4 SWMMAC kernel failed: %s\n", __func__, hipGetErrorString(err));
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
                for (int k = 0; k < 32; ++k) { ref += (long) A_logical[i][k] * B_dense[k][j]; }
                const Loc loc = swmmac24_d_loc(i, j);
                const int32_t got = dregs[loc.lane][loc.vgpr];
                max_abs = std::max(max_abs, (long) llabs((long long) got - ref));
            }
        }
        max_abs_over_all = std::max(max_abs_over_all, max_abs);
        if (max_abs != 0) { all_pass = false; }
        (void) hipFree(d_a); (void) hipFree(d_b); (void) hipFree(d_idx); (void) hipFree(d_dout);
    }
    GGML_LOG_INFO("%s: V_SWMMAC_I32_16X16X32_IU4 2:4-sparse, IDX-FIX applied, %d random trials, max_abs_err=%ld -> %s\n",
                   __func__, n_trials, max_abs_over_all, all_pass ? "PASS" : "FAIL");
    return all_pass;
}

#endif // defined(GGML_USE_HIP)

bool ggml_cuda_swmmac24_iu4_fixed_selftest() {
#if defined(GGML_USE_HIP)
    return test_iu4_swmmac_24_fixed_k32(20);
#else
    return true;
#endif
}
