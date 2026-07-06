// T89: native iu4 x iu4 W4A4 RDNA4 WMMA driver-completeness self-test.
// See iu4_w4a4.cuh for the full rationale/doctrine citation.
//
// Method: build a REAL dense int4 A (M=16xK=32) and B (N=16xK=32), signed
// range [-8,7], load them into ggml_cuda_mma::tile<16,4,int> operands via the
// file's own `load_generic` (same convention every other quant path uses),
// call the new `ggml_cuda_mma::mma_iu4()` (mma.cuh) which dispatches the real
// `__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12` hardware instruction, and
// compare the tile<16,16,int> accumulator (read back via the SAME
// `tile_C::get_i/get_j` convention production mmq.cuh code uses) against an
// exact host int32 reference dot-product. int4 arithmetic has NO rounding, so
// PASS requires an EXACT match (max_abs_err == 0), not "close enough".
//
// Packing convention: for row r, physical 32-bit word j (0..3), 8 signed
// 4-bit values are packed 2-per-byte (low nibble = even k, high nibble = odd
// k within that byte), little-endian across the 4 bytes of the int32 word,
// covering logical k = [8*j, 8*j+7]. This is an internal, self-consistent
// convention -- the reference sum is computed against the SAME encoding used
// to build the device buffers, so correctness is provable regardless of
// whatever internal lane/bit convention the real V_WMMA_I32_16X16X32_IU4
// instruction uses for K-reduction: a dot product's value is invariant to
// which physical slot each logical k is stored in, as long as A and B agree
// on the same slot<->k mapping (guaranteed here, both packed identically) --
// this is a well-defined systolic MAC, not order-sensitive floating point.
#include "iu4_w4a4.cuh"

#if defined(GGML_USE_HIP) && defined(RDNA4)

#include "common.cuh"
#include "mma.cuh"

#include <random>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

namespace ggml_cuda_iu4_w4a4 {

using namespace ggml_cuda_mma;

// Pack 32 signed int4 values (range [-8,7]) into 4 physical int32 words
// following the convention documented above.
static void pack_row_i4(const int vals[32], int phys[4]) {
    for (int j = 0; j < 4; ++j) {
        uint32_t word = 0;
        for (int b = 0; b < 4; ++b) {
            const int k_lo = 8*j + 2*b;
            const int k_hi = 8*j + 2*b + 1;
            const uint32_t nlo = (uint32_t)(vals[k_lo] & 0xF);
            const uint32_t nhi = (uint32_t)(vals[k_hi] & 0xF);
            word |= (nlo | (nhi << 4)) << (8*b);
        }
        phys[j] = (int) word;
    }
}

__global__ void k_iu4_dense_test(const int * __restrict__ dA, const int * __restrict__ dB, int * __restrict__ dOut) {
    tile<16, 4, int> A;
    tile<16, 4, int> B;
    load_generic(A, dA, 4);
    load_generic(B, dB, 4);

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
        dOut[i*16 + j] = D.x[l];
    }
}

static bool run_trial(std::mt19937 & rng) {
    std::uniform_int_distribution<int> valdist(-8, 7);

    int A_logical[16][32];
    int B_logical[16][32];
    for (int i = 0; i < 16; ++i) {
        for (int k = 0; k < 32; ++k) {
            A_logical[i][k] = valdist(rng);
            B_logical[i][k] = valdist(rng);
        }
    }

    int A_phys[16][4];
    int B_phys[16][4];
    for (int i = 0; i < 16; ++i) {
        pack_row_i4(A_logical[i], A_phys[i]);
        pack_row_i4(B_logical[i], B_phys[i]);
    }

    int * dA = nullptr;
    int * dB = nullptr;
    int * dOut = nullptr;
    if (hipMalloc(&dA, 16*4*sizeof(int)) != hipSuccess ||
        hipMalloc(&dB, 16*4*sizeof(int)) != hipSuccess ||
        hipMalloc(&dOut, 16*16*sizeof(int)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    CUDA_CHECK(hipMemcpy(dA, A_phys, 16*4*sizeof(int), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(dB, B_phys, 16*4*sizeof(int), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(dOut, 0, 16*16*sizeof(int)));

    hipLaunchKernelGGL(k_iu4_dense_test, dim3(1), dim3(32), 0, 0, dA, dB, dOut);
    const hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        GGML_LOG_ERROR("%s: iu4 W4A4 dense kernel failed: %s\n", __func__, hipGetErrorString(err));
        hipFree(dA); hipFree(dB); hipFree(dOut);
        return false;
    }

    std::vector<int> out(16*16);
    CUDA_CHECK(hipMemcpy(out.data(), dOut, 16*16*sizeof(int), hipMemcpyDeviceToHost));

    long max_abs = 0;
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 16; ++n) {
            long ref = 0;
            for (int k = 0; k < 32; ++k) {
                ref += (long) A_logical[m][k] * (long) B_logical[n][k];
            }
            const long got = out[m*16 + n];
            max_abs = std::max(max_abs, std::labs(got - ref));
        }
    }

    hipFree(dA); hipFree(dB); hipFree(dOut);
    return max_abs == 0;
}

static bool selftest_impl(int n_trials) {
    std::mt19937 rng(4242);
    bool all_pass = true;
    for (int t = 0; t < n_trials; ++t) {
        if (!run_trial(rng)) {
            all_pass = false;
        }
    }
    GGML_LOG_INFO("%s: V_WMMA_I32_16X16X32_IU4 dense W4A4, %d random trials -> %s\n",
                   __func__, n_trials, all_pass ? "PASS" : "FAIL");
    return all_pass;
}

} // namespace ggml_cuda_iu4_w4a4

bool ggml_cuda_iu4_w4a4_selftest() {
    return ggml_cuda_iu4_w4a4::selftest_impl(20);
}

#else // !(defined(GGML_USE_HIP) && defined(RDNA4))

bool ggml_cuda_iu4_w4a4_selftest() {
    // Not RDNA4 (or not HIP) -- nothing to test, vacuously true. The native
    // iu4 WMMA instruction this file targets does not exist on other archs.
    return true;
}

#endif // defined(GGML_USE_HIP) && defined(RDNA4)
