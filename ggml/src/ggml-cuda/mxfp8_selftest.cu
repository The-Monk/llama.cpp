// ROC8: MXFP8 selftest implementation. See mxfp8_selftest.cuh for the full
// rationale, gating discipline (T122), and env-var contract.
#include "mxfp8_selftest.cuh"

// T122 fix pattern (see iu4_w4a4.cu): gate the whole TU on GGML_USE_HIP only
// -- never on RDNA4/__GFX12__, which is a device-pass-only macro that is
// NEVER defined during the host compilation pass on any arch. A host-callable
// entry point gated that way silently compiles to a vacuous stub on every
// build, including real RDNA4 hardware.
#if defined(GGML_USE_HIP)

#include "common.cuh"
#include "ggml-quants.h"
#include "dequantize.cuh"
#include "vecdotq.cuh"
#include "mma.cuh"

#include <random>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace ggml_cuda_mxfp8_selftest_impl {

using namespace ggml_cuda_mma;

// ---------------------------------------------------------------------------
// Sub-test 1: CPU round-trip (quantize_row_mxfp8_ref -> dequantize_row_mxfp8).
// Pure host code, no GPU involved at all -- the literal "no weights, no big
// GPU" gate from the design doc. Exercises the real production CPU codec
// (ggml-quants.c), including the e8m0 power-of-two scale search.
// ---------------------------------------------------------------------------
static bool test_cpu_roundtrip(double * out_max_abs_err) {
    std::mt19937 rng(1234);
    // Deliberately varied magnitudes/signs per block (64 elements = 2 blocks
    // of QK_MXFP8=32) so the e8m0 per-block scale search is genuinely
    // exercised across different amax values, not just one lucky case.
    constexpr int n_blocks = 8;
    constexpr int n = n_blocks * QK_MXFP8;

    std::vector<float> src(n);
    std::uniform_real_distribution<float> big(-100.0f, 100.0f);
    std::uniform_real_distribution<float> small(-0.01f, 0.01f);
    std::uniform_real_distribution<float> mid(-3.0f, 3.0f);
    for (int b = 0; b < n_blocks; ++b) {
        for (int j = 0; j < QK_MXFP8; ++j) {
            const int idx = b*QK_MXFP8 + j;
            float v;
            switch (b % 4) {
                case 0: v = big(rng);   break; // near-saturating block
                case 1: v = small(rng); break; // near-zero block (tiny e8m0 exponent)
                case 2: v = mid(rng);   break;
                default: v = (j == 0) ? 0.0f : mid(rng); break; // a literal zero element
            }
            src[idx] = v;
        }
    }

    std::vector<block_mxfp8> q(n_blocks);
    quantize_row_mxfp8_ref(src.data(), q.data(), n);

    std::vector<float> out(n);
    dequantize_row_mxfp8(q.data(), out.data(), n);

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    for (int i = 0; i < n; ++i) {
        const double err = std::fabs((double) out[i] - (double) src[i]);
        max_abs_err = std::max(max_abs_err, err);
        const double denom = std::max(1e-6, std::fabs((double) src[i]));
        max_rel_err = std::max(max_rel_err, err / denom);
    }
    *out_max_abs_err = max_abs_err;

    // e4m3 has ~3 mantissa bits (relative quantization step ~1/16 = 6.25% at
    // worst case within a scale octave) PLUS the e8m0 power-of-two scale can
    // itself be up to 2x coarser than a continuous scale (the block's amax
    // maps somewhere in (224, 448], not always exactly at 448) -- so up to
    // ~2x the base e4m3 relative error is expected and NOT a bug. 20% relative
    // (worst-case element) is a generous but real correctness bound; this is
    // catching codec BUGS (wrong sign, wrong exponent bias, off-by-one scale
    // octave), not measuring production accuracy (that's the real-model PPL
    // test, Phase 3).
    const bool pass = max_rel_err < 0.20;
    GGML_LOG_INFO("%s: CPU roundtrip (%d blocks, %d elements): max_abs_err=%.6f max_rel_err=%.4f -> %s\n",
                   __func__, n_blocks, n, max_abs_err, max_rel_err, pass ? "PASS" : "FAIL");
    return pass;
}

// ---------------------------------------------------------------------------
// Sub-test 2: GPU dequant kernel (dequantize_mxfp8, dequantize.cuh) vs the
// SAME CPU reference (dequantize_row_mxfp8) on the SAME quantized bytes.
// Portable (no RDNA4 dependency -- ggml_cuda_e8m0_to_fp32/e4m3_to_fp32 both
// have non-hardware fallback paths), runs on any HIP target.
// ---------------------------------------------------------------------------
__global__ void k_dequant_mxfp8(const block_mxfp8 * bq, float * out, int nb) {
    const int ib  = blockIdx.x;
    const int iqs = threadIdx.x * 2; // dequantize_mxfp8 produces a float2 per call
    if (ib >= nb || iqs >= QK_MXFP8) {
        return;
    }
    float2 v;
    dequantize_mxfp8((const void *) bq, ib, iqs, v);
    out[ib*QK_MXFP8 + iqs + 0] = v.x;
    out[ib*QK_MXFP8 + iqs + 1] = v.y;
}

static bool test_gpu_dequant(double * out_max_abs_err) {
    constexpr int n_blocks = 8;
    constexpr int n = n_blocks * QK_MXFP8;

    std::mt19937 rng(5678);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::vector<float> src(n);
    for (int i = 0; i < n; ++i) {
        src[i] = dist(rng);
    }

    std::vector<block_mxfp8> q(n_blocks);
    quantize_row_mxfp8_ref(src.data(), q.data(), n);

    std::vector<float> cpu_out(n);
    dequantize_row_mxfp8(q.data(), cpu_out.data(), n);

    block_mxfp8 * d_q = nullptr;
    float * d_out = nullptr;
    if (hipMalloc(&d_q, n_blocks*sizeof(block_mxfp8)) != hipSuccess ||
        hipMalloc(&d_out, n*sizeof(float)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    CUDA_CHECK(hipMemcpy(d_q, q.data(), n_blocks*sizeof(block_mxfp8), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(k_dequant_mxfp8, dim3(n_blocks), dim3(QK_MXFP8/2), 0, 0, d_q, d_out, n_blocks);
    const hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        GGML_LOG_ERROR("%s: dequant kernel failed: %s\n", __func__, hipGetErrorString(err));
        hipFree(d_q); hipFree(d_out);
        return false;
    }

    std::vector<float> gpu_out(n);
    CUDA_CHECK(hipMemcpy(gpu_out.data(), d_out, n*sizeof(float), hipMemcpyDeviceToHost));
    hipFree(d_q); hipFree(d_out);

    // GPU decode must match CPU decode of the SAME bytes essentially exactly
    // (both paths do the same bit-manipulation; only rounding-mode-free
    // arithmetic is involved) -- this isolates KERNEL correctness from
    // QUANTIZATION accuracy (that's sub-test 1).
    double max_abs_err = 0.0;
    for (int i = 0; i < n; ++i) {
        max_abs_err = std::max(max_abs_err, (double) std::fabs(gpu_out[i] - cpu_out[i]));
    }
    *out_max_abs_err = max_abs_err;
    const bool pass = max_abs_err < 1e-4;
    GGML_LOG_INFO("%s: GPU dequant vs CPU dequant (%d elements): max_abs_err=%.8f -> %s\n",
                   __func__, n, max_abs_err, pass ? "PASS" : "FAIL");
    return pass;
}

// ---------------------------------------------------------------------------
// Sub-test 3: GPU mmvq decode dot (vec_dot_mxfp8_q8_1, vecdotq.cuh) vs a CPU
// reference computed with the EXACT SAME formula (e4m3 weight decode * e8m0
// scale, times int8 activation * q8_1 scale). Portable, no RDNA4 dependency.
// ---------------------------------------------------------------------------
// Builds the block_q8_1 activation block ON-DEVICE from raw int8 bytes + a
// float scale (rather than on the host) -- sidesteps any ambiguity about
// host-pass availability of HIP's make_half2()/half2 arithmetic, keeping
// every half-precision touch strictly inside a __global__ kernel like every
// other half2 use in this codebase (quantize.cu, fattn-wmma-f16.cu, etc.).
__global__ void k_decode_dot_mxfp8(
        const block_mxfp8 * bq, const int8_t * act, float act_scale, int32_t sum_act, float * out) {
    block_q8_1 bq8_1;
#pragma unroll
    for (int j = 0; j < QK_MXFP8; ++j) {
        bq8_1.qs[j] = act[j];
    }
    bq8_1.ds = make_half2(act_scale, act_scale * (float) sum_act);

    float sumf = 0.0f;
#pragma unroll
    for (int iqs = 0; iqs < QK_MXFP8/4; iqs += VDR_MXFP8_Q8_1_MMVQ) {
        sumf += vec_dot_mxfp8_q8_1_impl<VDR_MXFP8_Q8_1_MMVQ>((const void *) bq, &bq8_1, /*kbx=*/0, iqs);
    }
    *out = sumf;
}

static bool test_gpu_decode_dot(double * out_max_abs_err) {
    std::mt19937 rng(9012);
    std::uniform_real_distribution<float> wdist(-30.0f, 30.0f);
    std::uniform_int_distribution<int> adist(-127, 127);

    float w[QK_MXFP8];
    for (auto & v : w) v = wdist(rng);
    block_mxfp8 bq;
    quantize_row_mxfp8_ref(w, &bq, QK_MXFP8);

    // Synthetic int8 activation "block" -- built as plain host arrays; the
    // block_q8_1 container itself is assembled on-device (see
    // k_decode_dot_mxfp8) to avoid touching HIP half2 types from host code.
    int8_t act[QK_MXFP8];
    const float act_scale = 0.05f;
    for (int j = 0; j < QK_MXFP8; ++j) {
        act[j] = (int8_t) adist(rng);
    }
    int32_t sum_act = 0;
    for (int j = 0; j < QK_MXFP8; ++j) sum_act += act[j];

    // CPU reference: identical formula to vec_dot_mxfp8_q8_1_impl.
    const float d_w = ggml_e8m0_to_fp32(bq.e);
    double ref = 0.0;
    for (int j = 0; j < QK_MXFP8; ++j) {
        ref += (double) ggml_e4m3_to_fp32(bq.qs[j]) * (double) act[j];
    }
    ref *= (double) d_w * (double) act_scale;

    block_mxfp8 * d_bq = nullptr;
    int8_t * d_act = nullptr;
    float * d_out = nullptr;
    if (hipMalloc(&d_bq, sizeof(block_mxfp8)) != hipSuccess ||
        hipMalloc(&d_act, QK_MXFP8*sizeof(int8_t)) != hipSuccess ||
        hipMalloc(&d_out, sizeof(float)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    CUDA_CHECK(hipMemcpy(d_bq, &bq, sizeof(block_mxfp8), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_act, act, QK_MXFP8*sizeof(int8_t), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(k_decode_dot_mxfp8, dim3(1), dim3(1), 0, 0, d_bq, d_act, act_scale, sum_act, d_out);
    const hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        GGML_LOG_ERROR("%s: decode dot kernel failed: %s\n", __func__, hipGetErrorString(err));
        hipFree(d_bq); hipFree(d_act); hipFree(d_out);
        return false;
    }

    float gpu_out = 0.0f;
    CUDA_CHECK(hipMemcpy(&gpu_out, d_out, sizeof(float), hipMemcpyDeviceToHost));
    hipFree(d_bq); hipFree(d_act); hipFree(d_out);

    const double abs_err = std::fabs((double) gpu_out - ref);
    *out_max_abs_err = abs_err;
    // Loose-ish but real bound: fp32 accumulation across 32 terms, both sides
    // computed with the same per-term formula (fp32 device vs double host) --
    // any real kernel bug (wrong byte, wrong scale, wrong sign) produces
    // errors many orders of magnitude larger than fp32 accumulation noise.
    const bool pass = abs_err < 1e-2 * std::max(1.0, std::fabs(ref));
    GGML_LOG_INFO("%s: decode dot vs CPU ref: gpu=%.6f ref=%.6f abs_err=%.6f -> %s\n",
                   __func__, gpu_out, ref, abs_err, pass ? "PASS" : "FAIL");
    return pass;
}

// ---------------------------------------------------------------------------
// Sub-test 4: WMMA fp8xfp8 prefill compute (the same `mma()` overload
// vec_dot_mxfp8_mxfp8_mma, mmq.cuh, calls) fed with MXFP8-sourced weight
// bytes/e8m0 scale, vs a CPU reference. RDNA4-only (real hardware
// instruction, `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`) --
// gated by a RUNTIME cc check in the host wrapper below, not a compile-time
// macro (T122 discipline). Mirrors the M=16,N=16,K=32 tile shape and
// DATA_LAYOUT_J_MAJOR accumulator readback convention iu4_w4a4.cu's
// k_iu4_dense_test established (T123 fix) -- direct WMMA-level test rather
// than replicating the full mmq.cu grid/launch-parameter machinery, since the
// `mma()` call itself is BYTE-FOR-BYTE the same code F8E4M3 already validated
// in production; what's actually new here is load_tiles_mxfp8's e8m0 scale
// decode feeding that same call correctly.
// ---------------------------------------------------------------------------
__global__ void k_mxfp8_wmma_test(const int * dA_qs, const int * dB_qs, float * dOut) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
    tile<16, 8, int> A;
    tile<16, 8, int> B;
    load_generic(A, dA_qs, 8);
    load_generic(B, dB_qs, 8);

    tile<16, 16, float, DATA_LAYOUT_J_MAJOR> D; // T123 convention, see iu4_w4a4.cu
#pragma unroll
    for (int l = 0; l < D.ne; ++l) {
        D.x[l] = 0.0f;
    }

    mma(D, A, B);

#pragma unroll
    for (int l = 0; l < D.ne; ++l) {
        const int i = D.get_i(l);
        const int j = D.get_j(l);
        dOut[i*16 + j] = D.x[l];
    }
#else
    GGML_UNUSED_VARS(dA_qs, dB_qs, dOut);
    NO_DEVICE_CODE;
#endif
}

// Pack 32 raw bytes (e4m3 or arbitrary) into 8 int32 words, little-endian,
// matching get_int_b2's convention (4 consecutive qs[] bytes per int32).
static void pack_row_bytes(const uint8_t vals[32], int phys[8]) {
    for (int w = 0; w < 8; ++w) {
        uint32_t word = 0;
        for (int b = 0; b < 4; ++b) {
            word |= ((uint32_t) vals[4*w + b]) << (8*b);
        }
        phys[w] = (int) word;
    }
}

static bool run_wmma_trial(std::mt19937 & rng) {
    std::uniform_real_distribution<float> wdist(-40.0f, 40.0f);
    std::uniform_real_distribution<float> adist(-40.0f, 40.0f);

    // 16 weight rows, each its own MXFP8 block (K=32=QK_MXFP8).
    float W_logical[16][32];
    float A_logical[16][32];
    for (int i = 0; i < 16; ++i) {
        for (int k = 0; k < 32; ++k) {
            W_logical[i][k] = wdist(rng);
            A_logical[i][k] = adist(rng);
        }
    }

    block_mxfp8 W_q[16];
    for (int i = 0; i < 16; ++i) {
        quantize_row_mxfp8_ref(W_logical[i], &W_q[i], 32);
    }
    const float dW[16] = {
        ggml_e8m0_to_fp32(W_q[0].e),  ggml_e8m0_to_fp32(W_q[1].e),  ggml_e8m0_to_fp32(W_q[2].e),  ggml_e8m0_to_fp32(W_q[3].e),
        ggml_e8m0_to_fp32(W_q[4].e),  ggml_e8m0_to_fp32(W_q[5].e),  ggml_e8m0_to_fp32(W_q[6].e),  ggml_e8m0_to_fp32(W_q[7].e),
        ggml_e8m0_to_fp32(W_q[8].e),  ggml_e8m0_to_fp32(W_q[9].e),  ggml_e8m0_to_fp32(W_q[10].e), ggml_e8m0_to_fp32(W_q[11].e),
        ggml_e8m0_to_fp32(W_q[12].e), ggml_e8m0_to_fp32(W_q[13].e), ggml_e8m0_to_fp32(W_q[14].e), ggml_e8m0_to_fp32(W_q[15].e),
    };

    // Activation rows quantized to e4m3 the same way F8E4M3's real
    // quantize_mmq_f8e4m3 quantizer does (amax -> 448 per-row scale) --
    // reusing the exact math, not the kernel, to keep this a pure host
    // reference computation.
    uint8_t A_bytes[16][32];
    float dAct[16];
    for (int i = 0; i < 16; ++i) {
        float amax = 0.0f;
        for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(A_logical[i][k]));
        const float d = amax > 0.0f ? amax / 448.0f : 0.0f;
        const float id = d > 0.0f ? 1.0f/d : 0.0f;
        dAct[i] = d;
        for (int k = 0; k < 32; ++k) {
            A_bytes[i][k] = ggml_fp32_to_e4m3(A_logical[i][k]*id);
        }
    }

    int A_phys[16][8];
    int B_phys[16][8];
    for (int i = 0; i < 16; ++i) {
        pack_row_bytes(W_q[i].qs, A_phys[i]);
        pack_row_bytes(A_bytes[i], B_phys[i]);
    }

    int * dA = nullptr;
    int * dB = nullptr;
    float * dOut = nullptr;
    if (hipMalloc(&dA, 16*8*sizeof(int)) != hipSuccess ||
        hipMalloc(&dB, 16*8*sizeof(int)) != hipSuccess ||
        hipMalloc(&dOut, 16*16*sizeof(float)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    CUDA_CHECK(hipMemcpy(dA, A_phys, 16*8*sizeof(int), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(dB, B_phys, 16*8*sizeof(int), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(dOut, 0, 16*16*sizeof(float)));

    hipLaunchKernelGGL(k_mxfp8_wmma_test, dim3(1), dim3(32), 0, 0, dA, dB, dOut);
    const hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        GGML_LOG_ERROR("%s: WMMA kernel failed: %s\n", __func__, hipGetErrorString(err));
        hipFree(dA); hipFree(dB); hipFree(dOut);
        return false;
    }

    std::vector<float> out(16*16);
    CUDA_CHECK(hipMemcpy(out.data(), dOut, 16*16*sizeof(float), hipMemcpyDeviceToHost));
    hipFree(dA); hipFree(dB); hipFree(dOut);

    // Post-accumulate scale, exactly mirroring vec_dot_mxfp8_mxfp8_mma's
    // `sum += C.x[l]*dA*dB`.
    double max_abs_err = 0.0;
    double max_ref_mag = 0.0;
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 16; ++n) {
            double raw_dot = 0.0; // exact e4m3xe4m3 fp8 product sum, unscaled
            for (int k = 0; k < 32; ++k) {
                raw_dot += (double) ggml_e4m3_to_fp32(W_q[m].qs[k]) * (double) ggml_e4m3_to_fp32(A_bytes[n][k]);
            }
            const double ref = raw_dot * (double) dW[m] * (double) dAct[n];
            const double got = (double) out[m*16 + n] * (double) dW[m] * (double) dAct[n];
            max_abs_err = std::max(max_abs_err, std::fabs(got - ref));
            max_ref_mag = std::max(max_ref_mag, std::fabs(ref));
        }
    }

    // fp8 WMMA accumulates in fp32 hardware -- tolerate fp32 rounding only,
    // not e4m3 quantization error (both sides already consumed the SAME
    // quantized e4m3 bytes/e8m0 scale, so this isolates WMMA kernel
    // correctness, same "isolate kernel from quantization" principle as
    // sub-test 2/3).
    const bool pass = max_abs_err < 1e-2 * std::max(1.0, max_ref_mag);
    GGML_LOG_INFO("%s: WMMA fp8x fp8 16x16x32 trial: max_abs_err=%.6f (max_ref_mag=%.6f) -> %s\n",
                   __func__, max_abs_err, max_ref_mag, pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_gpu_wmma_prefill(int n_trials) {
    std::mt19937 rng(3456);
    bool all_pass = true;
    for (int t = 0; t < n_trials; ++t) {
        if (!run_wmma_trial(rng)) {
            all_pass = false;
        }
    }
    return all_pass;
}

} // namespace ggml_cuda_mxfp8_selftest_impl

bool ggml_cuda_mxfp8_selftest() {
    using namespace ggml_cuda_mxfp8_selftest_impl;

    bool all_pass = true;
    double err = 0.0;

    if (!test_cpu_roundtrip(&err))   { all_pass = false; }
    if (!test_gpu_dequant(&err))     { all_pass = false; }
    if (!test_gpu_decode_dot(&err))  { all_pass = false; }

    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        if (!test_gpu_wmma_prefill(20)) { all_pass = false; }
    } else {
        GGML_LOG_INFO("%s: skipping WMMA prefill sub-test (not RDNA4, cc=%d) -- vacuously true\n", __func__, cc);
    }

    GGML_LOG_INFO("%s: MXFP8 selftest overall -> %s\n", __func__, all_pass ? "PASS" : "FAIL");
    return all_pass;
}

#else // !defined(GGML_USE_HIP)

bool ggml_cuda_mxfp8_selftest() {
    // Not a HIP build -- nothing to test here (the WMMA sub-test is
    // AMD/RDNA4-only; the portable sub-tests are still meaningful on CUDA in
    // principle, but this fork's MXFP8 work targets gfx1201 specifically).
    return true;
}

#endif // defined(GGML_USE_HIP)
