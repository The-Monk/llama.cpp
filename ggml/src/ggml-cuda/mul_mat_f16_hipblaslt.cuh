// Phase-3 TOP INTEGRATION target (see ~/int4-research/kernel-coverage-map.md
// #6.1): route dense fp16/bf16 PREFILL matmuls (M > threshold) through AMD's
// tuned hipBLASLt Tensile GEMM (HH_SH: fp16 A/B -> fp32 D; BB_SB: bf16 A/B ->
// fp32 D) instead of the existing ggml_cuda_op_mul_mat_cublas() (hipblasGemmEx)
// path. Unlike the Q2_0 int8 route (dp4a was genuinely un-tuned), the fp16/
// bf16 weight case needs NO requant -- GGML_TYPE_F16/BF16 storage already IS
// the hipBLASLt A-operand dtype; only the F32 activation needs a cast to
// fp16/bf16 per call. D is fp32 directly (HH_SH/BB_SB), so no dequant
// epilogue kernel is needed either -- this route is much leaner than Q2_0's.
//
// Standalone stage-1 probe (~/int4-research/phase3-fp16bf16-route.md) found
// BOTH HH_SH and BB_SB select real, correct gfx1201 Tensile kernels -- but
// hipblasGemmEx (the CURRENT path, via cublasGemmEx->hipblasGemmEx) on both
// the qat714 (ROCm 7.13/14) and /opt/rocm-7.2.4 stacks is measured to
// *already* dispatch into hipBLASLt for this shape family on this hardware/
// library combo (parity or a current-path win in most shapes tested). This
// is the opposite of the Q2_0 case. This route exists for completeness/
// banking + the disk-persisted self-tuning cache still occasionally beats
// the single default hipblasGemmEx algo on FIXED unusual shapes -- gated off
// by default, opt-in only. See the phase-3 doc for the measured verdict
// before enabling in any ship candidate.
#pragma once

#include "common.cuh"

// In-scope check: F16/BF16 weight, F32 acts/dst, 2D (ne2==ne3==1), M above
// threshold (prefill only -- decode stays on mmvq/mmf), RDNA4.
bool ggml_cuda_f16_hipblaslt_prefill_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// Run the F16/BF16 prefill matmul through hipBLASLt fp16/bf16 GEMM. Returns
// false if unsupported/unavailable so the caller falls back to the existing
// cublas (hipblasGemmEx) path.
bool ggml_cuda_op_mul_mat_f16_hipblaslt(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
