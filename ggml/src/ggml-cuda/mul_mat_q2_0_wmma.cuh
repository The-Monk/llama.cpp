// Experimental decode-time lever: run Q2_0 (PrismML ternary, g128) decode
// matmuls through the native RDNA4 iu4 WMMA (V_WMMA_I32_16X16X32_IU4) instead
// of the dp4a-based mmvq.cu path. Ternary symbols {-1,0,+1} pack exactly into
// signed int4 (no precision loss on the WEIGHT side); the ACTIVATION side is
// additionally quantized online to int4 (same plain-RTN scheme
// mul_mat_iu4.cu's k_quantize_act_iu4 uses), which IS lossy relative to the
// q8_1 (int8) activations the dp4a path uses -- this is genuine W4A4, not
// W4A8. Motivation: at batch=1 decode the WMMA matrix cores are fully idle
// while dp4a/VALU is the measured bottleneck (see
// wiki/tech/rdna4-isa-optimization-audit.md); this reuses the SAME
// hardware-validated `ggml_cuda_mma::mma_iu4()` primitive iu4_w4a4.cu/
// mul_mat_iu4.cu already prove correct, just sourcing the weight operand
// on-the-fly from block_q2_0 instead of a native block_iu4 tensor.
//
// EXPERIMENTAL, opt-in, dormant by default: gated behind the
// GGML_HIP_Q2_0_WMMA_DECODE env var AND src1->ne[1]==1 (decode/batch=1 only
// -- prefill keeps using the validated mmq/mmvq path). No effect on any
// model/quant/kernel path unless both the env var is set and the call site
// is a genuine batch-1 decode matmul.
//
// Scope: single GPU, 2D weight tensors only (ne2==ne3==1), MUL_MAT only (no
// MUL_MAT_ID / MoE routing) -- same restrictions as mul_mat_iu4.cu.
#pragma once

#include "common.cuh"

// Returns false if the tensor shapes are outside this minimal kernel's scope
// (caller must fall back to the normal path in that case -- unlike IU4,
// Q2_0 has a working default path, so this is a soft/opt-in intercept).
bool ggml_cuda_op_mul_mat_q2_0_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Whether this call site is in-scope for the WMMA decode path (checked by
// the ggml_cuda_mul_mat() intercept before calling the op above).
bool ggml_cuda_q2_0_wmma_decode_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// T213: W2A4 dot8 decode -- same operands as the WMMA path, v_dot8_i32_iu4
// compute. M=1 only. Opt-in via GGML_HIP_Q2_0_DOT8_DECODE.
bool ggml_cuda_op_mul_mat_q2_0_dot8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool ggml_cuda_q2_0_dot8_decode_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// Concurrency PoC companion (see mul_mat_q2_0_wmma.cu for the full
// rationale): spawns a detached background thread that repeatedly launches
// a small WMMA-heavy kernel on its OWN hipStream_t, in the SAME process/HIP
// context as the caller, for `seconds` wall-clock time. Used to answer
// "does a VALU-bound decode kernel and a WMMA-bound kernel genuinely
// co-execute on separate streams of ONE GPU/context, or serialize?" at the
// real llama.cpp process level (same-context multi-stream, matching how the
// T112 async draft/verify pipeline is actually structured -- NOT a
// cross-process test, which is a materially different/worse-case scenario,
// see the write-up). Gated behind GGML_HIP_WMMA_CONCURRENCY_POC at
// ggml_backend_cuda_init(); zero effect unless that env var is set.
void ggml_cuda_start_wmma_concurrency_poc_companion(double seconds, long iters_per_launch);
