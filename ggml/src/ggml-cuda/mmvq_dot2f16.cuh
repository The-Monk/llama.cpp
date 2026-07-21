// Stage 25 (completeness-inventory mandate): v_dot2_f32_f16 (fdot2) dormant
// F16 decode GEMV -- wired as a TUNING KNOB, not a production win.
//
// VERDICT (measured, not guessed): LOSES to the production dp4a-int8
// decode path (Stage 19: raw ALU 0.66x, GEMV 0.92x) and its own naive
// scalar-fp16-FMA baseline is a wash at GEMV scale (0.98x) -- confirmed
// real hardware instruction (v_dot2_f32_f16), just not a decode-density
// win on this silicon. The dual-issue VOPD form (v_dual_dot2acc_f32_f16,
// pinned-register inline asm) was separately measured (Stage 22,
// ~/int4-research/pocs/dot2acc-dualissue/) to deliver ~1.04x over
// single-issue -- essentially NO real dual-issue gain despite emitting the
// genuine VOPD-bundle encoding (LLVM's auto-packer excludes this op from
// GFX12; the hand-bundled asm form doesn't recover it on real silicon).
//
// Wired DORMANT anyway (env-flag gated, off by default) per the
// completeness mandate: Stage 19/22 only measured pure batch-1 GEMV
// decode; a fusion context (e.g. norm+matvec, or an M>1 regime with
// different register pressure) wasn't excluded and might look different
// during a later dedicated tuning pass. This file exists so that lever is
// present and toggleable, not lost.
//
// GGML_HIP_F16_DOT2_DECODE unset/0 -> disabled, F16 MUL_MAT keeps using
// the production mmvf.cu mul_mat_vec_f path unchanged (packed half2 FMA,
// NOT fdot2 -- different instruction, different accumulation precision).
// GGML_HIP_F16_DOT2_DECODE=1 -> single-issue fdot2 GEMV (this file).
// GGML_HIP_F16_DOT2_DECODE=2 -> dual-issue VOPD DOT2ACC GEMV (this file).
// Gated additionally to src1->ne[1]==1 (M=1) -- M>1 always stays on the
// existing mmvf.cu path, same non-regression discipline as mmvq_iu4.cuh.
#pragma once

#include "common.cuh"

bool ggml_cuda_mmvq_dot2f16_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// mode: 1 = single-issue fdot2, 2 = dual-issue VOPD DOT2ACC.
bool ggml_cuda_op_mul_mat_vec_dot2f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int mode);

// Correctness self-test for BOTH modes, same style as
// ggml_cuda_mul_mat_vec_iu4_selftest(). Opt-in, see
// GGML_HIP_MUL_MAT_VEC_DOT2F16_SELFTEST in ggml-cuda.cu.
bool ggml_cuda_mul_mat_vec_dot2f16_selftest();
