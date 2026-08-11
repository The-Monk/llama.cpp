// T170: dot8 int4-drafter decode path -- the M=1 (batch=1 decode / spec-
// decode drafter) GEMV counterpart to mul_mat_iu4.cu's WMMA GEMM kernel.
//
// GGML_TYPE_IU4 (int4 weight x int4 activation) has TWO existing kernels
// (mul_mat_iu4.cu's one-warp-per-tile PoC, mul_mat_iu4_mmq.cu's multi-warp
// MMQ-grade GEMM), both WMMA 16x16-tile based via ggml_cuda_mma::mma_iu4()
// -- correct for compute-bound/larger-M regimes, but a 16x16 tile wastes
// 15/16 of its rows at M=1. This file adds the missing decode-shaped
// (batch=1) path: v_dot8_i32_iu4 (sudot8) VALU dot, ported verbatim from
// the correctness- and perf-validated isolated PoC at
// ~/int4-research/pocs/dot8-int4-decode/decode_poc.hip (Stage 20;
// Stage 19 first confirmed v_dot8_i32_iu4 as a real, winning primitive).
//
// Deliberately NOT threaded into the generic mmq.cuh/mmvq.cu Q8_1-based
// templated dispatch (vec_dot_TYPE_q8_1 style) -- that machinery assumes
// EVERY type's activation is quantized to block_q8_1 (int8) once, up
// front; IU4's whole point is an int4-quantized activation (block_iu4,
// amax/7 symmetric, NOT q8_1), which doesn't fit that shared assumption
// without restructuring code every other quantized type also depends on.
// Same "small, self-contained kernel, intercepted at the top of
// ggml_cuda_mul_mat()" pattern mul_mat_iu4.cu itself already uses for
// exactly this reason (see mul_mat_iu4.cuh's header comment) -- and the
// established house convention for new opt-in decode-shaped kernels in
// this tree (mmvf_qk.cu, mul_mat_q2_0_wmma.cu, etc: own file, own env-var
// gate, zero risk to any other path).
//
// EXPERIMENTAL / model-blocked, same status as the rest of the IU4 work:
// no rotation-free packed-int4 W4A4 model exists yet (see
// mul_mat_iu4.cuh's header) -- this proves the KERNEL is live, correct,
// and at PoC speed; it does not claim production quality against a real
// trained model. Opt-in via GGML_HIP_IU4_MMVQ_DECODE, gated additionally
// to src1->ne[1] == 1 (M=1) so it can NEVER intercept a prefill/M>1 call
// even if accidentally enabled -- those keep routing to the existing WMMA
// path unchanged.
#pragma once

#include "common.cuh"

// Shape/dtype gate: true only for IU4 weight x F32 activation x F32 dst,
// single 2D tensor (ne2==ne3==1), K a multiple of QK_IU4, AND src1->ne[1]==1
// (M=1, the decode/drafter case this kernel targets). M>1 always returns
// false here so the caller falls through to the existing WMMA path.
bool ggml_cuda_mmvq_iu4_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// The op itself: on-device int4 activation quantization (k_quantize_act_iu4,
// verbatim port of the same-named kernel in mul_mat_iu4.cu, kept as an
// intentional file-local duplicate rather than exposing that file's
// internals) + the dot8 GEMV (k_mmvq_dot8_iu4). Caller must have already
// checked ggml_cuda_mmvq_iu4_supports().
bool ggml_cuda_op_mul_mat_vec_iu4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Correctness self-test, same style/naming convention as
// ggml_cuda_mul_mat_iu4_selftest() (mul_mat_iu4.cu) -- hand-packs int4
// weight/activation operands, runs k_mmvq_dot8_iu4 directly (bypassing the
// ggml_tensor machinery), compares against a CPU int4 x int4 reference.
// Vacuously true off RDNA4. Opt-in, see GGML_HIP_MUL_MAT_VEC_IU4_SELFTEST
// in ggml-cuda.cu.
bool ggml_cuda_mul_mat_vec_iu4_selftest();
