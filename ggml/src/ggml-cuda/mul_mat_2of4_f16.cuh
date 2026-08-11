// card 141: RDNA4 2:4-structured-sparse fp16 SWMMAC, end-to-end. Minimal,
// dedicated GGML_OP_MUL_MAT path for GGML_TYPE_2OF4_F16 (block_2of4_f16
// sparse weight x dense fp16 activation, native RDNA4
// V_SWMMAC_F32_16X16X32_F16 via the VALIDATED swmmac24.cuh builtin call --
// see that file's self-test, GGML_HIP_SWMMAC24_SELFTEST=1, for the
// ground-truth per-lane operand layout this kernel re-packs
// block_2of4_f16/raw-fp16-activation into).
//
// This is the SAME "intercept before the generic switch" pattern as
// mul_mat_2of4_fp8.cu/mul_mat_iu4.cu (hooked directly at the top of
// ggml_cuda_mul_mat() in ggml-cuda.cu) -- NOT integrated into the generic
// mmq.cuh/mmvq.cu templated dispatch.
//
// Scope: single GPU, 2D weight tensors only (ne2==ne3==1), MUL_MAT only (no
// MUL_MAT_ID / MoE routing). The sparse WEIGHT operand (A) is native RDNA4
// 2:4-sparse fp16; the DENSE activation operand (B) is online-cast fp16 (a
// plain per-element float->half cast -- NO block-scale codec needed, unlike
// the fp8 sibling, because fp16 already carries full usable dynamic range;
// see block_2of4_f16's comment in ggml-common.h).
#pragma once

#include "common.cuh"

// Returns false if the tensor shapes are outside this minimal kernel's
// supported scope (caller should treat that as a hard error -- there is no
// fallback path for this type).
bool ggml_cuda_op_mul_mat_2of4_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
