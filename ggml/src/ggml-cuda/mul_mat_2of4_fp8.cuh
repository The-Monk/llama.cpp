// RDNA4 2:4-structured-sparse fp8 SWMMAC driver-completeness run: minimal,
// dedicated GGML_OP_MUL_MAT path for GGML_TYPE_2OF4_FP8 (block_2of4_fp8
// sparse weight x dense fp8 activation, native RDNA4 V_SWMMAC_F32_16X16X32_
// FP8_FP8 via the VALIDATED swmmac24.cuh builtin call -- see that file's
// self-test, GGML_HIP_SWMMAC24_SELFTEST=1, for the ground-truth per-lane
// operand layout this kernel re-packs block_2of4_fp8/block_f8e4m3 into).
//
// Mirrors mul_mat_iu4.cu's "intercept before the generic switch" pattern
// (hooked directly at the top of ggml_cuda_mul_mat() in ggml-cuda.cu) --
// NOT integrated into the generic mmq.cuh/mmvq.cu templated dispatch.
//
// Scope: single GPU, 2D weight tensors only (ne2==ne3==1), MUL_MAT only (no
// MUL_MAT_ID / MoE routing). The sparse WEIGHT operand (A) is native RDNA4
// 2:4-sparse fp8; the DENSE activation operand (B) is online-quantized fp8
// (plain per-block RTN, reusing the ggml_cuda_fp32_to_e4m3/ggml_cuda_e4m3_to_fp32
// codec from common.cuh -- same codec quantize_mmq_f8e4m3 uses for the dense
// fp8 MMQ path, just written to the plain block_f8e4m3 GGUF layout instead
// of that path's specialized block_q8_1_mmq container).
#pragma once

#include "common.cuh"

// Returns false if the tensor shapes are outside this minimal kernel's
// supported scope (caller should treat that as a hard error -- there is no
// fallback path for this type).
bool ggml_cuda_op_mul_mat_2of4_fp8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
