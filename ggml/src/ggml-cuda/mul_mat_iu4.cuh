// T89 driver-completeness run: minimal, dedicated GGML_OP_MUL_MAT path for
// GGML_TYPE_IU4 (int4 weight x int4 activation, native RDNA4 WMMA via
// ggml_cuda_mma::mma_iu4(), see mma.cuh / iu4_w4a4.cu). This is deliberately
// NOT integrated into the generic mmq.cuh/mmvq.cu templated dispatch (that
// would require threading a new operand-packing convention through every
// translation unit those headers touch) -- instead it is a small, self
// contained kernel hooked directly at the top of ggml_cuda_mul_mat() in
// ggml-cuda.cu, exactly the same "intercept before the generic switch"
// pattern already used there for GGML_HINT_SRC0_IS_HADAMARD.
//
// Scope: single GPU, 2D weight tensors only (ne2==ne3==1), MUL_MAT only (no
// MUL_MAT_ID / MoE routing). Quality is explicitly out of scope (plain RTN
// weights, per-block online RTN activations, no imatrix/rotation/SmoothQuant)
// -- the only goal is to prove mma_iu4() executes on real GGUF tensors
// end-to-end without crashing/asserting/NaN-ing. See the design doc at
// quark-fp8-bridge/w4a4/iu4-bridge-design.md and iu4_w4a4.cu's header.
#pragma once

#include "common.cuh"

// Returns false if the tensor shapes are outside this minimal kernel's
// supported scope (caller should treat that as a hard error for IU4 --
// there is no fallback path for this type).
bool ggml_cuda_op_mul_mat_iu4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
