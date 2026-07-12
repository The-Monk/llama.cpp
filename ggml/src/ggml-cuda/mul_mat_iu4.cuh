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
//
// EXPERIMENTAL / model-blocked, kept dormant on purpose (not removed):
//   (a) No rotation-free packed-int4 W4A4 model exists yet -- needs
//       QuaRot/SpinQuant online rotation or a co-trained BitNet-a4.8 (see
//       cards 122/133). Plain RTN quantization is a placeholder, not
//       expected to be production quality even once numerically correct.
//   (b) k_mul_mat_iu4's accumulator likely still carries the transposed-
//       readback bug 615f718ca fixed in the iu4_w4a4.cu selftest
//       (DATA_LAYOUT_J_MAJOR was never ported here) -- see the comment at
//       its `tile<16, 16, int> D` declaration in mul_mat_iu4.cu. Output is
//       expected to be garbage/RTN-at-best off the diagonal until that is
//       fixed.
// It compiles and runs finite (no crash) on real GGUF tensors but is NOT
// production. This dormancy is confined to GGML_TYPE_IU4's own dispatch
// path (ggml-cuda.cu's GGML_TYPE_IU4 intercepts) and does not affect any
// other type.
#pragma once

#include "common.cuh"

// Returns false if the tensor shapes are outside this minimal kernel's
// supported scope (caller should treat that as a hard error for IU4 --
// there is no fallback path for this type).
bool ggml_cuda_op_mul_mat_iu4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
