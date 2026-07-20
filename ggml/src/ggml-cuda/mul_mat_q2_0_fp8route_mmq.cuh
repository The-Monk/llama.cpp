// T165: route ternary (Q2_0) weights through the native RDNA4 fp8 e4m3 WMMA
// datapath instead of the iu4 int4xint4->int32 datapath. See
// mul_mat_q2_0_fp8route_mmq.cu for the full rationale (RC2 -- the
// int32->float rescale epilogue -- is structurally absent on the fp8 path
// since its WMMA accumulates natively in fp32).
//
// Standalone, opt-in (GGML_HIP_Q2_0_FP8ROUTE_MMQ), zero risk to the
// production dp4a Q2_0 path or the T162-T164 iu4-mmq paths.
#pragma once

#include "common.cuh"

bool ggml_cuda_op_mul_mat_q2_0_fp8route_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool ggml_cuda_q2_0_fp8route_mmq_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
