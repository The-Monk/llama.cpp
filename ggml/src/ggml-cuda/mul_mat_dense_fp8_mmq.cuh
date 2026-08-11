// T162 CAPSTONE dense-fp8 twin of mul_mat_2of4_fp8_mmq.cu -- IDENTICAL
// MMQ-grade cooperative-tile / double-buffered-LDS shape, dense WMMA math
// (v_wmma_f32_16x16x16_fp8_fp8, two calls/K32-chunk) instead of SWMMAC 2:4
// math, so the "kernel-quality factor" (this kernel / production dense)
// can be tracked independent of the sparsity question, at the SAME tile
// geometry as the sparse capstone kernel (apples-to-apples, matching the
// discipline established for the V3 pair).
#pragma once

#include "common.cuh"

bool ggml_cuda_op_mul_mat_dense_fp8_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
