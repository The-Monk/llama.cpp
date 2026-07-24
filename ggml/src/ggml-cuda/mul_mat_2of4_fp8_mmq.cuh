// T162 CAPSTONE (coordinator directive): library-grade 2:4-sparse fp8
// SWMMAC GEMM -- adapts mul_mat_iu4_mmq.cu's validated MMQ-style shape
// (cooperative multi-warp BM x BN output tile, register-blocked NTX
// SWMMAC-tiles/warp, double-buffered LDS K-pipeline, ONE __syncthreads()
// per K-chunk) to the SWMMAC 2:4-sparse fp8 instruction, closing the
// kernel-shape gap the T162 dense-V3 twin measured (dense-V3 1772 = 0.432x
// production dense 4098 -- NO LDS double-buffer, NO cooperative tiling).
// Keeps the validated V3 per-lane SWMMAC operand math (mul_mat_2of4_fp8.cu)
// but stages A (compressed weight+meta) and B (dense activation) into
// double-buffered LDS instead of private registers, so the per-row weight
// scale no longer needs a cross-lane __shfl_sync gather (any lane can
// directly index any row's staged scale in LDS).
#pragma once

#include "common.cuh"

bool ggml_cuda_op_mul_mat_2of4_fp8_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
