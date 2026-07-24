// Coordinator directive (T162 follow-up, "the missing measurement"): a
// DENSE-fp8 TWIN of mul_mat_2of4_fp8.cu's V3 kernel -- byte-for-byte the
// SAME shape (private registers, zero LDS, zero __syncthreads(), WARPS=32,
// ILP=4, same need_check/dispatch machinery) but computing over the FULL
// (uncompressed) fp8 weight via the dense RDNA4 WMMA instruction
// (v_wmma_f32_16x16x16_fp8_fp8, TWO calls per K=32 chunk) instead of the
// 2:4-sparse SWMMAC instruction (v_swmmac_f32_16x16x32_fp8_fp8, ONE call
// per K=32 chunk) over the compressed weight.
//
// Purpose: the whole-model comparison in the prior T162 writeup (2:4-V3
// ~2224 vs "dense fp8" ~4105) CONFOUNDED two different things -- (a)
// whether 2:4 sparsity helps at all, vs (b) whether the V3 kernel SHAPE is
// simply less tuned than the production dense-fp8 MMQ/WMMA path (mma.cuh's
// tile<>-based, LDS-double-buffered machinery) it was compared against.
// This kernel isolates (b): same hand-written shape, dense math instead of
// sparse math, so `dense-V3` vs `2:4-V3` measures ONLY the sparsity delta
// (both are equally "unlensed by the production tiling machinery"), and
// `dense-V3` vs `production dense (~4105)` measures ONLY the kernel-quality
// delta.
//
// Per-lane operand layout is NOT re-derived here -- it is copied directly
// from the ALREADY-VALIDATED production fp8 WMMA mma() overload in
// mma.cuh (tile<16,8,int,I_MAJOR> A/B, tile<16,16,float,J_MAJOR> D; the
// exact per-lane tile<16,8,int>::get_i/get_j formulas for AMD_WMMA_AVAILABLE
// -- row=lane%16, k_half=lane/16, 4 ints/lane = that lane's own k_half's 16
// bytes) -- this is the SAME hardware convention mul_mat_2of4_fp8.cu's V3
// kernel already uses for its (dense) activation operand, so no new
// derivation risk: it is a mechanical extension of already-working code.
#pragma once

#include "common.cuh"

// Flag-gated (GGML_HIP_DENSE_FP8_V3, off by default), opt-in intercept for
// GGML_TYPE_F8E4M3 x GGML_TYPE_F32 MUL_MAT -- see ggml-cuda.cu's dispatch
// hook. Falls through to the normal production dense-fp8 MMQ path when the
// env var is unset.
bool ggml_cuda_op_mul_mat_dense_fp8_v3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Isolated per-shape microbench, mirrors
// ggml_cuda_mul_mat_2of4_fp8_shape_bench() exactly (same 3 shapes, same
// WARPS/ILP sweep) so the two can be read side by side. Opt-in via
// GGML_HIP_DENSE_FP8_V3_SHAPE_BENCH.
bool ggml_cuda_mul_mat_dense_fp8_v3_shape_bench();
