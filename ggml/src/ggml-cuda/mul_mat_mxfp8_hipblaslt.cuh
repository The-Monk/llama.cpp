// mul_mat_mxfp8_hipblaslt.cuh - MXFP8 hipBLASLt prefill GEMM route header
//
// Route MXFP8 (8.25 bpw OCP MX: e4m3 elems + per-32-block UE8M0 shared scale.
// a 6-bit scale + 6-bit min, plus superblock d/dmin) through AMD's tuned hipBLASLt
// int8 (or fp8/e4m3) GEMM for prefill (M > M_THRESH), instead of the mmq/dp4a path.
//
// Requant strategy: FULLY dequant each weight (w = d*sc*q - dmin*m, min baked in)
// then requant to symmetric int8 with ONE per-output-channel scale. Because the
// asymmetric min is folded into the dequantized weight value, there is NO separate
// min/bias term at the GEMM level -- it is a plain symmetric int8 GEMM, identical
// to the Q1_0/Q2_0 routes. Lossy (8-bit per-row vs per-block 4.5-bit source) but
// PPL-gated. Same self-tuning per-shape algo cache + bounded int8 weight cache as
// the Q2_0 route (the tuned GEMM algo is shape-only, independent of source quant).

#pragma once

#include "common.cuh"

// In-scope check for the hipBLASLt prefill path (MXFP8 weight, F32 acts/dst, 2D,
// RDNA4, M > threshold). Soft/opt-in: a false return falls straight through to
// the unmodified mmq/dp4a path.
bool ggml_cuda_mxfp8_hipblaslt_prefill_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// Run the MXFP8 prefill matmul through hipBLASLt int8/fp8. Returns false if the
// build has no hipBLASLt (non-HIP / disabled) so the caller can fall back.
bool ggml_cuda_op_mul_mat_mxfp8_hipblaslt(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
