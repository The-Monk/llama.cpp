// T162: MMQ-grade native iu4 (int4 x int4 -> int32) WMMA GEMM for RDNA4.
//
// mul_mat_iu4.cu (k_mul_mat_iu4) and mul_mat_q2_0_wmma.cu
// (k_mul_mat_q2_0_wmma) are both a ONE-WARP-PER-16x16-TILE proof of concept:
// block(32,1,1), one wavefront computes exactly one output tile, single
// buffered shared staging, no register blocking. That leaves almost all of
// a CU idle (1 active warp of the ~waves a CU can co-schedule) and reloads
// operands from LDS with none of the K-chunk latency hidden -- it loses to
// the MMQ int8 (dp4a) path by 70%+ at prefill despite the iu4 WMMA
// instruction itself doing 2x the MACs/instruction of the int8 path.
//
// This file is the MMQ-style rebuild: multi-warp block, each warp owns
// several 16x16 output tiles (register blocking over both M and N),
// double-buffered shared-memory staging (the next K-chunk's global loads
// are issued before the current chunk's compute, so latency is hidden
// behind the WMMA/FMA work instead of serializing with it), and a single
// templated kernel body shared by BOTH the native GGML_TYPE_IU4 weight
// layout (block_iu4, load_iu4_words) and the GGML_TYPE_Q2_0 ternary layout
// (block_q2_0, unpack_q2_0_chunk_to_iu4_words) via a small loader trait --
// consolidating what used to be two byte-identical kernel clones.
//
// Kept as a SEPARATE opt-in path from the original one-warp kernels (which
// remain untouched behind GGML_HIP_Q2_0_WMMA_DECODE) so the two can be A/B
// benched on one binary. Gate: GGML_HIP_IU4_MMQ.
//
// Scope: single GPU, 2D weight tensors only (ne2==ne3==1), MUL_MAT only (no
// MUL_MAT_ID / MoE routing) -- same restrictions as the kernels it replaces.
#pragma once

#include "common.cuh"

bool ggml_cuda_op_mul_mat_iu4_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool ggml_cuda_op_mul_mat_q2_0_iu4_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool ggml_cuda_op_mul_mat_q1_0_iu4_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// T164: opt-in PER-CHANNEL weight-scale path (GGML_TYPE_Q2_0 only), gated by
// its own env var (GGML_HIP_IU4_MMQ_PERCHANNEL) in ggml-cuda.cu, separate
// from the T162/T163 production GGML_HIP_IU4_MMQ path. See
// mul_mat_iu4_mmq.cu's launch_perchannel()/k_requant_q2_0_to_iu4_perchannel().
bool ggml_cuda_op_mul_mat_q2_0_iu4_mmq_perchannel(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Shape/arch gate shared by both entry points above (RDNA4 + K%QK_IU4==0 +
// 2D F32 activation/output). Mirrors ggml_cuda_q2_0_wmma_decode_supports().
bool ggml_cuda_iu4_mmq_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// Synthetic correctness self-test (same style/CPU reference as
// ggml_cuda_mul_mat_iu4_selftest() in mul_mat_iu4.cu, but exercising the new
// multi-warp/double-buffered kernel body through BOTH loader traits).
// Opt-in, see GGML_HIP_MUL_MAT_IU4_MMQ_SELFTEST in ggml-cuda.cu.
bool ggml_cuda_mul_mat_iu4_mmq_selftest();
