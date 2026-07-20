// Phase-3(B) M<=2 route, GGML_TYPE_Q2_0 (PrismML ternary, real production
// weight type -- Bonsai-27B/8B) companion to mul_mat_iu4_gemv.cu.
//
// mul_mat_iu4_gemv's _supports() is GGML_TYPE_IU4-only (checked directly
// this session): GGML_TYPE_IU4 has no real GGUF producer (see
// mul_mat_iu4.cu's own doctrine comment), so the Phase-3(B) M-switch never
// actually fired on any real model before this file existed -- confirmed
// by measurement (llama-bench tg128 on Ternary-Bonsai-27B-Q2_0, switch on
// vs off: 51.34+-0.35 vs 51.49+-0.38 t/s, i.e. zero effect, GEMV was simply
// not reached). This file closes that gap for the real ternary production
// type (block_q2_0, QK2_0=128, ggml-common.h) using the SAME warp-per-row /
// one-lane-per-block-round-robin technique, correctness-gated the same way.
//
// Q2_0 block: 128 elements, 2 bits/element (codes 0/1/2 -> logical -1/0/+1,
// code 3 unused), one fp16 scale/block (34 bytes/block: 2 scale + 32 qs).
// Byte b of qs holds 4 elements (4b..4b+3), 2 bits each, LOW-to-HIGH bit
// order -> element order (matches unpack_q2_0_chunk_to_iu4_words in
// mul_mat_iu4_mmq.cu, the ONLY other place this format is read on this
// backend -- verified consistent, not re-derived).
#pragma once

#include "common.cuh"

bool ggml_cuda_op_mul_mat_q2_0_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool ggml_cuda_q2_0_gemv_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// Correctness-gated selftest (house pattern, mirrors
// ggml_cuda_mul_mat_iu4_gemv_selftest): CPU int64 reference over the
// logical {-1,0,+1} codes, M in {1,2,4,8,16} (M<=2 is the actual dispatch
// scope; M>2 included for the same completeness reason the IU4 selftest
// covers M it never dispatches). See GGML_HIP_MUL_MAT_Q2_0_GEMV_SELFTEST.
bool ggml_cuda_mul_mat_q2_0_gemv_selftest();

// Q1_0 companion (binary {-1,+1}, QK1_0=128, ggml-common.h) -- same
// technique, added alongside Q2_0 since the block size/kernel shape is
// identical and the marginal cost is small. Bit convention matches
// mul_mat_iu4_mmq.cu's unpack_q1_0_chunk_to_iu4_words / vecdotq.cuh's
// vec_dot_q1_0_q8_1 (the only other place this format is read): bit i of
// qs byte gb -> element 8*gb+i, bit set -> +1, clear -> -1 (verified
// algebraically equivalent to the subchunked mmq version, see the .cu
// doctrine comment).
bool ggml_cuda_op_mul_mat_q1_0_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool ggml_cuda_q1_0_gemv_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
bool ggml_cuda_mul_mat_q1_0_gemv_selftest();
