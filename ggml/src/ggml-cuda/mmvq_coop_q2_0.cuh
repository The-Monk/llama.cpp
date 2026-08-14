// T180 phase-pair 1: cooperative (persistent) quantize+GEMV for Q2_0 decode.
//
// The shipped decode path issues TWO dispatches per matmul-with-fresh-activations:
// quantize_q8_1 (grid over K) then mul_mat_vec_q (grid over rows, every block
// sweeping all of K). They cannot be fused conventionally -- the producer's
// parallel shape differs from the consumer's and each consumer block needs the
// WHOLE producer output -- so the fusion requires a grid-wide barrier.
//
// Measured justification (gfx1201, 2026-08-14):
//   grid.sync() ~0.81 us  vs  kernel-boundary gap ~2.17 us  => barrier 21x cheaper
//   dispatch gap is 1.35 ms of a 5.95 ms token (23% of wall) on Bonsai-8B
//   PoC on real 27B shapes: -32% (5120sq), -10% (gate/up), -15% (down)
// Opt-in via GGML_HIP_Q2_0_COOP_DECODE, M=1 only.
#pragma once
#include "common.cuh"

bool ggml_cuda_q2_0_coop_decode_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
bool ggml_cuda_op_mul_mat_q2_0_coop(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
