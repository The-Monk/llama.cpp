#pragma once

#include "common.cuh"

// N1: route Q2_0/Q1_0 prefill MUL_MATs (src1 batch >= 64) to the native int8 WMMA GEMM (n1_gemm_iu8.cuh).
// On by default (T412); GGML_N1_PREFILL=0 disables it. See mul_mat_n1.cu for the knobs.

bool ggml_cuda_n1_enabled();

// Coverage accounting: call for every MUL_MAT reaching ggml_cuda_mul_mat when enabled.
void ggml_cuda_n1_count(const ggml_tensor * src0, const ggml_tensor * src1);

// Returns true if N1 computed dst; false = caller falls through to the default path (fallback is counted).
bool ggml_cuda_n1_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// T399: N1 on a single-copy (GGML_TYPE_NK_Q2_0_W2ONLY) weight read in place, any batch; ignores GGML_N1_PREFILL and
// GGML_N1_MIN_N (the weight has no other path). Returns false only for an unsupported shape.
bool ggml_cuda_n1_mul_mat_w2only(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// [TAG_ACT_FUSE] T412: routing probe for the activation-fusion producers (act-fuse.cuh).
//   0 = N1 does not take this MUL_MAT (the default route decides)
//   1 = N1 takes it and reads the N1 activation layout from the act cache (GGML_CUDA_ACT_LAYOUT_N1)
//   2 = N1 takes it but cannot read a fused activation (token/row arms, or weight not converted yet)
// Uses the same predicate as ggml_cuda_n1_mul_mat; a weight that could not be converted is remembered,
// so the answer is stable for the process lifetime once the weight has been seen.
int ggml_cuda_n1_act_route(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// Bytes of the N1 activation buffer for src1: int8 X stream (Npad*K) then fp32 sx[K/128][Npad].
size_t ggml_cuda_n1_act_bytes(const ggml_tensor * src1);

// Reference quantizer (the unfused N1 activation quantize) into an N1 activation buffer; VERIFY only.
void ggml_cuda_n1_act_ref_quant(const float * x, int64_t s11, int64_t K, int64_t N, void * y, cudaStream_t stream);
