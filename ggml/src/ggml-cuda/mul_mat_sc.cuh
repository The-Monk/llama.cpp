#pragma once

// T399 single-copy weights (GGML_TYPE_NK_Q2_0_W2ONLY): every MUL_MAT on this type runs here.
//   batch N 1..7     multi-matrix GEMV straight from the W2 stream (up to 4 same-activation matrices per launch)
//   batch N 8..48    WMMA (iu8) straight from the W2 registers
//   batch N >= 49    the N1 int8 WMMA GEMM reading the tensor in place (no conversion, no cache copy)
// There is no other copy of the weight, so there is no fallback: supports_op claims exactly what runs here.

#include "common.cuh"

#define SC_MAX_GROUP 4

// supports_op for an op that has a W2ONLY source (MUL_MAT src0 only)
bool ggml_cuda_sc_supports_op(const ggml_tensor * op);

// largest batch the grouped decode / small-batch kernels take (above it: N1, one matrix per launch)
int64_t ggml_cuda_sc_group_max_n();

// n (1..SC_MAX_GROUP) consecutive MUL_MATs with W2ONLY weights of equal K reading the same src1
void ggml_cuda_sc_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, int n);

// act-fuse: activation layout the MUL_MAT `mm` (W2ONLY src0) reads, or -1 (nothing fused)
int ggml_cuda_sc_act_layout(const ggml_tensor * mm);
size_t ggml_cuda_sc_act_bytes(const ggml_tensor * src1);
void ggml_cuda_sc_act_ref_quant(const float * x, int64_t s11, int64_t K, int64_t N, void * y, cudaStream_t stream);
