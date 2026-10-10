#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// T447 GGML_GDN_GATED_NORM: RMS_NORM + MUL(weight) + SILU(z) + MUL in one launch (ncols == 128; false = declined).
bool ggml_cuda_op_rms_norm_mul_silu_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mulw,
        const ggml_tensor * silu, const ggml_tensor * mulg);

// T430 GGML_CUDA_FWHT_QUANT: decode RMS_NORM+MUL+MUL(signs)+FWHT1024+Q2_FIELD q8_1 in one launch (false = declined).
bool ggml_cuda_op_rms_norm_mul_fwht_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * mul_w,
        const ggml_tensor * signs, ggml_tensor * norm_dst, ggml_tensor * fwht_dst, float eps, void * qy);
