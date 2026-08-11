#include "common.cuh"

// ROC9 experimental RDNA4 decode lever (GGML_HIP_MMVF_QK, default OFF):
// direct fp32 dequant-FMA batch=1 decode kernel for K-quants (Q4_K/Q6_K),
// bypassing the quantize_row_q8_1_cuda int8 activation-packing kernel that
// mmvq.cu's generic dp4a path always launches even at ne11==1, where decode
// is bandwidth-bound and that extra kernel launch + int8 intermediate buffer
// is dead weight (root cause documented at mmvq.cu, ggml_cuda_op_mul_mat_vec_q
// call site). See ggml_cuda_should_use_mmvf_qk below for the exact gate.

bool ggml_cuda_should_use_mmvf_qk(enum ggml_type type, int cc, int64_t ne11);

void ggml_cuda_mul_mat_vec_qk(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
