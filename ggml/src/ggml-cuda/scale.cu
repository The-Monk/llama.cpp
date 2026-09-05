#include "scale.cuh"
#include "convert.cuh"

#define MAX_GRIDDIM_X 0x7FFFFFFF

template <typename src_t, typename dst_t>
static __global__ void scale_cuda_k(const src_t * x, dst_t * dst, const float scale, const float bias, const int64_t nelements) {
    ggml_cuda_pdl_lc();
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        const float v = scale * ggml_cuda_cast<float>(x[i]) + bias;
        dst[i] = ggml_cuda_cast<dst_t>(v);
    }
}

template <typename src_t, typename dst_t>
static void scale_cuda(const src_t * x, dst_t * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_cuda_k<src_t, dst_t>, launch_params, x, dst, scale, bias, nelements);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    // f32 is the hot path (attention/ffn scaling); f16 is added only to support
    // the recurrent-state zero-clear (build_rs's ggml_scale_inplace(state_zero, 0))
    // when GGML_RECURRENT_STATE_F16 stores the GDN/SSM cache in f16 -- see the
    // GDN-state ladder port. No other caller is known to need it today.
    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        scale_cuda((const float *) src0->data, (float *) dst->data, scale, bias, ggml_nelements(src0), stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        scale_cuda((const half *) src0->data, (half *) dst->data, scale, bias, ggml_nelements(src0), stream);
    } else {
        GGML_ABORT("ggml_cuda_op_scale: unsupported type combination (src0=%s, dst=%s)",
                    ggml_type_name(src0->type), ggml_type_name(dst->type));
    }
}
