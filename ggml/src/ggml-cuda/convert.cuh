#pragma once
#include "common.cuh"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template<typename T>
using to_t_cuda_t = void (*)(const void * x, T * y, int64_t k, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;
typedef to_t_cuda_t<nv_bfloat16> to_bf16_cuda_t;

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type);

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

// A5 (PORT-MANIFEST.md): the ggml_get_to_*_cuda() getters above are keyed
// on ggml_type alone and, like every other bare Q2_0 accessor, can only
// resolve the process-wide DEFAULT variant for GGML_TYPE_Q2_0 -- wrong for
// a src0 tensor bound to the OTHER variant. These direct entry points are
// for callers that HAVE the src0 tensor (ggml_cuda_q2_0_is_g64(), common.cuh)
// and need the g64-specific dequant explicitly, bypassing the type-keyed
// getters.
void ggml_cuda_q2_0_g64_to_fp32(const void * vx, float * y, int64_t k, cudaStream_t stream);
void ggml_cuda_q2_0_g64_to_fp16(const void * vx, half * y, int64_t k, cudaStream_t stream);
void ggml_cuda_q2_0_g64_to_bf16(const void * vx, nv_bfloat16 * y, int64_t k, cudaStream_t stream);
// Non-contiguous counterparts (a permuted/viewed src0 -- e.g.
// test-backend-ops' permuted-batch-dims MUL_MAT cases; a real loaded GGUF
// weight matrix is always contiguous).
void ggml_cuda_q2_0_g64_to_fp32_nc(const void * vx, float * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);
void ggml_cuda_q2_0_g64_to_fp16_nc(const void * vx, half * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);
void ggml_cuda_q2_0_g64_to_bf16_nc(const void * vx, nv_bfloat16 * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

// TODO more general support for non-contiguous inputs

template<typename T>
using to_t_nc_cuda_t = void (*)(const void * x, T * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

typedef to_t_nc_cuda_t<float> to_fp32_nc_cuda_t;
typedef to_t_nc_cuda_t<half> to_fp16_nc_cuda_t;
typedef to_t_nc_cuda_t<nv_bfloat16> to_bf16_nc_cuda_t;

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type);
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type);
to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type);

template<typename dst_t, typename src_t>
 __host__ __device__ inline dst_t ggml_cuda_cast(src_t x) {
    if constexpr (std::is_same_v<dst_t, src_t>) {
        return x;
    } else if constexpr(std::is_same_v<dst_t, nv_bfloat16>) {
        return __float2bfloat16(float(x));
    } else if constexpr(std::is_same_v<src_t, nv_bfloat16>) {
        return __bfloat162float(x);
    } else if constexpr(std::is_same_v<src_t, float2> && std::is_same_v<dst_t, half2>) {
        return __float22half2_rn(x);
    } else if constexpr(std::is_same_v<src_t, nv_bfloat162> && std::is_same_v<dst_t, float2>) {
#ifdef GGML_USE_HIP
        return make_float2(__bfloat162float(__low2bfloat16(x)), __bfloat162float(__high2bfloat16(x)));
#else
#if __CUDA_ARCH__ >= 800
        return __bfloat1622float2(x);
#else
        return make_float2(__bfloat162float(x.x), __bfloat162float(x.y));
#endif // __CUDA_ARCH__ >= 800
#endif // GGML_USE_HIP
    } else if constexpr(std::is_same_v<src_t, float2> && std::is_same_v<dst_t, nv_bfloat162>) {
        // bypass compile error on cuda 12.0.1
#ifdef GGML_USE_HIP
        return __float22bfloat162_rn(x);
#else
        return {x.x, x.y};
#endif // GGML_USE_HIP
    } else if constexpr(std::is_same_v<dst_t, int32_t>) {
        return int32_t(x);
    } else {
        return float(x);
    }
}
