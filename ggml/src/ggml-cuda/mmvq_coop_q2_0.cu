#include "mmvq_coop_q2_0.cuh"
#include "vecdotq.cuh"
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;

#define COOP_Q2_0_BLOCK 256

// Phase 1 = quantize f32 activations -> block_q8_1 (identical numerics to
// quantize_q8_1 in quantize.cu: amax/127 scale, ds = (d, sum(raw xi)), which is
// the d8*sum(u) term vec_dot_q2_0_q8_1 subtracts).
// Phase 2 = Q2_0 GEMV via the SAME vec_dot_q2_0_q8_1 the shipped path uses, so
// numerics are bit-identical by construction; only the dispatch structure differs.
__global__ void k_coop_q2_0_decode(
        const float * __restrict__ x, block_q8_1 * __restrict__ y,
        const void * __restrict__ vx, float * __restrict__ dst,
        const int64_t K, const int64_t N, const int64_t stride_row_x) {
    cg::grid_group grid = cg::this_grid();
    const int64_t nchunk = K / QK8_1;
    const int     warps  = blockDim.x / WARP_SIZE;
    const int     lane   = threadIdx.x % WARP_SIZE;

    for (int64_t b = (int64_t) blockIdx.x*warps + threadIdx.x/WARP_SIZE; b < nchunk; b += (int64_t) gridDim.x*warps) {
        const float xi = x[b*QK8_1 + lane];
        float amax = fabsf(xi);
        float sum  = xi;
        amax = warp_reduce_max<QK8_1>(amax);
        sum  = warp_reduce_sum<QK8_1>(sum);
        const float d = amax / 127.0f;
        y[b].qs[lane] = amax == 0.0f ? 0 : roundf(xi / d);
        if (lane == 0) {
            y[b].ds = make_half2(d, sum);
        }
    }

    grid.sync();   // <-- replaces a kernel boundary

    __shared__ float red[COOP_Q2_0_BLOCK];
    for (int64_t row = blockIdx.x; row < N; row += gridDim.x) {
        const int64_t kbx_off = row * stride_row_x;
        float acc = 0.0f;
        for (int64_t c = threadIdx.x; c < nchunk; c += blockDim.x) {
            const int kbl = (int) (c >> 2);
            const int iqs = (int) (c & 3);
            acc += vec_dot_q2_0_q8_1(vx, &y[kbl*4], (int) (kbx_off + kbl), iqs);
        }
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) dst[row] = red[0];
        __syncthreads();
    }
}

bool ggml_cuda_q2_0_coop_decode_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_Q2_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src1->ne[1] != 1) return false;                      // M=1 decode only
    if (src0->ne[0] != src1->ne[0] || src0->ne[0] % QK2_0 != 0) return false;
    if (!ggml_is_contiguous(src1)) return false;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return GGML_CUDA_CC_IS_RDNA4(cc);
}

bool ggml_cuda_op_mul_mat_q2_0_coop(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t stride_row_x = K / QK2_0;
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_q8_1> y(ctx.pool(), (size_t) (K / QK8_1));

    int blocks_per_cu = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, (const void *) k_coop_q2_0_decode, COOP_Q2_0_BLOCK, 0));
    const int nsm  = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    int grid = blocks_per_cu * nsm;
    if (grid <= 0) return false;
    if (grid > N) grid = (int) N;

    const float * xp = (const float *) src1->data;
    block_q8_1  * yp = y.get();
    const void  * wp = src0->data;
    float       * dp = (float *) dst->data;
    void * args[] = { (void*)&xp, (void*)&yp, (void*)&wp, (void*)&dp,
                      (void*)&K, (void*)&N, (void*)&stride_row_x };
    const hipError_t err = hipLaunchCooperativeKernel((const void *) k_coop_q2_0_decode,
            dim3(grid), dim3(COOP_Q2_0_BLOCK), args, 0, stream);
    return err == hipSuccess;
}
