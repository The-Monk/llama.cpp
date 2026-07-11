// RDNA4 2:4-structured-sparse fp8 SWMMAC driver-completeness run: see
// mul_mat_2of4_fp8.cuh for the full rationale.
//
// This is a from-scratch kernel (does NOT modify/reuse swmmac24.cuh's
// __global__ selftest kernels) that calls the SAME validated hardware
// builtin (__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32) using the SAME
// per-lane VGPR layout convention documented + hardware-validated in
// swmmac24.cuh (swmmac24_a_loc/swmmac24_b32_loc_8bit/swmmac24_d_loc,
// dataSizeBits==8). The per-lane operand math below is a direct, by-hand
// evaluation of those layout formulas for a wave32 launch (one warp = one
// 16x16 output tile, K accumulated 32-wide per SWMMAC call):
//
//   A (sparse weight, block_2of4_fp8, one block = 16 rows x 32 K):
//     lane = weight_row (0..15) for K-half-lo (groups 0..3, K 0..15)
//     lane = 16 + weight_row       for K-half-hi (groups 4..7, K 16..31)
//     -> thread tid<16 handles weight_row=tid, K-half-lo; tid>=16 handles
//        weight_row=tid-16, K-half-hi. a_arg.x/.y = 4 packed qs bytes each
//        (physical cols 0-3/4-7, or 8-11/12-15 for the hi half) -- i.e.
//        exactly qs[k_half*8 .. k_half*8+7] as two little-endian uint32
//        words. idx_arg = meta[2*k_half] | (meta[2*k_half+1] << 8) (the
//        block's meta bytes ARE the ISA's 16-bit sparsity_idx field, by
//        construction -- see ggml-common.h's block_2of4_fp8 comment).
//
//   B (dense activation, block_f8e4m3, one block = 32 K x 1 token/column):
//     lane = act_col (0..15) for K-half-lo (K 0..15)
//     lane = 16 + act_col          for K-half-hi (K 16..31)
//     -> thread tid<16 handles act_col=tid, K-half-lo; tid>=16 handles
//        act_col=tid-16, K-half-hi. b_arg.{x,y,z,w} = 4 packed qs bytes
//        each, i.e. qs[k_half*16 + 4*i .. +3] for i=0..3.
//
//   D (output, 16x16, row=weight_row 0..15, col=act_col 0..15):
//     lane = act_col for row<8, 16+act_col for row>=8; slot (v8f index)
//     l = row & 7. So thread tid<16 owns output rows [0,7] at col=tid,
//     thread tid>=16 owns rows [8,15] at col=tid-16 -- i.e. EXACTLY the same
//     (act_col, K-half) split used to build B, meaning each thread already
//     holds its own correct per-column activation scale locally; only the
//     8 per-row WEIGHT scales it needs are foreign data, gathered once per
//     K-chunk via 16 shared-memory slots.
#include "mul_mat_2of4_fp8.cuh"

#include <cstring>

static __device__ __forceinline__ int32_t pack4(const uint8_t * p) {
    int32_t w;
    memcpy(&w, p, 4);
    return w;
}

// Online dense fp8 activation quantizer -> plain block_f8e4m3 (per-32-elem
// block, symmetric RTN, amax -> 448.0 e4m3 max finite magnitude). Reuses the
// exact device-side e4m3 codec (common.cuh) the rest of the fp8 driver uses
// -- ggml_cuda_fp32_to_e4m3/ggml_cuda_e4m3_to_fp32, same functions
// quantize_mmq_f8e4m3's software fallback and the KV-cache fp8 write path
// call. One block (32 threads) per 32-elem chunk of one activation row.
static __global__ void k_quantize_act_f8e4m3(
        const float * __restrict__ x, block_f8e4m3 * __restrict__ y,
        const int64_t n_blocks_k, const int64_t row_stride_floats) {
    const int64_t c   = blockIdx.x; // which 32-elem block along K
    const int64_t m   = blockIdx.y; // which token/row
    const int     tid = threadIdx.x; // 0..31, element index within the block

    __shared__ float sh_val[32];
    __shared__ float sh_scale;

    const float v = x[m * row_stride_floats + c * 32 + tid];
    sh_val[tid] = fabsf(v);
    __syncthreads();

    if (tid == 0) {
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            amax = fmaxf(amax, sh_val[i]);
        }
        const float d = amax / 448.0f;
        sh_scale = d;
        y[m * n_blocks_k + c].d = __float2half(d);
    }
    __syncthreads();

    const float d  = sh_scale;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    y[m * n_blocks_k + c].qs[tid] = ggml_cuda_fp32_to_e4m3(v * id);
}

typedef int   v2i __attribute__((ext_vector_type(2)));
typedef int   v4i __attribute__((ext_vector_type(4)));
typedef float v8f __attribute__((ext_vector_type(8)));

// One warp (32 lanes) per 16(weight-row)x16(act-col) output tile, looping
// over K in 32-wide (one block_2of4_fp8 / one block_f8e4m3) chunks.
__launch_bounds__(32, 1)
static __global__ void k_mul_mat_2of4_fp8(
        const char * __restrict__ vweight, const block_f8e4m3 * __restrict__ act, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t nb01, const int64_t n_blocks_k,
        const int64_t dst_row_stride_floats) {
#if defined(RDNA4)
    const int64_t n0  = (int64_t) blockIdx.x * 16; // weight-row tile origin
    const int64_t m0  = (int64_t) blockIdx.y * 16; // activation-col (token) tile origin
    const int     tid = threadIdx.x;

    const int k_half     = (tid < 16) ? 0 : 1;
    const int local_idx  = (tid < 16) ? tid : (tid - 16); // weight_row (A) == act_col (B), same value

    __shared__ float sh_dw[16];

    float acc[8];
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        acc[l] = 0.0f;
    }

    for (int64_t c = 0; c < n_blocks_k; ++c) {
        v2i a_arg     = {0, 0};
        v4i b_arg     = {0, 0, 0, 0};
        unsigned idxv = 0;
        float    d_a_local = 0.0f;

        const int64_t weight_row = n0 + local_idx;
        if (weight_row < N) {
            const block_2of4_fp8 * blkw = (const block_2of4_fp8 *) (vweight + weight_row * nb01 + c * (int64_t) sizeof(block_2of4_fp8));
            a_arg.x = pack4(blkw->qs + k_half*8 + 0);
            a_arg.y = pack4(blkw->qs + k_half*8 + 4);
            idxv    = (unsigned) blkw->meta[2*k_half] | ((unsigned) blkw->meta[2*k_half + 1] << 8);
            const float d_w = __half2float(blkw->d);
            if (k_half == 0) {
                sh_dw[local_idx] = d_w;
            }
        } else if (k_half == 0) {
            sh_dw[local_idx] = 0.0f;
        }

        const int64_t act_col = m0 + local_idx;
        if (act_col < M) {
            const block_f8e4m3 & blka = act[act_col * n_blocks_k + c];
            b_arg.x    = pack4(blka.qs + k_half*16 + 0);
            b_arg.y    = pack4(blka.qs + k_half*16 + 4);
            b_arg.z    = pack4(blka.qs + k_half*16 + 8);
            b_arg.w    = pack4(blka.qs + k_half*16 + 12);
            d_a_local  = __half2float(blka.d);
        }
        __syncthreads(); // sh_dw fully populated before anyone reads it below

        v8f c0 = {0,0,0,0,0,0,0,0};
        v8f raw = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a_arg, b_arg, c0, idxv);

        const int out_row_base = (tid >= 16) ? 8 : 0;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            acc[l] += raw[l] * sh_dw[out_row_base + l] * d_a_local;
        }
        __syncthreads(); // sh_dw about to be overwritten next chunk
    }

    const int out_col      = local_idx;      // act-col, local to this tile
    const int out_row_base = (tid >= 16) ? 8 : 0;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int64_t n = n0 + out_row_base + l; // weight row / output feature
        const int64_t m = m0 + out_col;          // activation col / token
        if (m < M && n < N) {
            dst[m * dst_row_stride_floats + n] = acc[l];
        }
    }
#else
    GGML_UNUSED(vweight);
    GGML_UNUSED(act);
    GGML_UNUSED(dst);
    GGML_UNUSED(M);
    GGML_UNUSED(N);
    GGML_UNUSED(nb01);
    GGML_UNUSED(n_blocks_k);
    GGML_UNUSED(dst_row_stride_floats);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

bool ggml_cuda_op_mul_mat_2of4_fp8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_2OF4_FP8);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(src0->ne[0] % QK_2OF4_FP8 == 0);

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    GGML_ASSERT(GGML_CUDA_CC_IS_RDNA4(cc) && "GGML_TYPE_2OF4_FP8 requires RDNA4 (V_SWMMAC_F32_16X16X32_FP8_FP8)");

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t n_blocks_k = K / QK_2OF4_FP8;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_f8e4m3> act_q(ctx.pool(), (size_t) (M * n_blocks_k));

    {
        const int64_t row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
        const dim3 grid(n_blocks_k, M, 1);
        const dim3 block(32, 1, 1);
        k_quantize_act_f8e4m3<<<grid, block, 0, stream>>>((const float *) src1->data, act_q.get(), n_blocks_k, row_stride_floats);
    }

    {
        const int64_t nb01 = src0->nb[1];
        const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
        const dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
        const dim3 block(32, 1, 1);
        k_mul_mat_2of4_fp8<<<grid, block, 0, stream>>>(
                (const char *) src0->data, act_q.get(), (float *) dst->data,
                M, N, nb01, n_blocks_k, dst_row_stride_floats);
    }

    return true;
}
