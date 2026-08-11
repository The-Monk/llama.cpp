#include "ggml.h"
#include "common.cuh"
#include "mmvf_qk.cuh"

#include <cstdlib>

// ROC9 GGML_HIP_MMVF_QK (default OFF): see mmvf_qk.cuh + mmvq.cu file-top
// comment for the root cause this fixes. At ne11==1 (single-token decode)
// mmvq.cu's generic dp4a path always launches quantize_row_q8_1_cuda to pack
// the fp32 activations into int8 q8_1 before the dot -- dead weight at
// batch=1 since decode is bandwidth-bound on the WEIGHT bytes, not the
// (tiny, 1-token) activation vector. This file instead dequantizes the
// K-quant weight block directly to fp32 in-register and FMAs it against the
// fp32 activation vector y, single kernel launch, no q8_1 intermediate --
// mirrors what mmvf.cu already does for F32/F16/BF16 and what Vulkan's
// mul_mat_vec_q4_k.comp does for K-quants.
//
// Layout reference (do NOT re-derive, matches convert.cu's
// dequantize_block_q4_K / dequantize_block_q6_K exactly, just remapped from
// their {32,64}-thread/4-output-per-thread shape to a 256-thread/
// 1-output-per-thread shape so each thread owns one fixed
// superblock-relative output element and walks every QK_K super-block of
// the row):
//   Q4_K: tid = il*64 + rem (il 0..3, rem 0..63). outpos = 64*il+rem always
//         (low nibble for rem<32, high nibble for rem>=32 -- both land on
//         outpos = 64*il+rem, verified against dequantize_block_q4_K).
//   Q6_K: tid = ip*128 + group*32 + il (ip 0..1, group 0..3, il 0..31).
//         outpos = 128*ip + il + 32*group; ql byte reused across two
//         groups (0&2 share ql[+0], 1&3 share ql[+32]), qh shift = 2*group,
//         scale index = 8*ip + il/16 + 2*group.

static __device__ __forceinline__ void get_scale_min_k4_qk(const int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

#define MMVF_QK_BLOCK_SIZE 256

template <typename block_t, bool is_q6_k>
static __global__ void mul_mat_vec_qk_f32(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int64_t ncols,
        const int stride_row,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {
    const int row         = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst  = blockIdx.z;
    const int tid         = threadIdx.x;

    const int channel_x = (int) fastdiv((uint32_t) channel_dst, channel_ratio);
    const int sample_x  = (int) fastdiv((uint32_t) sample_dst, sample_ratio);

    const block_t * x = (const block_t *) vx
        + int64_t(sample_x)*stride_sample_x + channel_x*stride_channel_x + int64_t(row)*stride_row;
    const float * yp = y + int64_t(sample_dst)*stride_sample_y + channel_dst*stride_channel_y;
    float       * dstp = dst + int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst;

    const int64_t nb = ncols / QK_K;

    float sumf = 0.0f;

    if constexpr (!is_q6_k) {
        const int il  = tid >> 6;          // 0..3
        const int rem = tid & 63;          // 0..63
        const bool hi = rem >= 32;
        const int qoff   = 32*il + (rem & 31);
        const int outpos = 64*il + rem;
        const int is     = 2*il + (hi ? 1 : 0);

        for (int64_t i = 0; i < nb; ++i) {
            const block_t * xb = x + i;
            const float dall = __low2float(xb->dm);
            const float dmin = __high2float(xb->dm);
            uint8_t sc, m;
            get_scale_min_k4_qk(is, xb->scales, sc, m);
            const uint8_t qbyte = xb->qs[qoff];
            const float nib = hi ? float(qbyte >> 4) : float(qbyte & 0xF);
            const float val = dall*sc*nib - dmin*m;
            sumf += val * yp[i*QK_K + outpos];
        }
    } else {
        const int ip    = tid >> 7;             // 0..1
        const int rem   = tid & 127;            // 0..127
        const int group = rem >> 5;              // 0..3
        const int il    = rem & 31;              // 0..31
        const bool hi      = group >= 2;
        const bool second  = (group & 1) != 0;
        const int qh_shift = 2*group;
        const int sc_idx   = 8*ip + il/16 + 2*group;
        const int outpos   = 128*ip + il + 32*group;

        for (int64_t i = 0; i < nb; ++i) {
            const block_t * xb = x + i;
            const float d = xb->d;
            const uint8_t ql_byte = xb->ql[64*ip + il + (second ? 32 : 0)];
            const uint8_t qh_byte = xb->qh[32*ip + il];
            const uint8_t nib4 = hi ? (ql_byte >> 4) : (ql_byte & 0xF);
            const int8_t  q6   = (int8_t)(nib4 | (((qh_byte >> qh_shift) & 3) << 4)) - 32;
            const float val = d * xb->scales[sc_idx] * q6;
            sumf += val * yp[i*QK_K + outpos];
        }
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = MMVF_QK_BLOCK_SIZE / warp_size;

    sumf = warp_reduce_sum<warp_size>(sumf);

    __shared__ float buf[n_warps];
    if (tid % warp_size == 0) {
        buf[tid / warp_size] = sumf;
    }
    __syncthreads();

    if (tid < warp_size) {
        float v = tid < n_warps ? buf[tid] : 0.0f;
        v = warp_reduce_sum<warp_size>(v);
        if (tid == 0) {
            dstp[row] = v;
        }
    }
}

bool ggml_cuda_should_use_mmvf_qk(enum ggml_type type, int cc, int64_t ne11) {
    static const bool enabled = (getenv("GGML_HIP_MMVF_QK") != nullptr);
    if (!enabled) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_AMD(cc) || !GGML_CUDA_CC_IS_RDNA4(cc)) {
        return false;
    }
    if (ne11 != 1) {
        return false;
    }
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K;
}

void ggml_cuda_mul_mat_vec_qk(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K);

    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(ne00 % QK_K == 0);
    GGML_ASSERT(ne10 == ne00);
    GGML_ASSERT(ne1 == 1); // decode only: this path is gated to ne11 == 1, no MUL_MAT_ID
    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(nb00 == ts_src0);
    GGML_ASSERT(nb10 == ts_src1);
    GGML_ASSERT(nb0  == ts_dst);

    const int stride_row         = (int) (src0->nb[1] / ts_src0);
    const int stride_channel_x   = (int) (src0->nb[2] / ts_src0);
    const int stride_channel_y   = (int) (src1->nb[2] / ts_src1);
    const int stride_channel_dst = (int) (dst->nb[2]  / ts_dst);
    const int stride_sample_x    = (int) (src0->nb[3] / ts_src0);
    const int stride_sample_y    = (int) (src1->nb[3] / ts_src1);
    const int stride_sample_dst  = (int) (dst->nb[3]  / ts_dst);

    const uint3 channel_ratio = init_fastdiv_values((uint32_t) (ne2 / ne02));
    const uint3 sample_ratio  = init_fastdiv_values((uint32_t) (ne3 / ne03));

    const dim3 block_nums(ne01, ne2, ne3);
    const dim3 block_dims(MMVF_QK_BLOCK_SIZE, 1, 1);

    const float * src1_d = (const float *) src1->data;
    float       * dst_d   = (float       *) dst->data;
    cudaStream_t stream = ctx.stream();

    if (src0->type == GGML_TYPE_Q4_K) {
        mul_mat_vec_qk_f32<block_q4_K, false><<<block_nums, block_dims, 0, stream>>>(
            src0->data, src1_d, dst_d, ne00, stride_row,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    } else {
        mul_mat_vec_qk_f32<block_q6_K, true><<<block_nums, block_dims, 0, stream>>>(
            src0->data, src1_d, dst_d, ne00, stride_row,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    }
}
