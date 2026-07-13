// card 141: RDNA4 2:4-structured-sparse fp16 SWMMAC, end-to-end. See
// mul_mat_2of4_f16.cuh for the full rationale. Mirrors mul_mat_2of4_fp8.cu's
// structure exactly (same per-lane operand math, same warp/tile shape), with
// two simplifications specific to fp16:
//   1. No per-block scale (`d`) at all -- fp16 values are stored/used raw,
//      so there is no `sh_dw` gather-and-multiply step the fp8 kernel needs
//      to turn a dequantized e4m3 code back into a real magnitude.
//   2. The dense activation operand is a plain online float->fp16 cast (no
//      RTN/amax block quantizer) into a flat fp16 buffer, not a specialized
//      block container.
//
// This is a from-scratch kernel (does NOT modify/reuse swmmac24.cuh's
// __global__ selftest kernels) that calls the SAME validated hardware
// builtin (__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32) using the SAME
// per-lane VGPR layout convention documented + hardware-validated in
// swmmac24.cuh (swmmac24_a_loc/swmmac24_b32_loc_16bit/swmmac24_d_loc,
// dataSizeBits==16). The per-lane operand math below is a direct, by-hand
// evaluation of those layout formulas for a wave32 launch (one warp = one
// 16x16 output tile, K accumulated 32-wide per SWMMAC call) -- IDENTICAL
// lane assignment to the fp8 kernel (dataSizeBits only changes how many
// bytes/lane each operand needs, not which lane owns which row/col):
//
//   A (sparse weight, block_2of4_f16, one block = 16 rows x 32 K):
//     lane = weight_row (0..15) for K-half-lo (groups 0..3, K 0..15)
//     lane = 16 + weight_row       for K-half-hi (groups 4..7, K 16..31)
//     -> thread tid<16 handles weight_row=tid, K-half-lo; tid>=16 handles
//        weight_row=tid-16, K-half-hi. a_arg = 8 packed fp16 values (qs[k_half*8
//        .. k_half*8+7], raw bit-for-bit, no scale). idx_arg = meta[2*k_half]
//        | (meta[2*k_half+1] << 8) (the block's meta bytes ARE the ISA's
//        16-bit sparsity_idx field, by construction -- see ggml-common.h's
//        block_2of4_f16 comment).
//
//   B (dense activation, flat fp16 buffer, one "block" = 32 K x 1 token/column):
//     lane = act_col (0..15) for K-half-lo (K 0..15)
//     lane = 16 + act_col          for K-half-hi (K 16..31)
//     -> thread tid<16 handles act_col=tid, K-half-lo; tid>=16 handles
//        act_col=tid-16, K-half-hi. b_arg = 16 packed fp16 values, i.e.
//        act_row[k_half*16 .. k_half*16+15].
//
//   D (output, 16x16, row=weight_row 0..15, col=act_col 0..15):
//     lane = act_col for row<8, 16+act_col for row>=8; slot (v8f index)
//     l = row & 7. So thread tid<16 owns output rows [0,7] at col=tid,
//     thread tid>=16 owns rows [8,15] at col=tid-16 -- no cross-thread
//     shared-memory gather needed at all (unlike the fp8 kernel's sh_dw),
//     since there's no per-row scale to look up.
#include "mul_mat_2of4_f16.cuh"

#include <cstring>

typedef _Float16 v8h  __attribute__((ext_vector_type(8)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float    v8f  __attribute__((ext_vector_type(8)));

// ggml_half is a raw IEEE-754 binary16 bit pattern (ggml-common.h) -- bit-for-bit
// identical to _Float16's in-memory representation, so this is a pure bitcast,
// not a value conversion.
static __device__ __forceinline__ _Float16 bits_to_half(ggml_half b) {
    _Float16 h;
    memcpy(&h, &b, sizeof(h));
    return h;
}

// Online dense fp16 activation caster: F32 -> raw fp16 bits, no scale/block
// codec at all (see block_2of4_f16 comment in ggml-common.h for why fp16
// doesn't need one, unlike the fp8 sibling kernel's k_quantize_act_f8e4m3).
// One thread per element, flat [token][k] layout.
static __global__ void k_cast_act_f16(
        const float * __restrict__ x, ggml_half * __restrict__ y,
        const int64_t n_per_row, const int64_t row_stride_floats) {
    const int64_t k = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    const int64_t m = blockIdx.y;
    if (k >= n_per_row) {
        return;
    }
    const _Float16 h = (_Float16) x[m * row_stride_floats + k];
    ggml_half b;
    memcpy(&b, &h, sizeof(b));
    y[m * n_per_row + k] = b;
}

// One warp (32 lanes) per 16(weight-row)x16(act-col) output tile, looping
// over K in 32-wide (one block_2of4_f16 / 32 activation elements) chunks.
__launch_bounds__(32, 1)
static __global__ void k_mul_mat_2of4_f16(
        const char * __restrict__ vweight, const ggml_half * __restrict__ act, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t nb01, const int64_t n_blocks_k,
        const int64_t K, const int64_t dst_row_stride_floats) {
#if defined(RDNA4)
    const int64_t n0  = (int64_t) blockIdx.x * 16; // weight-row tile origin
    const int64_t m0  = (int64_t) blockIdx.y * 16; // activation-col (token) tile origin
    const int     tid = threadIdx.x;

    const int k_half     = (tid < 16) ? 0 : 1;
    const int local_idx  = (tid < 16) ? tid : (tid - 16); // weight_row (A) == act_col (B), same value

    float acc[8];
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        acc[l] = 0.0f;
    }

    for (int64_t c = 0; c < n_blocks_k; ++c) {
        v8h  a_arg = {0,0,0,0,0,0,0,0};
        v16h b_arg = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        unsigned idxv = 0;

        const int64_t weight_row = n0 + local_idx;
        if (weight_row < N) {
            const block_2of4_f16 * blkw = (const block_2of4_f16 *) (vweight + weight_row * nb01 + c * (int64_t) sizeof(block_2of4_f16));
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                a_arg[j] = bits_to_half(blkw->qs[k_half*8 + j]);
            }
            idxv = (unsigned) blkw->meta[2*k_half] | ((unsigned) blkw->meta[2*k_half + 1] << 8);
        }

        const int64_t act_col = m0 + local_idx;
        if (act_col < M) {
            const ggml_half * arow = act + act_col * K + c * 32;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                b_arg[j] = bits_to_half(arow[k_half*16 + j]);
            }
        }

        v8f c0 = {0,0,0,0,0,0,0,0};
        v8f raw = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a_arg, b_arg, c0, idxv);

#pragma unroll
        for (int l = 0; l < 8; ++l) {
            acc[l] += raw[l];
        }
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
    GGML_UNUSED(K);
    GGML_UNUSED(dst_row_stride_floats);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

bool ggml_cuda_op_mul_mat_2of4_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_2OF4_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(src0->ne[0] % QK_2OF4_F16 == 0);

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    GGML_ASSERT(GGML_CUDA_CC_IS_RDNA4(cc) && "GGML_TYPE_2OF4_F16 requires RDNA4 (V_SWMMAC_F32_16X16X32_F16)");

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t n_blocks_k = K / QK_2OF4_F16;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<ggml_half> act_h(ctx.pool(), (size_t) (M * K));

    {
        const int64_t row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
        const dim3 block(256, 1, 1);
        const dim3 grid((K + block.x - 1) / block.x, M, 1);
        k_cast_act_f16<<<grid, block, 0, stream>>>((const float *) src1->data, act_h.get(), K, row_stride_floats);
    }

    {
        const int64_t nb01 = src0->nb[1];
        const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
        const dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
        const dim3 block(32, 1, 1);
        k_mul_mat_2of4_f16<<<grid, block, 0, stream>>>(
                (const char *) src0->data, act_h.get(), (float *) dst->data,
                M, N, nb01, n_blocks_k, K, dst_row_stride_floats);
    }

    return true;
}
