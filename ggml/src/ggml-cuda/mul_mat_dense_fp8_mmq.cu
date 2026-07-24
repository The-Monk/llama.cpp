// See mul_mat_dense_fp8_mmq.cuh -- dense-fp8 twin of mul_mat_2of4_fp8_mmq.cu,
// same MMQ-grade cooperative-tile/double-buffered-LDS shape, dense WMMA
// math instead of sparse SWMMAC.
#include "mul_mat_dense_fp8_mmq.cuh"

#include <cstring>

static __device__ __forceinline__ int32_t pack4_dmmq(const uint8_t * p) {
    int32_t w;
    memcpy(&w, p, 4);
    return w;
}

typedef float v8f_dmmq __attribute__((ext_vector_type(8)));

// T162 FIX 1: see mul_mat_2of4_fp8_mmq.cu's k_quantize_act_f8e4m3_mmq_fast
// for the full rationale -- byte-for-byte the same rewrite, duplicated here
// per this codebase's one-quantizer-per-kernel-file convention.
static __global__ void k_quantize_act_f8e4m3_dmmq_fast(
        const float * __restrict__ x, block_f8e4m3 * __restrict__ y,
        const int64_t n_blocks_k, const int64_t row_stride_floats) {
    const int      group_id = threadIdx.x >> 3;
    const int      lane8    = threadIdx.x & 7;
    const int64_t  c        = (int64_t) blockIdx.x * 16 + group_id;
    const int64_t  m        = blockIdx.y;

    if (c >= n_blocks_k) {
        return;
    }

    const int64_t elem0 = c * 32 + lane8 * 4;
    const float4  xi    = *((const float4 *) (x + m * row_stride_floats + elem0));

    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset, WARP_SIZE));
    }

    const float d  = amax / 448.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    block_f8e4m3 & blk = y[m * n_blocks_k + c];
    if (lane8 == 0) {
        blk.d = __float2half(d);
    }
    uint8_t * qs = blk.qs + lane8 * 4;
    qs[0] = ggml_cuda_fp32_to_e4m3(xi.x * id);
    qs[1] = ggml_cuda_fp32_to_e4m3(xi.y * id);
    qs[2] = ggml_cuda_fp32_to_e4m3(xi.z * id);
    qs[3] = ggml_cuda_fp32_to_e4m3(xi.w * id);
}

// Dense-fp8 twin of k_mul_mat_2of4_fp8_mmq -- same BM x BN cooperative tile,
// NWARPS warps, NTX register-blocked tiles/warp, double-buffered LDS,
// ONE __syncthreads()/chunk. Both A (weight) and B (activation) are full
// dense block_f8e4m3 rows now (8 ints/row, not 4 -- no compression, no
// metadata), and compute is TWO WMMA calls/chunk (copied from the
// validated mma.cuh fp8 mma() overload) instead of one SWMMAC call.
template <int BM, int BN, int NWARPS, bool need_check>
__launch_bounds__(NWARPS * 32, 1)
static __global__ void k_mul_mat_dense_fp8_mmq(
        const block_f8e4m3 * __restrict__ weight, const block_f8e4m3 * __restrict__ act, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t n_blocks_k,
        const int64_t dst_row_stride_floats) {
    constexpr int NTILES_N = BN / 16;
    constexpr int NTILES_M = BM / 16;
    constexpr int NTILES   = NTILES_M * NTILES_N;
    constexpr int NTX      = NTILES / NWARPS;
    static_assert(BM % 32 == 0 && BN % 32 == 0, "staging assumes one thread/row, 32 lanes/warp");
    static_assert(NTILES % NWARPS == 0, "tiles must divide evenly across warps");
    constexpr int WARPS_A = BM / 32;
    constexpr int WARPS_B = BN / 32;
    static_assert(WARPS_A + WARPS_B <= NWARPS, "not enough warps to cover staging");

#if defined(RDNA4)
    using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;

    const int64_t n0      = (int64_t) blockIdx.x * BN;
    const int64_t m0      = (int64_t) blockIdx.y * BM;
    const int     lane    = threadIdx.x;
    const int     warp_id = threadIdx.y;
    const int     warp_id_u = __builtin_amdgcn_readfirstlane(warp_id);

    // T162 FIX 2b tried and reverted -- see mul_mat_2of4_fp8_mmq.cu for the
    // measured result (small net regression, not a win).
    __shared__ int   sh_actq[2][BM][8]; // full dense row, 8 ints = 32 bytes
    __shared__ float sh_da[2][BM];
    __shared__ int   sh_wq[2][BN][8];   // full dense row, 8 ints = 32 bytes (no compression for dense)
    __shared__ float sh_dw[2][BN];

    auto load_chunk = [&] (int64_t c, int buf) {
        if (warp_id_u < WARPS_A) {
            const int     row = warp_id_u * 32 + lane;
            const int64_t m   = m0 + row;
            if (!need_check || m < M) {
                const block_f8e4m3 & blk = act[m * n_blocks_k + c];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    sh_actq[buf][row][i] = pack4_dmmq(blk.qs + 4 * i);
                }
                sh_da[buf][row] = __half2float(blk.d);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    sh_actq[buf][row][i] = 0;
                }
                sh_da[buf][row] = 0.0f;
            }
        } else if (warp_id_u < WARPS_A + WARPS_B) {
            const int     row = (warp_id_u - WARPS_A) * 32 + lane;
            const int64_t n   = n0 + row;
            if (!need_check || n < N) {
                const block_f8e4m3 & blk = weight[n * n_blocks_k + c];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    sh_wq[buf][row][i] = pack4_dmmq(blk.qs + 4 * i);
                }
                sh_dw[buf][row] = __half2float(blk.d);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    sh_wq[buf][row][i] = 0;
                }
                sh_dw[buf][row] = 0.0f;
            }
        }
    };

    float acc[NTX][8];
#pragma unroll
    for (int s = 0; s < NTX; ++s) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            acc[s][l] = 0.0f;
        }
    }

    load_chunk(0, 0);
    __syncthreads();

    const int k_half       = (lane < 16) ? 0 : 1;
    const int local_idx    = (lane < 16) ? lane : (lane - 16);
    const int out_row_base = (lane >= 16) ? 8 : 0;

    int cur = 0;
    for (int64_t c = 0; c < n_blocks_k; ++c) {
        const int nxt = cur ^ 1;
        if (c + 1 < n_blocks_k) {
            load_chunk(c + 1, nxt);
        }

#pragma unroll
        for (int s = 0; s < NTX; ++s) {
            const int t  = warp_id + s * NWARPS;
            const int mi = t / NTILES_N;
            const int ni = t % NTILES_N;

            const int w_row = ni * 16 + local_idx;
            const int a_col = mi * 16 + local_idx;

            const int32x2_t a_vec0 = { sh_wq[cur][w_row][k_half*4 + 0], sh_wq[cur][w_row][k_half*4 + 1] };
            const int32x2_t a_vec1 = { sh_wq[cur][w_row][k_half*4 + 2], sh_wq[cur][w_row][k_half*4 + 3] };
            const int32x2_t b_vec0 = { sh_actq[cur][a_col][k_half*4 + 0], sh_actq[cur][a_col][k_half*4 + 1] };
            const int32x2_t b_vec1 = { sh_actq[cur][a_col][k_half*4 + 2], sh_actq[cur][a_col][k_half*4 + 3] };

            v8f_dmmq accf = {0, 0, 0, 0, 0, 0, 0, 0};
            accf = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec0, b_vec0, accf);
            accf = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec1, b_vec1, accf);

#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float dw_row = sh_dw[cur][ni * 16 + out_row_base + l];
                const float da_col = sh_da[cur][a_col];
                acc[s][l] += accf[l] * dw_row * da_col;
            }
        }

        __syncthreads();
        cur = nxt;
    }

#pragma unroll
    for (int s = 0; s < NTX; ++s) {
        const int t  = warp_id + s * NWARPS;
        const int mi = t / NTILES_N;
        const int ni = t % NTILES_N;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int64_t n = n0 + ni * 16 + out_row_base + l;
            const int64_t m = m0 + mi * 16 + local_idx;
            if (!need_check || (m < M && n < N)) {
                dst[m * dst_row_stride_floats + n] = acc[s][l];
            }
        }
    }
#else
    GGML_UNUSED(weight);
    GGML_UNUSED(act);
    GGML_UNUSED(dst);
    GGML_UNUSED(M);
    GGML_UNUSED(N);
    GGML_UNUSED(n_blocks_k);
    GGML_UNUSED(dst_row_stride_floats);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

enum class DmmqGeom { G128x128x8, G64x64x4, G256x128x16, G128x256x16,
                       // T162 k/v-proj occupancy sweep (dense twin of the sparse sweep)
                       G64x64x8, G64x32x4, G64x32x8, G32x64x4, G32x64x8, G96x64x8, G64x96x8 };

static DmmqGeom ggml_cuda_dense_fp8_mmq_geom() {
    static const DmmqGeom g = [] {
        const char * env = getenv("GGML_HIP_DENSE_FP8_MMQ_GEOM");
        if (env == nullptr) { return DmmqGeom::G128x128x8; }
        if (strcmp(env, "64x64x4") == 0)    { return DmmqGeom::G64x64x4; }
        if (strcmp(env, "256x128x16") == 0) { return DmmqGeom::G256x128x16; }
        if (strcmp(env, "128x256x16") == 0) { return DmmqGeom::G128x256x16; }
        if (strcmp(env, "64x64x8") == 0)    { return DmmqGeom::G64x64x8; }
        if (strcmp(env, "64x32x4") == 0)    { return DmmqGeom::G64x32x4; }
        if (strcmp(env, "64x32x8") == 0)    { return DmmqGeom::G64x32x8; }
        if (strcmp(env, "32x64x4") == 0)    { return DmmqGeom::G32x64x4; }
        if (strcmp(env, "32x64x8") == 0)    { return DmmqGeom::G32x64x8; }
        if (strcmp(env, "96x64x8") == 0)    { return DmmqGeom::G96x64x8; }
        if (strcmp(env, "64x96x8") == 0)    { return DmmqGeom::G64x96x8; }
        return DmmqGeom::G128x128x8;
    }();
    return g;
}

bool ggml_cuda_op_mul_mat_dense_fp8_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F8E4M3);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(src0->ne[0] % QK_F8E4M3 == 0);

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    GGML_ASSERT(GGML_CUDA_CC_IS_RDNA4(cc) && "GGML_HIP_DENSE_FP8_MMQ requires RDNA4");

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t n_blocks_k = K / QK_F8E4M3;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_f8e4m3> act_q(ctx.pool(), (size_t) (M * n_blocks_k));
    {
        const int64_t row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
        const dim3 grid((n_blocks_k + 15) / 16, M, 1);
        const dim3 block(128, 1, 1);
        k_quantize_act_f8e4m3_dmmq_fast<<<grid, block, 0, stream>>>((const float *) src1->data, act_q.get(), n_blocks_k, row_stride_floats);
    }

    const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
    const block_f8e4m3 * d_act = act_q.get();
    const block_f8e4m3 * d_w   = (const block_f8e4m3 *) src0->data;
    float * d_dst = (float *) dst->data;

    // T162 k/v-proj occupancy fix, dense twin -- see mul_mat_2of4_fp8_mmq.cu
    // for the full rationale. Isolated sweep (N=1024/K=4096/M=512): 64x64x8
    // wins at 67.18us, BEATS stock's 74.13us (-9.4%) -- vs the flat
    // 128x128x8 default's measured 97.9us. Opt-in via
    // GGML_HIP_DENSE_FP8_MMQ_ADAPTIVE (default off).
    const bool adaptive = getenv("GGML_HIP_DENSE_FP8_MMQ_ADAPTIVE") != nullptr;
    const DmmqGeom geom = (adaptive && N <= 1024) ? DmmqGeom::G64x64x8 : ggml_cuda_dense_fp8_mmq_geom();

#define LAUNCH_DMMQ(BM_, BN_, NWARPS_)                                                          \
    do {                                                                                        \
        const dim3 grid_((N + (BN_) - 1) / (BN_), (M + (BM_) - 1) / (BM_), 1);                  \
        const dim3 block_(32, (NWARPS_), 1);                                                    \
        const bool need_check = (M % (BM_) != 0) || (N % (BN_) != 0);                           \
        if (need_check) {                                                                       \
            k_mul_mat_dense_fp8_mmq<BM_, BN_, NWARPS_, true><<<grid_, block_, 0, stream>>>(      \
                    d_w, d_act, d_dst, M, N, n_blocks_k, dst_row_stride_floats);                 \
        } else {                                                                                 \
            k_mul_mat_dense_fp8_mmq<BM_, BN_, NWARPS_, false><<<grid_, block_, 0, stream>>>(     \
                    d_w, d_act, d_dst, M, N, n_blocks_k, dst_row_stride_floats);                 \
        }                                                                                        \
    } while (0)

    switch (geom) {
        case DmmqGeom::G64x64x4:    LAUNCH_DMMQ(64, 64, 4);    break;
        case DmmqGeom::G256x128x16: LAUNCH_DMMQ(256, 128, 16); break;
        case DmmqGeom::G128x256x16: LAUNCH_DMMQ(128, 256, 16); break;
        case DmmqGeom::G64x64x8:    LAUNCH_DMMQ(64, 64, 8);    break;
        case DmmqGeom::G64x32x4:    LAUNCH_DMMQ(64, 32, 4);    break;
        case DmmqGeom::G64x32x8:    LAUNCH_DMMQ(64, 32, 8);    break;
        case DmmqGeom::G32x64x4:    LAUNCH_DMMQ(32, 64, 4);    break;
        case DmmqGeom::G32x64x8:    LAUNCH_DMMQ(32, 64, 8);    break;
        case DmmqGeom::G96x64x8:    LAUNCH_DMMQ(96, 64, 8);    break;
        case DmmqGeom::G64x96x8:    LAUNCH_DMMQ(64, 96, 8);    break;
        default:                    LAUNCH_DMMQ(128, 128, 8);  break;
    }
#undef LAUNCH_DMMQ

    return true;
}
