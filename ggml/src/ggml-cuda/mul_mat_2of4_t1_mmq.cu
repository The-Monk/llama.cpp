// See mul_mat_2of4_t1_mmq.cuh for the full rationale.
#include "mul_mat_2of4_t1_mmq.cuh"
#include "vecdotq.cuh" // ggml_cuda_2of4_t1_{sgn,msk}_lut -- the validated decode LUTs

#include <cstring>

typedef int v2i_t1 __attribute__((ext_vector_type(2)));
typedef int v4i_t1 __attribute__((ext_vector_type(4)));
typedef int v8i_t1 __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ int32_t pack4_t1(const uint8_t * p) {
    int32_t w;
    memcpy(&w, p, 4);
    return w;
}

// Online per-32 int8 activation quantizer. Mechanical clone of the fp8
// kernel's k_quantize_act_f8e4m3_mmq_fast shape (128 threads/block, float4
// loads, 8-lane shuffle amax reduction, zero LDS/sync); output is a flat
// int8 code array [M][K] plus a flat float scale array [M][K/32] instead of
// an AoS block, so the GEMM kernel's LDS staging can copy plain aligned
// 16-byte vectors.
static __global__ void k_quantize_act_q8_t1(
        const float * __restrict__ x, int8_t * __restrict__ yq, float * __restrict__ yd,
        const int64_t n_groups_k, const int64_t row_stride_floats) {
    const int      group_id = threadIdx.x >> 3; // 0..15
    const int      lane8    = threadIdx.x & 7;  // 0..7
    const int64_t  g        = (int64_t) blockIdx.x * 16 + group_id;
    const int64_t  m        = blockIdx.y;

    if (g >= n_groups_k) {
        return;
    }

    const int64_t elem0 = g * 32 + lane8 * 4;
    const float4  xi    = *((const float4 *) (x + m * row_stride_floats + elem0));

    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset, WARP_SIZE));
    }

    const float d  = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    if (lane8 == 0) {
        yd[m * n_groups_k + g] = d;
    }
    char4 q;
    q.x = (int8_t) roundf(xi.x * id);
    q.y = (int8_t) roundf(xi.y * id);
    q.z = (int8_t) roundf(xi.z * id);
    q.w = (int8_t) roundf(xi.w * id);
    *(char4 *) (yq + m * (n_groups_k * 32) + elem0) = q;
}

template <int BM, int BN, int NWARPS, bool need_check>
__launch_bounds__(NWARPS * 32, 1)
static __global__ void k_mul_mat_2of4_t1_mmq(
        const char * __restrict__ vweight, const int8_t * __restrict__ actq,
        const float * __restrict__ actd, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t nb01, const int64_t n_blocks_k,
        const int64_t dst_row_stride_floats) {
    constexpr int NTILES_N = BN / 16;
    constexpr int NTILES_M = BM / 16;
    constexpr int NTILES   = NTILES_M * NTILES_N;
    constexpr int NTX      = NTILES / NWARPS;
    static_assert(BM % 32 == 0 && BN % 32 == 0, "staging assumes one thread/row, 32 lanes/warp");
    static_assert(NTILES % NWARPS == 0, "tiles must divide evenly across warps");
    constexpr int WARPS_A = BM / 32; // warps that stage the activation operand
    constexpr int WARPS_B = BN / 32; // warps that stage the weight operand
    static_assert(WARPS_A + WARPS_B <= NWARPS, "not enough warps to cover staging");

#if defined(RDNA4)
    const int64_t n0      = (int64_t) blockIdx.x * BN;
    const int64_t m0      = (int64_t) blockIdx.y * BM;
    const int     lane    = threadIdx.x;
    const int     warp_id = threadIdx.y;
    // Warp-uniform staging dispatch (see mul_mat_2of4_fp8_mmq.cu).
    const int warp_id_u = __builtin_amdgcn_readfirstlane(warp_id);

    // One chunk = one block_2of4_t1 = K=128 = 4 SWMMAC K32 windows.
    // Row pitches are PADDED to break LDS bank conflicts: the natural
    // strides (32 ints = 128 B for activations, 16 ints for codes) put all
    // 16 rows of a fragment read -- and all 32 rows of a staging store --
    // on the same bank (128 B = exactly the 32-bank period). 36/18-int
    // pitches keep 16-byte vector alignment (both are 0 mod 4 / 0 mod 2 in
    // ints where the reads land) while spreading the 16-row access groups
    // across banks (36: 2-way worst case, 18: conflict-free for 16 rows).
    // Same fix, same reasoning as the standalone swmmac GEMM bench
    // (~/swmmac-bench/gemm_bench3.hip) whose padded pitches measured 250
    // TOPS where an unpadded port of this kernel measured 40.
    __shared__ int      sh_actq[BM][36];  // K128 of int8 activation codes (32 used + 4 pad)
    __shared__ float    sh_da[BM][4];     // per-32 activation scales
    __shared__ int      sh_wcode[BN][18]; // 64 survivor codes {+1,-1,0} (16 used + 2 pad)
    __shared__ unsigned sh_wmeta[BN][4];  // 32 meta nibbles (= ISA idx bits)
    __shared__ float    sh_dw[BN];        // per-128 ternary scale

    auto load_chunk = [&] (int64_t c) {
        if (warp_id_u < WARPS_A) {
            const int     row = warp_id_u * 32 + lane;
            const int64_t m   = m0 + row;
            if (!need_check || m < M) {
                const int8_t * src = actq + m * (n_blocks_k * 128) + c * 128;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *(v4i_t1 *) &sh_actq[row][4 * i] = *(const v4i_t1 *) (src + 16 * i);
                }
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    sh_da[row][w] = actd[m * (n_blocks_k * 4) + c * 4 + w];
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *(v4i_t1 *) &sh_actq[row][4 * i] = v4i_t1{0, 0, 0, 0};
                }
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    sh_da[row][w] = 0.0f;
                }
            }
        } else if (warp_id_u < WARPS_A + WARPS_B) {
            const int     row = (warp_id_u - WARPS_A) * 32 + lane;
            const int64_t n   = n0 + row;
            if (!need_check || n < N) {
                // 26-byte packed struct at arbitrary alignment: assemble by
                // bytes (memcpy -> the compiler picks safe loads), exactly
                // like the fp8 kernel's pack4.
                const block_2of4_t1 * blk = (const block_2of4_t1 *) (vweight + n * nb01 + c * (int64_t) sizeof(block_2of4_t1));
                const unsigned sig_lo = (unsigned) pack4_t1(blk->signs + 0);
                const unsigned sig_hi = (unsigned) pack4_t1(blk->signs + 4);
                // Assemble the survivor codes {+1,-1,0} ONCE per (row, chunk)
                // here at staging, instead of per (tile, window) in the hot
                // loop: kills 8x the sgn/msk LUT gather traffic a naive port
                // pays per SWMMAC. Indexing mirrors vec_dot_2of4_t1_q8_1
                // exactly (window w = iqs, meta byte m, sign nibble at 4*m).
                // NOTE: a software-pipelined fetch/commit split of this
                // loader (global loads for c+1 issued into registers before
                // chunk c's compute) was implemented and MEASURED: 75.5 vs
                // 100.3 TFLOPS at m=4096/n=512/k=14336 -- the +45 VGPRs cost
                // more occupancy than the hidden latency bought. Reverted.
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const unsigned mw = (unsigned) pack4_t1(blk->meta + 4 * w);
                    sh_wmeta[row][w] = mw;
                    const unsigned s16 = ((w & 2) ? sig_hi : sig_lo) >> ((w & 1) * 16);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const unsigned lane16 = mw >> (h * 16);
                        const unsigned sh8    = (s16 >> (h * 8)) & 0xFFu;
                        sh_wcode[row][4 * w + 2 * h + 0] =
                            (int) (ggml_cuda_2of4_t1_sgn_lut[sh8 & 0xFu] & ggml_cuda_2of4_t1_msk_lut[lane16 & 0xFFu]);
                        sh_wcode[row][4 * w + 2 * h + 1] =
                            (int) (ggml_cuda_2of4_t1_sgn_lut[sh8 >> 4]   & ggml_cuda_2of4_t1_msk_lut[(lane16 >> 8) & 0xFFu]);
                    }
                }
                sh_dw[row] = __half2float(*(const __half *) &blk->d);
            } else {
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    sh_wmeta[row][w] = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sh_wcode[row][4 * w + i] = 0;
                    }
                }
                sh_dw[row] = 0.0f;
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

    const int k_half       = (lane < 16) ? 0 : 1;
    const int local_idx    = (lane < 16) ? lane : (lane - 16);
    const int out_row_base = (lane >= 16) ? 8 : 0;

    for (int64_t c = 0; c < n_blocks_k; ++c) {
        load_chunk(c);
        __syncthreads();

#pragma unroll
        for (int s = 0; s < NTX; ++s) {
            const int t  = warp_id + s * NWARPS;
            const int mi = t / NTILES_N;
            const int ni = t % NTILES_N;

            const int w_row = ni * 16 + local_idx; // weight row within the staged tile
            const int a_col = mi * 16 + local_idx; // activation row (token)

            // Per-128 weight scale: constant across the 4 windows -> hoisted.
            float dw_l[8];
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                dw_l[l] = sh_dw[ni * 16 + out_row_base + l];
            }
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                // This lane's 16 idx bits = its k-half's 2 meta bytes,
                // unchanged from block storage (ISA sparsity_idx packing).
                const int idxv = (int) ((sh_wmeta[w_row][w] >> (k_half * 16)) & 0xFFFFu);
                const v2i_t1 a_arg = *(const v2i_t1 *) &sh_wcode[w_row][4 * w + 2 * k_half];
                const v4i_t1 b_arg = *(const v4i_t1 *) &sh_actq[a_col][8 * w + 4 * k_half];

                const v8i_t1 c0  = {0, 0, 0, 0, 0, 0, 0, 0};
                const v8i_t1 raw = __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(true, a_arg, true, b_arg, c0, idxv, false);

                const float da = sh_da[a_col][w];
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    acc[s][l] += (float) raw[l] * (dw_l[l] * da);
                }
            }
        }

        __syncthreads(); // compute -> next-load barrier (single buffer)
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
    GGML_UNUSED(vweight);
    GGML_UNUSED(actq);
    GGML_UNUSED(actd);
    GGML_UNUSED(dst);
    GGML_UNUSED(M);
    GGML_UNUSED(N);
    GGML_UNUSED(nb01);
    GGML_UNUSED(n_blocks_k);
    GGML_UNUSED(dst_row_stride_floats);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

bool ggml_cuda_op_mul_mat_2of4_t1_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_2OF4_T1);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(src0->ne[0] % QK_2OF4_T1 == 0);

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    GGML_ASSERT(GGML_CUDA_CC_IS_RDNA4(cc) && "2of4_t1 SWMMAC MMQ requires RDNA4");

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];          // weight rows (output features)
    const int64_t M = src1->ne[1];          // tokens
    const int64_t n_blocks_k = K / QK_2OF4_T1;
    const int64_t n_groups_k = K / 32;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<int8_t> act_q(ctx.pool(), (size_t) (M * K));
    ggml_cuda_pool_alloc<float>  act_d(ctx.pool(), (size_t) (M * n_groups_k));
    {
        const int64_t row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
        const dim3 grid((n_groups_k + 15) / 16, M, 1);
        const dim3 block(128, 1, 1);
        k_quantize_act_q8_t1<<<grid, block, 0, stream>>>(
                (const float *) src1->data, act_q.get(), act_d.get(), n_groups_k, row_stride_floats);
    }

    const int64_t nb01 = src0->nb[1];
    const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
    const char * d_w = (const char *) src0->data;
    float * d_dst = (float *) dst->data;

#define LAUNCH_T1_MMQ(BM_, BN_, NWARPS_)                                                              \
    do {                                                                                              \
        const dim3 grid_((N + (BN_) - 1) / (BN_), (M + (BM_) - 1) / (BM_), 1);                        \
        const dim3 block_(32, (NWARPS_), 1);                                                          \
        const bool need_check = (M % (BM_) != 0) || (N % (BN_) != 0);                                 \
        if (need_check) {                                                                             \
            k_mul_mat_2of4_t1_mmq<BM_, BN_, NWARPS_, true><<<grid_, block_, 0, stream>>>(             \
                    d_w, act_q.get(), act_d.get(), d_dst, M, N, nb01, n_blocks_k, dst_row_stride_floats); \
        } else {                                                                                      \
            k_mul_mat_2of4_t1_mmq<BM_, BN_, NWARPS_, false><<<grid_, block_, 0, stream>>>(            \
                    d_w, act_q.get(), act_d.get(), d_dst, M, N, nb01, n_blocks_k, dst_row_stride_floats); \
        }                                                                                             \
    } while (0)

    // Geometry: 128x128x8 default (the fp8 capstone winner); 64x64x8 for the
    // low-occupancy k/v-proj shape (N=1024), the fp8 sweep's measured winner
    // there. Env override for sweeps: GGML_HIP_2OF4_T1_MMQ_GEOM.
    const char * geom = getenv("GGML_HIP_2OF4_T1_MMQ_GEOM");
    if (geom != nullptr && strcmp(geom, "64x64x8") == 0) {
        LAUNCH_T1_MMQ(64, 64, 8);
    } else if (geom != nullptr && strcmp(geom, "64x128x8") == 0) {
        LAUNCH_T1_MMQ(64, 128, 8);
    } else if (geom != nullptr && strcmp(geom, "128x64x8") == 0) {
        LAUNCH_T1_MMQ(128, 64, 8);
    } else if (geom != nullptr && strcmp(geom, "256x128x16") == 0) {
        LAUNCH_T1_MMQ(256, 128, 16);
    } else if (N <= 1024) {
        LAUNCH_T1_MMQ(64, 64, 8);
    } else {
        LAUNCH_T1_MMQ(128, 128, 8);
    }
#undef LAUNCH_T1_MMQ

    return true;
}
