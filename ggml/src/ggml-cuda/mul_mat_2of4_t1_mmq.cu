// See mul_mat_2of4_t1_mmq.cuh for the full rationale.
#include "mul_mat_2of4_t1_mmq.cuh"
#include "vecdotq.cuh" // ggml_cuda_2of4_t1_{sgn,msk}_lut -- the validated decode LUTs
#include "hipblaslt_wcache.cuh" // [TAG_2OF4_T1_KPIPE] pre-pack cache invalidation

#include <cstring>
#include <mutex>
#include <unordered_map>

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
// [TAG_2OF4_T1_SCALE_HOIST] HOIST128: one scale per 128 values (= one
// block_2of4_t1 chunk). A warp's 32 lanes are 4 consecutive 32-groups, i.e.
// one aligned 128-value chunk (16 groups/block, K % 128 == 0), so the amax
// reduction just spans the warp; the per-32 scale array is still written,
// with the same d in all 4 slots, so the layout is unchanged.
template <bool HOIST128>
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
    for (int offset = HOIST128 ? 16 : 4; offset > 0; offset >>= 1) {
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

// [TAG_2OF4_T1_KPIPE] PIPE (env GGML_HIP_2OF4_T1_KPIPE, default 0):
//   0: shipped single-buffer staging, two barriers per chunk.
//   1: double-buffered LDS: chunk c+1 is staged into the other buffer, then
//      chunk c is computed; one barrier per chunk.
//   2: as 1, but the c+1 global loads are issued into registers BEFORE
//      computing c and written to LDS after it (fetch/commit split).
// The compute body is unchanged in all three, so results are bit-identical.
template <int BM, int BN, int NWARPS, bool need_check, bool HOIST128, int PIPE = 0>
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
    constexpr int NBUF = PIPE >= 1 ? 2 : 1; // [TAG_2OF4_T1_KPIPE] LDS buffers
    __shared__ int      sh_actq[NBUF][BM][36];  // K128 of int8 activation codes (32 used + 4 pad)
    __shared__ float    sh_da[NBUF][BM][4];     // per-32 activation scales
    __shared__ int      sh_wcode[NBUF][BN][18]; // 64 survivor codes {+1,-1,0} (16 used + 2 pad)
    __shared__ unsigned sh_wmeta[NBUF][BN][4];  // 32 meta nibbles (= ISA idx bits)
    __shared__ float    sh_dw[NBUF][BN];        // per-128 ternary scale

    auto load_chunk = [&] (int64_t c, int b) __attribute__((always_inline)) {
        if (warp_id_u < WARPS_A) {
            const int     row = warp_id_u * 32 + lane;
            const int64_t m   = m0 + row;
            if (!need_check || m < M) {
                const int8_t * src = actq + m * (n_blocks_k * 128) + c * 128;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *(v4i_t1 *) &sh_actq[b][row][4 * i] = *(const v4i_t1 *) (src + 16 * i);
                }
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    sh_da[b][row][w] = actd[m * (n_blocks_k * 4) + c * 4 + w];
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *(v4i_t1 *) &sh_actq[b][row][4 * i] = v4i_t1{0, 0, 0, 0};
                }
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    sh_da[b][row][w] = 0.0f;
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
                    sh_wmeta[b][row][w] = mw;
                    const unsigned s16 = ((w & 2) ? sig_hi : sig_lo) >> ((w & 1) * 16);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const unsigned lane16 = mw >> (h * 16);
                        const unsigned sh8    = (s16 >> (h * 8)) & 0xFFu;
                        sh_wcode[b][row][4 * w + 2 * h + 0] =
                            (int) (ggml_cuda_2of4_t1_sgn_lut[sh8 & 0xFu] & ggml_cuda_2of4_t1_msk_lut[lane16 & 0xFFu]);
                        sh_wcode[b][row][4 * w + 2 * h + 1] =
                            (int) (ggml_cuda_2of4_t1_sgn_lut[sh8 >> 4]   & ggml_cuda_2of4_t1_msk_lut[(lane16 >> 8) & 0xFFu]);
                    }
                }
                sh_dw[b][row] = __half2float(*(const __half *) &blk->d);
            } else {
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    sh_wmeta[b][row][w] = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sh_wcode[b][row][4 * w + i] = 0;
                    }
                }
                sh_dw[b][row] = 0.0f;
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

    // [TAG_2OF4_T1_KPIPE] compute body as a macro, not a lambda: wrapping it in a
    // lambda changes register allocation of the shipped PIPE 0 kernel.
#define T1_COMPUTE(b) \
_Pragma("unroll")                                                                                                                \
        for (int s = 0; s < NTX; ++s) {                                                                                          \
            const int t  = warp_id + s * NWARPS;                                                                                 \
            const int mi = t / NTILES_N;                                                                                         \
            const int ni = t % NTILES_N;                                                                                         \
            const int w_row = ni * 16 + local_idx; /* weight row within the staged tile */                                       \
            const int a_col = mi * 16 + local_idx; /* activation row (token) */                                                  \
            /* Per-128 weight scale: constant across the 4 windows -> hoisted. */                                                \
            float dw_l[8];                                                                                                       \
_Pragma("unroll")                                                                                                                \
            for (int l = 0; l < 8; ++l) {                                                                                        \
                dw_l[l] = sh_dw[b][ni * 16 + out_row_base + l];                                                                  \
            }                                                                                                                    \
            /* [TAG_2OF4_T1_SCALE_HOIST] With 128-value activation scales both */                                                \
            /* scales are constant over the chunk, so the int32 SWMMAC result is */                                              \
            /* carried across the 4 windows (passed back as C) and rescaled */                                                   \
            /* once per chunk instead of once per window. |raw| <= 128*127. */                                                   \
            v8i_t1 craw = {0, 0, 0, 0, 0, 0, 0, 0};                                                                              \
_Pragma("unroll")                                                                                                                \
            for (int w = 0; w < 4; ++w) {                                                                                        \
                /* This lane's 16 idx bits = its k-half's 2 meta bytes, */                                                       \
                /* unchanged from block storage (ISA sparsity_idx packing). */                                                   \
                const int idxv = (int) ((sh_wmeta[b][w_row][w] >> (k_half * 16)) & 0xFFFFu);                                     \
                const v2i_t1 a_arg = *(const v2i_t1 *) &sh_wcode[b][w_row][4 * w + 2 * k_half];                                  \
                const v4i_t1 b_arg = *(const v4i_t1 *) &sh_actq[b][a_col][8 * w + 4 * k_half];                                   \
                if constexpr (HOIST128) {                                                                                        \
                    craw = __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(true, a_arg, true, b_arg, craw, idxv, false);            \
                } else {                                                                                                         \
                    const v8i_t1 c0  = {0, 0, 0, 0, 0, 0, 0, 0};                                                                 \
                    const v8i_t1 raw = __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(true, a_arg, true, b_arg, c0, idxv, false);  \
                    const float da = sh_da[b][a_col][w];                                                                         \
_Pragma("unroll")                                                                                                                \
                    for (int l = 0; l < 8; ++l) {                                                                                \
                        acc[s][l] += (float) raw[l] * (dw_l[l] * da);                                                            \
                    }                                                                                                            \
                }                                                                                                                \
            }                                                                                                                    \
            if constexpr (HOIST128) {                                                                                            \
                const float da = sh_da[b][a_col][0]; /* == sh_da[b][a_col][1..3] */                                              \
_Pragma("unroll")                                                                                                                \
                for (int l = 0; l < 8; ++l) {                                                                                    \
                    acc[s][l] += (float) craw[l] * (dw_l[l] * da);                                                               \
                }                                                                                                                \
            }                                                                                                                    \
        }                                                                                                                        \
    do {} while (0)

    if constexpr (PIPE == 0) {
        for (int64_t c = 0; c < n_blocks_k; ++c) {
            load_chunk(c, 0);
            __syncthreads();
            T1_COMPUTE(0);
            __syncthreads(); // compute -> next-load barrier (single buffer)
        }
    } else if constexpr (PIPE == 1) {
        // [TAG_2OF4_T1_KPIPE] double buffer: the barrier closing iteration c
        // both publishes chunk c+1 and retires every read of chunk c, so the
        // buffer chunk c+2 overwrites is free. One barrier per chunk.
        load_chunk(0, 0);
        __syncthreads();
        for (int64_t c = 0; c < n_blocks_k; ++c) {
            const int b = (int) (c & 1);
            if (c + 1 < n_blocks_k) {
                load_chunk(c + 1, b ^ 1);
            }
            T1_COMPUTE(b);
            __syncthreads();
        }
    } else {
        // [TAG_2OF4_T1_KPIPE] fetch/commit split: chunk c+1's global loads go
        // into registers before compute(c), the LDS writes (and the weight
        // LUT gathers, which depend on the loaded bytes) after it. One
        // register set covers both roles (36 ints for an activation row,
        // 7 for a weight row).
        int pf[36];
        auto fetch = [&] (int64_t c) __attribute__((always_inline)) {
            if (warp_id_u < WARPS_A) {
                const int64_t m = m0 + warp_id_u * 32 + lane;
                if (!need_check || m < M) {
                    const int8_t * src = actq + m * (n_blocks_k * 128) + c * 128;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const v4i_t1 q = *(const v4i_t1 *) (src + 16 * i);
                        pf[4 * i + 0] = q[0]; pf[4 * i + 1] = q[1]; pf[4 * i + 2] = q[2]; pf[4 * i + 3] = q[3];
                    }
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        pf[32 + w] = __float_as_int(actd[m * (n_blocks_k * 4) + c * 4 + w]);
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < 36; ++i) {
                        pf[i] = 0;
                    }
                }
            } else if (warp_id_u < WARPS_A + WARPS_B) {
                const int64_t n = n0 + (warp_id_u - WARPS_A) * 32 + lane;
                if (!need_check || n < N) {
                    const block_2of4_t1 * blk = (const block_2of4_t1 *) (vweight + n * nb01 + c * (int64_t) sizeof(block_2of4_t1));
                    pf[0] = pack4_t1(blk->signs + 0);
                    pf[1] = pack4_t1(blk->signs + 4);
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        pf[2 + w] = pack4_t1(blk->meta + 4 * w);
                    }
                    pf[6] = (int) *(const unsigned short *) &blk->d;
                } else {
                    pf[6] = -1; // row out of range: commit writes zeros
                }
            }
        };
        auto commit = [&] (int b) __attribute__((always_inline)) {
            if (warp_id_u < WARPS_A) {
                const int row = warp_id_u * 32 + lane;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *(v4i_t1 *) &sh_actq[b][row][4 * i] = v4i_t1{pf[4 * i + 0], pf[4 * i + 1], pf[4 * i + 2], pf[4 * i + 3]};
                }
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    sh_da[b][row][w] = __int_as_float(pf[32 + w]);
                }
            } else if (warp_id_u < WARPS_A + WARPS_B) {
                const int row = (warp_id_u - WARPS_A) * 32 + lane;
                if (!need_check || pf[6] >= 0) {
                    const unsigned sig_lo = (unsigned) pf[0];
                    const unsigned sig_hi = (unsigned) pf[1];
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const unsigned mw = (unsigned) pf[2 + w];
                        sh_wmeta[b][row][w] = mw;
                        const unsigned s16 = ((w & 2) ? sig_hi : sig_lo) >> ((w & 1) * 16);
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const unsigned lane16 = mw >> (h * 16);
                            const unsigned sh8    = (s16 >> (h * 8)) & 0xFFu;
                            sh_wcode[b][row][4 * w + 2 * h + 0] =
                                (int) (ggml_cuda_2of4_t1_sgn_lut[sh8 & 0xFu] & ggml_cuda_2of4_t1_msk_lut[lane16 & 0xFFu]);
                            sh_wcode[b][row][4 * w + 2 * h + 1] =
                                (int) (ggml_cuda_2of4_t1_sgn_lut[sh8 >> 4]   & ggml_cuda_2of4_t1_msk_lut[(lane16 >> 8) & 0xFFu]);
                        }
                    }
                    __half_raw hr;
                    hr.x = (unsigned short) pf[6];
                    sh_dw[b][row] = __half2float(__half(hr));
                } else {
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        sh_wmeta[b][row][w] = 0;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            sh_wcode[b][row][4 * w + i] = 0;
                        }
                    }
                    sh_dw[b][row] = 0.0f;
                }
            }
        };
        load_chunk(0, 0);
        __syncthreads();
        for (int64_t c = 0; c < n_blocks_k; ++c) {
            const int  b    = (int) (c & 1);
            const bool more = c + 1 < n_blocks_k;
            if (more) {
                fetch(c + 1);
            }
            T1_COMPUTE(b);
            if (more) {
                commit(b ^ 1);
            }
            __syncthreads();
        }
    }
#undef T1_COMPUTE

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

// [TAG_2OF4_T1_KPIPE] Per-lane pre-packed weight layout (PIPE 3/4).
//
// Built once per weight tensor from the native blocks (k_prepack_2of4_t1,
// cached on the device, see t1_prepack_get) with the SAME LUT expansion the
// staging loader does, so the codes are bit-identical. For chunk c, row n,
// k-half h (Npad = N rounded up to 128, padded rows are zero):
//   codes: int8 [c][n][h][32]  window w = bytes 8w..8w+7 (= the lane's a_arg)
//   idx:   u16  [c][n][h][4]   window w = the lane's 16 sparsity_idx bits
//   d:     f32  [c][n]         per-128 ternary scale
// 84 bytes per (row, chunk) vs 26 native. Chunk-major so one warp's 16 rows
// x 2 halves read one contiguous 1 KB (codes) / 256 B (idx) span: the weight
// fragment is two global_load_b128 + one b64 straight into VGPRs, no LDS, no
// LUT, no byte assembly.
static __host__ __device__ __forceinline__ int64_t t1_pack_idx_off(int64_t nb, int64_t Npad) { return nb * Npad * 64; }
static __host__ __device__ __forceinline__ int64_t t1_pack_d_off  (int64_t nb, int64_t Npad) { return nb * Npad * 80; }
static __host__ __device__ __forceinline__ int64_t t1_pack_bytes  (int64_t nb, int64_t Npad) { return nb * Npad * 84; }

static __global__ void k_prepack_2of4_t1(
        const char * __restrict__ vweight, const int64_t nb01, const int64_t N, const int64_t Npad,
        const int64_t nb, char * __restrict__ out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Npad * nb) {
        return;
    }
    const int64_t c = i / Npad;
    const int64_t n = i % Npad;
    int      codes[2][8];
    unsigned idx[2][2];
    float    d = 0.0f;
    if (n < N) {
        const block_2of4_t1 * blk = (const block_2of4_t1 *) (vweight + n * nb01 + c * (int64_t) sizeof(block_2of4_t1));
        const unsigned sig_lo = (unsigned) pack4_t1(blk->signs + 0);
        const unsigned sig_hi = (unsigned) pack4_t1(blk->signs + 4);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const unsigned mw  = (unsigned) pack4_t1(blk->meta + 4 * w);
            const unsigned s16 = ((w & 2) ? sig_hi : sig_lo) >> ((w & 1) * 16);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const unsigned lane16 = mw >> (h * 16);
                const unsigned sh8    = (s16 >> (h * 8)) & 0xFFu;
                codes[h][2 * w + 0] = (int) (ggml_cuda_2of4_t1_sgn_lut[sh8 & 0xFu] & ggml_cuda_2of4_t1_msk_lut[lane16 & 0xFFu]);
                codes[h][2 * w + 1] = (int) (ggml_cuda_2of4_t1_sgn_lut[sh8 >> 4]   & ggml_cuda_2of4_t1_msk_lut[(lane16 >> 8) & 0xFFu]);
            }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned m0 = (unsigned) pack4_t1(blk->meta + 0) >> (h * 16);
            const unsigned m1 = (unsigned) pack4_t1(blk->meta + 4) >> (h * 16);
            const unsigned m2 = (unsigned) pack4_t1(blk->meta + 8) >> (h * 16);
            const unsigned m3 = (unsigned) pack4_t1(blk->meta + 12) >> (h * 16);
            idx[h][0] = (m0 & 0xFFFFu) | ((m1 & 0xFFFFu) << 16);
            idx[h][1] = (m2 & 0xFFFFu) | ((m3 & 0xFFFFu) << 16);
        }
        d = __half2float(*(const __half *) &blk->d);
    } else {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                codes[h][j] = 0;
            }
            idx[h][0] = idx[h][1] = 0;
        }
    }
    v4i_t1 * pc = (v4i_t1 *) (out + (c * Npad + n) * 64);
    pc[0] = v4i_t1{codes[0][0], codes[0][1], codes[0][2], codes[0][3]};
    pc[1] = v4i_t1{codes[0][4], codes[0][5], codes[0][6], codes[0][7]};
    pc[2] = v4i_t1{codes[1][0], codes[1][1], codes[1][2], codes[1][3]};
    pc[3] = v4i_t1{codes[1][4], codes[1][5], codes[1][6], codes[1][7]};
    v4i_t1 * pi = (v4i_t1 *) (out + t1_pack_idx_off(nb, Npad) + (c * Npad + n) * 16);
    pi[0] = v4i_t1{(int) idx[0][0], (int) idx[0][1], (int) idx[1][0], (int) idx[1][1]};
    ((float *) (out + t1_pack_d_off(nb, Npad)))[c * Npad + n] = d;
}

// One 16-byte activation segment into pf[4i..4i+3]; out-of-range rows load a
// clamped (valid) row and are zeroed by a select, never a branch.
#define T1_AFETCH_SEG(i, j, m)                                                                      \
    do {                                                                                            \
        const int64_t mm_ = need_check ? ((m) < M ? (m) : M - 1) : (m);                             \
        const v4i_t1  q_  = *(const v4i_t1 *) (actq + mm_ * (n_blocks_k * 128) + c * 128 + ((j) & 7) * 16); \
        const bool    ok_ = !need_check || (m) < M;                                                 \
        pf[4 * (i) + 0] = ok_ ? q_[0] : 0; pf[4 * (i) + 1] = ok_ ? q_[1] : 0;                      \
        pf[4 * (i) + 2] = ok_ ? q_[2] : 0; pf[4 * (i) + 3] = ok_ ? q_[3] : 0;                      \
    } while (0)

// PIPE 3: pre-packed weights, direct global->VGPR, prefetched one chunk ahead;
//         activations double-buffered in LDS (fetch/commit split, staged by
//         all NWARPS warps with a coalesced 16 B/thread mapping).
// PIPE 4: as 3, weights prefetched two chunks ahead (deeper K pipeline).
// Requires HOIST128 (one activation scale per chunk). The SWMMAC operands,
// their order and the epilogue are those of k_mul_mat_2of4_t1_mmq<HOIST128>,
// so the result is bit-identical to it.
template <int BM, int BN, int NWARPS, bool need_check, int PIPE>
__launch_bounds__(NWARPS * 32, 1)
static __global__ void k_mul_mat_2of4_t1_mmq_pp(
        const char * __restrict__ wpack, const int8_t * __restrict__ actq,
        const float * __restrict__ actd, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t Npad, const int64_t n_blocks_k,
        const int64_t dst_row_stride_floats) {
    constexpr int NTILES_N = BN / 16;
    constexpr int NTILES_M = BM / 16;
    constexpr int NTILES   = NTILES_M * NTILES_N;
    constexpr int NTX      = NTILES / NWARPS;
    constexpr int NT       = NWARPS * 32;
    constexpr int SEGS     = BM * 8 / NT; // 16-byte activation segments per thread per chunk
    static_assert(NTILES % NWARPS == 0, "tiles must divide evenly across warps");
    static_assert(NWARPS % NTILES_N == 0, "each warp must own one weight tile column (ni fixed)");
    static_assert((BM * 8) % NT == 0 && BM <= NT, "activation staging mapping");
    static_assert(PIPE == 3 || PIPE == 4, "pre-packed arms are PIPE 3 and 4");

#if defined(RDNA4)
    const int64_t n0      = (int64_t) blockIdx.x * BN;
    const int64_t m0      = (int64_t) blockIdx.y * BM;
    const int     lane    = threadIdx.x;
    const int     warp_id = threadIdx.y;
    const int     tid     = warp_id * 32 + lane;

    __shared__ int   sh_actq[2][BM][36]; // same padded pitch as the LDS kernel
    __shared__ float sh_da[2][BM];       // one activation scale per row per chunk (HOIST128)

    const int k_half       = (lane < 16) ? 0 : 1;
    const int local_idx    = (lane < 16) ? lane : (lane - 16);
    const int out_row_base = (lane >= 16) ? 8 : 0;
    const int ni           = warp_id % NTILES_N;

    const int8_t * wcodes = (const int8_t *) wpack;
    const char   * widx   = wpack + t1_pack_idx_off(n_blocks_k, Npad);
    const float  * wd     = (const float *) (wpack + t1_pack_d_off(n_blocks_k, Npad));
    const int64_t  n_lane = n0 + ni * 16 + local_idx;          // this lane's weight row
    const int64_t  n_dw   = n0 + ni * 16 + out_row_base;       // first of this lane's 8 output rows

    struct wfrag { v4i_t1 c0, c1; v2i_t1 ix; v4i_t1 d0, d1; };
    auto wfetch = [&] (int64_t c) __attribute__((always_inline)) {
        wfrag f;
        const int64_t r = c * Npad;
        const v4i_t1 * pc = (const v4i_t1 *) (wcodes + ((r + n_lane) * 2 + k_half) * 32);
        f.c0 = pc[0];
        f.c1 = pc[1];
        f.ix = *(const v2i_t1 *) (widx + ((r + n_lane) * 2 + k_half) * 8);
        const v4i_t1 * pd = (const v4i_t1 *) (wd + r + n_dw);
        f.d0 = pd[0];
        f.d1 = pd[1];
        return f;
    };

    int pf[SEGS * 4 + 1];
    auto afetch = [&] (int64_t c) __attribute__((always_inline)) {
#pragma unroll
        for (int i = 0; i < SEGS; ++i) {
            const int     j   = tid + i * NT;
            const int64_t m   = m0 + (j >> 3);
            T1_AFETCH_SEG(i, j, m);
        }
        {
            // unconditional (clamped) load: a load inside a divergent branch makes
            // the waitcnt pass drain every outstanding load at the join, i.e.
            // before compute, which serializes the prefetch.
            const int64_t m = m0 + (tid & (BM - 1));
            const float   d = actd[(need_check ? (m < M ? m : M - 1) : m) * (n_blocks_k * 4) + c * 4];
            pf[SEGS * 4] = (!need_check || m < M) ? __float_as_int(d) : 0;
        }
    };
    auto acommit = [&] (int b) __attribute__((always_inline)) {
#pragma unroll
        for (int i = 0; i < SEGS; ++i) {
            const int j = tid + i * NT;
            *(v4i_t1 *) &sh_actq[b][j >> 3][(j & 7) * 4] = v4i_t1{pf[4 * i + 0], pf[4 * i + 1], pf[4 * i + 2], pf[4 * i + 3]};
        }
        if (tid < BM) {
            sh_da[b][tid] = __int_as_float(pf[SEGS * 4]);
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

    auto compute = [&] (int b, const wfrag & f) __attribute__((always_inline)) {
        const int code[8] = {f.c0[0], f.c0[1], f.c0[2], f.c0[3], f.c1[0], f.c1[1], f.c1[2], f.c1[3]};
        float dw_l[8] = {
            __int_as_float(f.d0[0]), __int_as_float(f.d0[1]), __int_as_float(f.d0[2]), __int_as_float(f.d0[3]),
            __int_as_float(f.d1[0]), __int_as_float(f.d1[1]), __int_as_float(f.d1[2]), __int_as_float(f.d1[3]),
        };
#pragma unroll
        for (int s = 0; s < NTX; ++s) {
            const int t  = warp_id + s * NWARPS;
            const int mi = t / NTILES_N;
            const int a_col = mi * 16 + local_idx;
            v8i_t1 craw = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const int    idxv  = (int) (((unsigned) f.ix[w >> 1] >> ((w & 1) * 16)) & 0xFFFFu);
                const v2i_t1 a_arg = {code[2 * w + 0], code[2 * w + 1]};
                const v4i_t1 b_arg = *(const v4i_t1 *) &sh_actq[b][a_col][8 * w + 4 * k_half];
                craw = __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(true, a_arg, true, b_arg, craw, idxv, false);
            }
            const float da = sh_da[b][a_col];
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                acc[s][l] += (float) craw[l] * (dw_l[l] * da);
            }
        }
    };

    afetch(0);
    acommit(0);
    wfrag wc = wfetch(0);
    wfrag wn1, wn2;
    if constexpr (PIPE == 4) {
        if (n_blocks_k > 1) {
            wn1 = wfetch(1);
        }
    }
    __syncthreads();
    for (int64_t c = 0; c < n_blocks_k; ++c) {
        const int  b    = (int) (c & 1);
        const bool more = c + 1 < n_blocks_k;
        if constexpr (PIPE == 4) {
            if (c + 2 < n_blocks_k) {
                wn2 = wfetch(c + 2);
            }
        } else {
            if (more) {
                wn1 = wfetch(c + 1);
            }
        }
        if (more) {
            afetch(c + 1);
        }
        compute(b, wc);
        if (more) {
            acommit(b ^ 1);
        }
        if constexpr (PIPE == 4) {
            wc  = wn1;
            wn1 = wn2;
        } else {
            wc = wn1;
        }
        __syncthreads();
    }

#pragma unroll
    for (int s = 0; s < NTX; ++s) {
        const int t  = warp_id + s * NWARPS;
        const int mi = t / NTILES_N;
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
    GGML_UNUSED(wpack);
    GGML_UNUSED(actq);
    GGML_UNUSED(actd);
    GGML_UNUSED(dst);
    GGML_UNUSED(M);
    GGML_UNUSED(N);
    GGML_UNUSED(Npad);
    GGML_UNUSED(n_blocks_k);
    GGML_UNUSED(dst_row_stride_floats);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

// PIPE 5/6: PIPE 3 + 2D warp tiling. Each warp owns WN weight tile columns x
// WM activation tile rows (PIPE 3: 1 x 8), so every activation fragment read
// from LDS feeds WN SWMMACs: WN=2 halves the per-chunk LDS read volume (the
// k3 PC samples are ds_load_b128 + s_wait_dscnt bound). The weight scales
// move to LDS ([2][BN], staged with the activations) so the WN weight
// fragments in registers are only codes + idx. Addressing is uniform base
// (SGPR) + 32-bit lane offset. PIPE 5: WN=2, PIPE 6: WN=4. Same operands,
// same per-tile window order and epilogue -> bit-identical to PIPE 0.
template <int BM, int BN, int NWARPS, bool need_check, int WN>
__launch_bounds__(NWARPS * 32, 1)
static __global__ void k_mul_mat_2of4_t1_mmq_pp2(
        const char * __restrict__ wpack, const int8_t * __restrict__ actq,
        const float * __restrict__ actd, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t Npad, const int64_t n_blocks_k,
        const int64_t dst_row_stride_floats) {
    constexpr int NTILES_N = BN / 16;
    constexpr int NTILES_M = BM / 16;
    constexpr int NT       = NWARPS * 32;
    constexpr int SEGS     = BM * 8 / NT;
    constexpr int NW_N     = NTILES_N / WN;    // warps along N
    constexpr int NW_M     = NWARPS / NW_N;    // warps along M
    constexpr int WM       = NTILES_M / NW_M;  // activation tiles per warp
    static_assert(NTILES_N % WN == 0 && NWARPS % NW_N == 0 && NTILES_M % NW_M == 0, "warp tiling");
    static_assert((BM * 8) % NT == 0 && BM <= NT && BN / 4 <= NT && (BM & (BM - 1)) == 0 && (BN & (BN - 1)) == 0, "staging mapping");

#if defined(RDNA4)
    const int64_t n0      = (int64_t) blockIdx.x * BN;
    const int64_t m0      = (int64_t) blockIdx.y * BM;
    const int     lane    = threadIdx.x;
    const int     warp_id = threadIdx.y;
    const int     tid     = warp_id * 32 + lane;

    __shared__ int   sh_actq[2][BM][36];
    __shared__ float sh_da[2][BM];
    __shared__ float sh_dw[2][BN];

    const int k_half       = (lane < 16) ? 0 : 1;
    const int local_idx    = (lane < 16) ? lane : (lane - 16);
    const int out_row_base = (lane >= 16) ? 8 : 0;
    const int wn_idx       = warp_id % NW_N;
    const int wm_idx       = warp_id / NW_N;

    const int64_t  wc_stride = Npad * 64;
    const int64_t  wi_stride = Npad * 16;
    const char   * wcodes = wpack;
    const char   * widx   = wpack + t1_pack_idx_off(n_blocks_k, Npad);
    const float  * wd     = (const float *) (wpack + t1_pack_d_off(n_blocks_k, Npad));

    // per-lane 32-bit offsets inside one chunk slab of the pre-packed arrays
    unsigned off_c[WN], off_i[WN];
#pragma unroll
    for (int j = 0; j < WN; ++j) {
        const unsigned n_l = (unsigned) (n0 + (wn_idx * WN + j) * 16 + local_idx);
        off_c[j] = (n_l * 2 + k_half) * 32;
        off_i[j] = (n_l * 2 + k_half) * 8;
    }

    // weight fragments: [0] = chunk being computed, [1] = prefetched next chunk
    v4i_t1 wc0[2][WN], wc1[2][WN];
    v2i_t1 wix[2][WN];
    auto wfetch = [&] (int64_t c, int slot) __attribute__((always_inline)) {
        const char * cb = wcodes + c * wc_stride;
        const char * ib = widx   + c * wi_stride;
#pragma unroll
        for (int j = 0; j < WN; ++j) {
            wc0[slot][j] = *(const v4i_t1 *) (cb + off_c[j]);
            wc1[slot][j] = *(const v4i_t1 *) (cb + off_c[j] + 16);
            wix[slot][j] = *(const v2i_t1 *) (ib + off_i[j]);
        }
    };

    int pf[SEGS * 4 + 5];
    auto afetch = [&] (int64_t c) __attribute__((always_inline)) {
#pragma unroll
        for (int i = 0; i < SEGS; ++i) {
            const int     j = tid + i * NT;
            const int64_t m = m0 + (j >> 3);
            T1_AFETCH_SEG(i, j, m);
        }
        {
            // unconditional (clamped) loads, see k_mul_mat_2of4_t1_mmq_pp: every
            // thread loads one activation scale and one 4-float weight-scale
            // group; acommit stores only the ones it owns.
            const int64_t m  = m0 + (tid & (BM - 1));
            const float   da = actd[(need_check ? (m < M ? m : M - 1) : m) * (n_blocks_k * 4) + c * 4];
            pf[SEGS * 4] = (!need_check || m < M) ? __float_as_int(da) : 0;
            const v4i_t1 d = *(const v4i_t1 *) (wd + c * Npad + n0 + 4 * (tid & (BN / 4 - 1)));
            pf[SEGS * 4 + 1] = d[0]; pf[SEGS * 4 + 2] = d[1]; pf[SEGS * 4 + 3] = d[2]; pf[SEGS * 4 + 4] = d[3];
        }
    };
    auto acommit = [&] (int b) __attribute__((always_inline)) {
#pragma unroll
        for (int i = 0; i < SEGS; ++i) {
            const int j = tid + i * NT;
            *(v4i_t1 *) &sh_actq[b][j >> 3][(j & 7) * 4] = v4i_t1{pf[4 * i + 0], pf[4 * i + 1], pf[4 * i + 2], pf[4 * i + 3]};
        }
        if (tid < BM) {
            sh_da[b][tid] = __int_as_float(pf[SEGS * 4]);
        }
        if (tid < BN / 4) {
            *(v4i_t1 *) &sh_dw[b][4 * tid] = v4i_t1{pf[SEGS * 4 + 1], pf[SEGS * 4 + 2], pf[SEGS * 4 + 3], pf[SEGS * 4 + 4]};
        }
    };

    float acc[WM * WN][8];
#pragma unroll
    for (int s = 0; s < WM * WN; ++s) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            acc[s][l] = 0.0f;
        }
    }

    afetch(0);
    acommit(0);
    wfetch(0, 0);
    __syncthreads();
    for (int64_t c = 0; c < n_blocks_k; ++c) {
        const int  b    = (int) (c & 1);
        const bool more = c + 1 < n_blocks_k;
        if (more) {
            wfetch(c + 1, 1);
            afetch(c + 1);
        }
        float dw_l[WN][8];
#pragma unroll
        for (int j = 0; j < WN; ++j) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                dw_l[j][l] = sh_dw[b][(wn_idx * WN + j) * 16 + out_row_base + l];
            }
        }
#pragma unroll
        for (int k = 0; k < WM; ++k) {
            const int a_col = (wm_idx * WM + k) * 16 + local_idx;
            v4i_t1 b_arg[4];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                b_arg[w] = *(const v4i_t1 *) &sh_actq[b][a_col][8 * w + 4 * k_half];
            }
            const float da = sh_da[b][a_col];
#pragma unroll
            for (int j = 0; j < WN; ++j) {
                const int code[8] = {wc0[0][j][0], wc0[0][j][1], wc0[0][j][2], wc0[0][j][3],
                                     wc1[0][j][0], wc1[0][j][1], wc1[0][j][2], wc1[0][j][3]};
                v8i_t1 craw = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const int    idxv  = (int) (((unsigned) wix[0][j][w >> 1] >> ((w & 1) * 16)) & 0xFFFFu);
                    const v2i_t1 a_arg = {code[2 * w + 0], code[2 * w + 1]};
                    craw = __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(true, a_arg, true, b_arg[w], craw, idxv, false);
                }
                {
                    // keep the shipped rounding order craw * (dw * da): with
                    // -funsafe-math-optimizations the compiler otherwise
                    // reassociates to (craw * da) * dw here (da is shared by
                    // the 8 l), which is not bit-identical (gate job 140).
#pragma clang fp reassociate(off)
#pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        acc[k * WN + j][l] += (float) craw[l] * (dw_l[j][l] * da);
                    }
                }
            }
        }
        if (more) {
            acommit(b ^ 1);
#pragma unroll
            for (int j = 0; j < WN; ++j) {
                wc0[0][j] = wc0[1][j];
                wc1[0][j] = wc1[1][j];
                wix[0][j] = wix[1][j];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int k = 0; k < WM; ++k) {
#pragma unroll
        for (int j = 0; j < WN; ++j) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int64_t n = n0 + (wn_idx * WN + j) * 16 + out_row_base + l;
                const int64_t m = m0 + (wm_idx * WM + k) * 16 + local_idx;
                if (!need_check || (m < M && n < N)) {
                    dst[m * dst_row_stride_floats + n] = acc[k * WN + j][l];
                }
            }
        }
    }
#else
    GGML_UNUSED(wpack);
    GGML_UNUSED(actq);
    GGML_UNUSED(actd);
    GGML_UNUSED(dst);
    GGML_UNUSED(M);
    GGML_UNUSED(N);
    GGML_UNUSED(Npad);
    GGML_UNUSED(n_blocks_k);
    GGML_UNUSED(dst_row_stride_floats);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

// Device cache of pre-packed weights, keyed on the weight's device address and
// purged through the hipBLASLt wcache invalidation registry when the owning
// buffer is freed (same stale-address hazard and fix). COMPUTE-usage buffers
// are never cached: they get a per-call pool conversion instead.
namespace {
struct t1_pack_entry { char * p; size_t bytes; int64_t N, K, nb01; };
std::mutex                                        g_t1_pack_mtx;
std::unordered_map<const void *, t1_pack_entry> g_t1_pack;

void t1_pack_invalidate(const void * base, size_t size) {
    std::lock_guard<std::mutex> lk(g_t1_pack_mtx);
    const char * bb = (const char *) base;
    for (auto it = g_t1_pack.begin(); it != g_t1_pack.end(); ) {
        const char * k = (const char *) it->first;
        if (k >= bb && k < bb + size) {
            (void) cudaFree(it->second.p);
            it = g_t1_pack.erase(it);
        } else {
            ++it;
        }
    }
}
struct t1_pack_registrar { t1_pack_registrar() { ggml_hipblaslt_wcache_register(t1_pack_invalidate); } };
t1_pack_registrar g_t1_pack_registrar;
} // namespace

static const char * t1_prepack_get(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, int64_t Npad,
                                   ggml_cuda_pool_alloc<char> & pool_buf) {
    const int64_t K  = src0->ne[0];
    const int64_t N  = src0->ne[1];
    const int64_t nb = K / QK_2OF4_T1;
    const size_t  bytes = (size_t) t1_pack_bytes(nb, Npad);
    cudaStream_t  stream = ctx.stream();
    auto build = [&] (char * out) {
        const int64_t n_items = Npad * nb;
        k_prepack_2of4_t1<<<(unsigned) ((n_items + 255) / 256), 256, 0, stream>>>(
                (const char *) src0->data, src0->nb[1], N, Npad, nb, out);
    };
    const bool cacheable = src0->buffer == nullptr ||
        ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE;
    if (cacheable) {
        std::lock_guard<std::mutex> lk(g_t1_pack_mtx);
        auto it = g_t1_pack.find(src0->data);
        if (it != g_t1_pack.end() && it->second.N == N && it->second.K == K && it->second.nb01 == (int64_t) src0->nb[1]) {
            return it->second.p;
        }
        if (it != g_t1_pack.end()) {
            (void) cudaFree(it->second.p);
            g_t1_pack.erase(it);
        }
        char * p = nullptr;
        if (cudaMalloc((void **) &p, bytes) == cudaSuccess) {
            build(p);
            g_t1_pack.emplace(src0->data, t1_pack_entry{p, bytes, N, K, (int64_t) src0->nb[1]});
            return p;
        }
        (void) cudaGetLastError(); // fall through to the pool path
    }
    pool_buf.alloc(ctx.pool(), bytes);
    build(pool_buf.get());
    return pool_buf.get();
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

    // [TAG_2OF4_T1_SCALE_HOIST] on by default, GGML_HIP_2OF4_T1_SCALE_HOIST=0 opts out:
    // 128-value activation scales + one rescale per chunk. Not bit-exact
    // (activation granularity per32 -> per128), PPL-gated. The scale buffer
    // is call-local and quantizer and kernel read the same flag below.
    // Compile out with -DGGML_CUDA_NO_2OF4_T1_SCALE_HOIST.
#ifndef GGML_CUDA_NO_2OF4_T1_SCALE_HOIST
    static const bool hoist = [] {
        const char * e = getenv("GGML_HIP_2OF4_T1_SCALE_HOIST");
        return e == nullptr || strcmp(e, "0") != 0;
    }();
#else
    constexpr bool hoist = false;
#endif // GGML_CUDA_NO_2OF4_T1_SCALE_HOIST

    ggml_cuda_pool_alloc<int8_t> act_q(ctx.pool(), (size_t) (M * K));
    ggml_cuda_pool_alloc<float>  act_d(ctx.pool(), (size_t) (M * n_groups_k));
    {
        const int64_t row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
        const dim3 grid((n_groups_k + 15) / 16, M, 1);
        const dim3 block(128, 1, 1);
        if (hoist) {
            k_quantize_act_q8_t1<true><<<grid, block, 0, stream>>>(
                    (const float *) src1->data, act_q.get(), act_d.get(), n_groups_k, row_stride_floats);
        } else {
            k_quantize_act_q8_t1<false><<<grid, block, 0, stream>>>(
                    (const float *) src1->data, act_q.get(), act_d.get(), n_groups_k, row_stride_floats);
        }
    }

    const int64_t nb01 = src0->nb[1];
    const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
    const char * d_w = (const char *) src0->data;
    float * d_dst = (float *) dst->data;

    // [TAG_2OF4_T1_KPIPE] K-pipeline arms, default 0 (shipped kernel). 1/2 need
    // the double LDS footprint and are wired for the two default geometries
    // only; 3-6 (pre-packed weights) also need the hoist. Anything else falls
    // back to PIPE 0.
    static const int kpipe = [] {
        const char * e = getenv("GGML_HIP_2OF4_T1_KPIPE");
        return e == nullptr ? 0 : atoi(e);
    }();
    const char * geom = getenv("GGML_HIP_2OF4_T1_MMQ_GEOM");
    if (kpipe >= 1 && kpipe <= 6 && geom == nullptr && !(kpipe >= 3 && !hoist)) {
        const bool small = N <= 1024;
#define LAUNCH_T1_KP(BM_, BN_, NWARPS_, P_)                                                           \
        do {                                                                                          \
            const dim3 grid_((N + (BN_) - 1) / (BN_), (M + (BM_) - 1) / (BM_), 1);                    \
            const dim3 block_(32, (NWARPS_), 1);                                                      \
            const bool need_check = (M % (BM_) != 0) || (N % (BN_) != 0);                             \
            if (hoist) {                                                                              \
                if (need_check) {                                                                     \
                    k_mul_mat_2of4_t1_mmq<BM_, BN_, NWARPS_, true, true, P_><<<grid_, block_, 0, stream>>>( \
                        d_w, act_q.get(), act_d.get(), d_dst, M, N, nb01, n_blocks_k, dst_row_stride_floats); \
                } else {                                                                              \
                    k_mul_mat_2of4_t1_mmq<BM_, BN_, NWARPS_, false, true, P_><<<grid_, block_, 0, stream>>>( \
                        d_w, act_q.get(), act_d.get(), d_dst, M, N, nb01, n_blocks_k, dst_row_stride_floats); \
                }                                                                                     \
            } else {                                                                                  \
                if (need_check) {                                                                     \
                    k_mul_mat_2of4_t1_mmq<BM_, BN_, NWARPS_, true, false, P_><<<grid_, block_, 0, stream>>>( \
                        d_w, act_q.get(), act_d.get(), d_dst, M, N, nb01, n_blocks_k, dst_row_stride_floats); \
                } else {                                                                              \
                    k_mul_mat_2of4_t1_mmq<BM_, BN_, NWARPS_, false, false, P_><<<grid_, block_, 0, stream>>>( \
                        d_w, act_q.get(), act_d.get(), d_dst, M, N, nb01, n_blocks_k, dst_row_stride_floats); \
                }                                                                                     \
            }                                                                                         \
        } while (0)
#define LAUNCH_T1_PP(BM_, BN_, NWARPS_, P_)                                                           \
        do {                                                                                          \
            const dim3 grid_((N + (BN_) - 1) / (BN_), (M + (BM_) - 1) / (BM_), 1);                    \
            const dim3 block_(32, (NWARPS_), 1);                                                      \
            const bool need_check = (M % (BM_) != 0) || (N % (BN_) != 0);                             \
            if (need_check) {                                                                         \
                k_mul_mat_2of4_t1_mmq_pp<BM_, BN_, NWARPS_, true, P_><<<grid_, block_, 0, stream>>>(  \
                    wpack, act_q.get(), act_d.get(), d_dst, M, N, Npad, n_blocks_k, dst_row_stride_floats); \
            } else {                                                                                  \
                k_mul_mat_2of4_t1_mmq_pp<BM_, BN_, NWARPS_, false, P_><<<grid_, block_, 0, stream>>>( \
                    wpack, act_q.get(), act_d.get(), d_dst, M, N, Npad, n_blocks_k, dst_row_stride_floats); \
            }                                                                                         \
        } while (0)
#define LAUNCH_T1_PP2(BM_, BN_, NWARPS_, WN_)                                                         \
        do {                                                                                          \
            const dim3 grid_((N + (BN_) - 1) / (BN_), (M + (BM_) - 1) / (BM_), 1);                    \
            const dim3 block_(32, (NWARPS_), 1);                                                      \
            const bool need_check = (M % (BM_) != 0) || (N % (BN_) != 0);                             \
            if (need_check) {                                                                         \
                k_mul_mat_2of4_t1_mmq_pp2<BM_, BN_, NWARPS_, true, WN_><<<grid_, block_, 0, stream>>>( \
                    wpack, act_q.get(), act_d.get(), d_dst, M, N, Npad, n_blocks_k, dst_row_stride_floats); \
            } else {                                                                                  \
                k_mul_mat_2of4_t1_mmq_pp2<BM_, BN_, NWARPS_, false, WN_><<<grid_, block_, 0, stream>>>( \
                    wpack, act_q.get(), act_d.get(), d_dst, M, N, Npad, n_blocks_k, dst_row_stride_floats); \
            }                                                                                         \
        } while (0)
        if (kpipe <= 2) {
            if (kpipe == 1) {
                if (small) { LAUNCH_T1_KP(64, 64, 8, 1); } else { LAUNCH_T1_KP(128, 128, 8, 1); }
            } else {
                if (small) { LAUNCH_T1_KP(64, 64, 8, 2); } else { LAUNCH_T1_KP(128, 128, 8, 2); }
            }
        } else {
            const int64_t Npad = (N + 127) / 128 * 128;
            ggml_cuda_pool_alloc<char> wpack_pool;
            const char * wpack = t1_prepack_get(ctx, src0, Npad, wpack_pool);
            if (kpipe == 3) {
                if (small) { LAUNCH_T1_PP(64, 64, 8, 3); } else { LAUNCH_T1_PP(128, 128, 8, 3); }
            } else if (kpipe == 4) {
                if (small) { LAUNCH_T1_PP(64, 64, 8, 4); } else { LAUNCH_T1_PP(128, 128, 8, 4); }
            } else if (kpipe == 5) {
                if (small) { LAUNCH_T1_PP2(64, 64, 8, 2); } else { LAUNCH_T1_PP2(128, 128, 8, 2); }
            } else {
                if (small) { LAUNCH_T1_PP2(64, 64, 8, 2); } else { LAUNCH_T1_PP2(128, 128, 8, 4); } // 64x64x8 has no WN=4 tiling
            }
        }
#undef LAUNCH_T1_KP
#undef LAUNCH_T1_PP
#undef LAUNCH_T1_PP2
        return true;
    }

#define LAUNCH_T1_MMQ_H(BM_, BN_, NWARPS_, H_)                                                        \
    do {                                                                                              \
        const dim3 grid_((N + (BN_) - 1) / (BN_), (M + (BM_) - 1) / (BM_), 1);                        \
        const dim3 block_(32, (NWARPS_), 1);                                                          \
        const bool need_check = (M % (BM_) != 0) || (N % (BN_) != 0);                                 \
        if (need_check) {                                                                             \
            k_mul_mat_2of4_t1_mmq<BM_, BN_, NWARPS_, true, H_><<<grid_, block_, 0, stream>>>(         \
                    d_w, act_q.get(), act_d.get(), d_dst, M, N, nb01, n_blocks_k, dst_row_stride_floats); \
        } else {                                                                                      \
            k_mul_mat_2of4_t1_mmq<BM_, BN_, NWARPS_, false, H_><<<grid_, block_, 0, stream>>>(        \
                    d_w, act_q.get(), act_d.get(), d_dst, M, N, nb01, n_blocks_k, dst_row_stride_floats); \
        }                                                                                             \
    } while (0)

#define LAUNCH_T1_MMQ(BM_, BN_, NWARPS_)                                                              \
    do {                                                                                              \
        if (hoist) {                                                                                  \
            LAUNCH_T1_MMQ_H(BM_, BN_, NWARPS_, true);                                                 \
        } else {                                                                                      \
            LAUNCH_T1_MMQ_H(BM_, BN_, NWARPS_, false);                                                \
        }                                                                                             \
    } while (0)

    // Geometry: 128x128x8 default (the fp8 capstone winner); 64x64x8 for the
    // low-occupancy k/v-proj shape (N=1024), the fp8 sweep's measured winner
    // there. Env override for sweeps: GGML_HIP_2OF4_T1_MMQ_GEOM.
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
#undef LAUNCH_T1_MMQ_H

    return true;
}
