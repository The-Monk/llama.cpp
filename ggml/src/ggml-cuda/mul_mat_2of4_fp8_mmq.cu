// See mul_mat_2of4_fp8_mmq.cuh for the full rationale (T162 CAPSTONE).
#include "mul_mat_2of4_fp8_mmq.cuh"

#include <cstring>

static __device__ __forceinline__ int32_t pack4_mmq(const uint8_t * p) {
    int32_t w;
    memcpy(&w, p, 4);
    return w;
}

typedef int   v2i_mmq __attribute__((ext_vector_type(2)));
typedef int   v4i_mmq __attribute__((ext_vector_type(4)));
typedef float v8f_mmq __attribute__((ext_vector_type(8)));

// Same online fp8 activation quantizer as mul_mat_2of4_fp8.cu (duplicated
// locally, matching the one-quantizer-per-kernel-file convention already
// used throughout this codebase).
// T162 FIX 1 (coordinator directive): the old 32-thread/1-elem-per-thread/
// LDS+2-syncthreads quantizer above measured 2x SLOWER (36.3us vs 18.4us)
// than production's quantize_mmq_f8e4m3 (quantize.cu) at the same shape --
// this is a direct mechanical copy of that kernel's shape (128 threads/
// block, float4 vectorized loads = 4 elements/thread, warp-shuffle
// max-reduction within each aligned 8-lane group, ZERO LDS, ZERO
// __syncthreads()) -- the exact "shuffle beats LDS+sync" lesson this
// session already applied to the SWMMAC weight-scale gather, now applied
// here too. One 8-lane group (32 values, matching QK_F8E4M3) per group;
// 128 threads/block = 16 groups/block, so grid.x = ceil(n_blocks_k/16)
// instead of the old kernel's grid.x = n_blocks_k.
static __global__ void k_quantize_act_f8e4m3_mmq_fast(
        const float * __restrict__ x, block_f8e4m3 * __restrict__ y,
        const int64_t n_blocks_k, const int64_t row_stride_floats) {
    const int      group_id = threadIdx.x >> 3; // 0..15
    const int      lane8    = threadIdx.x & 7;  // 0..7, aligned to the physical warp's low 3 lane bits
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

// MMQ-grade cooperative-tile 2:4-sparse SWMMAC GEMM. BM x BN output tile
// per block, NWARPS warps/block, NTX = (BM/16)*(BN/16)/NWARPS
// register-blocked SWMMAC tiles/warp (the ILP>=4 lever, now via tile count
// instead of a single-warp inner loop). Double-buffered LDS staging for
// BOTH operands (mirrors k_mul_mat_iu4_mmq's shape exactly): one thread per
// staged row (BM/32 + BN/32 warps do the staging, warp-uniform dispatch via
// readfirstlane -- same divergence-analysis fix mul_mat_iu4_mmq.cu
// documents), ONE __syncthreads() per K-chunk (not V3's zero, not the dead
// V2's two -- the double-buffer swap genuinely needs exactly one, same as
// every other MMQ-shaped kernel in this codebase).
//
// Per-lane SWMMAC operand assembly is the validated V3 math
// (mul_mat_2of4_fp8.cu), just re-pointed at LDS instead of private
// registers -- and the weight-scale gather that needed __shfl_sync in V3
// (because V3 kept dw private-per-lane) is now a PLAIN LDS INDEX (any lane
// can read any row's staged scale directly), which is simpler AND removes
// 8*NTX shuffle instructions/chunk that V3 paid.
template <int BM, int BN, int NWARPS, bool need_check>
__launch_bounds__(NWARPS * 32, 1)
static __global__ void k_mul_mat_2of4_fp8_mmq(
        const char * __restrict__ vweight, const block_f8e4m3 * __restrict__ act, float * __restrict__ dst,
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
    // Same warp-uniform-dispatch fix mul_mat_iu4_mmq.cu documents:
    // threadIdx.y is architecturally wave-uniform (blockDim.x==32==wave
    // size) but readfirstlane makes that explicit to LLVM's divergence
    // analysis so the staging dispatch branch compiles to a plain scalar
    // s_cmp/s_cbranch instead of full exec-mask predication machinery.
    const int warp_id_u = __builtin_amdgcn_readfirstlane(warp_id);

    // T162 FIX 2b TRIED AND REVERTED: +1 int LDS row padding (mirroring
    // mmq.cuh's documented bank-conflict-avoidance rule, and a real
    // per-lane bank-collision derivation for this kernel's unpadded stride
    // -- see T162 writeup for the arithmetic) was implemented and
    // MEASURED: N=1024 k/v-proj shape went 97.9us -> 101.6us (worse, not
    // better), whole-model pp512 -2.2% to -2.8% vs Fix1-only across 3 fresh
    // runs on both dense-MMQ and 2:4-MMQ. A theoretically-sound fix that
    // measured a small net regression -- reverted, not kept, per this
    // session's "measure, don't guess" discipline.
    __shared__ int      sh_actq[2][BM][8]; // full dense fp8 row, 8 ints = 32 bytes
    __shared__ float    sh_da[2][BM];
    __shared__ int      sh_wq[2][BN][4];   // compressed 2:4 row, 4 ints = 16 bytes
    __shared__ unsigned sh_wmeta[2][BN];   // packed 4-byte sparsity metadata
    __shared__ float    sh_dw[2][BN];

    auto load_chunk = [&] (int64_t c, int buf) {
        if (warp_id_u < WARPS_A) {
            const int     row = warp_id_u * 32 + lane;
            const int64_t m   = m0 + row;
            if (!need_check || m < M) {
                const block_f8e4m3 & blk = act[m * n_blocks_k + c];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    sh_actq[buf][row][i] = pack4_mmq(blk.qs + 4 * i);
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
                const block_2of4_fp8 * blk = (const block_2of4_fp8 *) (vweight + n * nb01 + c * (int64_t) sizeof(block_2of4_fp8));
                sh_wq[buf][row][0] = pack4_mmq(blk->qs + 0);
                sh_wq[buf][row][1] = pack4_mmq(blk->qs + 4);
                sh_wq[buf][row][2] = pack4_mmq(blk->qs + 8);
                sh_wq[buf][row][3] = pack4_mmq(blk->qs + 12);
                sh_wmeta[buf][row] = (unsigned) blk->meta[0] | ((unsigned) blk->meta[1] << 8) |
                                      ((unsigned) blk->meta[2] << 16) | ((unsigned) blk->meta[3] << 24);
                sh_dw[buf][row] = __half2float(blk->d);
            } else {
                sh_wq[buf][row][0] = sh_wq[buf][row][1] = sh_wq[buf][row][2] = sh_wq[buf][row][3] = 0;
                sh_wmeta[buf][row] = 0;
                sh_dw[buf][row]    = 0.0f;
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

    const int k_half    = (lane < 16) ? 0 : 1;
    const int local_idx = (lane < 16) ? lane : (lane - 16);
    const int out_row_base = (lane >= 16) ? 8 : 0;

    int cur = 0;
    for (int64_t c = 0; c < n_blocks_k; ++c) {
        const int nxt = cur ^ 1;
        if (c + 1 < n_blocks_k) {
            // Issued before this chunk's SWMMAC+accumulate below, no
            // intervening sync -- overlaps DRAM latency with compute
            // (the double-buffer K-pipeline the coordinator asked for).
            load_chunk(c + 1, nxt);
        }

#pragma unroll
        for (int s = 0; s < NTX; ++s) {
            const int t  = warp_id + s * NWARPS;
            const int mi = t / NTILES_N;
            const int ni = t % NTILES_N;

            const int w_row = ni * 16 + local_idx; // row within sh_wq/sh_wmeta/sh_dw
            const int a_col = mi * 16 + local_idx; // row within sh_actq/sh_da

            const v2i_mmq a_arg = { sh_wq[cur][w_row][k_half*2 + 0], sh_wq[cur][w_row][k_half*2 + 1] };
            const unsigned idxv = (sh_wmeta[cur][w_row] >> (k_half * 16)) & 0xFFFFu;
            const v4i_mmq b_arg = {
                sh_actq[cur][a_col][k_half*4 + 0], sh_actq[cur][a_col][k_half*4 + 1],
                sh_actq[cur][a_col][k_half*4 + 2], sh_actq[cur][a_col][k_half*4 + 3]
            };

            v8f_mmq c0 = {0, 0, 0, 0, 0, 0, 0, 0};
            const v8f_mmq raw = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a_arg, b_arg, c0, idxv);

#pragma unroll
            for (int l = 0; l < 8; ++l) {
                // Plain LDS index, not a shuffle -- any lane can read any
                // row's staged scale directly now that it lives in LDS.
                const float dw_row = sh_dw[cur][ni * 16 + out_row_base + l];
                const float da_col = sh_da[cur][a_col];
                acc[s][l] += raw[l] * dw_row * da_col;
            }
        }

        __syncthreads(); // guards the LDS double-buffer swap
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

// T162 FIX 2a: coarser K-per-sync (coordinator directive). k_mul_mat_2of4_fp8_mmq
// above double-buffers but still syncs every single QK_2OF4_FP8=32-K chunk.
// Diagnostic finding: production mmq.cuh does NOT double-buffer at all --
// it's single-buffered (load->sync->compute->sync, no [2] dimension) but
// covers MMQ_ITER_K=256 (8 chunks) per sync-pair, paying ~4x FEWER
// syncthreads per K-element than our double-buffered-but-32-K-granular
// kernel. This kernel matches that shape: single LDS buffer (no double
// buffer -- the freed LDS budget goes toward staging KPS chunks at once
// instead), KPS chunks loaded/synced/computed together per outer
// iteration. Kept as a SEPARATE kernel (not a rewrite of the one above) so
// KPS=1 can be A/B'd cleanly against the double-buffered baseline instead
// of silently changing it.
template <int BM, int BN, int NWARPS, int KPS, bool need_check>
__launch_bounds__(NWARPS * 32, 1)
static __global__ void k_mul_mat_2of4_fp8_mmq_kps(
        const char * __restrict__ vweight, const block_f8e4m3 * __restrict__ act, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t nb01, const int64_t n_blocks_k,
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
    const int64_t n0      = (int64_t) blockIdx.x * BN;
    const int64_t m0      = (int64_t) blockIdx.y * BM;
    const int     lane    = threadIdx.x;
    const int     warp_id = threadIdx.y;
    const int     warp_id_u = __builtin_amdgcn_readfirstlane(warp_id);

    __shared__ int      sh_actq[BM][KPS][8]; // single-buffered, KPS chunks staged at once
    __shared__ float    sh_da[BM][KPS];
    __shared__ int      sh_wq[BN][KPS][4];
    __shared__ unsigned sh_wmeta[BN][KPS];
    __shared__ float    sh_dw[BN][KPS];

    auto load_group = [&] (int64_t c0) {
        if (warp_id_u < WARPS_A) {
            const int     row = warp_id_u * 32 + lane;
            const int64_t m   = m0 + row;
#pragma unroll
            for (int kk = 0; kk < KPS; ++kk) {
                const int64_t c = c0 + kk;
                if (c < n_blocks_k && (!need_check || m < M)) {
                    const block_f8e4m3 & blk = act[m * n_blocks_k + c];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        sh_actq[row][kk][i] = pack4_mmq(blk.qs + 4 * i);
                    }
                    sh_da[row][kk] = __half2float(blk.d);
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        sh_actq[row][kk][i] = 0;
                    }
                    sh_da[row][kk] = 0.0f;
                }
            }
        } else if (warp_id_u < WARPS_A + WARPS_B) {
            const int     row = (warp_id_u - WARPS_A) * 32 + lane;
            const int64_t n   = n0 + row;
#pragma unroll
            for (int kk = 0; kk < KPS; ++kk) {
                const int64_t c = c0 + kk;
                if (c < n_blocks_k && (!need_check || n < N)) {
                    const block_2of4_fp8 * blk = (const block_2of4_fp8 *) (vweight + n * nb01 + c * (int64_t) sizeof(block_2of4_fp8));
                    sh_wq[row][kk][0] = pack4_mmq(blk->qs + 0);
                    sh_wq[row][kk][1] = pack4_mmq(blk->qs + 4);
                    sh_wq[row][kk][2] = pack4_mmq(blk->qs + 8);
                    sh_wq[row][kk][3] = pack4_mmq(blk->qs + 12);
                    sh_wmeta[row][kk] = (unsigned) blk->meta[0] | ((unsigned) blk->meta[1] << 8) |
                                         ((unsigned) blk->meta[2] << 16) | ((unsigned) blk->meta[3] << 24);
                    sh_dw[row][kk] = __half2float(blk->d);
                } else {
                    sh_wq[row][kk][0] = sh_wq[row][kk][1] = sh_wq[row][kk][2] = sh_wq[row][kk][3] = 0;
                    sh_wmeta[row][kk] = 0;
                    sh_dw[row][kk]    = 0.0f;
                }
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

    for (int64_t c0 = 0; c0 < n_blocks_k; c0 += KPS) {
        load_group(c0);
        __syncthreads(); // load -> compute barrier (matches stock mmq.cuh's shape)

#pragma unroll
        for (int kk = 0; kk < KPS; ++kk) {
            if (c0 + kk >= n_blocks_k) { break; }
#pragma unroll
            for (int s = 0; s < NTX; ++s) {
                const int t  = warp_id + s * NWARPS;
                const int mi = t / NTILES_N;
                const int ni = t % NTILES_N;

                const int w_row = ni * 16 + local_idx;
                const int a_col = mi * 16 + local_idx;

                const v2i_mmq a_arg = { sh_wq[w_row][kk][k_half*2 + 0], sh_wq[w_row][kk][k_half*2 + 1] };
                const unsigned idxv = (sh_wmeta[w_row][kk] >> (k_half * 16)) & 0xFFFFu;
                const v4i_mmq b_arg = {
                    sh_actq[a_col][kk][k_half*4 + 0], sh_actq[a_col][kk][k_half*4 + 1],
                    sh_actq[a_col][kk][k_half*4 + 2], sh_actq[a_col][kk][k_half*4 + 3]
                };

                v8f_mmq c0v = {0, 0, 0, 0, 0, 0, 0, 0};
                const v8f_mmq raw = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a_arg, b_arg, c0v, idxv);

#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const float dw_row = sh_dw[ni * 16 + out_row_base + l][kk];
                    const float da_col = sh_da[a_col][kk];
                    acc[s][l] += raw[l] * dw_row * da_col;
                }
            }
        }

        __syncthreads(); // compute -> next-load barrier (single buffer, matches stock)
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

static int ggml_cuda_2of4_fp8_mmq_kps() {
    static const int v = [] {
        const char * env = getenv("GGML_HIP_2OF4_FP8_MMQ_KPS");
        const int x = env ? atoi(env) : 1;
        return (x == 1 || x == 2 || x == 4 || x == 8) ? x : 1;
    }();
    return v;
}

// BM/BN/NWARPS tile geometry, selectable via env for the sweep (compile-time
// template params -- a small fixed set of instantiations, matching
// mul_mat_iu4_mmq.cu's convention of picking ONE tuned constexpr geometry
// rather than a huge cross product).
// Each geometry must satisfy BM/32 + BN/32 <= NWARPS (staging coverage) and
// (BM/16)*(BN/16) % NWARPS == 0 (even tile division) -- checked at compile
// time by the kernel's own static_asserts either way.
enum class MmqGeom { G64x64x4, G128x128x8, G64x64x8, G128x64x8, G64x128x8,
                      G256x128x16, G128x256x16, G256x256x16, G256x64x16, G64x256x16,
                      G128x128x16, G128x128x32,
                      // T162 k/v-proj occupancy sweep (coordinator directive): small
                      // output tiles for the N=1024 low-occupancy shape ONLY, tested
                      // with the UNCHANGED double-buffered/KPS=1 kernel (Fix-2a/2b
                      // stay dead, not re-opened) -- pure tile-size occupancy lever.
                      G64x32x4, G64x32x8, G32x64x4, G32x64x8, G96x64x8, G64x96x8 };

static MmqGeom ggml_cuda_2of4_fp8_mmq_geom() {
    static const MmqGeom g = [] {
        const char * env = getenv("GGML_HIP_2OF4_FP8_MMQ_GEOM");
        if (env == nullptr) { return MmqGeom::G128x128x8; } // measured best so far, see T162 capstone writeup
        if (strcmp(env, "64x64x4") == 0)     { return MmqGeom::G64x64x4; }
        if (strcmp(env, "64x64x8") == 0)     { return MmqGeom::G64x64x8; }
        if (strcmp(env, "128x64x8") == 0)    { return MmqGeom::G128x64x8; }
        if (strcmp(env, "64x128x8") == 0)    { return MmqGeom::G64x128x8; }
        if (strcmp(env, "256x128x16") == 0)  { return MmqGeom::G256x128x16; }
        if (strcmp(env, "128x256x16") == 0)  { return MmqGeom::G128x256x16; }
        if (strcmp(env, "256x256x16") == 0)  { return MmqGeom::G256x256x16; }
        if (strcmp(env, "256x64x16") == 0)   { return MmqGeom::G256x64x16; }
        if (strcmp(env, "64x256x16") == 0)   { return MmqGeom::G64x256x16; }
        if (strcmp(env, "128x128x16") == 0)  { return MmqGeom::G128x128x16; }
        if (strcmp(env, "128x128x32") == 0)  { return MmqGeom::G128x128x32; }
        if (strcmp(env, "64x32x4") == 0)     { return MmqGeom::G64x32x4; }
        if (strcmp(env, "64x32x8") == 0)     { return MmqGeom::G64x32x8; }
        if (strcmp(env, "32x64x4") == 0)     { return MmqGeom::G32x64x4; }
        if (strcmp(env, "32x64x8") == 0)     { return MmqGeom::G32x64x8; }
        if (strcmp(env, "96x64x8") == 0)     { return MmqGeom::G96x64x8; }
        if (strcmp(env, "64x96x8") == 0)     { return MmqGeom::G64x96x8; }
        return MmqGeom::G128x128x8;
    }();
    return g;
}

bool ggml_cuda_op_mul_mat_2of4_fp8_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_2OF4_FP8);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(src0->ne[0] % QK_2OF4_FP8 == 0);

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    GGML_ASSERT(GGML_CUDA_CC_IS_RDNA4(cc) && "GGML_HIP_2OF4_FP8_MMQ requires RDNA4");

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t n_blocks_k = K / QK_2OF4_FP8;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_f8e4m3> act_q(ctx.pool(), (size_t) (M * n_blocks_k));
    {
        const int64_t row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
        const dim3 grid((n_blocks_k + 15) / 16, M, 1);
        const dim3 block(128, 1, 1);
        k_quantize_act_f8e4m3_mmq_fast<<<grid, block, 0, stream>>>((const float *) src1->data, act_q.get(), n_blocks_k, row_stride_floats);
    }

    const int64_t nb01 = src0->nb[1];
    const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
    const block_f8e4m3 * d_act = act_q.get();
    const char * d_w = (const char *) src0->data;
    float * d_dst = (float *) dst->data;

    // T162 FIX 2a: coarser K-per-sync, opt-in (GGML_HIP_2OF4_FP8_MMQ_KPS,
    // default 1 = falls through to the unmodified baseline kernel below).
    // GGML_HIP_2OF4_FP8_MMQ_KPS_GEOM picks which tile geometry the KPS
    // kernel uses ("128x128" default, matching the established capstone
    // winner, or "64x64" -- smaller tile, more resident blocks/CU -- for
    // testing the coarser-sync lever specifically against the low-occupancy
    // N=1024 k/v-proj shape).
    const int kps = ggml_cuda_2of4_fp8_mmq_kps();
    if (kps > 1) {
        const bool small_geom = getenv("GGML_HIP_2OF4_FP8_MMQ_KPS_GEOM") != nullptr &&
                                 strcmp(getenv("GGML_HIP_2OF4_FP8_MMQ_KPS_GEOM"), "64x64") == 0;
#define LAUNCH_MMQ_KPS(BM_, BN_, NWARPS_, KPS_)                                                           \
        do {                                                                                              \
            const dim3 grid_((N + (BN_) - 1) / (BN_), (M + (BM_) - 1) / (BM_), 1);                        \
            const dim3 block_(32, (NWARPS_), 1);                                                          \
            const bool need_check = (M % (BM_) != 0) || (N % (BN_) != 0);                                 \
            if (need_check) {                                                                             \
                k_mul_mat_2of4_fp8_mmq_kps<BM_, BN_, NWARPS_, KPS_, true><<<grid_, block_, 0, stream>>>(    \
                        d_w, d_act, d_dst, M, N, nb01, n_blocks_k, dst_row_stride_floats);                  \
            } else {                                                                                       \
                k_mul_mat_2of4_fp8_mmq_kps<BM_, BN_, NWARPS_, KPS_, false><<<grid_, block_, 0, stream>>>(   \
                        d_w, d_act, d_dst, M, N, nb01, n_blocks_k, dst_row_stride_floats);                  \
            }                                                                                              \
        } while (0)
        if (small_geom) {
            switch (kps) {
                case 2: LAUNCH_MMQ_KPS(64, 64, 8, 2); break;
                case 4: LAUNCH_MMQ_KPS(64, 64, 8, 4); break;
                default: LAUNCH_MMQ_KPS(64, 64, 8, 2); break;
            }
        } else {
            switch (kps) {
                case 2: LAUNCH_MMQ_KPS(128, 128, 8, 2); break;
                case 4: LAUNCH_MMQ_KPS(128, 128, 8, 4); break;
                case 8: LAUNCH_MMQ_KPS(128, 128, 8, 8); break;
                default: LAUNCH_MMQ_KPS(128, 128, 8, 2); break;
            }
        }
#undef LAUNCH_MMQ_KPS
        return true;
    }

    // T162 k/v-proj occupancy fix (coordinator directive): shape-adaptive
    // output tile, PURELY a tile-size lever -- same unmodified
    // double-buffered/KPS=1 k_mul_mat_2of4_fp8_mmq kernel as the big
    // shapes, just a smaller BM x BN for low-occupancy N. Isolated sweep
    // (rocprofv3 kernel-trace, N=1024/K=4096/M=512) found 64x64x8 the
    // winner: 69.89us, BEATS stock's 74.13us (-5.7%) -- vs the flat
    // 128x128x8 default's 97.9us (grid=32 blocks/64 CU=0.5/CU) and vs
    // 64x32x4's 73.72us (2nd place, still slightly behind stock).
    // 64x64x8 gives grid=(16,8)=128 blocks=2/CU. Opt-in via
    // GGML_HIP_2OF4_FP8_MMQ_ADAPTIVE (default off); N<=1024 threshold
    // matches exactly the one real low-occupancy shape in this model
    // (k/v-proj) without touching q/o-proj (N=4096) or the FFN (N=14336),
    // which stay on the unchanged 128x128x8 default.
    const bool adaptive = getenv("GGML_HIP_2OF4_FP8_MMQ_ADAPTIVE") != nullptr;
    const MmqGeom geom = (adaptive && N <= 1024) ? MmqGeom::G64x64x8 : ggml_cuda_2of4_fp8_mmq_geom();

#define LAUNCH_MMQ(BM_, BN_, NWARPS_)                                                                     \
    do {                                                                                                  \
        const dim3 grid_((N + (BN_) - 1) / (BN_), (M + (BM_) - 1) / (BM_), 1);                            \
        const dim3 block_(32, (NWARPS_), 1);                                                              \
        const bool need_check = (M % (BM_) != 0) || (N % (BN_) != 0);                                     \
        if (need_check) {                                                                                 \
            k_mul_mat_2of4_fp8_mmq<BM_, BN_, NWARPS_, true><<<grid_, block_, 0, stream>>>(                 \
                    d_w, d_act, d_dst, M, N, nb01, n_blocks_k, dst_row_stride_floats);                     \
        } else {                                                                                           \
            k_mul_mat_2of4_fp8_mmq<BM_, BN_, NWARPS_, false><<<grid_, block_, 0, stream>>>(                \
                    d_w, d_act, d_dst, M, N, nb01, n_blocks_k, dst_row_stride_floats);                     \
        }                                                                                                  \
    } while (0)

    switch (geom) {
        case MmqGeom::G64x64x4:     LAUNCH_MMQ(64, 64, 4);     break;
        case MmqGeom::G64x64x8:     LAUNCH_MMQ(64, 64, 8);     break;
        case MmqGeom::G128x64x8:    LAUNCH_MMQ(128, 64, 8);    break;
        case MmqGeom::G64x128x8:    LAUNCH_MMQ(64, 128, 8);    break;
        case MmqGeom::G256x128x16:  LAUNCH_MMQ(256, 128, 16);  break;
        case MmqGeom::G128x256x16:  LAUNCH_MMQ(128, 256, 16);  break;
        case MmqGeom::G256x256x16:  LAUNCH_MMQ(256, 256, 16);  break;
        case MmqGeom::G256x64x16:   LAUNCH_MMQ(256, 64, 16);   break;
        case MmqGeom::G64x256x16:   LAUNCH_MMQ(64, 256, 16);   break;
        case MmqGeom::G128x128x16:  LAUNCH_MMQ(128, 128, 16);  break;
        case MmqGeom::G128x128x32:  LAUNCH_MMQ(128, 128, 32);  break;
        case MmqGeom::G64x32x4:     LAUNCH_MMQ(64, 32, 4);     break;
        case MmqGeom::G64x32x8:     LAUNCH_MMQ(64, 32, 8);     break;
        case MmqGeom::G32x64x4:     LAUNCH_MMQ(32, 64, 4);     break;
        case MmqGeom::G32x64x8:     LAUNCH_MMQ(32, 64, 8);     break;
        case MmqGeom::G96x64x8:     LAUNCH_MMQ(96, 64, 8);     break;
        case MmqGeom::G64x96x8:     LAUNCH_MMQ(64, 96, 8);     break;
        default:                    LAUNCH_MMQ(128, 128, 8);   break;
    }
#undef LAUNCH_MMQ

    return true;
}
