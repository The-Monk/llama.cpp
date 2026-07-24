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
#include "ggml-quants.h"

#include <cstring>
#include <random>
#include <vector>

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

// GGML_HIP_2OF4_FP8_ILP (default unset/0 = baseline k_mul_mat_2of4_fp8
// below, unchanged): the baseline kernel issues exactly ONE SWMMAC per warp
// per K-chunk -- c0 is re-zeroed every iteration (no hardware accumulate-
// chaining), but the wavefront still has only one in-flight SWMMAC at a
// time, gated by two __syncthreads() per iteration that force full-latency
// drains before the next iteration's operand loads can start. That's ILP=1,
// the same shape the RDNA4 SWMMAC microbench measured at 84% of peak
// (642/765 TOP/s int8) -- >=4 independent accumulators (no data dependency
// between them) hid the latency and hit 100%.
//
// k_mul_mat_2of4_fp8_ilp<ILP> below applies the same fix here: each warp
// covers ILP consecutive 16-row weight tiles (same 16 activation columns)
// per K-chunk, loads the activation operand (B) ONCE (it doesn't depend on
// which weight tile), then issues ILP independent SWMMAC calls back-to-back
// -- raw[t] has no data dependency on raw[t-1], so the compiler can overlap
// their issue/latency instead of serializing on __syncthreads() after each
// one. Register cost scales with ILP (acc[ILP][8] + transient raw[ILP][8]);
// sweep 2/4/8 via the env var and take whichever measures fastest -- higher
// ILP can lose to register-pressure/occupancy cliffs same as any other
// unroll-width lever in this codebase (see mmvq.cu's VDR sweeps).
static int ggml_cuda_2of4_fp8_ilp() {
    static const int ilp = [] {
        const char * env = getenv("GGML_HIP_2OF4_FP8_ILP");
        if (env == nullptr) {
            return 0; // baseline (ILP=1 shape, original kernel)
        }
        const int v = atoi(env);
        if (v == 2 || v == 4 || v == 8) {
            return v;
        }
        return 0;
    }();
    return ilp;
}

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

// ILP>=2 variant: each warp covers ILP consecutive 16-row weight tiles
// against the SAME 16-column activation tile per K-chunk. B/d_a_local are
// loaded once (shared across all ILP tiles); the ILP SWMMAC calls are
// issued in one unrolled loop with no cross-dependency, then consumed in a
// second unrolled loop -- kept as two separate loops (not fused) so the
// compiler has no reason to serialize issue behind consumption.
template <int ILP>
__launch_bounds__(32, 1)
static __global__ void k_mul_mat_2of4_fp8_ilp(
        const char * __restrict__ vweight, const block_f8e4m3 * __restrict__ act, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t nb01, const int64_t n_blocks_k,
        const int64_t dst_row_stride_floats) {
#if defined(RDNA4)
    const int64_t n0  = (int64_t) blockIdx.x * (16 * ILP);
    const int64_t m0  = (int64_t) blockIdx.y * 16;
    const int     tid = threadIdx.x;

    const int k_half    = (tid < 16) ? 0 : 1;
    const int local_idx = (tid < 16) ? tid : (tid - 16);

    __shared__ float sh_dw[ILP][16];

    float acc[ILP][8];
#pragma unroll
    for (int t = 0; t < ILP; ++t) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            acc[t][l] = 0.0f;
        }
    }

    for (int64_t c = 0; c < n_blocks_k; ++c) {
        v4i b_arg      = {0, 0, 0, 0};
        float d_a_local = 0.0f;
        const int64_t act_col = m0 + local_idx;
        if (act_col < M) {
            const block_f8e4m3 & blka = act[act_col * n_blocks_k + c];
            b_arg.x   = pack4(blka.qs + k_half*16 + 0);
            b_arg.y   = pack4(blka.qs + k_half*16 + 4);
            b_arg.z   = pack4(blka.qs + k_half*16 + 8);
            b_arg.w   = pack4(blka.qs + k_half*16 + 12);
            d_a_local = __half2float(blka.d);
        }

        v2i a_arg[ILP];
        unsigned idxv[ILP];
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
            a_arg[t] = {0, 0};
            idxv[t]  = 0;
            const int64_t weight_row = n0 + (int64_t) t*16 + local_idx;
            if (weight_row < N) {
                const block_2of4_fp8 * blkw = (const block_2of4_fp8 *) (vweight + weight_row * nb01 + c * (int64_t) sizeof(block_2of4_fp8));
                a_arg[t].x = pack4(blkw->qs + k_half*8 + 0);
                a_arg[t].y = pack4(blkw->qs + k_half*8 + 4);
                idxv[t]    = (unsigned) blkw->meta[2*k_half] | ((unsigned) blkw->meta[2*k_half + 1] << 8);
                const float d_w = __half2float(blkw->d);
                if (k_half == 0) {
                    sh_dw[t][local_idx] = d_w;
                }
            } else if (k_half == 0) {
                sh_dw[t][local_idx] = 0.0f;
            }
        }
        __syncthreads(); // sh_dw fully populated before anyone reads it below

        // Issue ILP independent SWMMAC calls -- raw[t] has no data
        // dependency on raw[t-1], the ILP>=4 lever the microbench validated.
        v8f raw[ILP];
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
            v8f c0 = {0, 0, 0, 0, 0, 0, 0, 0};
            raw[t] = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a_arg[t], b_arg, c0, idxv[t]);
        }

        const int out_row_base = (tid >= 16) ? 8 : 0;
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                acc[t][l] += raw[t][l] * sh_dw[t][out_row_base + l] * d_a_local;
            }
        }
        __syncthreads(); // sh_dw about to be overwritten next chunk
    }

    const int out_col      = local_idx;
    const int out_row_base = (tid >= 16) ? 8 : 0;
#pragma unroll
    for (int t = 0; t < ILP; ++t) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int64_t n = n0 + (int64_t) t*16 + out_row_base + l;
            const int64_t m = m0 + out_col;
            if (m < M && n < N) {
                dst[m * dst_row_stride_floats + n] = acc[t][l];
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

// ============================================================================
// T162 V2: MMQ-grade redesign (GGML_HIP_2OF4_FP8_V2, default off).
//
// Root cause established by the ILP-only kernel above (raising ILP alone
// made throughput WORSE, measured -- this is NOT an issue-latency/ILP
// problem): it is a KERNEL-SHAPE problem. k_mul_mat_2of4_fp8[_ilp] launches
// blockDim.x==32 -- ONE block IS exactly one wavefront, there is no second
// wave in that block to synchronize with -- yet it pays two __syncthreads()
// per K-chunk. Those barriers exist only to move each of the 16 per-row
// weight scales (sh_dw[]) from the lane that loaded it to the *other* 8
// lanes whose D-matrix output slots need it. That is pure INTRA-WAVEFRONT
// lane-to-lane data motion (all 32 lanes are already one lock-step SIMD32
// unit) -- the correct tool is a cross-lane shuffle (one cycle, no drain),
// not an LDS round trip gated by two block-wide barriers that force a full
// load-latency drain every iteration.
//
// Three fixes, applied together:
//  1. __shfl_sync() replaces sh_dw+__syncthreads() x2 for the weight-scale
//     gather -- zero LDS, zero barrier, for data that was never actually
//     cross-WARP to begin with.
//  2. Multi-warp cooperative block (blockDim=(32,NWARPS)): each warp owns
//     its own private ILP consecutive 16-row weight tiles (register-blocked,
//     no LDS needed -- A is per-lane-private, never shared across warps) --
//     bigger, fewer block launches instead of NWARPS independent 32-thread
//     blocks.
//  3. The activation operand (B) IS genuinely shared across every warp in
//     the block (same M-tile, doesn't depend on which weight rows a warp
//     owns) -- staged once per chunk into a small double-buffered LDS array
//     by warp 0 only, so the other NWARPS-1 warps stop redundantly
//     re-issuing the same global load. This is the one place a real
//     multi-warp double buffer + a single (not double) __syncthreads() per
//     chunk earns its keep -- mirrors the validated double-buffered-LDS
//     shape already proven out in k_mul_mat_iu4_mmq (mul_mat_iu4_mmq.cu).
// ============================================================================

template <int ILP>
__launch_bounds__(1024, 1)
static __global__ void k_mul_mat_2of4_fp8_v2(
        const char * __restrict__ vweight, const block_f8e4m3 * __restrict__ act, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t nb01, const int64_t n_blocks_k,
        const int64_t dst_row_stride_floats) {
#if defined(RDNA4)
    const int lane      = threadIdx.x; // 0..31
    const int warp_id_u = __builtin_amdgcn_readfirstlane((int) threadIdx.y); // wave-uniform, see mul_mat_iu4_mmq.cu
    const int n_warps   = blockDim.y;

    const int k_half       = (lane < 16) ? 0 : 1;
    const int local_idx    = (lane < 16) ? lane : (lane - 16);
    const int out_row_base = (lane >= 16) ? 8 : 0;

    const int64_t n0      = (int64_t) blockIdx.x * ((int64_t) n_warps * ILP * 16) + (int64_t) warp_id_u * ILP * 16;
    const int64_t m0      = (int64_t) blockIdx.y * 16;
    const int64_t act_col = m0 + local_idx;

    __shared__ v4i   sh_b[2][32];
    __shared__ float sh_da[2][32];

    // Shared activation (B) operand: identical for every warp in this
    // block, so only warp 0 stages it -- no redundant re-fetch per warp.
    auto load_b = [&] (int64_t c, int buf) {
        if (warp_id_u == 0) {
            v4i   b  = {0, 0, 0, 0};
            float da = 0.0f;
            if (act_col < M) {
                const block_f8e4m3 & blka = act[act_col * n_blocks_k + c];
                b.x = pack4(blka.qs + k_half*16 + 0);
                b.y = pack4(blka.qs + k_half*16 + 4);
                b.z = pack4(blka.qs + k_half*16 + 8);
                b.w = pack4(blka.qs + k_half*16 + 12);
                da  = __half2float(blka.d);
            }
            sh_b[buf][lane]  = b;
            sh_da[buf][lane] = da;
        }
    };

    // Private per-warp weight (A) operand: each lane owns exactly one
    // (weight_row, k_half) pair, never shared cross-warp -- a direct
    // 2-deep register prefetch is the right tool, no LDS/sync needed here.
    v2i      a_cur[ILP], a_nxt[ILP];
    unsigned idx_cur[ILP], idx_nxt[ILP];
    float    dw_cur[ILP], dw_nxt[ILP];

    auto load_a = [&] (int64_t c, v2i (&a)[ILP], unsigned (&idx)[ILP], float (&dw)[ILP]) {
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
            a[t]   = v2i{0, 0};
            idx[t] = 0;
            dw[t]  = 0.0f;
            const int64_t weight_row = n0 + (int64_t) t*16 + local_idx;
            if (weight_row < N) {
                const block_2of4_fp8 * blkw = (const block_2of4_fp8 *) (vweight + weight_row * nb01 + c * (int64_t) sizeof(block_2of4_fp8));
                a[t].x = pack4(blkw->qs + k_half*8 + 0);
                a[t].y = pack4(blkw->qs + k_half*8 + 4);
                idx[t] = (unsigned) blkw->meta[2*k_half] | ((unsigned) blkw->meta[2*k_half + 1] << 8);
                dw[t]  = __half2float(blkw->d);
            }
        }
    };

    float acc[ILP][8];
#pragma unroll
    for (int t = 0; t < ILP; ++t) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            acc[t][l] = 0.0f;
        }
    }

    if (n_blocks_k > 0) {
        load_b(0, 0);
        load_a(0, a_cur, idx_cur, dw_cur);
    }
    __syncthreads(); // sh_b[0]/sh_da[0] (warp 0's write) visible to all warps before first read

    int cur = 0;
    for (int64_t c = 0; c < n_blocks_k; ++c) {
        const int  nxt       = cur ^ 1;
        const bool have_next = (c + 1 < n_blocks_k);
        if (have_next) {
            // Issued before this chunk's SWMMAC+accumulate below, no
            // intervening sync -- overlaps DRAM latency with compute
            // instead of the baseline's fully-serialized load->sync->
            // compute->sync shape.
            load_b(c + 1, nxt);
            load_a(c + 1, a_nxt, idx_nxt, dw_nxt);
        }

        const v4i   b_cur  = sh_b[cur][lane];
        const float da_cur = sh_da[cur][lane];

        v8f raw[ILP];
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
            v8f c0 = {0, 0, 0, 0, 0, 0, 0, 0};
            raw[t] = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a_cur[t], b_cur, c0, idx_cur[t]);
        }
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                // Intra-wavefront gather, no LDS: row out_row_base+l's scale
                // lives in lane (out_row_base+l), always < 16 (see file
                // header for the derivation -- both k_half=0 and k_half=1
                // lanes redundantly compute the SAME weight_row's dw, so the
                // low-16-lanes copy alone is sufficient as the broadcast
                // source for every reader, regardless of its own k_half).
                const float dw_row = __shfl_sync(0xFFFFFFFFu, dw_cur[t], out_row_base + l, WARP_SIZE);
                acc[t][l] += raw[t][l] * dw_row * da_cur;
            }
        }

        if (have_next) {
#pragma unroll
            for (int t = 0; t < ILP; ++t) {
                a_cur[t]   = a_nxt[t];
                idx_cur[t] = idx_nxt[t];
                dw_cur[t]  = dw_nxt[t];
            }
        }
        __syncthreads(); // guards sh_b/sh_da double-buffer swap (warp0 writer vs all-warp readers)
        cur = nxt;
    }

    const int out_col = local_idx;
#pragma unroll
    for (int t = 0; t < ILP; ++t) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int64_t n = n0 + (int64_t) t*16 + out_row_base + l;
            const int64_t m = m0 + out_col;
            if (m < M && n < N) {
                dst[m * dst_row_stride_floats + n] = acc[t][l];
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

// V3: isolated-hypothesis A/B kernel (T162 method discipline -- V2's
// LDS-shared-B design was tested and MEASURED WORSE than the plain
// single-tile baseline at every WARPS/ILP setting, falsifying the "LDS
// double-buffer earns its keep for B" half of the V2 design; see the T162
// writeup for the numbers). V3 isolates the OTHER half of the hypothesis
// (shfl-based dw gather removes the need for a barrier at all) by making
// EVERY operand -- A and B both -- fully private per-warp registers, with
// ZERO __shared__ memory and ZERO __syncthreads() anywhere in the kernel
// (every warp in the block is then a fully independent computation; the
// only thing multi-warp buys here is fewer/bigger block launches). This is
// the true minimal test of "does removing the two-syncthreads/iteration
// barrier, on its own, beat the baseline" -- kept separate from V2 rather
// than folded in as a template bool, specifically so it can be measured
// and reported as its own row instead of silently blended into the V2
// sweep.
template <int ILP, bool need_check>
__launch_bounds__(1024, 1)
static __global__ void k_mul_mat_2of4_fp8_v3(
        const char * __restrict__ vweight, const block_f8e4m3 * __restrict__ act, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t nb01, const int64_t n_blocks_k,
        const int64_t dst_row_stride_floats) {
#if defined(RDNA4)
    const int lane      = threadIdx.x; // 0..31
    const int warp_id_u = __builtin_amdgcn_readfirstlane((int) threadIdx.y);
    const int n_warps   = blockDim.y;

    const int k_half       = (lane < 16) ? 0 : 1;
    const int local_idx    = (lane < 16) ? lane : (lane - 16);
    const int out_row_base = (lane >= 16) ? 8 : 0;

    const int64_t n0      = (int64_t) blockIdx.x * ((int64_t) n_warps * ILP * 16) + (int64_t) warp_id_u * ILP * 16;
    const int64_t m0      = (int64_t) blockIdx.y * 16;
    const int64_t act_col = m0 + local_idx;

    v2i      a_cur[ILP], a_nxt[ILP];
    unsigned idx_cur[ILP], idx_nxt[ILP];
    float    dw_cur[ILP], dw_nxt[ILP];
    v4i      b_cur, b_nxt;
    float    da_cur, da_nxt;

    // need_check=false (grid exactly tiles M/N, true whenever M%16==0 and
    // N%(n_warps*ILP*16)==0 -- both hold for every real Llama-3.1-8B linear
    // layer at pp512's M=512 and this kernel's WARPS=32/ILP=4 default,
    // 4096%2048==0 / 14336%2048==0) drops every bounds branch from the hot
    // loop -- every lane's row/col is provably in range by construction.
    auto load_chunk = [&] (int64_t c, v2i (&a)[ILP], unsigned (&idx)[ILP], float (&dw)[ILP], v4i & b, float & da) {
        if (!need_check || act_col < M) {
            const block_f8e4m3 & blka = act[act_col * n_blocks_k + c];
            b.x = pack4(blka.qs + k_half*16 + 0);
            b.y = pack4(blka.qs + k_half*16 + 4);
            b.z = pack4(blka.qs + k_half*16 + 8);
            b.w = pack4(blka.qs + k_half*16 + 12);
            da  = __half2float(blka.d);
        } else {
            b  = v4i{0, 0, 0, 0};
            da = 0.0f;
        }
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
            const int64_t weight_row = n0 + (int64_t) t*16 + local_idx;
            if (!need_check || weight_row < N) {
                const block_2of4_fp8 * blkw = (const block_2of4_fp8 *) (vweight + weight_row * nb01 + c * (int64_t) sizeof(block_2of4_fp8));
                a[t].x = pack4(blkw->qs + k_half*8 + 0);
                a[t].y = pack4(blkw->qs + k_half*8 + 4);
                idx[t] = (unsigned) blkw->meta[2*k_half] | ((unsigned) blkw->meta[2*k_half + 1] << 8);
                dw[t]  = __half2float(blkw->d);
            } else {
                a[t]   = v2i{0, 0};
                idx[t] = 0;
                dw[t]  = 0.0f;
            }
        }
    };

    float acc[ILP][8];
#pragma unroll
    for (int t = 0; t < ILP; ++t) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            acc[t][l] = 0.0f;
        }
    }

    if (n_blocks_k > 0) {
        load_chunk(0, a_cur, idx_cur, dw_cur, b_cur, da_cur);
    }

    for (int64_t c = 0; c < n_blocks_k; ++c) {
        const bool have_next = (c + 1 < n_blocks_k);
        if (have_next) {
            load_chunk(c + 1, a_nxt, idx_nxt, dw_nxt, b_nxt, da_nxt);
        }

        v8f raw[ILP];
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
            v8f c0 = {0, 0, 0, 0, 0, 0, 0, 0};
            raw[t] = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a_cur[t], b_cur, c0, idx_cur[t]);
        }
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float dw_row = __shfl_sync(0xFFFFFFFFu, dw_cur[t], out_row_base + l, WARP_SIZE);
                acc[t][l] += raw[t][l] * dw_row * da_cur;
            }
        }

        if (have_next) {
#pragma unroll
            for (int t = 0; t < ILP; ++t) {
                a_cur[t]   = a_nxt[t];
                idx_cur[t] = idx_nxt[t];
                dw_cur[t]  = dw_nxt[t];
            }
            b_cur  = b_nxt;
            da_cur = da_nxt;
        }
    }

    const int out_col = local_idx;
#pragma unroll
    for (int t = 0; t < ILP; ++t) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int64_t n = n0 + (int64_t) t*16 + out_row_base + l;
            const int64_t m = m0 + out_col;
            if (!need_check || (m < M && n < N)) {
                dst[m * dst_row_stride_floats + n] = acc[t][l];
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

static bool ggml_cuda_2of4_fp8_v3_enabled() {
    static const bool v = getenv("GGML_HIP_2OF4_FP8_V3") != nullptr;
    return v;
}

// GGML_HIP_2OF4_FP8_V2=1 selects the kernel above; GGML_HIP_2OF4_FP8_V2_WARPS
// (default 4) sets warps/block (runtime, blockDim.y -- no template bloat),
// GGML_HIP_2OF4_FP8_V2_ILP (default 4, must be 2/4/8) sets the per-warp
// register-blocked tile count (template, for unrolling).
static bool ggml_cuda_2of4_fp8_v2_enabled() {
    static const bool v = getenv("GGML_HIP_2OF4_FP8_V2") != nullptr;
    return v;
}
static int ggml_cuda_2of4_fp8_v2_warps() {
    static const int w = [] {
        const char * env = getenv("GGML_HIP_2OF4_FP8_V2_WARPS");
        // Default 32 (max block size, 1024 threads) -- measured best for V3:
        // WARPS=32/ILP=4 = 2171-2213 t/s vs the best of every other combo
        // tried (WARPS=16/ILP=4 = 1259, WARPS=4/ILP=4 = 1156) -- see T162
        // writeup. Env override kept for re-sweeping, not because 32 is
        // ever expected to lose.
        const int v = env ? atoi(env) : 32;
        return (v >= 1 && v <= 32) ? v : 32;
    }();
    return w;
}
static int ggml_cuda_2of4_fp8_v2_ilp() {
    static const int v = [] {
        const char * env = getenv("GGML_HIP_2OF4_FP8_V2_ILP");
        const int x = env ? atoi(env) : 4;
        return (x == 1 || x == 2 || x == 3 || x == 4 || x == 5 || x == 6 || x == 8) ? x : 4;
    }();
    return v;
}

// T162 occupancy-starvation follow-up (coordinator directive): the isolated
// per-shape sweep (GGML_HIP_2OF4_FP8_SHAPE_BENCH, ~/mul_mat_2of4_fp8.cu
// run_2of4_fp8_shape_bench) found the "N=4096 => 1 block/CU at BN=2048 =>
// starved" story is only HALF right. It holds for q/o-proj (N=4096,
// K=4096): WARPS=4/ILP=4 (BN=256, 8 blocks/CU) measured 0.6144ms vs
// WARPS=32/ILP=4's 0.6240ms -- a real but SMALL 1.6% win. But down-proj is
// ALSO N=4096 (same "1 block/CU" grid shape at BN=2048) with K=14336, and
// there WARPS=32/ILP=4 measured 2.1783ms vs WARPS=4/ILP=4's 4.5223ms --
// WARPS=32 wins by 2.1x. The determining variable is K (loop depth: how
// many chunks amortize the block-launch/occupancy cost), NOT N alone --
// short-K shapes want more, smaller blocks; long-K shapes want fewer,
// bigger ones regardless of how few blocks/CU that leaves. gate/up-proj
// (N=14336,K=4096) was already best at WARPS=32/ILP=4 in both the isolated
// sweep and the whole-model tune. Opt-in via GGML_HIP_2OF4_FP8_V3_ADAPTIVE
// (default off -- the measured whole-model win is small, see T162 writeup)
// so it can be A/B'd against the flat WARPS=32 default without disturbing it.
static int ggml_cuda_2of4_fp8_v3_warps_for_shape(int64_t N, int64_t K) {
    if (getenv("GGML_HIP_2OF4_FP8_V3_ADAPTIVE") == nullptr) {
        return ggml_cuda_2of4_fp8_v2_warps();
    }
    // BUG FIX (measured): keying on K alone also caught gate/up-proj
    // (N=14336, K=4096) into the WARPS=4 bucket, where the isolated sweep
    // showed it's 32% SLOWER (5.0810ms vs 3.4564ms) -- that single
    // misclassification collapsed whole-model pp512 from 2192 to 1230
    // (-44%) despite the intended q/o-proj fix being a real, if small,
    // per-shape win. Must key on BOTH N and K: only the specific
    // short-N/short-K shape (q/o-proj, N<=4096 && K<=4096) gets WARPS=4;
    // every other shape (gate/up's long-N, down-proj's long-K) stays on
    // WARPS=32, matching what the isolated sweep actually measured.
    return (N <= 4096 && K <= 4096) ? 4 : 32;
}

// T162 follow-up: isolated per-shape occupancy sweep (coordinator directive
// "fix the N=4096 occupancy starvation"). Constructs synthetic device
// tensors for a given (M,N,K) using the REAL host quantizer
// (quantize_row_2of4_fp8_ref, the exact function llama-quantize calls) for
// the sparse weight, and the production k_quantize_act_f8e4m3 device kernel
// for the activation -- so this measures the SAME kernel, on the SAME data
// format, as the real model, just isolated to one GEMM shape at a time
// (mirrors run_ceiling_bench's pattern in mul_mat_iu4_mmq.cu). Sweeps
// WARPS in {4,8,16,32} x ILP in {2,4,8} so the WARPSxILP peak is re-found
// PER SHAPE, not assumed to be the whole-model-tuned (32,4) pair.
static double run_2of4_fp8_shape_bench(int64_t M, int64_t N, int64_t K, int n_warps, int ilp) {
    std::mt19937 rng(162162);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int64_t n_blocks_k = K / QK_2OF4_FP8;

    std::vector<float> w_f32((size_t) (N * K));
    for (auto & v : w_f32) { v = dist(rng); }
    std::vector<block_2of4_fp8> w_blocks((size_t) (N * n_blocks_k));
    for (int64_t n = 0; n < N; ++n) {
        quantize_row_2of4_fp8_ref(w_f32.data() + n * K, w_blocks.data() + n * n_blocks_k, K);
    }

    std::vector<float> act_f32((size_t) (M * K));
    for (auto & v : act_f32) { v = dist(rng); }

    void * d_w = nullptr;
    float * d_act_f32 = nullptr;
    block_f8e4m3 * d_act_q = nullptr;
    float * d_dst = nullptr;
    if (hipMalloc(&d_w, w_blocks.size() * sizeof(block_2of4_fp8)) != hipSuccess ||
        hipMalloc(&d_act_f32, act_f32.size() * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_act_q, (size_t) (M * n_blocks_k) * sizeof(block_f8e4m3)) != hipSuccess ||
        hipMalloc(&d_dst, (size_t) (M * N) * sizeof(float)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed (M=%ld N=%ld K=%ld)\n", __func__, (long) M, (long) N, (long) K);
        return -1.0;
    }
    CUDA_CHECK(hipMemcpy(d_w, w_blocks.data(), w_blocks.size() * sizeof(block_2of4_fp8), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_act_f32, act_f32.data(), act_f32.size() * sizeof(float), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(d_dst, 0, (size_t) (M * N) * sizeof(float)));

    // T162 activation-quant fusion check (coordinator directive): measure
    // the unfused quantize pre-pass cost in isolation (same
    // k_quantize_act_f8e4m3 kernel every dispatch path -- 2:4-V3, dense-V3,
    // AND production mmq.cu's quantize_mmq_f8e4m3_cuda -- pays; confirmed
    // by code inspection, mmq.cu:191 dispatches it as its own separate
    // kernel too, so this is a SYMMETRIC fixed cost, not an asymmetry that
    // explains any of the kernel-vs-kernel deltas above).
    {
        const dim3 grid(n_blocks_k, M, 1);
        const dim3 block(32, 1, 1);
        hipEvent_t qev0, qev1;
        hipEventCreate(&qev0);
        hipEventCreate(&qev1);
        for (int r = 0; r < 3; ++r) { k_quantize_act_f8e4m3<<<grid, block>>>(d_act_f32, d_act_q, n_blocks_k, K); }
        CUDA_CHECK(hipDeviceSynchronize());
        hipEventRecord(qev0);
        for (int r = 0; r < 10; ++r) { k_quantize_act_f8e4m3<<<grid, block>>>(d_act_f32, d_act_q, n_blocks_k, K); }
        hipEventRecord(qev1);
        CUDA_CHECK(hipDeviceSynchronize());
        float q_ms = 0.0f;
        hipEventElapsedTime(&q_ms, qev0, qev1);
        hipEventDestroy(qev0);
        hipEventDestroy(qev1);
        if (n_warps == 32 && ilp == 4) { // log once per shape, not once per WARPS/ILP combo
            GGML_LOG_INFO("%s:   [quant-prepass M=%ld K=%ld] %.4f ms/call (unfused, separate kernel launch -- same structural cost paid by production mmq.cu:191 quantize_mmq_f8e4m3_cuda)\n",
                          __func__, (long) M, (long) K, (double) q_ms / 10.0);
        }
    }

    const int64_t nb01 = n_blocks_k * (int64_t) sizeof(block_2of4_fp8);
    const dim3 block(32, n_warps, 1);
    const int64_t bn = (int64_t) n_warps * ilp * 16;
    const dim3 grid((N + bn - 1) / bn, (M + 15) / 16, 1);
    const bool exact = (M % 16 == 0) && (N % bn == 0);

    auto launch = [&] () {
        switch (ilp) {
            case 2:
                if (exact) k_mul_mat_2of4_fp8_v3<2, false><<<grid, block>>>((const char *) d_w, d_act_q, d_dst, M, N, nb01, n_blocks_k, N);
                else       k_mul_mat_2of4_fp8_v3<2, true> <<<grid, block>>>((const char *) d_w, d_act_q, d_dst, M, N, nb01, n_blocks_k, N);
                break;
            case 8:
                if (exact) k_mul_mat_2of4_fp8_v3<8, false><<<grid, block>>>((const char *) d_w, d_act_q, d_dst, M, N, nb01, n_blocks_k, N);
                else       k_mul_mat_2of4_fp8_v3<8, true> <<<grid, block>>>((const char *) d_w, d_act_q, d_dst, M, N, nb01, n_blocks_k, N);
                break;
            default:
                if (exact) k_mul_mat_2of4_fp8_v3<4, false><<<grid, block>>>((const char *) d_w, d_act_q, d_dst, M, N, nb01, n_blocks_k, N);
                else       k_mul_mat_2of4_fp8_v3<4, true> <<<grid, block>>>((const char *) d_w, d_act_q, d_dst, M, N, nb01, n_blocks_k, N);
                break;
        }
    };

    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);

    for (int r = 0; r < 3; ++r) { launch(); } // warmup
    CUDA_CHECK(hipDeviceSynchronize());

    const int n_reps = 10;
    hipEventRecord(ev0);
    for (int r = 0; r < n_reps; ++r) { launch(); }
    hipEventRecord(ev1);
    CUDA_CHECK(hipDeviceSynchronize());

    float t_ms = 0.0f;
    hipEventElapsedTime(&t_ms, ev0, ev1);
    hipEventDestroy(ev0);
    hipEventDestroy(ev1);
    (void) hipFree(d_w);
    (void) hipFree(d_act_f32);
    (void) hipFree(d_act_q);
    (void) hipFree(d_dst);

    return (double) t_ms / n_reps; // ms/call
}

// Env-gated driver: GGML_HIP_2OF4_FP8_SHAPE_BENCH=1 -> sweeps the two
// dominant Llama-3.1-8B linear-layer shapes (N=4096/K=4096 [q/o-proj, the
// occupancy-starved case: grid=64=1 block/CU at BN=2048] and N=14336/K=4096
// [gate/up-proj, ~3.5 blocks/CU already]; also N=4096/K=14336 for
// down-proj, the OTHER N=4096 shape) x WARPS in {4,8,16,32} x ILP in
// {2,4,8}, logging ms/call + an implied "pp-equivalent" M/t_s throughput
// for each combo so the WARPSxILP peak can be re-found per shape instead of
// assumed from the whole-model tune.
bool ggml_cuda_mul_mat_2of4_fp8_shape_bench() {
    struct Shape { const char * name; int64_t M, N, K; };
    const Shape shapes[] = {
        { "N=4096/K=4096 (q/o-proj, 1 blk/CU @BN2048)",  512, 4096,  4096 },
        { "N=4096/K=14336 (down-proj, 1 blk/CU @BN2048)", 512, 4096, 14336 },
        { "N=14336/K=4096 (gate/up-proj, ~3.5 blk/CU)",   512, 14336, 4096 },
    };
    const int warps_sweep[] = { 4, 8, 16, 32 };
    const int ilp_sweep[]   = { 2, 4, 8 };
    const int cu_count = 64; // R9700 gfx1201, rocminfo-confirmed

    bool ok = true;
    for (const Shape & s : shapes) {
        double best_ms = 1e300;
        int    best_w = 0, best_i = 0;
        GGML_LOG_INFO("%s: --- shape %s (M=%ld N=%ld K=%ld) ---\n", __func__, s.name, (long) s.M, (long) s.N, (long) s.K);
        for (int w : warps_sweep) {
            for (int i : ilp_sweep) {
                const int64_t bn = (int64_t) w * i * 16;
                const int64_t grid_x = (s.N + bn - 1) / bn;
                const int64_t grid_y = (s.M + 15) / 16;
                const int64_t nblocks = grid_x * grid_y;
                const double  blocks_per_cu = (double) nblocks / cu_count;
                const double  ms = run_2of4_fp8_shape_bench(s.M, s.N, s.K, w, i);
                if (ms < 0.0) { ok = false; continue; }
                const double pp_equiv = ms > 0.0 ? (double) s.M / (ms / 1000.0) : 0.0;
                if (ms < best_ms) { best_ms = ms; best_w = w; best_i = i; }
                GGML_LOG_INFO("%s:   WARPS=%2d ILP=%d BN=%4ld blocks=%3ld (%.2f/CU) -> %.4f ms/call, pp-equiv=%.1f t/s\n",
                              __func__, w, i, (long) bn, (long) nblocks, blocks_per_cu, ms, pp_equiv);
            }
        }
        const double best_pp = best_ms > 0.0 ? (double) s.M / (best_ms / 1000.0) : 0.0;
        GGML_LOG_INFO("%s: BEST for %s -> WARPS=%d ILP=%d, %.4f ms/call, pp-equiv=%.1f t/s\n",
                      __func__, s.name, best_w, best_i, best_ms, best_pp);
    }
    return ok;
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

    if (ggml_cuda_2of4_fp8_v3_enabled()) {
        const int64_t nb01 = src0->nb[1];
        const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
        const int n_warps = ggml_cuda_2of4_fp8_v3_warps_for_shape(N, K);
        const int ilp     = ggml_cuda_2of4_fp8_v2_ilp();
        const dim3 block(32, n_warps, 1);
        const int64_t bn = (int64_t) n_warps * ilp * 16;
        const dim3 grid((N + bn - 1) / bn, (M + 15) / 16, 1);
        // MEASURED FALSIFICATION (T162): tried a need_check=false fast path
        // (grid exactly tiles M/N for every real Llama-3.1-8B linear layer,
        // 4096/14336 both % 2048 == 0) expecting the dropped bounds branches
        // to win. Reproducibly the OPPOSITE: need_check=false measured
        // 1483-1489 t/s vs need_check=true's 2170-2213 t/s at the SAME
        // WARPS=32/ILP=4 shape -- the branchy version is ~46% FASTER, not
        // slower (likely a compiler scheduling/register-allocation artifact
        // of removing the check, not the branch itself; not chased further).
        // need_check=true stays the always-correct AND measured-faster
        // default -- GGML_HIP_2OF4_FP8_V3_FORCE_CHECK is now a no-op kept
        // only so the exact-tiling instantiation below stays reachable for
        // any future re-test.
        const bool exact = (M % 16 == 0) && (N % bn == 0) && (getenv("GGML_HIP_2OF4_FP8_V3_FORCE_EXACT") != nullptr);
        switch (ilp) {
            case 1:
                k_mul_mat_2of4_fp8_v3<1, true><<<grid, block, 0, stream>>>(
                        (const char *) src0->data, act_q.get(), (float *) dst->data,
                        M, N, nb01, n_blocks_k, dst_row_stride_floats);
                break;
            case 2:
                k_mul_mat_2of4_fp8_v3<2, true><<<grid, block, 0, stream>>>(
                        (const char *) src0->data, act_q.get(), (float *) dst->data,
                        M, N, nb01, n_blocks_k, dst_row_stride_floats);
                break;
            case 3:
                k_mul_mat_2of4_fp8_v3<3, true><<<grid, block, 0, stream>>>(
                        (const char *) src0->data, act_q.get(), (float *) dst->data,
                        M, N, nb01, n_blocks_k, dst_row_stride_floats);
                break;
            case 5:
                k_mul_mat_2of4_fp8_v3<5, true><<<grid, block, 0, stream>>>(
                        (const char *) src0->data, act_q.get(), (float *) dst->data,
                        M, N, nb01, n_blocks_k, dst_row_stride_floats);
                break;
            case 6:
                k_mul_mat_2of4_fp8_v3<6, true><<<grid, block, 0, stream>>>(
                        (const char *) src0->data, act_q.get(), (float *) dst->data,
                        M, N, nb01, n_blocks_k, dst_row_stride_floats);
                break;
            case 8:
                k_mul_mat_2of4_fp8_v3<8, true><<<grid, block, 0, stream>>>(
                        (const char *) src0->data, act_q.get(), (float *) dst->data,
                        M, N, nb01, n_blocks_k, dst_row_stride_floats);
                break;
            default:
                if (exact) {
                    k_mul_mat_2of4_fp8_v3<4, false><<<grid, block, 0, stream>>>(
                            (const char *) src0->data, act_q.get(), (float *) dst->data,
                            M, N, nb01, n_blocks_k, dst_row_stride_floats);
                } else {
                    k_mul_mat_2of4_fp8_v3<4, true><<<grid, block, 0, stream>>>(
                            (const char *) src0->data, act_q.get(), (float *) dst->data,
                            M, N, nb01, n_blocks_k, dst_row_stride_floats);
                }
                break;
        }
        return true;
    }

    if (ggml_cuda_2of4_fp8_v2_enabled()) {
        const int64_t nb01 = src0->nb[1];
        const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
        const int n_warps = ggml_cuda_2of4_fp8_v2_warps();
        const int ilp     = ggml_cuda_2of4_fp8_v2_ilp();
        const dim3 block(32, n_warps, 1);
        const int64_t bn = (int64_t) n_warps * ilp * 16;
        const dim3 grid((N + bn - 1) / bn, (M + 15) / 16, 1);
        switch (ilp) {
            case 1:
                k_mul_mat_2of4_fp8_v2<1><<<grid, block, 0, stream>>>(
                        (const char *) src0->data, act_q.get(), (float *) dst->data,
                        M, N, nb01, n_blocks_k, dst_row_stride_floats);
                break;
            case 2:
                k_mul_mat_2of4_fp8_v2<2><<<grid, block, 0, stream>>>(
                        (const char *) src0->data, act_q.get(), (float *) dst->data,
                        M, N, nb01, n_blocks_k, dst_row_stride_floats);
                break;
            case 8:
                k_mul_mat_2of4_fp8_v2<8><<<grid, block, 0, stream>>>(
                        (const char *) src0->data, act_q.get(), (float *) dst->data,
                        M, N, nb01, n_blocks_k, dst_row_stride_floats);
                break;
            default:
                k_mul_mat_2of4_fp8_v2<4><<<grid, block, 0, stream>>>(
                        (const char *) src0->data, act_q.get(), (float *) dst->data,
                        M, N, nb01, n_blocks_k, dst_row_stride_floats);
                break;
        }
        return true;
    }

    {
        const int64_t nb01 = src0->nb[1];
        const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
        const dim3 block(32, 1, 1);
        const int ilp = ggml_cuda_2of4_fp8_ilp();
        switch (ilp) {
            case 2: {
                const dim3 grid((N + 31) / 32, (M + 15) / 16, 1);
                k_mul_mat_2of4_fp8_ilp<2><<<grid, block, 0, stream>>>(
                        (const char *) src0->data, act_q.get(), (float *) dst->data,
                        M, N, nb01, n_blocks_k, dst_row_stride_floats);
                break;
            }
            case 4: {
                const dim3 grid((N + 63) / 64, (M + 15) / 16, 1);
                k_mul_mat_2of4_fp8_ilp<4><<<grid, block, 0, stream>>>(
                        (const char *) src0->data, act_q.get(), (float *) dst->data,
                        M, N, nb01, n_blocks_k, dst_row_stride_floats);
                break;
            }
            case 8: {
                const dim3 grid((N + 127) / 128, (M + 15) / 16, 1);
                k_mul_mat_2of4_fp8_ilp<8><<<grid, block, 0, stream>>>(
                        (const char *) src0->data, act_q.get(), (float *) dst->data,
                        M, N, nb01, n_blocks_k, dst_row_stride_floats);
                break;
            }
            default: {
                const dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
                k_mul_mat_2of4_fp8<<<grid, block, 0, stream>>>(
                        (const char *) src0->data, act_q.get(), (float *) dst->data,
                        M, N, nb01, n_blocks_k, dst_row_stride_floats);
                break;
            }
        }
    }

    return true;
}
