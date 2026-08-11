// T162 int4-2:4 pivot (coordinator directive, "the real prize"): isolated
// correctness+throughput probe for a library-grade (MMQ-cooperative-tile,
// double-buffered LDS) 2:4-sparse int4 GEMM, reusing the mul_mat_2of4_fp8_mmq.cu
// tiling machinery verbatim and swapping only the instruction + operand
// widths (int4 packs 2x denser than fp8/int8, so A/B operand int-counts halve).
//
// STANDALONE / NOT WIRED INTO GGML DISPATCH: there is no int4-2:4 ggml_type
// or GGUF format in this codebase (verified by search -- see T162 writeup,
// Step 0). Building one (new block struct registered in ggml.c/ggml.h,
// host quantizer, llama-quantize wiring, re-quantizing an 8B model) is out
// of scope for this pass; this file answers the narrower, still-decisive
// question -- "does the ISA-level int4-2:4 throughput ceiling (1345-1551
// TOP/s microbench) survive in a real, library-grade TILED GEMM" -- via a
// synthetic-data isolated kernel probe at the SAME (M,N,K) shapes used
// throughout the rest of the T162 arc, with a CPU-reference correctness
// check on every run (not just the tiny 16x16 self-test).
//
// CORRECTNESS-CRITICAL NOTE (read before trusting any number from this
// file): a CK-library-derived "idx swap + XOR-1" fix for the iu4 sparse
// SWMMAC metadata encoding (~/int4-research/FINDINGS.md Stage 17d) was
// TRIED against this driver's OWN (already-shipped, swmmac24.cuh) per-lane
// addressing convention and MEASURED WRONG (max_abs_err=385, see
// swmmac24_iu4_fixed.cu) -- re-running the EXISTING, unmodified
// swmmac24.cu::test_iu4_swmmac_24 self-test confirms it already passes
// (max_abs_err=0) with the NATURAL (unswapped, no-XOR) encoding. Conclusion:
// the CK bug was specific to CK's own internal register-packing convention
// (pk_int4_t / compress_a_impl), not a hardware truth that transfers to any
// encoder -- this driver's own from-scratch-derived encoding was already
// correct. This file therefore uses the SAME natural idx encoding as
// block_2of4_fp8/mul_mat_2of4_fp8_mmq.cu (meta[2*k_half] |
// (meta[2*k_half+1]<<8), no swap, no XOR), K32 SWMMAC only (the only form
// this driver has ever hardware-validated -- K64 was never independently
// derived/tested here, only via CK's now-known-different convention, so it
// is NOT used).
#include "common.cuh"
#include "ggml-impl.h" // ggml_fp32_to_fp16/ggml_fp16_to_fp32 (host)

#include <cstring>
#include <random>
#include <vector>

#if defined(GGML_USE_HIP)

// Local-only 2:4-sparse int4 block, QK=32 (8 groups of 4, keep 2/group =
// 16 kept int4 values). Layout derived directly from the validated
// swmmac24.cuh a_loc(4,...)/b32_loc_4bit(...) per-lane addressing (see file
// header): qs[g] (g=0..7, one byte/group) holds {low nibble = lo-survivor's
// value, high nibble = hi-survivor's value}; meta mirrors block_2of4_fp8's
// exact packing (meta[2*k_half]|(meta[2*k_half+1]<<8), 4 bits/group: 2 bits
// lo-position + 2 bits hi-position).
#define QK_2OF4_IU4 32
struct block_2of4_iu4 {
    ggml_half d;
    uint8_t   qs[QK_2OF4_IU4 / 8];   // 8 bytes: 1 byte/group, low nibble=lo val, high nibble=hi val
    uint8_t   meta[QK_2OF4_IU4 / 8]; // 4 bytes: packed 2-bit-per-index sparsity metadata
};
static_assert(sizeof(block_2of4_iu4) == sizeof(ggml_half) + QK_2OF4_IU4/8 + QK_2OF4_IU4/8, "bad block_2of4_iu4 size");

// Plain dense int4 activation block, QK=32, 2 values/byte (matches block_iu4's
// existing packing convention: low nibble = even index, high nibble = odd).
#define QK_ACT_IU4 32
struct block_act_iu4 {
    ggml_half d;
    uint8_t   qs[QK_ACT_IU4 / 2]; // 16 bytes
};

typedef int   v1i_p __attribute__((ext_vector_type(1)));
typedef int   v2i_p __attribute__((ext_vector_type(2)));
typedef int   v8i_p __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ int32_t pack4_p(const uint8_t * p) {
    int32_t w;
    memcpy(&w, p, 4);
    return w;
}
static __device__ __forceinline__ int32_t pack2_p(const uint8_t * p) {
    int32_t w = 0;
    memcpy(&w, p, 2);
    return w;
}

// ============================================================================
// MMQ-grade cooperative-tile 2:4-sparse int4 GEMM. Byte-for-byte the SAME
// shape as k_mul_mat_2of4_fp8_mmq (mul_mat_2of4_fp8_mmq.cu) -- BM x BN tile,
// NWARPS warps, NTX register-blocked tiles/warp, double-buffered LDS, ONE
// __syncthreads()/chunk -- only the operand widths (halved: int4 packs 2x
// denser) and the compute instruction (v_swmmac_i32_16x16x32_iu4, int32
// accumulate) differ.
// ============================================================================
template <int BM, int BN, int NWARPS, bool need_check>
__launch_bounds__(NWARPS * 32, 1)
static __global__ void k_mul_mat_2of4_iu4_mmq(
        const block_2of4_iu4 * __restrict__ weight, const block_act_iu4 * __restrict__ act, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t n_blocks_k, const int64_t dst_row_stride_floats) {
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

    // Double-buffered LDS staging: A (weight, compressed 2:4) needs only 2
    // ints/row (8 bytes, half of fp8's 4-int form -- int4 packs 2x denser);
    // B (activation, dense) needs 4 ints/row (16 bytes, half of fp8's 8-int
    // form, same reason).
    __shared__ int      sh_actq2[2][BM][4];
    __shared__ float    sh_da2[2][BM];
    __shared__ int      sh_wq2[2][BN][2];
    __shared__ unsigned sh_wmeta2[2][BN];
    __shared__ float    sh_dw2[2][BN];

    auto load_chunk = [&] (int64_t c, int buf) {
        if (warp_id_u < WARPS_A) {
            const int     row = warp_id_u * 32 + lane;
            const int64_t m   = m0 + row;
            if (!need_check || m < M) {
                const block_act_iu4 & blk = act[m * n_blocks_k + c];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sh_actq2[buf][row][i] = pack4_p(blk.qs + 4 * i);
                }
                sh_da2[buf][row] = __half2float(blk.d);
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) { sh_actq2[buf][row][i] = 0; }
                sh_da2[buf][row] = 0.0f;
            }
        } else if (warp_id_u < WARPS_A + WARPS_B) {
            const int     row = (warp_id_u - WARPS_A) * 32 + lane;
            const int64_t n   = n0 + row;
            if (!need_check || n < N) {
                const block_2of4_iu4 & blk = weight[n * n_blocks_k + c];
                sh_wq2[buf][row][0] = pack4_p(blk.qs + 0);
                sh_wq2[buf][row][1] = pack4_p(blk.qs + 4);
                sh_wmeta2[buf][row] = (unsigned) blk.meta[0] | ((unsigned) blk.meta[1] << 8) |
                                       ((unsigned) blk.meta[2] << 16) | ((unsigned) blk.meta[3] << 24);
                sh_dw2[buf][row] = __half2float(blk.d);
            } else {
                sh_wq2[buf][row][0] = sh_wq2[buf][row][1] = 0;
                sh_wmeta2[buf][row] = 0;
                sh_dw2[buf][row]    = 0.0f;
            }
        }
    };

    float acc[NTX][8];
#pragma unroll
    for (int s = 0; s < NTX; ++s) {
#pragma unroll
        for (int l = 0; l < 8; ++l) { acc[s][l] = 0.0f; }
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

            // A: 1 int/lane (k_half selects which of the 2 staged ints).
            const int      a_arg = sh_wq2[cur][w_row][k_half];
            const unsigned idxv  = (sh_wmeta2[cur][w_row] >> (k_half * 16)) & 0xFFFFu;
            // B: v2i/lane (k_half selects which pair of the 4 staged ints).
            const v2i_p b_arg = { sh_actq2[cur][a_col][k_half*2 + 0], sh_actq2[cur][a_col][k_half*2 + 1] };

            v8i_p c0v = {0, 0, 0, 0, 0, 0, 0, 0};
            const v8i_p raw = __builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32(
                    /*A sign*/ 1, a_arg, /*B sign*/ 1, b_arg, c0v, idxv, /*clamp*/ 0);

#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float dw_row = sh_dw2[cur][ni * 16 + out_row_base + l];
                const float da_col = sh_da2[cur][a_col];
                acc[s][l] += (float) raw[l] * dw_row * da_col;
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
    GGML_UNUSED(weight); GGML_UNUSED(act); GGML_UNUSED(dst);
    GGML_UNUSED(M); GGML_UNUSED(N); GGML_UNUSED(n_blocks_k); GGML_UNUSED(dst_row_stride_floats);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

// ============================================================================
// Host-side: synthetic 2:4-structured data generation + quantization + CPU
// reference + hipEvent-timed launch. Mirrors run_2of4_fp8_shape_bench's
// exact structure (mul_mat_2of4_fp8_mmq.cu).
// ============================================================================
static void quantize_2of4_iu4_row(const float * x, block_2of4_iu4 * y, int64_t k, std::mt19937 & rng) {
    const int nb = (int) (k / QK_2OF4_IU4);
    for (int i = 0; i < nb; ++i) {
        for (int g = 0; g < 8; ++g) {
            const float * grp = x + i*QK_2OF4_IU4 + g*4;
            int best0 = -1, best1 = -1; float best0v = -1.0f, best1v = -1.0f;
            for (int j = 0; j < 4; ++j) {
                const float av = fabsf(grp[j]);
                if (av > best0v) { best1v = best0v; best1 = best0; best0v = av; best0 = j; }
                else if (av > best1v) { best1v = av; best1 = j; }
            }
            if (best0 < 0) { best0 = 0; }
            if (best1 < 0) { best1 = (best0 == 0) ? 1 : 0; }
            const int lo = best0 < best1 ? best0 : best1;
            const int hi = best0 < best1 ? best1 : best0;
            const float vlo = grp[lo], vhi = grp[hi];
            float amax = fmaxf(fabsf(vlo), fabsf(vhi));
            for (int gg = 0; gg < 8; ++gg) {
                if (gg == g) continue; // only this group's amax matters for THIS group's own scale below (per-block scale uses whole block though)
            }
            (void) amax;
            const uint8_t nib = (uint8_t) ((lo & 0x3) | ((hi & 0x3) << 2));
            if ((g & 1) == 0) { y[i].meta[g/2] = nib; } else { y[i].meta[g/2] = (uint8_t)(y[i].meta[g/2] | (nib << 4)); }
        }
        // per-block scale over all 16 kept values
        float amax = 0.0f;
        float kept[16]; int kept_n = 0;
        for (int g = 0; g < 8; ++g) {
            const int lo = (y[i].meta[g/2] >> ((g&1)*4)) & 0x3;
            const int hi = (y[i].meta[g/2] >> ((g&1)*4 + 2)) & 0x3;
            const float * grp = x + i*QK_2OF4_IU4 + g*4;
            kept[kept_n++] = grp[lo];
            kept[kept_n++] = grp[hi];
            amax = fmaxf(amax, fmaxf(fabsf(grp[lo]), fabsf(grp[hi])));
        }
        const float d = amax / 7.0f; // signed 4-bit range [-8,7], symmetric to 7
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = ggml_fp32_to_fp16(d);
        for (int g = 0; g < 8; ++g) {
            const int lo = (y[i].meta[g/2] >> ((g&1)*4)) & 0x3;
            const int hi = (y[i].meta[g/2] >> ((g&1)*4 + 2)) & 0x3;
            const float * grp = x + i*QK_2OF4_IU4 + g*4;
            int qlo = (int) rintf(grp[lo] * id); qlo = qlo < -8 ? -8 : (qlo > 7 ? 7 : qlo);
            int qhi = (int) rintf(grp[hi] * id); qhi = qhi < -8 ? -8 : (qhi > 7 ? 7 : qhi);
            y[i].qs[g] = (uint8_t) ((qlo & 0xF) | ((qhi & 0xF) << 4));
        }
    }
    (void) rng;
}

static void quantize_act_iu4_row(const float * x, block_act_iu4 * y, int64_t k) {
    const int nb = (int) (k / QK_ACT_IU4);
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK_ACT_IU4; ++j) { amax = fmaxf(amax, fabsf(x[i*QK_ACT_IU4+j])); }
        const float d = amax / 7.0f;
        const float id = d != 0.0f ? 1.0f/d : 0.0f;
        y[i].d = ggml_fp32_to_fp16(d);
        for (int j = 0; j < QK_ACT_IU4; j += 2) {
            int q0 = (int) rintf(x[i*QK_ACT_IU4+j+0]*id); q0 = q0<-8?-8:(q0>7?7:q0);
            int q1 = (int) rintf(x[i*QK_ACT_IU4+j+1]*id); q1 = q1<-8?-8:(q1>7?7:q1);
            y[i].qs[j/2] = (uint8_t) ((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }
}

// Dequant helpers for the CPU reference.
static float dequant_iu4_nibble(uint8_t byte, int hi) {
    int v = hi ? (byte >> 4) : (byte & 0xF);
    if (v & 0x8) { v -= 16; } // sign-extend 4-bit
    return (float) v;
}

double run_2of4_iu4_shape_probe(int64_t M, int64_t N, int64_t K, int n_warps, int ilp_bm_bn /*unused, geometry fixed 128x128x8 for this probe*/, bool check_correctness, long * max_abs_err_out) {
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int64_t n_blocks_k = K / QK_2OF4_IU4;

    std::vector<float> w_f32((size_t)(N*K));
    for (auto & v : w_f32) v = dist(rng);
    std::vector<block_2of4_iu4> w_blocks((size_t)(N*n_blocks_k));
    for (int64_t n = 0; n < N; ++n) { quantize_2of4_iu4_row(w_f32.data() + n*K, w_blocks.data() + n*n_blocks_k, K, rng); }

    std::vector<float> act_f32((size_t)(M*K));
    for (auto & v : act_f32) v = dist(rng);
    std::vector<block_act_iu4> act_blocks((size_t)(M*n_blocks_k));
    for (int64_t m = 0; m < M; ++m) { quantize_act_iu4_row(act_f32.data() + m*K, act_blocks.data() + m*n_blocks_k, K); }

    block_2of4_iu4 * d_w = nullptr; block_act_iu4 * d_act = nullptr; float * d_dst = nullptr;
    if (hipMalloc(&d_w, w_blocks.size()*sizeof(block_2of4_iu4)) != hipSuccess ||
        hipMalloc(&d_act, act_blocks.size()*sizeof(block_act_iu4)) != hipSuccess ||
        hipMalloc(&d_dst, (size_t)(M*N)*sizeof(float)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return -1.0;
    }
    CUDA_CHECK(hipMemcpy(d_w, w_blocks.data(), w_blocks.size()*sizeof(block_2of4_iu4), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_act, act_blocks.data(), act_blocks.size()*sizeof(block_act_iu4), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(d_dst, 0, (size_t)(M*N)*sizeof(float)));

    constexpr int BM = 128, BN = 128, NWARPS = 8;
    const dim3 block(32, NWARPS, 1);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
    const bool need_check = (M % BM != 0) || (N % BN != 0);
    (void) n_warps; (void) ilp_bm_bn;

    auto launch = [&] () {
        if (need_check) {
            k_mul_mat_2of4_iu4_mmq<BM, BN, NWARPS, true><<<grid, block>>>(d_w, d_act, d_dst, M, N, n_blocks_k, N);
        } else {
            k_mul_mat_2of4_iu4_mmq<BM, BN, NWARPS, false><<<grid, block>>>(d_w, d_act, d_dst, M, N, n_blocks_k, N);
        }
    };

    for (int r = 0; r < 3; ++r) { launch(); }
    CUDA_CHECK(hipDeviceSynchronize());

    if (check_correctness) {
        std::vector<float> dst_host((size_t)(M*N));
        launch();
        CUDA_CHECK(hipDeviceSynchronize());
        CUDA_CHECK(hipMemcpy(dst_host.data(), d_dst, (size_t)(M*N)*sizeof(float), hipMemcpyDeviceToHost));

        long max_abs = 0;
        std::uniform_int_distribution<int64_t> pickm(0, M-1), pickn(0, N-1);
        for (int t = 0; t < 200; ++t) {
            const int64_t m = pickm(rng), n = pickn(rng);
            double ref = 0.0;
            for (int64_t c = 0; c < n_blocks_k; ++c) {
                const block_2of4_iu4 & wb = w_blocks[n*n_blocks_k + c];
                const block_act_iu4 & ab = act_blocks[m*n_blocks_k + c];
                const float dw = ggml_fp16_to_fp32(wb.d);
                const float da = ggml_fp16_to_fp32(ab.d);
                for (int g = 0; g < 8; ++g) {
                    const int lo = (wb.meta[g/2] >> ((g&1)*4)) & 0x3;
                    const int hi = (wb.meta[g/2] >> ((g&1)*4 + 2)) & 0x3;
                    const float wvlo = dequant_iu4_nibble(wb.qs[g], 0) * dw;
                    const float wvhi = dequant_iu4_nibble(wb.qs[g], 1) * dw;
                    const int kbase = g*4;
                    const float avlo = dequant_iu4_nibble(ab.qs[(kbase+lo)/2], (kbase+lo)%2 != 0) * da;
                    const float avhi = dequant_iu4_nibble(ab.qs[(kbase+hi)/2], (kbase+hi)%2 != 0) * da;
                    ref += (double) wvlo * avlo + (double) wvhi * avhi;
                }
            }
            const float got = dst_host[m*N + n];
            const long err = (long) llabs((long long) llroundf(got - (float) ref));
            max_abs = std::max(max_abs, err);
        }
        if (max_abs_err_out) { *max_abs_err_out = max_abs; }
    }

    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0); hipEventCreate(&ev1);
    const int n_reps = 10;
    hipEventRecord(ev0);
    for (int r = 0; r < n_reps; ++r) { launch(); }
    hipEventRecord(ev1);
    CUDA_CHECK(hipDeviceSynchronize());
    float t_ms = 0.0f;
    hipEventElapsedTime(&t_ms, ev0, ev1);
    hipEventDestroy(ev0); hipEventDestroy(ev1);
    (void) hipFree(d_w); (void) hipFree(d_act); (void) hipFree(d_dst);
    return (double) t_ms / n_reps;
}

bool ggml_cuda_int4_24_probe() {
    struct Shape { const char * name; int64_t M, N, K; };
    const Shape shapes[] = {
        { "N=4096/K=4096 (q/o-proj)",      512, 4096,  4096 },
        { "N=4096/K=14336 (down-proj)",    512, 4096, 14336 },
        { "N=14336/K=4096 (gate/up-proj)", 512, 14336, 4096 },
    };
    bool ok = true;
    for (const Shape & s : shapes) {
        long max_abs_err = -1;
        const double ms = run_2of4_iu4_shape_probe(s.M, s.N, s.K, 0, 0, true, &max_abs_err);
        if (ms < 0.0) { ok = false; continue; }
        const double pp_equiv = (double) s.M / (ms / 1000.0);
        const double flops = 2.0 * s.M * s.N * s.K;
        const double top_s = flops / (ms / 1000.0) / 1e12;
        const bool correct = (max_abs_err <= 2); // int32 accumulate + fp32 rescale rounding tolerance
        if (!correct) { ok = false; }
        GGML_LOG_INFO("%s: %s -> %.4f ms/call, pp-equiv=%.1f t/s, %.2f TOP/s, max_abs_err(200 spot checks)=%ld -> %s\n",
                      __func__, s.name, ms, pp_equiv, top_s, max_abs_err, correct ? "CORRECT" : "WRONG");
    }
    return ok;
}

#else // !defined(GGML_USE_HIP)

bool ggml_cuda_int4_24_probe() { return true; }

#endif // defined(GGML_USE_HIP)
