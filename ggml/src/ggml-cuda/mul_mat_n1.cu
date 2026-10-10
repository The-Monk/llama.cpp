// N1 native int8 WMMA prefill GEMM (T404 harness, default ON since T412).
//
// Routes Q2_0 (g128) / Q1_0 prefill MUL_MATs (src1 batch N >= GGML_N1_MIN_N, default 64) to the native int8 WMMA
// GEMM N1 (n1_gemm_iu8.cuh = native-kernels/dense_wmma/gemm_iu8.cuh + engine-only epilogue/scale additions).
// AMD RDNA4 only. Each routed weight gets a 2-bit + fp16-scale copy (converted once, cached): VRAM cost ~0.9x
// the Q2_0/Q1_0 weight bytes on top of the model.
//
// Env:
//   GGML_N1_PREFILL=0      disable (unset / anything else = on)
//   GGML_N1_STATS=1        print coverage stats at exit
//   GGML_N1_ACT=g128|token activation scale granularity for the g128-weight arm (default g128):
//                            g128  = per-(128-K block, token) scale, numerically the same class as the default
//                                    MMQ per-128 hoist path (kernel FL bit3, +1 mul per element per segment)
//                            token = per-token scale = the kernel exactly as measured standalone (results.md job 236)
//   GGML_N1_ROW=ffn        ARM B: ffn_* tensors use ROW mode (M0 weight fold, int8 copy, rs0 kernel, per-token act);
//                          all other tensors stay on the g128 arm. Int8 copies are VRAM-guarded per tensor.
//   GGML_N1_MIN_N=<n>      minimum src1 batch (default 64)
//   GGML_N1_DUMP=<dir>     T400 gate: write every runtime-converted g128 weight (W2 stream, fp16 sw) to <dir>
//   GGML_N1_TRANSIENT=1    T400 single-copy study: re-pack W2 + sw into pool scratch on every call (no cache)
//   GGML_N1_M1=0           T420: the g128 arm uses the release kernel (gemm_iu8 FL=9) instead of gemm_xd (X direct-to-VGPR,
//                          bias-initialised accumulators; bit-identical output, +13% standalone)
//   GGML_N1_M2=1           T422 (default OFF, DIFFERENT MATH, KLD-gated): for K % 256 == 0 the g128 arm runs CONTRACT
//                          mode ROW weights (M0 fold: W' = code * m[g][f], row scale S_f; m/S_f derived on device from the
//                          W2 stream + fp16 sw, for cached and native weights alike) x per-256 activation scales (T402
//                          g256) on gemm_fold<8,1,1,2> (n1_m2_fold.cuh: 256 tok x 64 feat, rescale every 256 K with one
//                          pairable sub + fmac). The activation is quantized here (no act-fuse producer for g256 yet).
//   GGML_N1_M2=2           T424 (default OFF, DIFFERENT MATH again, KLD-gated): the M2 path with the bias carried through the
//                          rescale fma (one fmac per element per 256 K instead of sub + fmac; BIASF * sx removed every 8
//                          groups and in the epilogue). Rounding noise ~5e-4 of the output rms at K = 17408 (standalone).
//
// T400 N4: weights loaded from a native dual-blob GGUF (llama-model-loader, GGML_TYPE_NK_*_W2) arrive as a compact
// Q2_0/Q1_0 view whose view_src is the dual tensor; its W2 region and its companion's sw are used in place (no
// conversion, no cache copy).
//
// Weights (g128 arm): 2-bit packed stream (LAYOUT.md section 3, pack_tiles_w2 with NT=256) + fp16 sw[K/128][F]; config
// t128_f128_w4x2_rs1_mb1_w2 (verified bit-exact in job 235). Converted once on a side stream and cached per tensor.
// Unsupported shapes / memory refusals fall back to the default path and are counted; stats are printed at exit.

#include "mul_mat_n1.cuh"
#include "act-fuse.cuh"
#include "hipblaslt_wcache.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#if defined(GGML_USE_HIP)
#include "n1_gemm_iu8.cuh"
#include "n1_gemm_xd.cuh"
#include "n1_m2_fold.cuh"
#include "n1_act.cuh"

namespace {

constexpr int N1_BT = 128, N1_BF = 128, N1_NT = 256;

struct n1_env {
    bool    on      = false;
    bool    stats   = false;
    bool    act_tok = false;
    bool    row_ffn = false, row_all = false;
    int64_t min_n   = 64;
    const char * dump = nullptr;
    bool    transient = false;
    bool    m1 = true;
    bool    m2 = false;
    bool    m3 = false;     // GGML_N1_M3=1 (T434, default OFF, DIFFERENT MATH, KLD-gated): Hadamard-model activations (src1 = FWHT node)
                            // quantised per TOKEN, folded int8 weights, ONE int32 accumulation over the whole K (no in-loop rescale)
    int     m3_segs = 1;
    int     m3_feed = 0;       // GGML_N1_M3_FEED (T435): 1 = wave tile 64 tok x 64 feat (W LDS reads per WMMA halved), 2 = 32 tok x 128 feat (X loads per WMMA halved), 3 = tile 1 + s_setprio 2 around each k-block's WMMA burst (T440); all on a 256 x 128 block
    bool    m3_wide = false;   // GGML_N1_M3_WIDE=1: always the 256-token tile (A/B only)
    int     m2bc_flush = 8;
    bool    m2bc = false;   // GGML_N1_M2=2 (T424): M2 path + bias-carry rescale (one fmac per element per 256 K)
    n1_env() {
        const char * m1e = getenv("GGML_N1_M1");
        m1 = !(m1e && strcmp(m1e, "0") == 0);
        const char * m2e = getenv("GGML_N1_M2");
        m2 = m2e && (strcmp(m2e, "1") == 0 || strcmp(m2e, "2") == 0);
        m2bc = m2e && strcmp(m2e, "2") == 0;
        const char * m3e = getenv("GGML_N1_M3");
        m3 = !(m3e && strcmp(m3e, "0") == 0);   // default ON (GGML_N1_M3=0 disables); engages only when src1 is a prism.hadamard FWHT node
        m3_wide = getenv("GGML_N1_M3_WIDE") != nullptr;
        const char * m3s = getenv("GGML_N1_M3_SEGS");
        m3_segs = m3s && atoi(m3s) == 2 ? 2 : 1;
        const char * m3f = getenv("GGML_N1_M3_FEED");
        m3_feed = !m3f ? 3 : ((atoi(m3f) >= 1 && atoi(m3f) <= 3) ? atoi(m3f) : 0);   // T440: default 3 (64x64 wave tile + s_setprio around each k-block WMMA burst); 1 = T435 tile; GGML_N1_M3_FEED=0 disables
        const char * fe = getenv("GGML_N1_M2BC_FLUSH");
        m2bc_flush = fe ? atoi(fe) : 8;
        dump = getenv("GGML_N1_DUMP");
        const char * tr = getenv("GGML_N1_TRANSIENT");
        transient = tr && strcmp(tr, "1") == 0;
        const char * e = getenv("GGML_N1_PREFILL");
        on = !(e && strcmp(e, "0") == 0);
        if (on) {
            int dev = 0;
            on = cudaGetDevice(&dev) == cudaSuccess && GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[dev].cc);
        }
        const char * st = getenv("GGML_N1_STATS");
        stats = st && strcmp(st, "0") != 0;
        const char * a = getenv("GGML_N1_ACT");
        act_tok = a && strcmp(a, "token") == 0;
        const char * r = getenv("GGML_N1_ROW");
        row_ffn = r && strcmp(r, "ffn") == 0;
        row_all = r && strcmp(r, "all") == 0;   // test-only: ROW mode for every routed tensor (test-backend-ops)
        const char * m = getenv("GGML_N1_MIN_N");
        if (m) min_n = atoll(m);
        if (on) {
            fprintf(stderr, "[N1] Q2_0/Q1_0 prefill on native int8 WMMA (GGML_N1_PREFILL=0 disables): act=%s row=%s min_n=%lld dump=%s transient=%d g128_kernel=%s\n",
                    act_tok ? "token" : "g128", row_all ? "all" : row_ffn ? "ffn" : "none", (long long) min_n, dump ? dump : "-", (int) transient,
                    m2 ? "fold+g256(m2, K%256==0; else xd)" : m1 ? "xd(m1)" : "iu8(release)");
        }
    }
};
const n1_env & env() { static n1_env e; return e; }

// ---------------- stats ----------------
struct n1_stats {
    std::atomic<uint64_t> mm_total{0}, fl_total{0};        // all MUL_MATs with N >= min_n
    std::atomic<uint64_t> mm_q{0},     fl_q{0};            // ... of which Q2_0/Q1_0 weights
    std::atomic<uint64_t> mm_n1{0},    fl_n1{0};           // ... routed to N1 (g128 arm)
    std::atomic<uint64_t> mm_row{0},   fl_row{0};          // ... routed to N1 (ROW arm)
    std::atomic<uint64_t> fb_shape{0}, fl_fb_shape{0};     // fallback: unsupported shape / layout
    std::atomic<uint64_t> fb_mem{0},   fl_fb_mem{0};       // fallback: weight copy refused (VRAM)
    std::atomic<uint64_t> conv_w2{0}, conv_row{0};
    std::atomic<uint64_t> bytes_w2{0}, bytes_row{0};
    std::atomic<uint64_t> act_hit{0}, act_miss{0};          // N1 activation from the act-fuse cache (hit) / quantized here
    std::atomic<uint64_t> act_glu{0}, act_norm{0};          // ... hits written by the GLU / norm producers
    std::atomic<uint64_t> mm_nk{0}, mm_transient{0}, dumped{0};
    std::atomic<uint64_t> mm_m2{0}, fl_m2{0}, conv_fold{0};  // T422: g128-arm calls on the fold+g256 kernel
    std::atomic<uint64_t> mm_m3{0}, fl_m3{0}, m3_hit{0};      // T434: one-GEMM calls (m3_hit: activation from the FWHT producer)
    ~n1_stats() {
        if (!env().on || !env().stats) return;
        if (env().m3) {
            fprintf(stderr, "[N1] stats: m3 one-gemm %llu ops / %.3f TFLOP, activation from the fused FWHT producer %llu\n",
                    (unsigned long long) mm_m3.load(), fl_m3.load() * 1e-12, (unsigned long long) m3_hit.load());
        }
        if (env().m2) {
            fprintf(stderr, "[N1] stats: m2 fold+g256 %llu ops / %.3f TFLOP, fold side-tables %llu\n",
                    (unsigned long long) mm_m2.load(), fl_m2.load() * 1e-12, (unsigned long long) conv_fold.load());
        }
        fprintf(stderr, "[N1] stats: activation cache hit %llu (from glu producer %llu, norm producer %llu) miss %llu\n",
                (unsigned long long) act_hit.load(), (unsigned long long) act_glu.load(),
                (unsigned long long) act_norm.load(), (unsigned long long) act_miss.load());
        fprintf(stderr, "[N1] stats: native in-place weight calls %llu, transient re-pack calls %llu, dumped tensors %llu\n",
                (unsigned long long) mm_nk.load(), (unsigned long long) mm_transient.load(), (unsigned long long) dumped.load());
        auto pct = [](uint64_t a, uint64_t b) { return b ? 100.0 * (double) a / (double) b : 0.0; };
        fprintf(stderr,
                "[N1] stats: prefill MUL_MAT (N>=%lld) total %llu ops / %.3f TFLOP; Q2_0/Q1_0 %llu ops / %.3f TFLOP\n"
                "[N1] stats: N1 g128 %llu ops / %.3f TFLOP, N1 row %llu ops / %.3f TFLOP -> coverage %.2f%% of all prefill GEMM FLOPs\n"
                "[N1] stats: fallback shape %llu ops / %.3f TFLOP, fallback mem %llu ops / %.3f TFLOP\n"
                "[N1] stats: converted w2 %llu tensors / %.3f GB, row int8 %llu tensors / %.3f GB\n",
                (long long) env().min_n, (unsigned long long) mm_total.load(), fl_total.load() * 1e-12,
                (unsigned long long) mm_q.load(), fl_q.load() * 1e-12,
                (unsigned long long) mm_n1.load(), fl_n1.load() * 1e-12, (unsigned long long) mm_row.load(), fl_row.load() * 1e-12,
                pct(fl_n1.load() + fl_row.load(), fl_total.load()),
                (unsigned long long) fb_shape.load(), fl_fb_shape.load() * 1e-12, (unsigned long long) fb_mem.load(), fl_fb_mem.load() * 1e-12,
                (unsigned long long) conv_w2.load(), bytes_w2.load() * 1e-9, (unsigned long long) conv_row.load(), bytes_row.load() * 1e-9);
    }
};
n1_stats g_stats;

uint64_t mm_flops(const ggml_tensor * src0, const ggml_tensor * src1) {
    return 2ull * (uint64_t) src0->ne[0] * (uint64_t) src0->ne[1] * (uint64_t) src1->ne[1] * (uint64_t) src1->ne[2] * (uint64_t) src1->ne[3];
}

// ---------------- layout helpers (LAYOUT.md section 1; n1_off is in n1_act.cuh) ----------------

// code in {0..3} (value = code-1) of weight j (0..127) of a 128-weight block
template <int TYPE>
__device__ __forceinline__ uint32_t n1_code(const uint8_t * qs, int j) {
    if constexpr (TYPE == GGML_TYPE_Q2_0) {
        return (qs[j >> 2] >> (2 * (j & 3))) & 3u;
    } else {   // Q1_0: bit 1 -> +1 (code 2), bit 0 -> -1 (code 0)
        return ((qs[j >> 3] >> (j & 7)) & 1u) ? 2u : 0u;
    }
}
template <int TYPE> constexpr int n1_bs() { return TYPE == GGML_TYPE_Q2_0 ? (int) sizeof(block_q2_0) : (int) sizeof(block_q1_0); }

// 2-bit packed weight stream (LAYOUT.md section 3, == pack_tiles_w2<256>): one thread per u32 word.
template <int TYPE>
__global__ void k_n1_pack_w2(const char * __restrict__ w, size_t nb01, int F, int K, uint32_t * __restrict__ out) {
    const int S = K / 128;
    const size_t total = (size_t) (F / N1_BF) * S * 1024;
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int    wi = (int) (idx % 1024);
    const size_t ts = idx / 1024;
    const int s = (int) (ts % S), tr = (int) (ts / S);
    const int t = wi >> 2, d = wi & 3;
    const int q = t + d * N1_NT;   // c = 0 (PWL = 1)
    const int ri = q / 128, kb = (q / 32) % 4, lane = q % 32;
    const int row = tr * N1_BF + ri * 16 + (lane & 15);
    const int j0 = kb * 32 + 16 * (lane >> 4);
    const uint8_t * qs = (const uint8_t *) (w + (size_t) row * nb01 + (size_t) s * n1_bs<TYPE>() + 2);
    uint32_t p = 0;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        p |= n1_code<TYPE>(qs, j0 + e) << (8 * (e & 3) + 2 * (e >> 2));
    }
    out[idx] = p;
}

// sw[s][f] = d(block s of row f)
template <int TYPE>
__global__ void k_n1_pack_sw(const char * __restrict__ w, size_t nb01, int F, int K, __half * __restrict__ sw) {
    const int S = K / 128;
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t) S * F) return;
    const int s = (int) (idx / F), f = (int) (idx % F);
    __half d;
    memcpy(&d, w + (size_t) f * nb01 + (size_t) s * n1_bs<TYPE>(), sizeof(d));
    sw[idx] = d;
}

// ROW fold (M0, fold_twin.py): per row S_f = max_g|d|/Mmax (Mmax 63 if the row uses code 3, else 127), m = rint(d/S_f),
// |m| >= 1 keeping sign, m = 0 iff d == 0. One block (256) per row.
template <int TYPE>
__global__ void k_n1_row_fold(const char * __restrict__ w, size_t nb01, int K, float * __restrict__ rowS, int8_t * __restrict__ mbuf) {
    const int S = K / 128, f = blockIdx.x;
    const char * rw = w + (size_t) f * nb01;
    float amax = 0.f; int c3 = 0;
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        __half dh; memcpy(&dh, rw + (size_t) s * n1_bs<TYPE>(), sizeof(dh));
        amax = fmaxf(amax, fabsf(__half2float(dh)));
        if constexpr (TYPE == GGML_TYPE_Q2_0) {
            const uint8_t * qs = (const uint8_t *) (rw + (size_t) s * n1_bs<TYPE>() + 2);
            for (int b = 0; b < 32; ++b) { const uint32_t v = qs[b]; c3 |= ((v & (v >> 1)) & 0x55u) != 0; }
        }
    }
    __shared__ float sa[256]; __shared__ int sc[256];
    sa[threadIdx.x] = amax; sc[threadIdx.x] = c3;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) { sa[threadIdx.x] = fmaxf(sa[threadIdx.x], sa[threadIdx.x + o]); sc[threadIdx.x] |= sc[threadIdx.x + o]; }
        __syncthreads();
    }
    float Sf = sa[0] / (sc[0] ? 63.f : 127.f);
    if (Sf == 0.f) Sf = 1.f;
    if (threadIdx.x == 0) rowS[f] = Sf;
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        __half dh; memcpy(&dh, rw + (size_t) s * n1_bs<TYPE>(), sizeof(dh));
        const float d = __half2float(dh);
        float m = rintf(d / Sf);
        if (fabsf(m) < 1.f) m = d < 0.f ? -1.f : 1.f;
        if (d == 0.f) m = 0.f;
        mbuf[(size_t) f * S + s] = (int8_t) m;
    }
}

// int8 stream (LAYOUT.md section 1, == pack_tiles B=128): one thread per 16-byte chunk; value = (code-1)*m.
template <int TYPE>
__global__ void k_n1_pack_row(const char * __restrict__ w, size_t nb01, int F, int K, const int8_t * __restrict__ mbuf, uint4 * __restrict__ out) {
    const int S = K / 128;
    const size_t total = (size_t) F * K / 16;
    const size_t ci = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= total) return;
    const int lane = (int) (ci % 32), kb = (int) ((ci / 32) % 4), ri = (int) ((ci / 128) % 8);
    const size_t ts = ci / 1024;
    const int s = (int) (ts % S), tile = (int) (ts / S);
    const int row = tile * 128 + ri * 16 + (lane & 15);
    const int j0 = kb * 32 + 16 * (lane >> 4);
    const uint8_t * qs = (const uint8_t *) (w + (size_t) row * nb01 + (size_t) s * n1_bs<TYPE>() + 2);
    const int m = mbuf[(size_t) row * S + s];
    uint32_t v[4] = {0, 0, 0, 0};
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int x = ((int) n1_code<TYPE>(qs, j0 + e) - 1) * m;
        v[e >> 2] |= ((uint32_t) (uint8_t) (int8_t) x) << (8 * (e & 3));
    }
    out[ci] = make_uint4(v[0], v[1], v[2], v[3]);
}

// Activation quantisation into the X stream: one block per (group g, token t), group G = 128 or K.
// d = amax/127, q = roundf(x/d) (same rounding as quantize_q8_1). sx: G==K -> sx[t], else sx[g*Npad + t].
// Rows t >= N (tile padding) are written as zeros with d = 0.
__global__ void k_n1_quant_act(const char * __restrict__ src, size_t nb1, int N, int K, int G, int Npad,
                               int8_t * __restrict__ X, float * __restrict__ sx) {
    const int g = blockIdx.x, t = blockIdx.y, S = K / 128, U = G / 4;
    const float4 * row = (const float4 *) (src + (size_t) t * nb1) + (size_t) g * U;
    float amax = 0.f;
    if (t < N) {
        for (int u = threadIdx.x; u < U; u += blockDim.x) {
            const float4 v = row[u];
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
    __shared__ float red[32];
    if (blockDim.x > 32) {
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
        __syncthreads();
        amax = 0.f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) amax = fmaxf(amax, red[i]);
    }
    const float d = n1_scale(amax);
    if (threadIdx.x == 0) {
        if (G == K) sx[t] = d; else sx[(size_t) g * Npad + t] = d;
    }
    for (int u = threadIdx.x; u < U; u += blockDim.x) {
        uint32_t p = 0;
        if (t < N && d > 0.f) {
            p = n1_quant4(row[u], d);   // n1_act.cuh: shared with the act-fuse store (bit-identical bytes)
        }
        const int k = g * G + 4 * u;
        *(uint32_t *) (X + n1_off(t, k, S)) = p;
    }
}

// T422 M2: per-256 activation quantisation (T402 g256): one block of 64 threads per (256-K group g, token t); d =
// amax/127 over the 256 values, q = n1_quant4 (same rounding as k_n1_quant_act); the scale is written into BOTH
// 128-slots of the group (sx keeps the [K/128][Npad] layout; gemm_fold reads the group's first slot).
__global__ void k_n1_quant_act256(const char * __restrict__ src, size_t nb1, int N, int K, int Npad,
                                  int8_t * __restrict__ X, float * __restrict__ sx) {
    const int g = blockIdx.x, t = blockIdx.y, S = K / 128;
    const float4 * row = (const float4 *) (src + (size_t) t * nb1) + (size_t) g * 64;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (t < N) v = row[threadIdx.x];
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
    __shared__ float red[2];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = fmaxf(red[0], red[1]);
    const float d = n1_scale(amax);
    if (threadIdx.x == 0) {
        sx[(size_t) (2 * g) * Npad + t]     = d;
        sx[(size_t) (2 * g + 1) * Npad + t] = d;
    }
    const uint32_t p = (t < N && d > 0.f) ? n1_quant4(v, d) : 0u;
    *(uint32_t *) (X + n1_off(t, g * 256 + 4 * (int) threadIdx.x, S)) = p;
}

// T422 M2: ROW fold side-table from the W2 stream + fp16 sw (same rule as k_n1_row_fold: S_f = max_g|d| / (63 if the
// row uses code 3 else 127), m = rint(d / S_f), |m| >= 1 keeping sign, m = 0 iff d == 0). mq is [S][F] (gemm_fold's
// layout). One 256-thread block per row f; the row's 8 packed dwords per segment are read to detect code 3.
__global__ void k_n1_fold_sw(const uint32_t * __restrict__ w2, const __half * __restrict__ sw, int F, int K,
                             float * __restrict__ rowS, int8_t * __restrict__ mq) {
    const int S = K / 128, f = blockIdx.x;
    const int tile = f / 128, r = f % 128, ri = r / 16, l16 = r % 16;
    float amax = 0.f; int c3 = 0;
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        amax = fmaxf(amax, fabsf(__half2float(sw[(size_t) s * F + f])));
        const uint32_t * base = w2 + ((size_t) tile * S + s) * 1024;
#pragma unroll
        for (int kb = 0; kb < 4; ++kb)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int q = ri * 128 + kb * 32 + h * 16 + l16;
                const uint32_t p = base[(q % 256) * 4 + q / 256];
                c3 |= (p & (p >> 1) & 0x55555555u) != 0;
            }
    }
    __shared__ float sa[256]; __shared__ int sc[256];
    sa[threadIdx.x] = amax; sc[threadIdx.x] = c3;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) { sa[threadIdx.x] = fmaxf(sa[threadIdx.x], sa[threadIdx.x + o]); sc[threadIdx.x] |= sc[threadIdx.x + o]; }
        __syncthreads();
    }
    float Sf = sa[0] / (sc[0] ? 63.f : 127.f);
    if (Sf == 0.f) Sf = 1.f;
    if (threadIdx.x == 0) rowS[f] = Sf;
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        const float d = __half2float(sw[(size_t) s * F + f]);
        float m = rintf(d / Sf);
        if (fabsf(m) < 1.f) m = d < 0.f ? -1.f : 1.f;
        if (d == 0.f) m = 0.f;
        mq[(size_t) s * F + f] = (int8_t) m;
    }
}

// ---------------- weight cache ----------------
struct n1_w {
    int    mode  = 0;   // 1 = w2 g128, 2 = row int8
    void * w     = nullptr;
    void * sw    = nullptr;
    size_t bytes = 0;
};
std::mutex g_mtx;
std::unordered_map<const void *, n1_w> g_cache;
std::unordered_set<const void *> g_refused;   // conversion refused (VRAM): stays on the default route, not retried
size_t g_cache_bytes = 0;
// T422 M2: ROW fold side-tables (w = mq int8 [S][F], sw = rowS fp32 [F]), keyed like g_cache
std::unordered_map<const void *, n1_w> g_fold;

void n1_invalidate(const void * base, size_t size) {
    std::lock_guard<std::mutex> lk(g_mtx);
    const char * b = (const char *) base;
    for (auto it = g_fold.begin(); it != g_fold.end();) {
        const char * k = (const char *) it->first;
        if (k >= b && k < b + size) {
            if (it->second.w)  (void) hipFree(it->second.w);
            if (it->second.sw) (void) hipFree(it->second.sw);
            g_cache_bytes -= it->second.bytes;
            it = g_fold.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = g_refused.begin(); it != g_refused.end();) {
        const char * k = (const char *) *it;
        it = (k >= b && k < b + size) ? g_refused.erase(it) : std::next(it);
    }
    for (auto it = g_cache.begin(); it != g_cache.end();) {
        const char * k = (const char *) it->first;
        if (k >= b && k < b + size) {
            if (it->second.w)  (void) hipFree(it->second.w);
            if (it->second.sw) (void) hipFree(it->second.sw);
            g_cache_bytes -= it->second.bytes;
            it = g_cache.erase(it);
        } else {
            ++it;
        }
    }
}
struct n1_registrar {
    n1_registrar()  { ggml_hipblaslt_wcache_register(n1_invalidate); }
    ~n1_registrar() { ggml_hipblaslt_wcache_unregister(n1_invalidate); }
};
n1_registrar g_reg;

hipStream_t conv_stream() {
    static hipStream_t s = [] { hipStream_t x = nullptr; CUDA_CHECK(hipStreamCreateWithFlags(&x, hipStreamNonBlocking)); return x; }();
    return s;
}

bool vram_ok(size_t need) {
    size_t freeb = 0, totb = 0;
    if (hipMemGetInfo(&freeb, &totb) != hipSuccess) return false;
    return freeb >= need + (size_t) (2ull << 30);   // 2 GB headroom for compute/pool buffers
}

template <int TYPE>
bool convert(const ggml_tensor * src0, int mode, n1_w & c) {
    const int F = (int) src0->ne[1], K = (int) src0->ne[0], S = K / 128;
    const char * w = (const char *) src0->data;
    const size_t nb01 = src0->nb[1];
    hipStream_t st = conv_stream();
    c.mode = mode;
    if (mode == 1) {
        const size_t wb = (size_t) F * K / 4, sb = (size_t) S * F * sizeof(__half);
        if (!vram_ok(wb + sb)) return false;
        if (hipMalloc(&c.w, wb) != hipSuccess) return false;
        if (hipMalloc(&c.sw, sb) != hipSuccess) { (void) hipFree(c.w); c.w = nullptr; return false; }
        c.bytes = wb + sb;
        const size_t nw = (size_t) (F / N1_BF) * S * 1024;
        k_n1_pack_w2<TYPE><<<(unsigned) ((nw + 255) / 256), 256, 0, st>>>(w, nb01, F, K, (uint32_t *) c.w);
        k_n1_pack_sw<TYPE><<<(unsigned) (((size_t) S * F + 255) / 256), 256, 0, st>>>(w, nb01, F, K, (__half *) c.sw);
    } else {
        const size_t wb = (size_t) F * K, sb = (size_t) F * sizeof(float);
        if (!vram_ok(wb + sb + (size_t) F * S)) return false;
        if (hipMalloc(&c.w, wb) != hipSuccess) return false;
        if (hipMalloc(&c.sw, sb) != hipSuccess) { (void) hipFree(c.w); c.w = nullptr; return false; }
        int8_t * mbuf = nullptr;
        if (hipMalloc(&mbuf, (size_t) F * S) != hipSuccess) { (void) hipFree(c.w); (void) hipFree(c.sw); c.w = c.sw = nullptr; return false; }
        c.bytes = wb + sb;
        k_n1_row_fold<TYPE><<<F, 256, 0, st>>>(w, nb01, K, (float *) c.sw, mbuf);
        const size_t nc = (size_t) F * K / 16;
        k_n1_pack_row<TYPE><<<(unsigned) ((nc + 255) / 256), 256, 0, st>>>(w, nb01, F, K, mbuf, (uint4 *) c.w);
        CUDA_CHECK(hipStreamSynchronize(st));
        (void) hipFree(mbuf);
    }
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipStreamSynchronize(st));
    if (env().dump && mode == 1) {   // T400 gate (a): the runtime cache, byte for byte
        const size_t wb = (size_t) F * K / 4, sb = (size_t) S * F * sizeof(__half);
        std::string buf(wb > sb ? wb : sb, '\0');
        const char * part[2] = {".w2", ".sw"};
        const void * dev[2]  = {c.w, c.sw};
        const size_t len[2]  = {wb, sb};
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(hipMemcpy(buf.data(), dev[i], len[i], hipMemcpyDeviceToHost));
            const std::string path = std::string(env().dump) + "/" + src0->name + part[i];
            FILE * f = fopen(path.c_str(), "wb");
            if (!f || fwrite(buf.data(), 1, len[i], f) != len[i]) {
                fprintf(stderr, "[N1] dump FAILED: %s\n", path.c_str());
            }
            if (f) fclose(f);
        }
        g_stats.dumped++;
    }
    return true;
}

// T400 N4: native dual-blob weight (loader view over GGML_TYPE_NK_*_W2) -> W2 region + companion sw, in place
bool nk_dual_weight(const ggml_tensor * src0, const void ** w, const void ** sw) {
    if (src0->type == GGML_TYPE_NK_Q2_0_W2ONLY) {   // T399 single copy: W2 stream, then sw[K/128][F]
        *w  = src0->data;
        *sw = (const char *) src0->data + (size_t) src0->ne[1] * src0->ne[0] / 4;
        return true;
    }
    const ggml_tensor * d = src0->view_src;
    if (!d || src0->view_offs != 0 || (d->type != GGML_TYPE_NK_Q2_0_W2 && d->type != GGML_TYPE_NK_Q1_0_W2)) return false;
    const ggml_tensor * comp = (const ggml_tensor *) d->extra;
    GGML_ASSERT(comp != nullptr && "NK dual tensor without companion (the loader refuses these)");
    const size_t cb = d->type == GGML_TYPE_NK_Q2_0_W2 ? 34 : 18;
    const size_t compact = (size_t) d->ne[1] * (d->ne[0] / 128) * cb;
    *w  = (const char *) d->data + compact;
    *sw = (const char *) comp->data + 256;   // scales_offset, enforced == 256 by the loader
    return true;
}

const n1_w * get_weight(const ggml_tensor * src0, int mode) {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_cache.find(src0->data);
    if (it != g_cache.end()) return it->second.mode == mode ? &it->second : nullptr;
    if (g_refused.count(src0->data)) return nullptr;
    n1_w c;
    const bool ok = src0->type == GGML_TYPE_Q2_0 ? convert<GGML_TYPE_Q2_0>(src0, mode, c) : convert<GGML_TYPE_Q1_0>(src0, mode, c);
    if (!ok) { g_refused.insert(src0->data); return nullptr; }
    g_cache_bytes += c.bytes;
    if (mode == 1) { g_stats.conv_w2++;  g_stats.bytes_w2  += c.bytes; }
    else           { g_stats.conv_row++; g_stats.bytes_row += c.bytes; }
    return &g_cache.emplace(src0->data, c).first->second;
}

bool n1_has_fold(const ggml_tensor * src0) {
    std::lock_guard<std::mutex> lk(g_mtx);
    return g_fold.count(src0->data) != 0;
}

// T422 M2: the fold side-table of a g128-arm weight (W2 stream + fp16 sw already resident), built once on the
// conversion stream. nullptr = refused (VRAM): the caller stays on the exact g128 kernel.
const n1_w * get_fold(const ggml_tensor * src0, const void * W2, const void * SW) {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_fold.find(src0->data);
    if (it != g_fold.end()) return &it->second;
    const int F = (int) src0->ne[1], K = (int) src0->ne[0], S = K / 128;
    const size_t mb = (size_t) S * F, rb = (size_t) F * sizeof(float);
    if (!vram_ok(mb + rb)) return nullptr;
    n1_w c; c.mode = 3;
    if (hipMalloc(&c.w, mb) != hipSuccess) return nullptr;
    if (hipMalloc(&c.sw, rb) != hipSuccess) { (void) hipFree(c.w); return nullptr; }
    c.bytes = mb + rb;
    hipStream_t st = conv_stream();
    k_n1_fold_sw<<<F, 256, 0, st>>>((const uint32_t *) W2, (const __half *) SW, F, K, (float *) c.sw, (int8_t *) c.w);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipStreamSynchronize(st));
    g_cache_bytes += c.bytes;
    g_stats.conv_fold++;
    return &g_fold.emplace(src0->data, c).first->second;
}

// BC: 0 = exact sub + fmac rescale (GGML_N1_M2=1); else m2::BCARRY | (log2(flush groups) << 25)
template <int KT, int BC>
void launch_m2_kt(const void * X, const void * W, const n1_w * fo, const float * sx, float * Y, int N, int Npad, int F, int K, int ldy, hipStream_t st) {
    const int ntile = Npad / 128;
    const dim3 grid((ntile + 1) / 2, F / 64);
    m2::gemm_fold<8, 1, 1, 2, KT | BC><<<grid, 256, 0, st>>>((const uint4 *) X, (const uint4 *) W, (const int8_t *) fo->w,
        (const float *) fo->sw, sx, Y, N, ntile, F, K, ldy);
}
template <int BC>
void launch_m2_bc(const void * X, const void * W, const n1_w * fo, const float * sx, float * Y, int N, int Npad, int F, int K, int ldy, hipStream_t st) {
    // the last template slot of gemm_fold is its variant mask (bit LPF = 4, BCARRY = 1 << 24); KT tags K in the kernel name
    // for rocprofv3 with values that never set bit 2 (K / 1024 << 3)
    switch (K) {
        case 5120:  launch_m2_kt<(5120 / 1024) << 3,  BC>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st); break;
        case 6144:  launch_m2_kt<(6144 / 1024) << 3,  BC>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st); break;
        case 17408: launch_m2_kt<(17408 / 1024) << 3, BC>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st); break;
        default:    launch_m2_kt<0,                   BC>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st); break;
    }
}
// T434 M3: ONE int32 accumulation over all of K (gemm_fold<ONEACC>, RS = 1), per-token scale sx[t], epilogue (acc * sx) * S_f
// WTn = 8: 256 tokens x 64 features per workgroup (the fast tile); WTn = 4, WFn = 2: 128 tokens x 128 features (odd token-tile
// counts: the 256-token tile would waste a whole half tile; 3% slower per FLOP in the standalone harness)
// FEED: 0 = wave tile 32 tok x 64 feat; m2::FT4 = 64 x 64; m2::FF8 = 32 x 128 (T435). BT = 16 * FT * WTn tokens, BF = 16 * FF * WFn features per workgroup.
template <int KT, int WTn, int WFn, int SEGS, int FEED>
void launch_m3_kt(const void * X, const void * W, const n1_w * fo, const float * sx, float * Y, int N, int Npad, int F, int K, int ldy, hipStream_t st) {
    constexpr int FTv = (FEED & m2::FT4) ? 4 : 2, FFv = (FEED & m2::FF8) ? 8 : 4;
    constexpr int TPB = 16 * FTv * WTn / 128;   // 128-token tiles per workgroup
    const int ntile = Npad / 128;
    const dim3 grid((ntile + TPB - 1) / TPB, F / (16 * FFv * WFn));
    m2::gemm_fold<WTn, WFn, SEGS, 1, KT | m2::ONEACC | FEED><<<grid, WTn * WFn * 32, 0, st>>>((const uint4 *) X, (const uint4 *) W, (const int8_t *) fo->w,
        (const float *) fo->sw, sx, Y, N, ntile, F, K, ldy);
}
template <int WTn, int WFn, int SEGS, int FEED>
void launch_m3_s(const void * X, const void * W, const n1_w * fo, const float * sx, float * Y, int N, int Npad, int F, int K, int ldy, hipStream_t st) {
    switch (K) {
        case 5120:  launch_m3_kt<(5120 / 1024) << 3,  WTn, WFn, SEGS, FEED>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st); break;
        case 6144:  launch_m3_kt<(6144 / 1024) << 3,  WTn, WFn, SEGS, FEED>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st); break;
        case 17408: launch_m3_kt<(17408 / 1024) << 3, WTn, WFn, SEGS, FEED>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st); break;
        default:    launch_m3_kt<0,                   WTn, WFn, SEGS, FEED>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st); break;
    }
}
void launch_m3(const void * X, const void * W, const n1_w * fo, const float * sx, float * Y, int N, int Npad, int F, int K, int ldy, hipStream_t st) {
    const bool odd = (Npad / 128) % 2 == 1 && !env().m3_wide;
    if (odd) launch_m3_s<4, 2, 1, 0>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st);
    else if (env().m3_feed == 1) launch_m3_s<4, 2, 1, m2::FT4>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st);
    else if (env().m3_feed == 3) launch_m3_s<4, 2, 1, m2::FT4 | m2::PRW>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st);
    else if (env().m3_feed == 2) launch_m3_s<8, 1, 1, m2::FF8>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st);
    else if (env().m3_segs == 2 && (K / 128) % 2 == 0) launch_m3_s<8, 1, 2, 0>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st);
    else launch_m3_s<8, 1, 1, 0>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st);
}

void launch_m2(const void * X, const void * W, const n1_w * fo, const float * sx, float * Y, int N, int Npad, int F, int K, int ldy, hipStream_t st) {
    if (!env().m2bc) { launch_m2_bc<0>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st); return; }
    switch (env().m2bc_flush) {   // GGML_N1_M2BC_FLUSH = 1, 2, 4 or 8 (default 8)
        case 1:  launch_m2_bc<m2::BCARRY | (0 << 25)>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st); break;
        case 2:  launch_m2_bc<m2::BCARRY | (1 << 25)>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st); break;
        case 4:  launch_m2_bc<m2::BCARRY | (2 << 25)>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st); break;
        default: launch_m2_bc<m2::BCARRY | (3 << 25)>(X, W, fo, sx, Y, N, Npad, F, K, ldy, st); break;
    }
}

// ---------------- GEMM launch ----------------
template <int RS, int FL, int KT>
void launch_kt(const void * X, const void * W, const void * sw, const float * sx, float * Y, int N, int Npad, int F, int K, int ldy, hipStream_t st) {
    const dim3 grid(Npad / N1_BT, F / N1_BF);
    dw::gemm_iu8<N1_BT, N1_BF, 4, 2, RS, 1, FL, KT><<<grid, N1_NT, 0, st>>>(
        (const uint4 *) X, (const uint4 *) W, sw, sx, Y, N, F, K, ldy);
}
template <int RS, int FL>
void launch(const void * X, const void * W, const void * sw, const float * sx, float * Y, int N, int Npad, int F, int K, int ldy, hipStream_t st) {
    switch (K) {
        case 5120:  launch_kt<RS, FL, 5120 >(X, W, sw, sx, Y, N, Npad, F, K, ldy, st); break;
        case 6144:  launch_kt<RS, FL, 6144 >(X, W, sw, sx, Y, N, Npad, F, K, ldy, st); break;
        case 17408: launch_kt<RS, FL, 17408>(X, W, sw, sx, Y, N, Npad, F, K, ldy, st); break;
        default:    launch_kt<RS, FL, 0    >(X, W, sw, sx, Y, N, Npad, F, K, ldy, st); break;
    }
}

// T420: g128 arm on gemm_xd (KT tags the K for rocprofv3, as launch_kt does)
template <int KT>
void launch_xd_kt(const void * X, const void * W, const void * sw, const float * sx, float * Y, int N, int Npad, int F, int K, int ldy, hipStream_t st) {
    const dim3 grid(Npad / N1_BT, F / N1_BF);
    dw::gemm_xd<KT><<<grid, N1_NT, 0, st>>>((const uint4 *) X, (const uint4 *) W, sw, sx, Y, N, F, K, ldy);
}
void launch_xd(const void * X, const void * W, const void * sw, const float * sx, float * Y, int N, int Npad, int F, int K, int ldy, hipStream_t st) {
    switch (K) {
        case 5120:  launch_xd_kt<5120 >(X, W, sw, sx, Y, N, Npad, F, K, ldy, st); break;
        case 6144:  launch_xd_kt<6144 >(X, W, sw, sx, Y, N, Npad, F, K, ldy, st); break;
        case 17408: launch_xd_kt<17408>(X, W, sw, sx, Y, N, Npad, F, K, ldy, st); break;
        default:    launch_xd_kt<0    >(X, W, sw, sx, Y, N, Npad, F, K, ldy, st); break;
    }
}

// The one routing predicate (shape side), shared by the GEMM entry and the act-fuse probe.
// src1 [K, n, s2, s3] with a 2D weight and contiguous rows collapses to N = n*s2*s3 tokens (e.g. the GDN ssm_out
// projection arrives as [K, n_seq_tokens, n_seqs]); dst is then [F, N] contiguous as well.
bool n1_shape_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    const bool collapse = src1->ne[2] * src1->ne[3] == 1 ||
        (src1->nb[2] == src1->nb[1] * src1->ne[1] && src1->nb[3] == src1->nb[2] * src1->ne[2] &&
         dst->nb[2] == dst->nb[1] * dst->ne[1] && dst->nb[3] == dst->nb[2] * dst->ne[2]);
    return (src0->type != GGML_TYPE_Q2_0 || ggml_q2_0_variant_of(src0) == GGML_Q2_0_VARIANT_G128) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && collapse &&
        src0->ne[0] == src1->ne[0] && src0->ne[0] % 128 == 0 && src0->ne[1] % N1_BF == 0 &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 && ((uintptr_t) src1->data % 16) == 0 &&
        ggml_is_contiguous(dst) && ggml_is_contiguous(src0) && src1->ne[1] <= (1 << 20);
}

// T434: src1 is the output of a prism.hadamard FWHT node (llama_mul_mat_hadamard: MUL_MAT with the HADAMARD hint), i.e. the
// activation of a rotation-folded weight
// (llama_mul_mat_hadamard: MUL_MAT(rot, x2d) with the HADAMARD hint, then a RESHAPE view back to the activation shape).
// Returns the MUL_MAT node (the act-cache key of the FWHT producer) or nullptr.
const ggml_tensor * n1_hada_base(const ggml_tensor * src1) {
    const ggml_tensor * mm = src1->op == GGML_OP_RESHAPE && src1->src[0] ? src1->src[0] : src1;
    return mm->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD ? mm : nullptr;
}
bool n1_hada_act(const ggml_tensor * src1) { return n1_hada_base(src1) != nullptr; }

bool n1_use_row(const ggml_tensor * src0) {
    if (src0->type == GGML_TYPE_NK_Q2_0_W2ONLY) {
        return false;   // T399: no compact blocks to fold from
    }
    return env().row_all || (env().row_ffn && strstr(src0->name, "ffn_") != nullptr);
}

// 1 = converted for `mode`, 0 = not seen yet, -1 = refused or converted for another mode
int n1_weight_state(const ggml_tensor * src0, int mode) {
    const void * w = nullptr, * sw = nullptr;
    if (mode == 1 && nk_dual_weight(src0, &w, &sw)) return 1;   // T400: native weight, used in place
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_cache.find(src0->data);
    if (it != g_cache.end()) return it->second.mode == mode ? 1 : -1;
    return g_refused.count(src0->data) ? -1 : 0;
}

} // namespace

bool ggml_cuda_n1_enabled() { return env().on; }

// T434: this MUL_MAT will run the one-GEMM path (GGML_N1_M3) and read the N1TOK cache entry of its FWHT src1. Mirrors the
// routing of n1_mul_mat_impl (weight already converted, so the fold side-table can be built from it).
bool ggml_cuda_n1_m3_ready(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return env().on && env().m3 && !env().act_tok && !env().transient && n1_hada_act(src1) &&
        (src0->type == GGML_TYPE_Q2_0 || src0->type == GGML_TYPE_Q1_0) && src1->ne[1] >= env().min_n &&
        n1_shape_ok(src0, src1, dst) && !n1_use_row(src0) && n1_weight_state(src0, 1) == 1;
}

void ggml_cuda_n1_count(const ggml_tensor * src0, const ggml_tensor * src1) {
    if (src1->ne[1] < env().min_n) return;
    const uint64_t fl = mm_flops(src0, src1);
    g_stats.mm_total++; g_stats.fl_total += fl;
    if (src0->type == GGML_TYPE_Q2_0 || src0->type == GGML_TYPE_Q1_0 || src0->type == GGML_TYPE_NK_Q2_0_W2ONLY) {
        g_stats.mm_q++; g_stats.fl_q += fl;
    }
}

static bool n1_mul_mat_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

bool ggml_cuda_n1_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q1_0) return false;
    if (src1->ne[1] < env().min_n) return false;
    return n1_mul_mat_impl(ctx, src0, src1, dst);
}

// T399: single-copy weights have no other path; the caller (mul_mat_sc.cu) decides the batch threshold and N1 runs
// regardless of GGML_N1_PREFILL / GGML_N1_MIN_N
bool ggml_cuda_n1_mul_mat_w2only(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_NK_Q2_0_W2ONLY);
    ggml_cuda_n1_count(src0, src1);
    return n1_mul_mat_impl(ctx, src0, src1, dst);
}

static bool n1_mul_mat_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const uint64_t fl = mm_flops(src0, src1);
    if (!n1_shape_ok(src0, src1, dst)) {
        g_stats.fb_shape++; g_stats.fl_fb_shape += fl;
        static std::atomic<int> nlog{0};
        if (nlog.fetch_add(1) < 12) {
            fprintf(stderr, "[N1] fallback shape: %s type=%s var=%d K=%lld F=%lld N=%lld ne2=%lld src1 nb1=%zu align=%d dst_contig=%d\n",
                    src0->name, ggml_type_name(src0->type), src0->type == GGML_TYPE_Q2_0 ? (int) ggml_q2_0_variant_of(src0) : 0,
                    (long long) src0->ne[0], (long long) src0->ne[1], (long long) src1->ne[1], (long long) src1->ne[2],
                    src1->nb[1], (int) ((uintptr_t) src1->data % 16), (int) ggml_is_contiguous(dst));
        }
        return false;
    }

    const bool row = n1_use_row(src0);
    const int F = (int) src0->ne[1], K = (int) src0->ne[0], N = (int) (src1->ne[1] * src1->ne[2] * src1->ne[3]), S = K / 128;
    const int Npad = (N + N1_BT - 1) / N1_BT * N1_BT;
    hipStream_t st = ctx.stream();

    const void * Wp  = nullptr;
    const void * SWp = nullptr;
    ggml_cuda_pool_alloc<uint32_t> tw(ctx.pool());
    ggml_cuda_pool_alloc<__half>   tsw(ctx.pool());
    if (!row && nk_dual_weight(src0, &Wp, &SWp)) {
        g_stats.mm_nk++;
    } else if (!row && env().transient) {
        const size_t nw = (size_t) (F / N1_BF) * S * 1024;
        tw.alloc(nw);
        tsw.alloc((size_t) S * F);
        if (src0->type == GGML_TYPE_Q2_0) {
            k_n1_pack_w2<GGML_TYPE_Q2_0><<<(unsigned) ((nw + 255) / 256), 256, 0, st>>>((const char *) src0->data, src0->nb[1], F, K, tw.get());
            k_n1_pack_sw<GGML_TYPE_Q2_0><<<(unsigned) (((size_t) S * F + 255) / 256), 256, 0, st>>>((const char *) src0->data, src0->nb[1], F, K, tsw.get());
        } else {
            k_n1_pack_w2<GGML_TYPE_Q1_0><<<(unsigned) ((nw + 255) / 256), 256, 0, st>>>((const char *) src0->data, src0->nb[1], F, K, tw.get());
            k_n1_pack_sw<GGML_TYPE_Q1_0><<<(unsigned) (((size_t) S * F + 255) / 256), 256, 0, st>>>((const char *) src0->data, src0->nb[1], F, K, tsw.get());
        }
        Wp = tw.get(); SWp = tsw.get();
        g_stats.mm_transient++;
    } else {
        const n1_w * cw = get_weight(src0, row ? 2 : 1);
        if (!cw) {
            g_stats.fb_mem++; g_stats.fl_fb_mem += fl;
            // a producer only fuses for this consumer once the weight is converted (ggml_cuda_n1_act_route), so a
            // pending GLU can never be left for the default route here
            GGML_ASSERT(ctx.act_pending != src1 && "N1: pending GLU consumer fell back");
            GGML_ASSERT((ctx.act_nofp32 == nullptr || ctx.act_nofp32 != n1_hada_base(src1)) && "N1: FWHT consumer fell back without fp32");
            return false;
        }
        Wp = cw->w; SWp = cw->sw;
    }

    const bool per_token = row || env().act_tok;
    float * Y = (float *) dst->data;
    const int ldy = (int) (dst->nb[1] / sizeof(float));

    // T434 M3 (one-GEMM): Hadamard-model activations (src1 is the FWHT node) quantised per token, folded int8 weights, no in-loop
    // rescale. Bonsai 1 (no Hadamard node) never takes this branch.
    if (env().m3 && !per_token && n1_hada_act(src1) && !env().transient) {
        const n1_w * fo3 = get_fold(src0, Wp, SWp);
        if (fo3) {
            const size_t nbytes   = n1_act_bytes(K, N);
            const int    mask     = ggml_cuda_act_fuse_mask();
            const ggml_tensor * hb = n1_hada_base(src1);   // the cache is keyed on the FWHT MUL_MAT node
            const bool   act_hit  = mask && ctx.act_cache_tensor == hb && ctx.act_cache_buf &&
                ctx.act_cache_layout == GGML_CUDA_ACT_LAYOUT_N1TOK && ctx.act_cache_bytes == nbytes;
            const bool   act_cache = act_hit || (mask & GGML_ACT_FUSE_DEDUP);
            GGML_ASSERT(ctx.act_pending != src1 && "N1 M3: pending GLU consumer");
            GGML_ASSERT((act_hit || ctx.act_nofp32 != hb) && "N1 M3: FWHT fp32 output was skipped but the cache missed");
            act_hit ? g_stats.act_hit++ : g_stats.act_miss++;
            if (act_hit) g_stats.m3_hit++;
            ggml_cuda_pool_alloc<char> local(ctx.pool(), act_cache ? 0 : nbytes);
            if (act_cache && !act_hit) {
                ctx.act_cache_buf.reset();
                ctx.act_cache_buf    = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), nbytes);
                ctx.act_cache_tensor = hb;
                ctx.act_cache_layout = GGML_CUDA_ACT_LAYOUT_N1TOK;
                ctx.act_cache_bytes  = nbytes;
            }
            char * buf = act_cache ? ctx.act_cache_buf->get() : local.get();
            int8_t * X  = (int8_t *) buf;
            float  * sx = (float *) (buf + (size_t) Npad * K);
            if (!act_hit) {
                k_n1_quant_act<<<dim3(1, Npad), 256, 0, st>>>((const char *) src1->data, src1->nb[1], N, K, K, Npad, X, sx);
            }
            launch_m3(X, Wp, fo3, sx, Y, N, Npad, F, K, ldy, st);
            g_stats.mm_n1++; g_stats.fl_n1 += fl;
            g_stats.mm_m3++; g_stats.fl_m3 += fl;
            CUDA_CHECK(hipGetLastError());
            return true;
        }
    }
    GGML_ASSERT((ctx.act_nofp32 == nullptr || ctx.act_nofp32 != n1_hada_base(src1)) &&
                "N1: FWHT fp32 output was skipped but this consumer cannot take the N1TOK cache");

    const n1_w * fold = (!per_token && env().m2 && K % 256 == 0) ? get_fold(src0, Wp, SWp) : nullptr;
    if (fold) {
        // T422 M2: per-256 activation (own layout tag in the dedup cache: siblings sharing src1 quantize once; the act-fuse
        // producers never write it, ggml_cuda_n1_act_route returns 2 for these consumers)
        constexpr int LAYOUT_N1G256 = GGML_CUDA_ACT_LAYOUT_N1G256;
        const size_t nbytes = n1_act_bytes(K, N);
        const int    mask   = ggml_cuda_act_fuse_mask();
        const bool   act_hit = mask && ctx.act_cache_tensor == src1 && ctx.act_cache_buf &&
            ctx.act_cache_layout == LAYOUT_N1G256 && ctx.act_cache_bytes == nbytes;
        const bool   act_cache = act_hit || (mask & GGML_ACT_FUSE_DEDUP);
        if (ctx.act_pending == src1) {
            GGML_ASSERT(act_hit && "act-fuse: pending GLU consumer missed the N1G256 cache");
            ctx.act_pending = nullptr;
        }
        act_hit ? g_stats.act_hit++ : g_stats.act_miss++;
        if (act_hit && src1->op == GGML_OP_GLU) g_stats.act_glu++;
        if (act_hit && src1->op == GGML_OP_MUL) g_stats.act_norm++;
        ggml_cuda_pool_alloc<char> local(ctx.pool(), act_cache ? 0 : nbytes);
        if (act_cache && !act_hit) {
            ctx.act_cache_buf.reset();
            ctx.act_cache_buf    = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), nbytes);
            ctx.act_cache_tensor = src1;
            ctx.act_cache_layout = LAYOUT_N1G256;
            ctx.act_cache_bytes  = nbytes;
        }
        char * buf = act_cache ? ctx.act_cache_buf->get() : local.get();
        int8_t * X  = (int8_t *) buf;
        float  * sx = (float *) (buf + (size_t) Npad * K);
        if (!act_hit) {
            k_n1_quant_act256<<<dim3(K / 256, Npad), 64, 0, st>>>((const char *) src1->data, src1->nb[1], N, K, Npad, X, sx);
        }
        launch_m2(X, Wp, fold, sx, Y, N, Npad, F, K, ldy, st);
        g_stats.mm_n1++; g_stats.fl_n1 += fl;
        g_stats.mm_m2++; g_stats.fl_m2 += fl;
        CUDA_CHECK(hipGetLastError());
        return true;
    }

    if (!per_token) {
        // [TAG_ACT_FUSE] T412: the g128 arm reads its activation through the act-fuse cache (N1 layout), so a fused
        // producer (GLU / norm) feeds it directly and siblings sharing src1 quantize once. Mirrors mmq.cu.
        const size_t nbytes   = n1_act_bytes(K, N);
        const int    mask     = ggml_cuda_act_fuse_mask();
        const bool   act_hit  = mask && ctx.act_cache_tensor == src1 && ctx.act_cache_buf &&
            ctx.act_cache_layout == GGML_CUDA_ACT_LAYOUT_N1 && ctx.act_cache_bytes == nbytes;
        const bool   act_cache = act_hit || (mask & GGML_ACT_FUSE_DEDUP);
        if (ctx.act_pending == src1) {
            GGML_ASSERT(act_hit && "act-fuse: pending GLU consumer missed the N1 cache");
            ctx.act_pending = nullptr;
        }
        if (act_cache) {
            act_hit ? ctx.act_stat_hit++ : ctx.act_stat_miss++;
            act_hit ? g_stats.act_hit++ : g_stats.act_miss++;
            if (act_hit && src1->op == GGML_OP_GLU) g_stats.act_glu++;
            if (act_hit && src1->op == GGML_OP_MUL) g_stats.act_norm++;
        } else {
            g_stats.act_miss++;
        }
        ggml_cuda_pool_alloc<char> local(ctx.pool(), act_cache ? 0 : nbytes);
        if (act_cache && !act_hit) {
            ctx.act_cache_buf.reset();
            ctx.act_cache_buf    = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), nbytes);
            ctx.act_cache_tensor = src1;
            ctx.act_cache_layout = GGML_CUDA_ACT_LAYOUT_N1;
            ctx.act_cache_bytes  = nbytes;
        }
        char * buf = act_cache ? ctx.act_cache_buf->get() : local.get();
        int8_t * X  = (int8_t *) buf;
        float  * sx = (float *) (buf + (size_t) Npad * K);
        if (!act_hit) {
            k_n1_quant_act<<<dim3(S, Npad), 32, 0, st>>>((const char *) src1->data, src1->nb[1], N, K, 128, Npad, X, sx);
        }
        if (env().m1) {
            launch_xd(X, Wp, SWp, sx, Y, N, Npad, F, K, ldy, st);
        } else {
            launch<1, 9>(X, Wp, SWp, sx, Y, N, Npad, F, K, ldy, st);
        }
        g_stats.mm_n1++; g_stats.fl_n1 += fl;
        CUDA_CHECK(hipGetLastError());
        return true;
    }

    GGML_ASSERT(ctx.act_pending != src1 && "N1: pending GLU reached a per-token arm");
    ggml_cuda_pool_alloc<int8_t> X(ctx.pool(), (size_t) Npad * K);
    ggml_cuda_pool_alloc<float>  sx(ctx.pool(), (size_t) Npad);
    k_n1_quant_act<<<dim3(1, Npad), 256, 0, st>>>((const char *) src1->data, src1->nb[1], N, K, K, Npad, X.get(), sx.get());
    if (row) {
        launch<0, 0>(X.get(), Wp, SWp, sx.get(), Y, N, Npad, F, K, ldy, st);
        g_stats.mm_row++; g_stats.fl_row += fl;
    } else {
        launch<1, 1>(X.get(), Wp, SWp, sx.get(), Y, N, Npad, F, K, ldy, st);
        g_stats.mm_n1++; g_stats.fl_n1 += fl;
    }
    CUDA_CHECK(hipGetLastError());
    return true;
}

int ggml_cuda_n1_act_route(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!env().on || (src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q1_0) || src1->ne[1] < env().min_n ||
        !n1_shape_ok(src0, src1, dst)) {
        return 0;
    }
    const bool row = n1_use_row(src0);
    const int  ws  = n1_weight_state(src0, row ? 2 : 1);
    if (ws < 0) {
        return 0;   // refused: ggml_cuda_n1_mul_mat falls back to the default route
    }
    if (!row && !env().act_tok && env().m2 && src0->ne[0] % 256 == 0) {
        // T422 M2: the fold+g256 arm reads the N1G256 layout once its fold side-table exists (built on first use)
        return ws == 1 && n1_has_fold(src0) ? 3 : 2;
    }
    return ws == 1 && !row && !env().act_tok ? 1 : 2;
}

int ggml_cuda_n1_w2only_act_layout(const ggml_tensor * src0) {
    if (env().m2 && src0->ne[0] % 256 == 0) {
        return n1_has_fold(src0) ? GGML_CUDA_ACT_LAYOUT_N1G256 : -1;
    }
    return GGML_CUDA_ACT_LAYOUT_N1;
}

void ggml_cuda_n1_act_ref_quant256(const float * x, int64_t s11, int64_t K, int64_t N, void * y, cudaStream_t stream) {
    const int64_t Npad = n1_npad(N);
    k_n1_quant_act256<<<dim3((unsigned) (K / 256), (unsigned) Npad), 64, 0, stream>>>((const char *) x, (size_t) s11 * sizeof(float),
        (int) N, (int) K, (int) Npad, (int8_t *) y, (float *) ((char *) y + (size_t) Npad * K));
}

// T434: reference per-token quantizer (the unfused M3 producer: k_n1_quant_act with G = K) for the VERIFY of the FWHT producer
void ggml_cuda_n1_act_ref_quant_tok(const float * x, int64_t s11, int64_t K, int64_t N, void * y, cudaStream_t stream) {
    const int64_t Npad = n1_npad(N);
    k_n1_quant_act<<<dim3(1, (unsigned) Npad), 256, 0, stream>>>((const char *) x, (size_t) s11 * sizeof(float),
        (int) N, (int) K, (int) K, (int) Npad, (int8_t *) y, (float *) ((char *) y + (size_t) Npad * K));
}

size_t ggml_cuda_n1_act_bytes(const ggml_tensor * src1) {
    return n1_act_bytes(src1->ne[0], src1->ne[1] * src1->ne[2] * src1->ne[3]);
}

void ggml_cuda_n1_act_ref_quant(const float * x, int64_t s11, int64_t K, int64_t N, void * y, cudaStream_t stream) {
    const int64_t Npad = n1_npad(N);
    k_n1_quant_act<<<dim3((unsigned) (K / 128), (unsigned) Npad), 32, 0, stream>>>((const char *) x, (size_t) s11 * sizeof(float),
        (int) N, (int) K, 128, (int) Npad, (int8_t *) y, (float *) ((char *) y + (size_t) Npad * K));
}

#else

bool ggml_cuda_n1_enabled() { return false; }
bool ggml_cuda_n1_m3_ready(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) { return false; }
void ggml_cuda_n1_count(const ggml_tensor *, const ggml_tensor *) {}
bool ggml_cuda_n1_mul_mat(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) { return false; }
bool ggml_cuda_n1_mul_mat_w2only(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) { return false; }
int ggml_cuda_n1_act_route(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) { return 0; }
size_t ggml_cuda_n1_act_bytes(const ggml_tensor *) { return 0; }
void ggml_cuda_n1_act_ref_quant(const float *, int64_t, int64_t, int64_t, void *, cudaStream_t) {}
int ggml_cuda_n1_w2only_act_layout(const ggml_tensor *) { return -1; }
void ggml_cuda_n1_act_ref_quant256(const float *, int64_t, int64_t, int64_t, void *, cudaStream_t) {}
void ggml_cuda_n1_act_ref_quant_tok(const float *, int64_t, int64_t, int64_t, void *, cudaStream_t) {}

#endif
