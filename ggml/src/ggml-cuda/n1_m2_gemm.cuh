// M2 (T422) N1 g128 GEMM, structural variants of M1's gemm_xd (X direct-to-VGPR, W widened into LDS, SWAP operand
// roles, bias-initialised accumulators). SAME MATH, bit-identical to gemm_xd / the release kernel:
//   Y[t][f] = sum_g fma-chain( ((float)I_g(t,f)) * sx[g][t] , (float)sw[g][f] ),  I_g = exact int32 dot over 128 K.
//
// Template:
//   WTn x WFn waves; each wave owns 32 tokens x 64 features (FT = 2, FF = 4 16x16 fragments) => BT = 32*WTn tokens,
//   BF = 64*WFn features per workgroup.  (4,2) = M1's t128_f128 tile; (8,1) = 256 tok x 64 feat (the weight is shared
//   by 8 waves along M, X is loaded by exactly one wave); (8,2) = 256 x 128 with 16 waves.
//   SEGS: 128-K segments per LDS stage (DepthU = 128*SEGS): one barrier per SEGS segments.
//   V bits: BIAS (as M1), SWL (weight scales converted to fp32 once per workgroup into LDS; the rescale fma becomes a
//   VOPD-pairable v_fmac_f32 instead of v_fma_mix_f32), LPF (W fragments of k-block kb+1 read from LDS before kb's WMMAs).
//
// Global load schedule (all global loads are inline asm; loadcnt retires in order; counts derived for the steady state):
//   stage s:  P0 = W(s+1) [NWL loads], SW(s+1) [SWL ? SEGS : 0]
//             per segment g: SEG(g) = sx [FT] (+ sw fp16 [FF] if !SWL); then per k-block kb: use X ring slot kb, reload
//             slot kb with the next segment's k-block kb [FT].
//   before using slot kb of segment g: newer loads = 3FT + SEGL + (g == 0 ? NWL + NSW : 0)
//   before rescaling segment g (needs SEG(g)): newer = 4FT
//   before the W/SW commit (needs P0): newer = SEGS * (SEGL + 4FT)
// T422 M2: engine copy of native-kernels/dense_wmma/m2/m2_gemm.cuh (helpers used by n1_m2_fold.cuh; gemm itself unused).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace m2 {
using i32x2 = __attribute__((__vector_size__(8))) int;
using i32x8 = __attribute__((__vector_size__(32))) int;
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));

constexpr int BIAS = 1, SWL = 2, LPF = 4;
constexpr float BIASF = 12582912.0f;   // 1.5 * 2^23
constexpr int BIASI = 0x4B400000;
constexpr uint32_t W2_LUT = 0x020100FFu;

__device__ __forceinline__ i32x8 wmma_iu8(i32x2 a, i32x2 b, i32x8 c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}
__device__ __forceinline__ void bar_signal() { asm volatile("s_wait_dscnt 0\n\ts_barrier_signal -1" ::: "memory"); }
__device__ __forceinline__ void bar_wait() { asm volatile("s_barrier_wait -1" ::: "memory"); }
__device__ __forceinline__ void lds_barrier() {
    asm volatile("s_wait_dscnt 0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
}
__device__ __forceinline__ u32x4 widen2(uint32_t p) {
    u32x4 o;
    o.x = __builtin_amdgcn_perm(0u, W2_LUT, p & 0x03030303u);
    o.y = __builtin_amdgcn_perm(0u, W2_LUT, (p >> 2) & 0x03030303u);
    o.z = __builtin_amdgcn_perm(0u, W2_LUT, (p >> 4) & 0x03030303u);
    o.w = __builtin_amdgcn_perm(0u, W2_LUT, (p >> 6) & 0x03030303u);
    return o;
}

// Xt: N1 activation stream (BT = 128 tiles: [tile][seg][row16][kb][lane][16B]). Wt: W2 stream (pack_tiles_w2<256>,
// [tile128][seg][256 threads][4 dwords]). sw fp16 [S][F]. sx fp32 [S][Npad]. ntile = Npad / 128.
template <int WTn, int WFn, int SEGS, int V>
__global__ __launch_bounds__(WTn * WFn * 32, 1)
void gemm(const uint4* __restrict__ Xt, const uint4* __restrict__ Wt, const __half* __restrict__ sw,
          const float* __restrict__ sx, float* __restrict__ Y, int Ntok, int ntile, int F, int K, int ldy) {
    constexpr int NT = WTn * WFn * 32, FT = 2, FF = 4, KBN = 4;
    constexpr int BT = 32 * WTn, BF = 64 * WFn, XB = 128 * 128, WSEG = BF * 128;   // LDS bytes of one widened W segment
    constexpr int DWN = BF * 8 / NT;            // packed W dwords per thread per segment
    static_assert(DWN == 2 || DWN == 4, "W load is b64 or b128 per thread");
    static_assert(WTn % 4 == 0 || WTn == 4, "token waves: whole 128-token tiles");
    constexpr bool kBias = V & BIAS, kSwl = V & SWL, kLpf = V & LPF;
    constexpr int NWL = SEGS;                    // W loads per stage
    constexpr int NSW = kSwl ? SEGS : 0;         // fp16 weight-scale loads per stage (SWL)
    constexpr int SEGL = FT + (kSwl ? 0 : FF);   // per-segment scale loads
    constexpr int WAIT_X0 = 3 * FT + SEGL + NWL + NSW, WAIT_XG = 3 * FT + SEGL;
    constexpr int WAIT_R = 4 * FT, WAIT_C = SEGS * (SEGL + 4 * FT);
    constexpr int WSTAGE = SEGS * WSEG;
    constexpr int SWST = SEGS * BF * 4;          // LDS bytes of one stage of fp32 weight scales
    // fp32 weight scales are TRIPLE-buffered: the last segment's rescale reads them after bar_signal, so a wave that has
    // already passed the barrier may commit the next stage's scales while a slower wave still reads this stage's
    __shared__ uint4 lds[(2 * WSTAGE + (kSwl ? 3 * SWST : 0)) / 16];

    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wt = wave % WTn, wf = wave / WTn;
    const int S = K / 128;
    // token side: this wave's 128-token tile (clamped; waves past the last tile compute a duplicate and do not store)
    const int tile_raw = blockIdx.x * (BT / 128) + wt / 4;
    const bool tok_ok = tile_raw < ntile;
    const int tile = tok_ok ? tile_raw : ntile - 1;
    const int tok0 = tile * 128 + (wt % 4) * 32;
    // feature side
    const int fbase = blockIdx.y * BF;               // first feature of the workgroup
    const int ftile = fbase / 128, fhalf = (fbase % 128) / 64;
    const int fea0 = fbase + wf * 64;
    // packed W: pack-thread t (0..255) and its dword range [d0, d0 + DWN) within the 128-feature tile
    const int pt = tid & 255;
    const int d0 = (BF == 64) ? 2 * fhalf : (DWN == 2 ? 2 * (tid >> 8) : 0);
    const int dl0 = (BF == 64) ? 0 : d0;            // dword index relative to this workgroup's LDS segment

    const char* xg = (const char*)Xt;
    const char* wg = (const char*)Wt + (size_t)ftile * S * 4096;
    const uint32_t xvoff = (uint32_t)(((size_t)tile * S * XB) + ((wt % 4) * FT * KBN * 32 + lane) * 16);
    const uint32_t wvoff = (uint32_t)(pt * 16 + d0 * 4);
    const uint32_t swvoff = (uint32_t)((fea0 + 8 * (lane >> 4)) * 2);   // !SWL: + j*32
    const uint32_t sw1off = (uint32_t)((fbase + (tid % BF)) * 2);         // SWL: one fp16 per thread
    const uint32_t sxvoff = (uint32_t)((tok0 + (lane & 15)) * 4);
    const int Npad = ntile * 128;
    const uint32_t lds0 = (uint32_t)(uintptr_t)(__attribute__((address_space(3))) char*)lds;
    const uint32_t ldsw = lds0 + (uint32_t)(pt * 16 + dl0 * 4096);   // + d * 4096 for this thread's dword d
    const uint32_t ldss = lds0 + 2 * WSTAGE + (uint32_t)((tid % BF) * 4);
    const float* swlds = (const float*)((const char*)lds + 2 * WSTAGE);

    float accf[FT][FF][8];
#pragma unroll
    for (int i = 0; i < FT; ++i)
#pragma unroll
        for (int j = 0; j < FF; ++j)
#pragma unroll
            for (int l = 0; l < 8; ++l) accf[i][j][l] = 0.f;

    u32x4 xr[KBN][FT];
    uint32_t rw[SEGS][DWN];
    uint32_t rsw[SEGS];
    uint32_t swh[FF][4];
    float sxa[FT];

#define XLOAD(G, kb)                                                                                         \
    { const char* xb_ = xg + (size_t)(G) * XB;                                                                \
      _Pragma("unroll") for (int i = 0; i < FT; ++i)                                                           \
          asm volatile("global_load_b128 %0, %1, %2 offset:%3" : "=v"(xr[kb][i]) : "v"(xvoff), "s"(xb_), "i"((i * KBN + (kb)) * 512) : "memory"); }
#define P0LOAD(stg)                                                                                          \
    { _Pragma("unroll") for (int g = 0; g < SEGS; ++g) {                                                       \
          const char* wb_ = wg + (size_t)((stg) * SEGS + g) * 4096;                                             \
          if constexpr (DWN == 4) { u32x4 t_;                                                                  \
              asm volatile("global_load_b128 %0, %1, %2 offset:0" : "=v"(t_) : "v"(wvoff), "s"(wb_) : "memory"); \
              rw[g][0] = t_.x; rw[g][1] = t_.y; rw[g][(DWN - 2)] = t_.z; rw[g][DWN - 1] = t_.w; } \
          else { u32x2 t_;                                                                                     \
              asm volatile("global_load_b64 %0, %1, %2 offset:0" : "=v"(t_) : "v"(wvoff), "s"(wb_) : "memory");  \
              rw[g][0] = t_.x; rw[g][1] = t_.y; } }                                                             \
      if constexpr (kSwl) { _Pragma("unroll") for (int g = 0; g < SEGS; ++g) {                                 \
          const char* sb_ = (const char*)(sw + (size_t)((stg) * SEGS + g) * F);                                 \
          asm volatile("global_load_u16 %0, %1, %2 offset:0" : "=v"(rsw[g]) : "v"(sw1off), "s"(sb_) : "memory"); } } }
#define SEGLOAD(G)                                                                                           \
    { if constexpr (!kSwl) { const char* swb_ = (const char*)(sw + (size_t)(G) * F);                           \
          _Pragma("unroll") for (int j = 0; j < FF; ++j) { u32x4 t_;                                            \
              asm volatile("global_load_b128 %0, %1, %2 offset:%3" : "=v"(t_) : "v"(swvoff), "s"(swb_), "i"(j * 32) : "memory"); \
              swh[j][0] = t_.x; swh[j][1] = t_.y; swh[j][2] = t_.z; swh[j][3] = t_.w; } }                       \
      const char* sxb_ = (const char*)(sx + (size_t)(G) * Npad);                                                \
      _Pragma("unroll") for (int i = 0; i < FT; ++i)                                                           \
          asm volatile("global_load_b32 %0, %1, %2 offset:%3" : "=v"(sxa[i]) : "v"(sxvoff), "s"(sxb_), "i"(i * 64) : "memory"); }
#define COMMIT(buf, sbuf, waitn)                                                                                   \
    { asm volatile("s_wait_loadcnt %0" :: "i"(waitn) : "memory");                                              \
      _Pragma("unroll") for (int g = 0; g < SEGS; ++g) {                                                       \
          _Pragma("unroll") for (int d = 0; d < DWN; ++d) asm volatile("" : "+v"(rw[g][d]));                    \
          if constexpr (kSwl) asm volatile("" : "+v"(rsw[g]));                                                  \
          const uint32_t a_ = ldsw + (uint32_t)(buf) * WSTAGE + g * WSEG;                                        \
          _Pragma("unroll") for (int d = 0; d < DWN; ++d) {                                                     \
              const u32x4 w8 = widen2(rw[g][d]);                                                                \
              asm volatile("ds_store_b128 %0, %1 offset:%2" :: "v"(a_), "v"(w8), "i"(d * 4096) : "memory"); } \
          if constexpr (kSwl) {                                                                                 \
              const float f_ = __half2float(__ushort_as_half((unsigned short)(rsw[g] & 0xFFFFu)));               \
              asm volatile("ds_store_b32 %0, %1 offset:%2" :: "v"(ldss + (uint32_t)(sbuf) * SWST), "v"(f_), "i"(g * BF * 4) : "memory"); } } }

    // prologue: W(0)/SW(0) into LDS buffer 0, X(seg 0, all k-blocks) into registers
    P0LOAD(0)
    COMMIT(0, 0, 0)
#pragma unroll
    for (int kb = 0; kb < KBN; ++kb) XLOAD(0, kb)
    lds_barrier();

    const i32x8 zero8 = i32x8{0, 0, 0, 0, 0, 0, 0, 0};
    const i32x8 bias8 = i32x8{BIASI, BIASI, BIASI, BIASI, BIASI, BIASI, BIASI, BIASI};
    i32x8 acc[FT][FF];
    const int NST = S / SEGS;

    auto rescale = [&](int buf, int g) {
        asm volatile("s_wait_loadcnt %0" :: "i"(WAIT_R) : "memory");
#pragma unroll
        for (int i = 0; i < FT; ++i) asm volatile("" : "+v"(sxa[i]));
        if constexpr (!kSwl) {
#pragma unroll
            for (int j = 0; j < FF; ++j)
#pragma unroll
                for (int u = 0; u < 4; ++u) asm volatile("" : "+v"(swh[j][u]));
        }
#pragma unroll
        for (int j = 0; j < FF; ++j) {
            float swf[8];
            if constexpr (kSwl) {
                const float4* p = (const float4*)(swlds + buf * (SWST / 4) + g * BF + wf * 64 + 16 * j + 8 * (lane >> 4));
                const float4 a = p[0], b = p[1];
                swf[0] = a.x; swf[1] = a.y; swf[2] = a.z; swf[3] = a.w; swf[4] = b.x; swf[5] = b.y; swf[6] = b.z; swf[7] = b.w;
            } else {
#pragma unroll
                for (int l = 0; l < 8; ++l)
                    swf[l] = __half2float(__ushort_as_half((unsigned short)(swh[j][l >> 1] >> (16 * (l & 1)))));
            }
#pragma unroll
            for (int i = 0; i < FT; ++i) {
                asm volatile("" : "+v"(acc[i][j]));
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const int ai = acc[i][j][l];
                    float x = kBias ? __int_as_float(ai) - BIASF : (float)ai;
                    if constexpr (kBias) asm volatile("" : "+v"(x));
                    accf[i][j][l] = __builtin_fmaf(x * sxa[i], swf[l], accf[i][j][l]);
                }
#pragma unroll
                for (int l = 0; l < 8; ++l) asm volatile("" : "+v"(accf[i][j][l]));
            }
        }
    };

    for (int s = 0; s < NST; ++s) {
        const int sn = s + 1 < NST ? s + 1 : s;
        P0LOAD(sn)
        const char* bW = (const char*)lds + (s & 1) * WSTAGE;
#pragma unroll
        for (int g = 0; g < SEGS; ++g) {
            const int G = s * SEGS + g;
            const int Gn = (g + 1 < SEGS) ? G + 1 : sn * SEGS;   // next segment for the X ring (dummy at the end)
            SEGLOAD(G)
            const uint4* b = (const uint4*)(bW + g * WSEG);
            uint4 w[2][FF];
            if constexpr (kLpf) {
#pragma unroll
                for (int j = 0; j < FF; ++j) w[0][j] = b[((wf * FF + j) * KBN + 0) * 32 + lane];
            }
#pragma unroll
            for (int kb = 0; kb < KBN; ++kb) {
                uint4* wc = w[kLpf ? (kb & 1) : 0];
                if constexpr (kLpf) {
                    if (kb + 1 < KBN) {
#pragma unroll
                        for (int j = 0; j < FF; ++j) w[(kb + 1) & 1][j] = b[((wf * FF + j) * KBN + kb + 1) * 32 + lane];
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < FF; ++j) wc[j] = b[((wf * FF + j) * KBN + kb) * 32 + lane];
                }
                if (g == 0) asm volatile("s_wait_loadcnt %0" :: "i"(WAIT_X0) : "memory");
                else asm volatile("s_wait_loadcnt %0" :: "i"(WAIT_XG) : "memory");
#pragma unroll
                for (int i = 0; i < FT; ++i) asm volatile("" : "+v"(xr[kb][i]));
#pragma unroll
                for (int i = 0; i < FT; ++i)
#pragma unroll
                    for (int j = 0; j < FF; ++j) {
                        i32x8 c0 = (kb == 0) ? (kBias ? bias8 : zero8) : acc[i][j];
                        const i32x2 x0{(int)xr[kb][i].x, (int)xr[kb][i].y}, x1{(int)xr[kb][i].z, (int)xr[kb][i].w};
                        const i32x2 w0{(int)wc[j].x, (int)wc[j].y}, w1{(int)wc[j].z, (int)wc[j].w};
                        c0 = wmma_iu8(w0, x0, c0);
                        acc[i][j] = wmma_iu8(w1, x1, c0);
                    }
                __builtin_amdgcn_sched_barrier(0);
                XLOAD(Gn, kb)
            }
            if (g + 1 < SEGS) {
                rescale(s % 3, g);
            } else {
#pragma unroll
                for (int i = 0; i < FT; ++i)
#pragma unroll
                    for (int j = 0; j < FF; ++j) asm volatile("" : "+v"(acc[i][j]));
                COMMIT((s + 1) & 1, (s + 1) % 3, WAIT_C)
                bar_signal();
                rescale(s % 3, g);
                bar_wait();
            }
        }
    }
    asm volatile("s_wait_loadcnt 0" ::: "memory");
#undef XLOAD
#undef P0LOAD
#undef SEGLOAD
#undef COMMIT

    if (!tok_ok) return;
#pragma unroll
    for (int i = 0; i < FT; ++i) {
        const int t = tok0 + i * 16 + (lane & 15);
        if (t >= Ntok) continue;
#pragma unroll
        for (int j = 0; j < FF; ++j) {
            const int f0 = fea0 + 16 * j + 8 * (lane >> 4);
            float o[8];
#pragma unroll
            for (int l = 0; l < 8; ++l) o[l] = accf[i][j][l] * 1.0f;
            float4* yp = (float4*)(Y + (size_t)t * ldy + f0);
            yp[0] = make_float4(o[0], o[1], o[2], o[3]);
            yp[1] = make_float4(o[4], o[5], o[6], o[7]);
        }
    }
}
}  // namespace m2
