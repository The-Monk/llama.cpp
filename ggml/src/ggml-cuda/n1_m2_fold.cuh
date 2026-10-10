// M2 (T422) fold + per-256 path: CONTRACT mode ROW weights (W' = code * m[g][f], m int8 per (128-K group, row), one
// fp32 row scale S_f) x activation scales per RS*128 K (RS = 1: per-128, RS = 2: per-256; the kernel reads
// sx[G0][t] of each RS-group's first 128-segment, i.e. the T402 emulation layout where the producer writes the same
// scale into each 128-slot of a group).  DIFFERENT MATH from the release g128 kernel (gated: M0 fold GO, T402 g256 OK).
//   Y[t][f] = S_f * fma-chain over RS-groups h ( (float)I_h(t,f) , sx[h][t] ),  I_h = exact int32 dot over RS*128 K of
//   x * (code * m).  Bias-initialised accumulators: exact while |I_h| < 2^22; worst case for RS = 2 with operands in
//   [-128,127] x [-127,127]: 256 * 128 * 127 = 4,161,536 < 4,194,304 (margin 0.8%) -> exact for every in-range input.
// Structure = m2::gemm (X direct-to-VGPR ring, W widened into LDS by a per-(row, segment) LUT {-m, 0, m, 2m}).
// Rescale per element per RS-group: v_sub (bias) + v_fmac (sx), both VOPD-pairable; no weight-scale multiply.
//
// Load schedule (asm, loadcnt in order), P0N = SEGS * (1 + DWN) (W b64/b128 + one u8 m per packed dword):
//   stage: P0(next stage) ; per segment p (= G % RS): [p == 0: sx FT loads] ; per kb: use ring slot, reload [FT]
// T422 M2: engine copy of native-kernels/dense_wmma/m2/m2_fold.cuh (gemm_fold); routed by GGML_N1_M2=1.
#pragma once
#include "n1_m2_gemm.cuh"

// Expert scheduling mode (-mllvm -amdgpu-expert-scheduling-mode): the compiler emits s_wait_alu depctr waits for its
// own instructions only; with inline-asm loads/stores every kernel lost exactness (M2 jobs 27, 36: a vm_vsrc wait after
// each asm ds_store is NOT enough). M2_ESM_FULL (all dependency counters drained before and after every asm global load
// and after every asm ds_store) is exact (job 37). M2_ESM_PRE: drained before each asm load and after each store only.
#if defined(M2_ESM_FULL)
#define M2_ESM_STORE_WAIT asm volatile("s_wait_alu 0x0" ::: "memory");
#define M2_EPRE M2_ESM_STORE_WAIT
#define M2_EPOST M2_ESM_STORE_WAIT
#elif defined(M2_ESM_PRE)
#define M2_ESM_STORE_WAIT asm volatile("s_wait_alu 0x0" ::: "memory");
#define M2_EPRE M2_ESM_STORE_WAIT
#define M2_EPOST
#elif defined(M2_ESM)
#define M2_ESM_STORE_WAIT asm volatile("s_wait_alu depctr_vm_vsrc(0)" ::: "memory");
#define M2_EPRE
#define M2_EPOST
#else
#define M2_ESM_STORE_WAIT
#define M2_EPRE
#define M2_EPOST
#endif

namespace m2 {

// T435 timing-only ablations (results are WRONG by design; the harness skips verification): ABX no X reload in the loop, ABW no W-fragment
// LDS reads (stage-0 fragment reused), ABC no W widen/ds_store/P0 loads in the loop, ABB no per-segment barrier
constexpr int ABX = 1 << 29, ABW = 1 << 30, ABC = 1 << 9, ABB = 1 << 10;
constexpr int FF8 = 1 << 11, FT4 = 1 << 12;
constexpr int ABCV = 1 << 13, ABCM = 1 << 14, ABCS = 1 << 15;   // commit sub-ablations: no widen VALU (raw dword splat), no m u8 loads, no ds_store (which also kills the widen)
constexpr int LDSA = 1 << 28;  // T435: W fragments read by inline-asm ds_load_b128, ONE s_wait_dscnt per k-block (the compiler emits one per fragment)
constexpr int ONEACC = 1 << 27;  // T434 variant bit: ONE int32 accumulation over the whole K (RS = 1), epilogue acc * sx[t] * S_f; no in-loop rescale
constexpr int BCARRY = 1 << 24;   // T424 variant bit: bias-carry rescale (GGML_N1_M2=2); the low bits tag K (K / 1024 << 3)

__device__ __forceinline__ uint32_t fold_lut(uint32_t mb) {   // mb = m as a zero-extended byte; LUT bytes {-m,0,m,2m}
    return ((0u - mb) & 0xFFu) | ((mb & 0xFFu) << 16) | (((mb << 1) & 0xFFu) << 24);
}
__device__ __forceinline__ u32x4 widen2l(uint32_t p, uint32_t lut) {
    u32x4 o;
    o.x = __builtin_amdgcn_perm(0u, lut, p & 0x03030303u);
    o.y = __builtin_amdgcn_perm(0u, lut, (p >> 2) & 0x03030303u);
    o.z = __builtin_amdgcn_perm(0u, lut, (p >> 4) & 0x03030303u);
    o.w = __builtin_amdgcn_perm(0u, lut, (p >> 6) & 0x03030303u);
    return o;
}

// mq: int8 [S][F] fold multipliers; rowS: fp32 [F]
template <int WTn, int WFn, int SEGS, int RS, int V>
__global__ __launch_bounds__(WTn * WFn * 32, 1)
void gemm_fold(const uint4* __restrict__ Xt, const uint4* __restrict__ Wt, const int8_t* __restrict__ mq,
               const float* __restrict__ rowS, const float* __restrict__ sx, float* __restrict__ Y,
               int Ntok, int ntile, int F, int K, int ldy) {
    // FF8 (T435): 32 tokens x 128 features per wave (X bytes per WMMA halved); FT4: 64 tokens x 64 features per wave (W LDS reads per WMMA halved)
    constexpr int NT = WTn * WFn * 32, FT = (V & FT4) ? 4 : 2, FF = (V & FF8) ? 8 : 4, KBN = 4;
    static_assert(!((V & FT4) && (V & FF8)), "FT4 and FF8 together need 256 accumulator VGPRs");
    constexpr int WPT = 8 / FT;   // waves per 128-token tile
    constexpr int BT = 16 * FT * WTn, BF = 16 * FF * WFn, XB = 128 * 128, WSEG = BF * 128;
    constexpr int DWN = BF * 8 / NT;
    static_assert(DWN == 1 || DWN == 2 || DWN == 4, "W load is b32, b64 or b128 per thread");
    static_assert(RS == 1 || RS == 2, "RS");
    static_assert(SEGS == 1 || SEGS == 2, "SEGS");
    static_assert(RS >= SEGS || RS == 1, "a stage may not split... RS < SEGS handled per segment");
    constexpr bool kLpf = V & LPF, kBc = V & BCARRY, kOne = V & ONEACC, kLa = V & LDSA;
    constexpr bool kAbX = V & ABX, kAbW = V & ABW, kAbC = V & ABC, kAbB = V & ABB, kAbCV = V & ABCV, kAbCM = V & ABCM, kAbCS = V & ABCS;
    static_assert(!kOne || RS == 1, "ONEACC: per-token scale, no rescale groups");
    constexpr int U = (RS > SEGS) ? RS / SEGS : 1;   // stages unrolled per loop iteration
    constexpr int P0N = SEGS * (1 + DWN);
    constexpr int WSTAGE = SEGS * WSEG;
    __shared__ uint4 lds[2 * WSTAGE / 16];

    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wt = wave % WTn, wf = wave / WTn;
    const int S = K / 128;
    const int tile_raw = blockIdx.x * (BT / 128) + wt / WPT;
    const bool tok_ok = tile_raw < ntile;
    const int tile = tok_ok ? tile_raw : ntile - 1;
    const int tok0 = tile * 128 + (wt % WPT) * (16 * FT);
    const int fbase = blockIdx.y * BF;
    const int ftile = fbase / 128, fhalf = (fbase % 128) / 64;
    const int fea0 = fbase + wf * 16 * FF;
    const int pt = tid & 255;
    const int d0 = (BF == 64) ? 2 * fhalf + (DWN == 1 ? (tid >> 8) : 0) : (DWN == 2 ? 2 * (tid >> 8) : 0);
    const int dl0 = (BF == 64) ? (DWN == 1 ? (tid >> 8) : 0) : d0;

    const char* xg = (const char*)Xt;
    const char* wg = (const char*)Wt + (size_t)ftile * S * 4096;
    const uint32_t xvoff = (uint32_t)(((size_t)tile * S * XB) + ((wt % WPT) * FT * KBN * 32 + lane) * 16);
    const uint32_t wvoff = (uint32_t)(pt * 16 + d0 * 4);
    // fold multiplier byte of packed dword d (q = pt + 256 d): row in the 128-feature tile = (q >> 7) * 16 + (q & 15)
    uint32_t mvoff[DWN];
#pragma unroll
    for (int d = 0; d < DWN; ++d) {
        const int q = pt + 256 * (d0 + d);
        mvoff[d] = (uint32_t)(ftile * 128 + (q >> 7) * 16 + (q & 15));
    }
    const uint32_t sxvoff = (uint32_t)((tok0 + (lane & 15)) * 4);
    const int Npad = ntile * 128;
    const uint32_t lds0 = (uint32_t)(uintptr_t)(__attribute__((address_space(3))) char*)lds;
    const uint32_t ldsw = lds0 + (uint32_t)(pt * 16 + dl0 * 4096);

    float accf[FT][FF][8];
#pragma unroll
    for (int i = 0; i < FT; ++i)
#pragma unroll
        for (int j = 0; j < FF; ++j)
#pragma unroll
            for (int l = 0; l < 8; ++l) accf[i][j][l] = 0.f;

    u32x4 xr[KBN][FT];
    uint32_t rw[SEGS][DWN], rm[SEGS][DWN];
    if constexpr (kAbCM) {   // ablation: a valid, runtime-opaque fold multiplier so the widened weights keep realistic data (zero weights would cut power and raise the clock)
#pragma unroll
        for (int g = 0; g < SEGS; ++g)
#pragma unroll
            for (int d = 0; d < DWN; ++d) { rm[g][d] = 64u + (uint32_t)(lane & 7); asm volatile("" : "+v"(rm[g][d])); }
    }
    float sxa[FT];
    float sxs[FT] = {0.f, 0.f};   // BC (T424): running sum of the group scales; BIASF * sxs is subtracted every BC_FLUSH groups and in the epilogue
    int nres = 0;
    constexpr int BC_FLUSH = 1 << ((V >> 25) & 3);   // 1, 2, 4 or 8 rescale groups between bias flushes (variant bits 25-26)

#define XLOAD(G, kb)                                                                                       \
    { const char* xb_ = xg + (size_t)(G) * XB;                                                                \
      _Pragma("unroll") for (int i = 0; i < FT; ++i)                                                           \
          { M2_EPRE asm volatile("global_load_b128 %0, %1, %2 offset:%3" : "=v"(xr[kb][i]) : "v"(xvoff), "s"(xb_), "i"((i * KBN + (kb)) * 512) : "memory"); M2_EPOST } }
#define P0LOAD(stg)                                                                                          \
    { _Pragma("unroll") for (int g = 0; g < SEGS; ++g) {                                                       \
          const char* wb_ = wg + (size_t)((stg) * SEGS + g) * 4096;                                             \
          if constexpr (DWN == 4) { u32x4 t_;                                                                  \
              M2_EPRE asm volatile("global_load_b128 %0, %1, %2 offset:0" : "=v"(t_) : "v"(wvoff), "s"(wb_) : "memory"); M2_EPOST \
              rw[g][0] = t_.x; rw[g][1] = t_.y; rw[g][DWN - 2] = t_.z; rw[g][DWN - 1] = t_.w; }                   \
          else if constexpr (DWN == 2) { u32x2 t_;                                                             \
              M2_EPRE asm volatile("global_load_b64 %0, %1, %2 offset:0" : "=v"(t_) : "v"(wvoff), "s"(wb_) : "memory"); M2_EPOST \
              rw[g][0] = t_.x; rw[g][1] = t_.y; }                                                               \
          else { uint32_t t_;                                                                                  \
              M2_EPRE asm volatile("global_load_b32 %0, %1, %2 offset:0" : "=v"(t_) : "v"(wvoff), "s"(wb_) : "memory"); M2_EPOST \
              rw[g][0] = t_; }                                                                                 \
          const char* mb_ = (const char*)(mq + (size_t)((stg) * SEGS + g) * F);                                 \
          if constexpr (!kAbCM) {                                                                              \
          _Pragma("unroll") for (int d = 0; d < DWN; ++d)                                                       \
              { M2_EPRE asm volatile("global_load_u8 %0, %1, %2 offset:0" : "=v"(rm[g][d]) : "v"(mvoff[d]), "s"(mb_) : "memory"); M2_EPOST } } } }
#define SXLOAD(G)                                                                                            \
    { const char* sxb_ = (const char*)(sx + (size_t)(G) * Npad);                                                \
      _Pragma("unroll") for (int i = 0; i < FT; ++i)                                                           \
          { M2_EPRE asm volatile("global_load_b32 %0, %1, %2 offset:%3" : "=v"(sxa[i]) : "v"(sxvoff), "s"(sxb_), "i"(i * 64) : "memory"); M2_EPOST } }
#define COMMIT(buf, waitn)                                                                                   \
    { asm volatile("s_wait_loadcnt %0" :: "i"(waitn) : "memory");                                              \
      _Pragma("unroll") for (int g = 0; g < SEGS; ++g) {                                                       \
          _Pragma("unroll") for (int d = 0; d < DWN; ++d) { asm volatile("" : "+v"(rw[g][d])); asm volatile("" : "+v"(rm[g][d])); } \
          const uint32_t a_ = ldsw + (uint32_t)(buf) * WSTAGE + g * WSEG;                                        \
          _Pragma("unroll") for (int d = 0; d < DWN; ++d) {                                                     \
              u32x4 w8;                                                                                         \
              if constexpr (kAbCV) w8 = u32x4{rw[g][d], rw[g][d], rw[g][d], rw[g][d]};                           \
              else w8 = widen2l(rw[g][d], fold_lut(rm[g][d]));                                                 \
              if constexpr (!kAbCS) { asm volatile("ds_store_b128 %0, %1 offset:%2" :: "v"(a_), "v"(w8), "i"(d * 4096) : "memory"); } \
              M2_ESM_STORE_WAIT } } }

    if (kOne) SXLOAD(0)   // T434: the per-token scale, loaded once (sx[0][t]); COMMIT(0, 0) drains it with the first W stage
    P0LOAD(0)
    COMMIT(0, 0)
    if (kOne) {
#pragma unroll
        for (int i = 0; i < FT; ++i) asm volatile("" : "+v"(sxa[i]));
    }
#pragma unroll
    for (int kb = 0; kb < KBN; ++kb) XLOAD(0, kb)
    lds_barrier();
    u32x4 wfix[FF];
    if constexpr (kAbW) {   // stage-0 fragments, read once; the loop reuses them
        const uint32_t r0_ = lds0 + (uint32_t)(wf * FF * KBN * 512 + lane * 16);
#pragma unroll
        for (int j = 0; j < FF; ++j) asm volatile("ds_load_b128 %0, %1 offset:%2" : "=v"(wfix[j]) : "v"(r0_), "i"(j * KBN * 512) : "memory");
        asm volatile("s_wait_dscnt 0" ::: "memory");
#pragma unroll
        for (int j = 0; j < FF; ++j) asm volatile("" : "+v"(wfix[j]));
    }

    const i32x8 bias8 = i32x8{BIASI, BIASI, BIASI, BIASI, BIASI, BIASI, BIASI, BIASI};
    i32x8 acc[FT][FF];
    if (kOne) {
#pragma unroll
        for (int i = 0; i < FT; ++i)
#pragma unroll
            for (int j = 0; j < FF; ++j) acc[i][j] = i32x8{0, 0, 0, 0, 0, 0, 0, 0};
    }
    const int NST = S / SEGS;

    auto rescale = [&](int waitsel) {
        // waitsel: loads issued after this group's sx (see header), compile-time via the two call sites below
        if (waitsel == 0) asm volatile("s_wait_loadcnt %0" :: "i"(4 * FT * RS) : "memory");
        else asm volatile("s_wait_loadcnt %0" :: "i"(4 * FT * RS + P0N) : "memory");
#pragma unroll
        for (int i = 0; i < FT; ++i) { asm volatile("" : "+v"(sxa[i])); if constexpr (kBc) sxs[i] += sxa[i]; }
#pragma unroll
        for (int i = 0; i < FT; ++i)
#pragma unroll
            for (int j = 0; j < FF; ++j) {
                asm volatile("" : "+v"(acc[i][j]));
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    if constexpr (kBc) {
                        // one fmac per element per group: BIASF * sx rides in accf and is removed by the flush / epilogue
                        accf[i][j][l] = __builtin_fmaf(__int_as_float(acc[i][j][l]), sxa[i], accf[i][j][l]);
                    } else {
                        float x = __int_as_float(acc[i][j][l]) - BIASF;
                        asm volatile("" : "+v"(x));
                        accf[i][j][l] = __builtin_fmaf(x, sxa[i], accf[i][j][l]);
                    }
                }
#pragma unroll
                for (int l = 0; l < 8; ++l) asm volatile("" : "+v"(accf[i][j][l]));
            }
        if constexpr (kBc) {
            // every BC_FLUSH groups remove the carried bias so the fp32 accumulator stays below ~BC_FLUSH * BIASF * sx
            if ((++nres & (BC_FLUSH - 1)) == 0) {
#pragma unroll
                for (int i = 0; i < FT; ++i) {
#pragma unroll
                    for (int j = 0; j < FF; ++j)
#pragma unroll
                        for (int l = 0; l < 8; ++l) accf[i][j][l] = __builtin_fmaf(-BIASF, sxs[i], accf[i][j][l]);
                    sxs[i] = 0.f;
                }
            }
        }
    };

    for (int s2 = 0; s2 < NST; s2 += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int s = s2 + u;
            const int sn = s + 1 < NST ? s + 1 : s;
            if constexpr (!kAbC) P0LOAD(sn)
            const char* bW = (const char*)lds + (s & 1) * WSTAGE;
#pragma unroll
            for (int g = 0; g < SEGS; ++g) {
                const int p = (u * SEGS + g) % RS;          // position inside the rescale group (compile-time)
                const int G = s * SEGS + g;
                const int Gn = (g + 1 < SEGS) ? G + 1 : sn * SEGS;
                const bool sxl = !kOne && p == 0;
                if (sxl) SXLOAD(G)
                const u32x4* b = (const u32x4*)(bW + g * WSEG);
                u32x4 w[2][FF];
                const uint32_t rd_ = lds0 + (uint32_t)((wf * FF * KBN * 512 + lane * 16) + ((s & 1) * WSTAGE + g * WSEG));
#define LDSLOAD(bf, kbv)                                                                                       \
    { _Pragma("unroll") for (int j = 0; j < FF; ++j)                                                          \
          asm volatile("ds_load_b128 %0, %1 offset:%2" : "=v"(w[bf][j]) : "v"(rd_), "i"(j * KBN * 512 + (kbv) * 512) : "memory"); }
                if constexpr (kAbW) {
                } else if constexpr (kLa) {
                    LDSLOAD(0, 0)
                } else if constexpr (kLpf) {
#pragma unroll
                    for (int j = 0; j < FF; ++j) w[0][j] = b[((wf * FF + j) * KBN + 0) * 32 + lane];
                }
#pragma unroll
                for (int kb = 0; kb < KBN; ++kb) {
                    u32x4* wc = w[(kLpf || kLa) ? (kb & 1) : 0];
                    if constexpr (kAbW) wc = wfix;
                    if constexpr (kAbW) {
                    } else if constexpr (kLa) {
                        if (kb + 1 < KBN) { LDSLOAD((kb + 1) & 1, kb + 1) asm volatile("s_wait_dscnt 4" ::: "memory"); }
                        else asm volatile("s_wait_dscnt 0" ::: "memory");
#pragma unroll
                        for (int j = 0; j < FF; ++j) asm volatile("" : "+v"(wc[j]));
                    } else if constexpr (kLpf) {
                        if (kb + 1 < KBN) {
#pragma unroll
                            for (int j = 0; j < FF; ++j) w[(kb + 1) & 1][j] = b[((wf * FF + j) * KBN + kb + 1) * 32 + lane];
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < FF; ++j) wc[j] = b[((wf * FF + j) * KBN + kb) * 32 + lane];
                    }
                    // newer than X(G, kb): (3-kb)FT + [g == 0: P0N] + [p == 0: FT] + kb*FT
                    if (g == 0 && sxl) asm volatile("s_wait_loadcnt %0" :: "i"(3 * FT + P0N + FT) : "memory");
                    else if (g == 0) asm volatile("s_wait_loadcnt %0" :: "i"(3 * FT + P0N) : "memory");
                    else if (sxl) asm volatile("s_wait_loadcnt %0" :: "i"(3 * FT + FT) : "memory");
                    else asm volatile("s_wait_loadcnt %0" :: "i"(3 * FT) : "memory");
#pragma unroll
                    for (int i = 0; i < FT; ++i) asm volatile("" : "+v"(xr[kb][i]));
#pragma unroll
                    for (int i = 0; i < FT; ++i)
#pragma unroll
                        for (int j = 0; j < FF; ++j) {
                            i32x8 c0 = (!kOne && kb == 0 && p == 0) ? bias8 : acc[i][j];
                            const i32x2 x0{(int)xr[kb][i].x, (int)xr[kb][i].y}, x1{(int)xr[kb][i].z, (int)xr[kb][i].w};
                            const i32x2 w0{(int)wc[j].x, (int)wc[j].y}, w1{(int)wc[j].z, (int)wc[j].w};
                            c0 = wmma_iu8(w0, x0, c0);
                            acc[i][j] = wmma_iu8(w1, x1, c0);
                        }
                    __builtin_amdgcn_sched_barrier(0);
                    if constexpr (!kAbX) XLOAD(Gn, kb)
                }
                const bool resc = (p == RS - 1);
                // a P0 lies between this group's sx and now only when the group spans a stage boundary (SEGS = 1, RS = 2)
                const int ws = (SEGS == 1 && RS == 2) ? 1 : 0;
                if (g + 1 < SEGS) {
                    if (resc && !kOne) rescale(ws);
                } else {
#pragma unroll
                    for (int i = 0; i < FT; ++i)
#pragma unroll
                        for (int j = 0; j < FF; ++j) asm volatile("" : "+v"(acc[i][j]));
                    // newer than the last P0 load: this stage's sx loads + 4 FT reloads per segment
                    constexpr int SXN = kOne ? 0 : (RS == 1) ? SEGS * FT : FT;   // RS == SEGS == 2: one sx group per stage
                    if constexpr (!kAbC) {
                        if (SEGS == 1 && RS == 2 && p == 1) COMMIT((s + 1) & 1, 4 * FT)
                        else COMMIT((s + 1) & 1, SXN + SEGS * 4 * FT)
                    }
                    if constexpr (!kAbB) bar_signal();
                    if (resc && !kOne) rescale(ws);
                    if constexpr (!kAbB) bar_wait();
                }
            }
        }
    }
    asm volatile("s_wait_loadcnt 0" ::: "memory");
#undef XLOAD
#undef P0LOAD
#undef SXLOAD
#undef COMMIT
#undef LDSLOAD

    if (!tok_ok) return;
#pragma unroll
    for (int j = 0; j < FF; ++j) {
        const int f0 = fea0 + 16 * j + 8 * (lane >> 4);
        const float4 r0 = *(const float4*)(rowS + f0), r1 = *(const float4*)(rowS + f0 + 4);
        const float rs[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
        for (int i = 0; i < FT; ++i) {
            const int t = tok0 + i * 16 + (lane & 15);
            if (t >= Ntok) continue;
            float o[8];
#pragma unroll
            for (int l = 0; l < 8; ++l) o[l] = kOne ? ((float) acc[i][j][l] * sxa[i]) * rs[l] : (kBc ? __builtin_fmaf(-BIASF, sxs[i], accf[i][j][l]) : accf[i][j][l]) * rs[l];
            float4* yp = (float4*)(Y + (size_t)t * ldy + f0);
            yp[0] = make_float4(o[0], o[1], o[2], o[3]);
            yp[1] = make_float4(o[4], o[5], o[6], o[7]);
        }
    }
}
}  // namespace m2
