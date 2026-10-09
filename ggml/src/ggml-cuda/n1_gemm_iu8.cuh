// Dense prefill GEMM on v_wmma_i32_16x16x16_iu8 (gfx1201), pre-tiled "fragment order" operands.
//
//   Y[t][f] = sx[t] * sum_g sw[g][f] * ( sum_{k in segment g} X[t][k] * W[f][k] )     (int32 inside a segment)
//
// X (activations, int8, per-token fp32 scale sx[t]) is the WMMA A operand, W (weights, int8, per-(128*RS)-K
// fp16 scale per output feature) the WMMA B operand. So the accumulator lane layout is: lane = feature (col),
// element l of 8 = token 8*(lane/16)+l  -> the weight scale is ONE scalar per lane per feature fragment.
// See LAYOUT.md for the byte layout of Xt / Wt / sw.
//
// T404 engine-harness copy of native-kernels/dense_wmma/gemm_iu8.cuh (final binary of jobs 235/236). The main loop
// is unchanged. Engine-only additions, all outside the K loop unless noted:
//   * FL bit3 (PA): per-(128-K segment, token) activation scale sx[seg][Npad] folded into the per-segment rescale
//     (+1 v_mul per accumulator element per segment, inside the loop's rescale step; RS>0 only). Epilogue then skips sx.
//   * RS==0 reads the per-row weight scale as fp32 (sw is passed as void*): the fold scale S_f can be < 6e-5 (fp16 subnormal).
//   * epilogue guards t < Ntok (tokens padded to BT in X) and writes with row stride ldy.
//   * KT: unused tag so rocprofv3 separates kernels with equal grids but different K.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace dw {
using i32x2 = __attribute__((__vector_size__(8))) int;
using i32x8 = __attribute__((__vector_size__(32))) int;
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ i32x8 wmma_iu8(i32x2 a, i32x2 b, i32x8 c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}

// BT/BF: tokens/features per workgroup. WTn/WFn: wave grid. RS: 128-K stages per rescale segment.
__device__ __forceinline__ void bar_signal() { asm volatile("s_wait_dscnt 0\n\ts_barrier_signal -1" ::: "memory"); }
__device__ __forceinline__ void bar_wait() { asm volatile("s_barrier_wait -1" ::: "memory"); }
__device__ __forceinline__ void lds_barrier() {
    asm volatile("s_wait_dscnt 0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
}

// 2-bit -> int8 widen: dword p holds 16 codes; output dword k = LUT[(p >> 2k) & 0x03030303] bytewise (v_perm_b32 as a
// 4-entry byte LUT). Packing (see LAYOUT.md): code of output (dword k, byte b) lives at bit 8b+2k of p.
// LUT {-1,0,1,2}: value = code - 1 (ternary offset-binary); the converter may use any 4 signed bytes.
constexpr uint32_t W2_LUT = 0x020100FFu;
__device__ __forceinline__ u32x4 widen2(uint32_t p) {
    u32x4 o;
    o.x = __builtin_amdgcn_perm(0u, W2_LUT, p & 0x03030303u);
    o.y = __builtin_amdgcn_perm(0u, W2_LUT, (p >> 2) & 0x03030303u);
    o.z = __builtin_amdgcn_perm(0u, W2_LUT, (p >> 4) & 0x03030303u);
    o.w = __builtin_amdgcn_perm(0u, W2_LUT, (p >> 6) & 0x03030303u);
    return o;
}

template <int BT, int BF, int WTn, int WFn, int RS, int MINB, int W2 = 0, int KBN = 4>
struct Cfg {
    static constexpr int NW = WTn * WFn, NT = NW * 32;
    static constexpr int WT = BT / WTn, WF = BF / WFn;      // wave tile (tokens, features)
    static constexpr int FT = WT / 16, FF = WF / 16;        // fragments per wave
    static constexpr int XB = BT * 32 * KBN, WB = BF * 32 * KBN;   // bytes per stage (BK = 32*KBN K; KBN=4 -> 128)
    static constexpr int STAGE = XB + WB;
    static constexpr int XCH = XB / 16 / NT, WCH = WB / 16 / NT;  // 16B (int8) chunks per thread per stage
    static constexpr int PWL = W2 ? WB / 64 / NT : 0;              // W2: packed b128 loads per thread per stage (4 chunks each)
    static_assert(XB / 16 % NT == 0 && WB / 16 % NT == 0, "tile must divide thread count");
    static_assert(!W2 || (WB / 64 % NT == 0 && WB / 64 / NT >= 1), "packed W tile must be whole b128 per thread");
    static_assert(BT % (WTn * 16) == 0 && BF % (WFn * 16) == 0, "wave tile must be whole fragments");
};

template <int BT, int BF, int WTn, int WFn, int RS, int MINB, int FL = 0, int KT = 0>
__global__ __launch_bounds__((WTn * WFn * 32), MINB)
void gemm_iu8(const uint4* __restrict__ Xt, const uint4* __restrict__ Wt, const void* __restrict__ swp,
              const float* __restrict__ sx, float* __restrict__ Y, int Ntok, int F, int K, int ldy) {
    constexpr int W2 = FL & 1;   // FL bit0: weights 2-bit packed in global, widened to int8 on the way into LDS
    constexpr int PA = (FL & 8) ? 1 : 0;   // FL bit3: per-(segment,token) activation scales (engine harness)
    static_assert(!PA || RS > 0, "PA needs a segmented (RS>0) kernel");
    const __half* __restrict__ sw = (const __half*)swp;
    const int Npad = gridDim.x * BT;
    constexpr int KBN = (FL & 4) ? 2 : 4;   // FL bit2: BK=64 stages (half-size LDS buffers) instead of BK=128
    static_assert(!W2 || KBN == 4, "2-bit packed stream is defined for BK=128 only");
    using C = Cfg<BT, BF, WTn, WFn, RS, MINB, W2, KBN>;
    constexpr int FT = C::FT, FF = C::FF;
    __shared__ uint4 lds[2 * C::STAGE / 16];

    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wt = wave % WTn, wf = wave / WTn;
    const int tb = blockIdx.x, fb = blockIdx.y;             // token tile fastest: sibling tiles share W in L2
    const int S = K / (32 * KBN);                            // stages
    const uint4* xg = Xt + (size_t)tb * S * (C::XB / 16);
    constexpr int WSTR = W2 ? C::WB / 64 : C::WB / 16;       // uint4 per stage of the W stream (packed or int8)
    const uint4* wg = Wt + (size_t)fb * S * WSTR;
    const int fbase = fb * BF + wf * C::WF + (lane & 15);    // this lane's feature for fragment 0

    float accf[FT][FF][8];
#pragma unroll
    for (int i = 0; i < FT; ++i)
#pragma unroll
        for (int j = 0; j < FF; ++j)
#pragma unroll
            for (int l = 0; l < 8; ++l) accf[i][j][l] = 0.f;

    u32x4 rx[C::XCH], rw[W2 ? C::PWL : C::WCH];
// Staging copies use inline asm: the AMDGPU scheduler otherwise sinks plain global loads down to their ds_store
// (exposing the whole DRAM/L2 latency per chunk). The wait is explicit (s_wait_loadcnt 0) right before the stores.
    const uint32_t voff = (uint32_t)tid * 16u;
    const uint32_t lds_base = (uint32_t)(uintptr_t)(__attribute__((address_space(3))) char*)lds + (uint32_t)tid * 16u;
#define GLOAD(stage)                                                                                         \
    { const uint4* xb_ = xg + (size_t)(stage) * (C::XB / 16);                                                  \
      const uint4* wb_ = wg + (size_t)(stage) * WSTR;                                                           \
      _Pragma("unroll") for (int c = 0; c < C::XCH; ++c)                                                       \
          asm volatile("global_load_b128 %0, %1, %2 offset:%3" : "=v"(rx[c]) : "v"(voff), "s"(xb_), "i"(c * C::NT * 16) : "memory"); \
      _Pragma("unroll") for (int c = 0; c < (W2 ? C::PWL : C::WCH); ++c)                                       \
          asm volatile("global_load_b128 %0, %1, %2 offset:%3" : "=v"(rw[c]) : "v"(voff), "s"(wb_), "i"(c * C::NT * 16) : "memory"); }
#define LSTORE(buf)                                                                                          \
    { const uint32_t a_ = lds_base + (uint32_t)(buf) * C::STAGE;                                               \
      asm volatile("s_wait_loadcnt 0x0" ::: "memory");                                                         \
      /* make every staging register a fresh output of the wait: widening VALU must not float above it */    \
      _Pragma("unroll") for (int c = 0; c < C::XCH; ++c) asm volatile("" : "+v"(rx[c]));                       \
      _Pragma("unroll") for (int c = 0; c < (W2 ? C::PWL : C::WCH); ++c) asm volatile("" : "+v"(rw[c]));       \
      _Pragma("unroll") for (int c = 0; c < C::XCH; ++c)                                                       \
          asm volatile("ds_store_b128 %0, %1 offset:%2" :: "v"(a_), "v"(rx[c]), "i"(c * C::NT * 16) : "memory"); \
      if constexpr (W2) {                                                                                      \
          _Pragma("unroll") for (int c = 0; c < C::PWL; ++c)                                                   \
              _Pragma("unroll") for (int d = 0; d < 4; ++d) {                                                  \
                  const u32x4 w8 = widen2(rw[c][d]);                                                           \
                  asm volatile("ds_store_b128 %0, %1 offset:%2" :: "v"(a_), "v"(w8), "i"(C::XB + (4 * c + d) * C::NT * 16) : "memory"); \
              }                                                                                                \
      } else {                                                                                                 \
          _Pragma("unroll") for (int c = 0; c < C::WCH; ++c)                                                   \
              asm volatile("ds_store_b128 %0, %1 offset:%2" :: "v"(a_), "v"(rw[c]), "i"(C::XB + c * C::NT * 16) : "memory"); \
      } }

    GLOAD(0)
    LSTORE(0)
    lds_barrier();

    constexpr int RSE = RS > 0 ? RS * (4 / KBN) : 1;   // stages per rescale segment (128*RS K)
    const int NSEG = RS > 0 ? S / RSE : 1;
    const int NST = RS > 0 ? RSE : S;   // stages per segment (RS==0: whole K in one segment)
    i32x8 acc[FT][FF];
    if constexpr (RS == 0) {
#pragma unroll
        for (int i = 0; i < FT; ++i)
#pragma unroll
            for (int j = 0; j < FF; ++j) acc[i][j] = i32x8{0, 0, 0, 0, 0, 0, 0, 0};
    }
    for (int seg = 0; seg < NSEG; ++seg) {
        float swv[FF];
#pragma unroll
        for (int j = 0; j < FF; ++j) swv[j] = __half2float(sw[(size_t)seg * F + fbase + 16 * j]);
        float sxa[PA ? FT : 1][8];
        if constexpr (PA) {
#pragma unroll
            for (int i = 0; i < FT; ++i) {
                const float4* p = (const float4*)(sx + (size_t)seg * Npad + tb * BT + wt * C::WT + i * 16 + 8 * (lane >> 4));
                const float4 v0 = p[0], v1 = p[1];
                sxa[i][0] = v0.x; sxa[i][1] = v0.y; sxa[i][2] = v0.z; sxa[i][3] = v0.w;
                sxa[i][4] = v1.x; sxa[i][5] = v1.y; sxa[i][6] = v1.z; sxa[i][7] = v1.w;
            }
        }
        for (int rr = 0; rr < NST; rr += (RS > 0 ? 1 : 1)) {   // RS>0: NST == RSE (small, unrolled below); RS==0: runtime loop
            if constexpr (RS > 0) {
#pragma unroll
                for (int r = 0; r < RSE; ++r) {
                    const int s = seg * RSE + r;
                    GLOAD(s + 1 < S ? s + 1 : s)
                    const uint4* b = lds + (s & 1) * (C::STAGE / 16);
#pragma unroll
                    for (int kb = 0; kb < KBN; ++kb) {
                        uint4 a[FT], w[FF];
#pragma unroll
                        for (int i = 0; i < FT; ++i) a[i] = b[((wt * FT + i) * KBN + kb) * 32 + lane];
#pragma unroll
                        for (int j = 0; j < FF; ++j) w[j] = b[C::XB / 16 + ((wf * FF + j) * KBN + kb) * 32 + lane];
#pragma unroll
                        for (int i = 0; i < FT; ++i)
#pragma unroll
                            for (int j = 0; j < FF; ++j) {
                                i32x8 c0;
                                if (r == 0 && kb == 0) c0 = i32x8{0, 0, 0, 0, 0, 0, 0, 0}; else c0 = acc[i][j];
                                c0 = wmma_iu8(i32x2{(int)a[i].x, (int)a[i].y}, i32x2{(int)w[j].x, (int)w[j].y}, c0);
                                acc[i][j] = wmma_iu8(i32x2{(int)a[i].z, (int)a[i].w}, i32x2{(int)w[j].z, (int)w[j].w}, c0);
                            }
                    }
                    {
#pragma unroll
                        for (int i = 0; i < FT; ++i)
#pragma unroll
                            for (int j = 0; j < FF; ++j) asm volatile("" : "+v"(acc[i][j]));
                        LSTORE((s + 1) & 1)
                        bar_signal();
                    }
                    if (r == RSE - 1) {   // segment done: rescale (cvt + fma per element) while the barrier is pending
#pragma unroll
                        for (int i = 0; i < FT; ++i)
#pragma unroll
                            for (int j = 0; j < FF; ++j) {
                                asm volatile("" : "+v"(acc[i][j]));
#pragma unroll
                                for (int l = 0; l < 8; ++l) {
                                    if constexpr (PA) accf[i][j][l] = __builtin_fmaf((float)acc[i][j][l] * sxa[i][l], swv[j], accf[i][j][l]);
                                    else              accf[i][j][l] = __builtin_fmaf((float)acc[i][j][l], swv[j], accf[i][j][l]);
                                }
#pragma unroll
                                for (int l = 0; l < 8; ++l) asm volatile("" : "+v"(accf[i][j][l]));
                            }
                    }
                    bar_wait();
                }
            } else {
                const int s = rr;
                GLOAD(s + 1 < S ? s + 1 : s)
                const uint4* b = lds + (s & 1) * (C::STAGE / 16);
#pragma unroll
                for (int kb = 0; kb < KBN; ++kb) {
                    uint4 a[FT], w[FF];
#pragma unroll
                    for (int i = 0; i < FT; ++i) a[i] = b[((wt * FT + i) * KBN + kb) * 32 + lane];
#pragma unroll
                    for (int j = 0; j < FF; ++j) w[j] = b[C::XB / 16 + ((wf * FF + j) * KBN + kb) * 32 + lane];
#pragma unroll
                    for (int i = 0; i < FT; ++i)
#pragma unroll
                        for (int j = 0; j < FF; ++j) {
                            i32x8 c0 = wmma_iu8(i32x2{(int)a[i].x, (int)a[i].y}, i32x2{(int)w[j].x, (int)w[j].y}, acc[i][j]);
                            acc[i][j] = wmma_iu8(i32x2{(int)a[i].z, (int)a[i].w}, i32x2{(int)w[j].z, (int)w[j].w}, c0);
                        }
                }
                {
#pragma unroll
                    for (int i = 0; i < FT; ++i)
#pragma unroll
                        for (int j = 0; j < FF; ++j) asm volatile("" : "+v"(acc[i][j]));
                    LSTORE((s + 1) & 1)
                    bar_signal();
                }
                bar_wait();
            }
            if constexpr (RS > 0) break;   // the unrolled r-loop above consumed the whole segment
        }
    }
    if constexpr (RS == 0) {   // whole-K: per-row weight scale, one cvt+mul at the end
#pragma unroll
        for (int j = 0; j < FF; ++j) {
            const float swj = ((const float*)swp)[fbase + 16 * j];   // engine copy: fp32 row scale
#pragma unroll
            for (int i = 0; i < FT; ++i)
#pragma unroll
                for (int l = 0; l < 8; ++l) accf[i][j][l] = (float)acc[i][j][l] * swj;
        }
    }

    // epilogue: per-token activation scale, store Y[t][f] (f contiguous)
#pragma unroll
    for (int i = 0; i < FT; ++i)
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int t = tb * BT + wt * C::WT + i * 16 + 8 * (lane >> 4) + l;
            if (t >= Ntok) continue;
            const float st = PA ? 1.0f : sx[t];
#pragma unroll
            for (int j = 0; j < FF; ++j) Y[(size_t)t * ldy + fbase + 16 * j] = accf[i][j][l] * st;
        }
}
}  // namespace dw
