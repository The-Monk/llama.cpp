// T420 M1: N1 g128 GEMM with X direct-to-VGPR ("XD"). Engine copy of native-kernels/dense_wmma/m1/m1_xd.cuh
// (gemm_xd<BIAS=1>), certified bit-identical to n1_gemm_iu8.cuh's gemm_iu8<128,128,4,2,1,1,FL=9> on every output
// element (same compiler flags) and +13% TOPS standalone on wrx90.
//
// Same operands, layouts and math as the g128 arm (W2 2-bit weights + fp16 sw[S][F], per-(128-K, token) activation
// scale sx[S][Npad]); differences are structural only:
//   * the W fragment is the WMMA A operand and the X fragment the B operand, so the accumulator lane = token and
//     element = feature (activation scale = one VGPR per token fragment, output stored as 8 contiguous floats per lane);
//   * X fragments go global -> VGPR directly (each wave's fragment is one contiguous 512 B of the N1 stream), so the
//     LDS holds only the widened W: 2 x 16 KB per workgroup instead of 2 x 32 KB;
//   * accumulators start each 128-K segment at 0x4B400000 (1.5 * 2^23 as fp32): the exact int32 segment sum is
//     float_bits(acc) - 12582912.0f, exact while |sum| < 2^22 (here |sum| <= 128 * 127 * 2 = 32512), replacing
//     v_cvt_f32_i32 by a VOPD-pairable v_sub_f32. The rescale keeps the release op order: fma(x * sx, sw, accf).
// Every global load is inline asm, so every s_wait_loadcnt is explicit (loadcnt retires in order):
//   per stage, issue order: scales(s) [FF sw + FT sx], W(s+1) [1], then X(s+1, kb) [FT] after each k-block kb.
//   before k-block kb reads X(s, kb): newer loads = 3 FT + FF + FT + 1 (constant); before the W commit: 4 FT.
#pragma once
#include "n1_gemm_iu8.cuh"

namespace dw {

constexpr float XD_BIASF = 12582912.0f;   // 1.5 * 2^23
constexpr int   XD_BIASI = 0x4B400000;

template <int KT = 0>
__global__ __launch_bounds__(256, 1)
void gemm_xd(const uint4 * __restrict__ Xt, const uint4 * __restrict__ Wt, const void * __restrict__ swp,
             const float * __restrict__ sx, float * __restrict__ Y, int Ntok, int F, int K, int ldy) {
    constexpr int WTn = 4, WFn = 2, NT = 256, BT = 128, BF = 128;
    constexpr int WT = BT / WTn, WF = BF / WFn, FT = WT / 16, FF = WF / 16, KBN = 4;
    constexpr int XB = BT * 128, WB = BF * 128;
    constexpr int WAIT_X = 3 * FT + FF + FT + 1;
    constexpr int WAIT_W = 4 * FT;
    __shared__ uint4 lds[2 * WB / 16];

    const __half * __restrict__ sw = (const __half *) swp;
    const int Npad = gridDim.x * BT;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wt = wave % WTn, wf = wave / WTn;
    const int tb = blockIdx.x, fb = blockIdx.y;
    const int S = K / 128;
    const char * xg = (const char *) (Xt + (size_t) tb * S * (XB / 16));
    const char * wg = (const char *) (Wt + (size_t) fb * S * (WB / 64));
    const int tok0 = tb * BT + wt * WT, fea0 = fb * BF + wf * WF;

    float accf[FT][FF][8];
#pragma unroll
    for (int i = 0; i < FT; ++i)
#pragma unroll
        for (int j = 0; j < FF; ++j)
#pragma unroll
            for (int l = 0; l < 8; ++l) accf[i][j][l] = 0.f;

    u32x4 xr[KBN][FT], rw;
    uint32_t swh[FF][4];   // 8 fp16 weight scales (features 8*(lane>>4)..+8 of fragment j)
    float sxa[FT];         // activation scale of this lane's token, per token fragment
    const uint32_t xvoff  = (uint32_t) ((wt * FT * KBN * 32 + lane) * 16);
    const uint32_t wvoff  = (uint32_t) tid * 16u;
    const uint32_t swvoff = (uint32_t) ((fea0 + 8 * (lane >> 4)) * 2);
    const uint32_t sxvoff = (uint32_t) ((tok0 + (lane & 15)) * 4);
    const uint32_t lds_base = (uint32_t) (uintptr_t) (__attribute__((address_space(3))) char *) lds + (uint32_t) tid * 16u;

#define XD_XLOAD(stage, kb)                                                                                  \
    { const char * xb_ = xg + (size_t) (stage) * XB;                                                           \
      _Pragma("unroll") for (int i = 0; i < FT; ++i)                                                           \
          asm volatile("global_load_b128 %0, %1, %2 offset:%3" : "=v"(xr[kb][i]) : "v"(xvoff), "s"(xb_), "i"((i * KBN + (kb)) * 512) : "memory"); }
#define XD_WLOAD(stage)                                                                                      \
    { const char * wb_ = wg + (size_t) (stage) * (WB / 4);                                                     \
      asm volatile("global_load_b128 %0, %1, %2 offset:0" : "=v"(rw) : "v"(wvoff), "s"(wb_) : "memory"); }
#define XD_SLOAD(seg)                                                                                        \
    { const char * swb_ = (const char *) (sw + (size_t) (seg) * F);                                            \
      const char * sxb_ = (const char *) (sx + (size_t) (seg) * Npad);                                         \
      _Pragma("unroll") for (int j = 0; j < FF; ++j) {                                                         \
          u32x4 t_;                                                                                            \
          asm volatile("global_load_b128 %0, %1, %2 offset:%3" : "=v"(t_) : "v"(swvoff), "s"(swb_), "i"(j * 32) : "memory"); \
          swh[j][0] = t_.x; swh[j][1] = t_.y; swh[j][2] = t_.z; swh[j][3] = t_.w; }                            \
      _Pragma("unroll") for (int i = 0; i < FT; ++i)                                                           \
          asm volatile("global_load_b32 %0, %1, %2 offset:%3" : "=v"(sxa[i]) : "v"(sxvoff), "s"(sxb_), "i"(i * 64) : "memory"); }
#define XD_WCOMMIT(buf, waitn)                                                                               \
    { const uint32_t a_ = lds_base + (uint32_t) (buf) * WB;                                                    \
      asm volatile("s_wait_loadcnt %0" :: "i"(waitn) : "memory");                                              \
      asm volatile("" : "+v"(rw));                                                                             \
      _Pragma("unroll") for (int d = 0; d < 4; ++d) {                                                          \
          const u32x4 w8 = widen2(rw[d]);                                                                      \
          asm volatile("ds_store_b128 %0, %1 offset:%2" :: "v"(a_), "v"(w8), "i"(d * NT * 16) : "memory");     \
      } }

    XD_WLOAD(0)
    XD_WCOMMIT(0, 0)
#pragma unroll
    for (int kb = 0; kb < KBN; ++kb) XD_XLOAD(0, kb)
    lds_barrier();

    const i32x8 bias8 = i32x8{XD_BIASI, XD_BIASI, XD_BIASI, XD_BIASI, XD_BIASI, XD_BIASI, XD_BIASI, XD_BIASI};
    i32x8 acc[FT][FF];
    for (int s = 0; s < S; ++s) {
        XD_SLOAD(s)
        const int sn = s + 1 < S ? s + 1 : s;
        XD_WLOAD(sn)
        const uint4 * b = lds + (s & 1) * (WB / 16);
#pragma unroll
        for (int kb = 0; kb < KBN; ++kb) {
            uint4 w[FF];
#pragma unroll
            for (int j = 0; j < FF; ++j) w[j] = b[((wf * FF + j) * KBN + kb) * 32 + lane];
            asm volatile("s_wait_loadcnt %0" :: "i"(WAIT_X) : "memory");
#pragma unroll
            for (int i = 0; i < FT; ++i) asm volatile("" : "+v"(xr[kb][i]));
#pragma unroll
            for (int i = 0; i < FT; ++i)
#pragma unroll
                for (int j = 0; j < FF; ++j) {
                    i32x8 c0 = (kb == 0) ? bias8 : acc[i][j];
                    const i32x2 x0{(int) xr[kb][i].x, (int) xr[kb][i].y}, x1{(int) xr[kb][i].z, (int) xr[kb][i].w};
                    const i32x2 w0{(int) w[j].x, (int) w[j].y}, w1{(int) w[j].z, (int) w[j].w};
                    c0 = wmma_iu8(w0, x0, c0);
                    acc[i][j] = wmma_iu8(w1, x1, c0);
                }
            __builtin_amdgcn_sched_barrier(0);   // the reload below must not move above this k-block's WMMAs
            XD_XLOAD(sn, kb)
        }
        {
#pragma unroll
            for (int i = 0; i < FT; ++i)
#pragma unroll
                for (int j = 0; j < FF; ++j) asm volatile("" : "+v"(acc[i][j]));
            XD_WCOMMIT((s + 1) & 1, WAIT_W)
            // scales of segment s are older than W(s+1): retired by the wait above
#pragma unroll
            for (int j = 0; j < FF; ++j)
#pragma unroll
                for (int u = 0; u < 4; ++u) asm volatile("" : "+v"(swh[j][u]));
#pragma unroll
            for (int i = 0; i < FT; ++i) asm volatile("" : "+v"(sxa[i]));
            bar_signal();
        }
#pragma unroll
        for (int i = 0; i < FT; ++i)
#pragma unroll
            for (int j = 0; j < FF; ++j) {
                asm volatile("" : "+v"(acc[i][j]));
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    // copy the element to a scalar first: __builtin_bit_cast on a vector-element lvalue read element 0
                    // for every l (hipcc 7.14, T420)
                    const int ai = acc[i][j][l];
                    float x = __int_as_float(ai) - XD_BIASF;
                    asm volatile("" : "+v"(x));   // -funsafe-math: never re-associate into bits*sx - B*sx
                    const float swl = __half2float(__ushort_as_half((unsigned short) (swh[j][l >> 1] >> (16 * (l & 1)))));
                    accf[i][j][l] = __builtin_fmaf(x * sxa[i], swl, accf[i][j][l]);
                }
#pragma unroll
                for (int l = 0; l < 8; ++l) asm volatile("" : "+v"(accf[i][j][l]));
            }
        bar_wait();
    }
    asm volatile("s_wait_loadcnt 0" ::: "memory");   // drain the dummy prefetch of the last stage
#undef XD_XLOAD
#undef XD_WLOAD
#undef XD_SLOAD
#undef XD_WCOMMIT

#pragma unroll
    for (int i = 0; i < FT; ++i) {
        const int t = tok0 + i * 16 + (lane & 15);
        if (t >= Ntok) continue;
#pragma unroll
        for (int j = 0; j < FF; ++j) {
            const int f0 = fea0 + 16 * j + 8 * (lane >> 4);
            float4 * yp = (float4 *) (Y + (size_t) t * ldy + f0);
            yp[0] = make_float4(accf[i][j][0], accf[i][j][1], accf[i][j][2], accf[i][j][3]);
            yp[1] = make_float4(accf[i][j][4], accf[i][j][5], accf[i][j][6], accf[i][j][7]);
        }
    }
}
}  // namespace dw
