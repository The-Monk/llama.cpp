#pragma once

// [TAG_ACT_FUSE] T399: the single-copy (NK_Q2_0_W2ONLY) decode / small-batch activation layout, shared by the SC
// kernels (mul_mat_sc.cu) and the act-fuse producers (act-fuse.cu).
// Buffer = int8 xq[NP][K] (natural k order, token n at n*K) followed by half2 ds[K/32][NP] = (d8, s8) per 32-chunk,
// token-contiguous; tokens N..NP-1 are zero. NP = N for the GEMV (N <= 7), N rounded up to 16 for the WMMA kernel.
// The arithmetic is ggml quantize_q8_1's per-32 scheme with N1's explicit reciprocal: d = amax*(1/127), q =
// round(x * rcp(d)), d8 = half(d), s8 = half(sum x) in the 32-lane xor butterfly order (16, 8, 4, 2, 1); producers and
// the unfused k_sc_quant write identical bytes. (ggml divides x / d: a q may differ by 1 in rare ulp cases.)

#include "common.cuh"

#define SC_GEMV_MAX_N 7

static inline int64_t sc_np(int64_t n) { return n <= SC_GEMV_MAX_N ? n : (n + 15) / 16 * 16; }

static inline size_t sc_act_bytes(int64_t K, int64_t N) {
    const int64_t NP = sc_np(N);
    return (size_t) NP * K + (size_t) (K / 32) * NP * 4;
}

// The one quantization of the layout, shared by k_sc_quant and sc_act_store so their bytes are identical. As in
// n1_act.cuh: under -funsafe-math-optimizations `x / d` (and __fdiv_rn, which is `x / d` in HIP) may compile to a true
// divide in one kernel and a reciprocal multiply in another (measured here: 123..1269 of 1e7-1e8 bytes differed); an
// explicit reciprocal multiply compiles the same way everywhere.
// The reciprocal is the hardware v_rcp_f32 (no divide for the compiler to choose a lowering for) and the partial sums
// go through an opaque register barrier so -fassociative-math cannot re-pair them differently per kernel.
static __device__ __forceinline__ float sc_scale(const float amax) { return amax * (1.0f / 127.0f); }
static __device__ __forceinline__ float sc_inv(const float d) { return __builtin_amdgcn_rcpf(d); }
static __device__ __forceinline__ int sc_q(const float x, const float id) { return (int) roundf(x * id); }
static __device__ __forceinline__ float sc_pin(float x) { asm volatile("" : "+v"(x)); return x; }

// Store functor with the act-fuse warp-collective contract: every lane of a warp calls it with 4 consecutive values,
// lane L holding columns [4L, 4L+4) of a 128-aligned span, so a 32-chunk is lanes 8c..8c+7 (element e = 4*(L&7) + i).
// The sum reproduces the 32-lane butterfly: per component, xor 4/2/1 across lanes = element xor 16/8/4, then
// (c0 + c2) + (c1 + c3) = element xor 2, xor 1. amax is order-free.
struct sc_act_store {
    int8_t  * xq;
    __half2 * ds;
    int       K;
    int       NP;

    __device__ __forceinline__ void operator()(const int64_t i1, const int64_t i0, const float4 v) const {
        if (i0 >= K) {   // producers run to GGML_PAD(K, 512); K % 128 == 0, so whole warps return together
            return;
        }
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        float c[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, o, 32));
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                c[i] = sc_pin(c[i] + __shfl_xor_sync(0xFFFFFFFF, c[i], o, 32));
            }
        }
        const float sum = sc_pin(sc_pin(c[0] + c[2]) + sc_pin(c[1] + c[3]));
        const float d   = sc_scale(amax);
        const float id  = amax == 0.0f ? 0.0f : sc_inv(d);
        const float x[4] = {v.x, v.y, v.z, v.w};
        uint32_t p = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = amax == 0.0f ? 0 : sc_q(x[i], id);
            p |= (uint32_t) (uint8_t) (int8_t) q << (8 * i);
        }
        const int t = (int) i1;
        *(uint32_t *) (xq + (size_t) t * K + i0) = p;
        if ((i0 & 31) == 0) {
            ds[(size_t) (i0 >> 5) * NP + t] = __halves2half2(__float2half(d), __float2half(sum));
        }
    }
};

static inline sc_act_store sc_act_store_make(void * y, int64_t K, int64_t N) {
    const int64_t NP = sc_np(N);
    sc_act_store s;
    s.xq = (int8_t *) y;
    s.ds = (__half2 *) ((char *) y + (size_t) NP * K);
    s.K  = (int) K;
    s.NP = (int) NP;
    return s;
}
