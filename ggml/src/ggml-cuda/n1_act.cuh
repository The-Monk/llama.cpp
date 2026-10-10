#pragma once

// [TAG_ACT_FUSE] T412: the N1 activation layout, shared by the N1 GEMM (mul_mat_n1.cu) and the act-fuse producers
// (act-fuse.cu). Buffer = int8 X stream [Npad x K] in the tiled order of n1_off (LAYOUT.md section 1), followed by
// fp32 sx[K/128][Npad] (one scale per (128-K segment, token)). Npad = N rounded up to 128.

#include "common.cuh"

static __device__ __forceinline__ size_t n1_off(int r, int k, int S) {
    const int tile = r >> 7, s = k >> 7, ri = (r & 127) >> 4, kb = (k & 127) >> 5, h = (k & 31) >> 4, e = k & 15;
    const int lane = 16 * h + (r & 15);
    return (((((size_t) tile * S + s) * 8 + ri) * 4 + kb) * 32 + lane) * 16 + e;
}

// The one int8 quantization of 4 values with scale d (d > 0), shared by k_n1_quant_act and n1_act_store so their
// bytes are identical: the HIP build uses -funsafe-math-optimizations, under which `x / d` may compile to a true
// divide in one kernel and a reciprocal multiply in another (seen: ~6e-7 of bytes differ). An explicit reciprocal
// multiply compiles the same way everywhere.
static __device__ __forceinline__ float n1_scale(const float amax) { return amax * (1.0f / 127.0f); }

static __device__ __forceinline__ uint32_t n1_quant4(const float4 v, const float d) {
    const float id = 1.0f / d;
    const int q0 = (int) roundf(v.x * id), q1 = (int) roundf(v.y * id), q2 = (int) roundf(v.z * id), q3 = (int) roundf(v.w * id);
    return (uint32_t) (uint8_t) (int8_t) q0 | ((uint32_t) (uint8_t) (int8_t) q1 << 8) |
           ((uint32_t) (uint8_t) (int8_t) q2 << 16) | ((uint32_t) (uint8_t) (int8_t) q3 << 24);
}

static inline int64_t n1_npad(int64_t n) { return (n + 127) / 128 * 128; }

static inline size_t n1_act_bytes(int64_t K, int64_t N) {
    const int64_t Npad = n1_npad(N);
    return (size_t) Npad * K + (size_t) (K / 128) * Npad * sizeof(float);
}

// Store functor with the act-fuse warp-collective contract: called by every lane of a warp with 4 consecutive
// values, lane L holding columns [4L, 4L+4) of a 128-aligned span (one N1 segment). Bytes equal k_n1_quant_act
// (G = 128) for the same fp32 values: d = amax/127, n1_quant4, d == 0 -> zeros.
struct n1_act_store {
    int8_t * X;
    float  * sx;
    int      K;
    int      S;
    int      Npad;

    __device__ __forceinline__ void operator()(const int64_t i1, const int64_t i0, const float4 v) const {
        if (i0 >= K) {   // producers run to GGML_PAD(K, 512); K % 128 == 0, so whole warps return together
            return;
        }
        float amax = 0.f;
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, o, 32));
        }
        const float d = n1_scale(amax);
        const int   t = (int) i1;
        if ((i0 & 127) == 0) {
            sx[(size_t) (i0 >> 7) * Npad + t] = d;
        }
        *(uint32_t *) (X + n1_off(t, (int) i0, S)) = d > 0.f ? n1_quant4(v, d) : 0u;
    }
};

// T422 M2: per-256 activation scale (layout GGML_CUDA_ACT_LAYOUT_N1G256, same buffer shape as N1). Called like
// n1_act_store plus `vp` = the 4 values 128 columns away (column c ^ 128, the other half of the 256-group), which the
// producer computes with the same expression; the warp's amax then covers the whole group, so both warps of a group
// derive the same d without any cross-warp exchange. Each warp writes its own 128-slot of sx (both slots = d), X gets
// only the warp's own values. Bytes equal k_n1_quant_act256 for the same fp32 values.
struct n1_act_store256 {
    int8_t * X;
    float  * sx;
    int      K;
    int      S;
    int      Npad;

    __device__ __forceinline__ void operator()(const int64_t i1, const int64_t i0, const float4 v, const float4 vp) const {
        if (i0 >= K) {
            return;
        }
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(vp.x), fabsf(vp.y)), fmaxf(fabsf(vp.z), fabsf(vp.w))));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, o, 32));
        }
        const float d = n1_scale(amax);
        const int   t = (int) i1;
        if ((i0 & 127) == 0) {
            sx[(size_t) (i0 >> 7) * Npad + t] = d;
        }
        *(uint32_t *) (X + n1_off(t, (int) i0, S)) = d > 0.f ? n1_quant4(v, d) : 0u;
    }
};

static inline n1_act_store256 n1_act_store256_make(void * y, int64_t K, int64_t N) {
    const int64_t Npad = n1_npad(N);
    n1_act_store256 s;
    s.X    = (int8_t *) y;
    s.sx   = (float *) ((char *) y + (size_t) Npad * K);
    s.K    = (int) K;
    s.S    = (int) (K / 128);
    s.Npad = (int) Npad;
    return s;
}

static inline n1_act_store n1_act_store_make(void * y, int64_t K, int64_t N) {
    const int64_t Npad = n1_npad(N);
    n1_act_store s;
    s.X    = (int8_t *) y;
    s.sx   = (float *) ((char *) y + (size_t) Npad * K);
    s.K    = (int) K;
    s.S    = (int) (K / 128);
    s.Npad = (int) Npad;
    return s;
}
