#pragma once
#include "common.cuh"

// T430 GGML_CUDA_FWHT_QUANT device half: one warp holds a 1024-element row as 32 contiguous elements per lane
// (lane j = elements j*32..j*32+31 = exactly one q8_1 block). Runs the Walsh-Hadamard butterfly network in the
// order fwht_cuda does (h = 1, 2, 4, ..., 512: the low 5 index bits are in registers, the high 5 across lanes;
// only the element->thread mapping differs, so the result is bit-identical), stores the transformed row to
// `dst` and writes the Q2_FIELD-layout q8_1 block of this lane to `yb` -- the bytes
// quantize_q8_1<Q8_1_LAYOUT_Q2_FIELD> writes reading `dst` back. The block sum uses warp_reduce_sum's
// xor-tree association (offsets 16, 8, 4, 2, 1), so ds.y is bit-identical too.
static __device__ __forceinline__ void ggml_cuda_fwht1024_q8_1_lane(float (&reg)[QK8_1], const int lane, float * dst, block_q8_1 * yb) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == QK8_1, "one q8_1 block per lane");

    // stages h = 1..16: in-register
#pragma unroll
    for (int h = 1; h < QK8_1; h *= 2) {
#pragma unroll
        for (int j = 0; j < QK8_1; j += 2*h) {
#pragma unroll
            for (int k = 0; k < h; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + h];
                reg[j + k]     = x + y;
                reg[j + k + h] = x - y;
            }
        }
    }
    // stages h = 32..512: across lanes (lane bit = h/32)
    // (lane bits 1..8 stay inside a 16-lane row: DPP row_xmask is a plain VALU operand, no LDS round trip)
#define GGML_FWHT_LANE_STAGE(B, XOR_EXPR) \
    _Pragma("unroll") \
    for (int j = 0; j < QK8_1; ++j) { \
        const float val  = reg[j]; \
        const float val2 = XOR_EXPR; \
        reg[j] = (lane & (B)) == 0 ? val + val2 : val2 - val; \
    }
#if defined(GGML_USE_HIP) && defined(__gfx1201__)
    GGML_FWHT_LANE_STAGE(1,  dpp_row_xmask_f32<1>(val))
    GGML_FWHT_LANE_STAGE(2,  dpp_row_xmask_f32<2>(val))
    GGML_FWHT_LANE_STAGE(4,  dpp_row_xmask_f32<4>(val))
    GGML_FWHT_LANE_STAGE(8,  dpp_row_xmask_f32<8>(val))
#else
    GGML_FWHT_LANE_STAGE(1,  __shfl_xor_sync(0xFFFFFFFF, val, 1, warp_size))
    GGML_FWHT_LANE_STAGE(2,  __shfl_xor_sync(0xFFFFFFFF, val, 2, warp_size))
    GGML_FWHT_LANE_STAGE(4,  __shfl_xor_sync(0xFFFFFFFF, val, 4, warp_size))
    GGML_FWHT_LANE_STAGE(8,  __shfl_xor_sync(0xFFFFFFFF, val, 8, warp_size))
#endif
    GGML_FWHT_LANE_STAGE(16, __shfl_xor_sync(0xFFFFFFFF, val, 16, warp_size))
#undef GGML_FWHT_LANE_STAGE

    float4 * d4 = (float4 *) dst;
#pragma unroll
    for (int k = 0; k < QK8_1/4; ++k) {
        d4[k] = make_float4(reg[4*k], reg[4*k + 1], reg[4*k + 2], reg[4*k + 3]);
    }

    // The block sum must add in warp_reduce_sum's exact tree. The build uses -funsafe-math-optimizations, which would
    // re-associate a chain of plain adds (and fold them into the butterfly above), so every operand is made opaque.
    float xq[QK8_1];
#pragma unroll
    for (int e = 0; e < QK8_1; ++e) {
        xq[e] = reg[e];
        asm volatile("" : "+v"(xq[e]));
    }
    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < QK8_1; ++e) {
        amax = fmaxf(amax, fabsf(xq[e]));
    }
    float t[QK8_1/2];
#pragma unroll
    for (int e = 0; e < QK8_1/2; ++e) {
        t[e] = xq[e] + xq[e ^ 16];
        asm volatile("" : "+v"(t[e]));
    }
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
#pragma unroll
        for (int e = 0; e < off; ++e) {
            t[e] = t[e] + t[e ^ off];
            asm volatile("" : "+v"(t[e]));
        }
    }
    const float sum = t[0];

    // quantize_q8_1 compiles `d = amax / 127.0f; q = roundf(xi / d)` (one division per thread, -funsafe-math-optimizations)
    // to d = amax * (1/127), q = roundf(xi * rcp(d)). Written out explicitly, behind opaque values, so the 32 divisions
    // sharing one d cannot be re-expressed differently here (that moved d across a half-precision rounding tie).
    float amax_o = amax;
    asm volatile("" : "+v"(amax_o));
    float d = amax_o / 127.0f;
    asm volatile("" : "+v"(d));
    const float rd = __builtin_amdgcn_rcpf(d);
    uint32_t * qw = (uint32_t *) yb->qs;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            uint32_t w = 0;
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const float  xi = xq[16*h + 4*g + f];
                const int8_t q  = amax == 0.0f ? 0 : roundf(xi * rd);
                w |= (uint32_t) (uint8_t) q << (8*g);
            }
            qw[4*h + f] = w;
        }
    }
    yb->ds = make_half2(d, sum);
}
