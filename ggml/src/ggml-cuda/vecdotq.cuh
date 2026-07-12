#pragma once

#include "common.cuh"

#include <cstdint>

static __device__ __forceinline__ int get_int_b1(const void * x, const int & i32) {
    const uint8_t * x8 = (const uint8_t *) x;

    int x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    return x32;
}

static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

// q4 contains 8 indices with 4 bit each.
// This function selects those bytes from table that are at those indices and returns them as int2.
// The first int contains the bytes with even indices in q4, the second int contains the bytes with odd indices in q4.
static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table) {
#if defined(GGML_USE_HIP)
    // Load the 16-byte table into four 32-bit unsigned integers.
    const uint32_t *values = (const uint32_t *)table;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    // Perform lookups in the lower half of the table (indices 0-7).
    uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707);

    // Perform lookups in the upper half of the table (indices 8-15).
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], q_odd & 0x07070707);

    // Select between the low and high results based on the MSB of each index nibble.
    uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    uint32_t res_x = __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even);
    uint32_t mask_odd = 0x03020100 | ((q_odd & 0x08080808) >> 1);
    uint32_t res_y = __builtin_amdgcn_perm(v_odd_high, v_odd_low, mask_odd);

    return make_int2(res_x, res_y);
#elif !defined(GGML_USE_MUSA)
    // CUDA does not have an instruction for selecting bytes with 4 bit indices.
    // However, __byte_perm is an instruction that selects bytes with 3 bit indices that can be used instead.
    const uint32_t * table32 = (const uint32_t *) table;

    // __byte_perm selects bytes based on the lower 16 bits in its third argument.
    // Therefore, do 2 iterations over the 32 bits in q4 with 0 and 16 shift.
    // To handle the fourth bit, first call _byte_perm both for the low and the high 64 bit of table, using the low 3 bits.
    // Then, call __byte_perm again to select from the low and high bytes based on the fourth bit.
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;

        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }

    // tmp contains the bytes from tyble in the same order as the 4 bit indices in q4.
    // However, for the result we need ints with all even/odd 4 bit indices in q4.
    // Therefore, 2 more calls to __byte_perm to put the bytes in the correct order.
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
#else
    // Generic implementation.
    const int      q0_32  = (q4 >> 0) & 0x0F0F0F0F;
    const int8_t * q0_8   = (const int8_t *) &q0_32;
    const char4    val0_8 = make_char4(
        table[q0_8[0]], table[q0_8[1]], table[q0_8[2]], table[q0_8[3]]);

    const int      q1_32  = (q4 >> 4) & 0x0F0F0F0F;
    const int8_t * q1_8   = (const int8_t *) &q1_32;
    const char4    val1_8 = make_char4(
        table[q1_8[0]], table[q1_8[1]], table[q1_8[2]], table[q1_8[3]]);

    return make_int2(*((const int *) &val0_8), *((const int *) &val1_8));
#endif
}

static __device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    // v is a 7 bit int, with the 8th sign being encodable as popcnt
    // with xor we can "correct" the bit instead of having to mask
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    // broadcast over uint to allow for 0x08040201 / 0x80402010 as selectors
    return s * 0x01010101;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

#define VDR_Q1_0_Q8_1_MMVQ 1  // Process one 32-element chunk at a time for parallelism
#define VDR_Q1_0_Q8_1_MMQ  4  // Q1_0 has 128 bits (4 ints) per block
#define VDR_Q2_0_Q8_1_MMVQ 1  // Process one 32-element chunk at a time
#define VDR_Q2_0_Q8_1_MMQ  4  // Q2_0 has 256 bits (8 ints) per block, 4 32-element chunks

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(__hmul2(dm4, ds8));
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;
#else
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
#endif // FAST_FP16_AVAILABLE

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(
    const int * vl, const int * vh, const int * u, const half2 & dm5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(__hmul2(dm5, ds8));
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;
#else
    const float2 dm5f = __half22float2(dm5);
    const float2 ds8f = __half22float2(ds8);
    const float d5d8 = dm5f.x * ds8f.x;
    const float m5s8 = dm5f.y * ds8f.y;
#endif // FAST_FP16_AVAILABLE

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <typename T, int vdr> static __device__ __forceinline__ T vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const T & d8_0, const T & d8_1) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

    return d8_0*d8_1 * ((T) sumi);
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm8, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(__hmul2(dm8, ds8));
    const float d8d8 = tmp.x;
    const float m8s8 = tmp.y;
#else
    const float2 dm8f = __half22float2(dm8);
    const float2 ds8f = __half22float2(ds8);
    const float d8d8 = dm8f.x * ds8f.x;
    const float m8s8 = dm8f.y * ds8f.y;
#endif // FAST_FP16_AVAILABLE

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_0_16_q8_1_impl(
    const int * v, const int * u, const float * d8_0, const float & d8_1) {

    float sumf = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < vdr; i0 += QI8_0/2) {
        int sumi = 0;

#pragma unroll
        for (int i = i0; i < i0 + QI8_0/2; ++i) {
            // SIMD dot product of quantized values
            sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
        }

        sumf += d8_0[i0/(QI8_0/2)]*sumi;
    }

    return d8_1*sumf;
}

#define VDR_MXFP4_Q8_1_MMVQ 2
#define VDR_MXFP4_Q8_1_MMQ  4

// NOTE (ROC8 fp4-hw-cvt investigation, gfx1201): this path already avoids a
// software fp4->f16 dequant -- get_int_from_table_16() is a 4-bit->int8 LUT
// that feeds ggml_cuda_dp4a() directly (int8 SIMD dot), no f16 conversion
// step exists here to replace. The RDNA4-native alternative considered was
// AMD's hardware fp4 decoder, __builtin_amdgcn_cvt_scalef32_pk(8)_fp4_f16
// (packs 2 or 8 fp4 lanes -> f16 with a scale multiply). CONFIRMED NOT
// PRESENT on gfx1201/RDNA4 silicon: the RDNA4 ISA manual (doc 70651) defines
// no FP4 datatype and no CVT_*_FP4 instruction anywhere in its 697 pages
// (only F8/BF8 conversions exist, see CVT_PK_FP8_F32 etc.); clang requires
// target feature `fp4-cvt-scale-insts` (2-lane pk_ form) / `gfx1250-insts`
// (8-lane pk8_ form) for these builtins, and gfx1201 does not carry either
// by default. Force-enabling `+fp4-cvt-scale-insts` for --offload-arch=
// gfx1201 crashes the LLVM backend at codegen (SIInstrInfo::getInstSizeIn
// Bytes assert in BranchRelaxation) -- there is no instruction encoding for
// gfx12 at all, only for gfx1250 (a distinct, newer target not present in
// this box's device inventory; same class of finding as the wide-K fp8 WMMA
// dead end, see wiki/tech/widek-fp8-wmma-notes.md). Do not re-attempt this
// wiring on gfx1201 without new silicon; the LUT+dp4a path above is already
// the fast route and was NOT changed. See wiki/tech/rdna4-isa-optimization
// -audit.md for the full writeup.
static __device__ __forceinline__ float vec_dot_mxfp4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_mxfp4 * bq4 = (const block_mxfp4 *) vbq + kbx;

    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b1(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);

        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }

    const float d = ggml_cuda_e8m0_to_fp32(bq4->e) * 0.5f * __low2float(bq8_1->ds);
    return d * sumi;
}

#define VDR_NVFP4_Q8_1_MMVQ 4
#define VDR_NVFP4_Q8_1_MMQ  8

// NOTE (ROC8 fp4-hw-cvt investigation, gfx1201): same finding as
// vec_dot_mxfp4_q8_1 above -- this path is also LUT(get_int_from_table_16)
// + dp4a, no f16 dequant to accelerate, and the candidate hardware
// instruction (cvt_scalef32_pk(8)_fp4_f16) is gfx1250-exclusive, absent on
// RDNA4/gfx1201 (verified via ISA manual, clang target-feature gate, and an
// LLVM backend crash when force-enabled -- see the mxfp4 comment above for
// full detail). Not wired here for the same reason.
static __device__ __forceinline__ float vec_dot_nvfp4_q8_1(
                                        const void * __restrict__ vbq,
                                        const block_q8_1 * __restrict__ bq8_1,
                                        const int32_t & kbx,
                                        const int32_t & iqs) {

    const block_nvfp4 * bq4 = (const block_nvfp4 *) vbq + kbx;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; i++) {
        const int32_t iqs0 = iqs + 2*i;
        const int32_t iqs1 = iqs0 + 1;
        const int32_t is = iqs0 >> 1;
        const int2 v0 = get_int_from_table_16(get_int_b4(bq4->qs, iqs0), kvalues_mxfp4);
        const int2 v1 = get_int_from_table_16(get_int_b4(bq4->qs, iqs1), kvalues_mxfp4);
        const block_q8_1 * bq8 = bq8_1 + (is >> 1);
        const int32_t i8 = ((is & 1) << 2);

        int sumi = ggml_cuda_dp4a(v0.x, get_int_b4(bq8->qs, i8 + 0), 0);
        sumi = ggml_cuda_dp4a(v0.y, get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi = ggml_cuda_dp4a(v1.x, get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi = ggml_cuda_dp4a(v1.y, get_int_b4(bq8->qs, i8 + 3), sumi);

        const float d = ggml_cuda_ue4m3_to_fp32(bq4->d[is]) * __low2float(bq8->ds);
        sum += d * float(sumi);
    }

    return sum;
}
#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q2_K_Q8_1_MMQ  4

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

// contiguous v/x + u/y values
template <int ns8>
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const half2 * dm2, const float & d8, const half2 * s8) {

    float sumf    = 0.0f;
    float sumf_d8 = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < QR2_K*VDR_Q2_K_Q8_1_MMQ; i0 += QI8_1) {
        const float2 dm2f0 = __half22float2(dm2[i0/(QI8_1/2) + 0]);
        int sumi_d0 = 0;

        const float2 dm2f1 = __half22float2(dm2[i0/(QI8_1/2) + 1]);
        int sumi_d1 = 0;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d0 = ggml_cuda_dp4a(v[i], u[i], sumi_d0);
        }
        sumf_d8 += dm2f0.x * sumi_d0;

#pragma unroll
        for (int i = i0 + QI8_1/2; i < i0 + QI8_1; ++i) {
            sumi_d1 = ggml_cuda_dp4a(v[i], u[i], sumi_d1);
        }
        sumf_d8 += dm2f1.x * sumi_d1;

        if (i0/QI8_1 < ns8) {
            const float2 s8f = __half22float2(s8[i0/QI8_1]);
            sumf -= dm2f0.y*s8f.x;
            sumf -= dm2f1.y*s8f.y;
        } else {
            int sumi_m0 = 0;
#pragma unroll
            for (int i = i0; i < i0 + QI8_1/2; ++i) {
                sumi_m0 = ggml_cuda_dp4a(0x01010101, u[i], sumi_m0);
            }
            sumf_d8 -= dm2f0.y * sumi_m0;

            int sumi_m1 = 0;
#pragma unroll
            for (int i = i0 + QI8_1/2; i < i0 + QI8_1; ++i) {
                sumi_m1 = ggml_cuda_dp4a(0x01010101, u[i], sumi_m1);
            }
            sumf_d8 -= dm2f1.y * sumi_m1;
        }
    }

    return sumf + d8*sumf_d8;
}

#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const int & scale_offset, const float & d3, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;

        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2*i)) & 0x03030303;

        const int vih = ((vh >> i) << 2) & 0x04040404;

        const int vi = __vsubss4(vil, vih);

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d3 * sumf;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d3, const float & d8) {

    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        int sumi_sc = 0;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = ggml_cuda_dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    return d3*d8 * sumi;
}

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = ggml_cuda_dp4a((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = ggml_cuda_dp4a(v[i*QI8_1 + j], u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ sc,
    const float & d6, const float * __restrict__ d8) {

    float sumf_d = 0.0f;

    const int      sc_packed = get_int_b4(sc, 0);
    const int8_t * sc_reg    = (const int8_t *) &sc_packed;

#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x = ggml_cuda_dp4a(v[2*i+0], u[2*i+0], sumi_d.x); // SIMD dot product
            sumi_d.x = ggml_cuda_dp4a(v[2*i+1], u[2*i+1], sumi_d.x); // SIMD dot product

            sumi_d.y = ggml_cuda_dp4a(v[2*i+4], u[2*i+4], sumi_d.y); // SIMD dot product
            sumi_d.y = ggml_cuda_dp4a(v[2*i+5], u[2*i+5], sumi_d.y); // SIMD dot product
        }

        sumf_d += d8[i0/4] * (sc_reg[i0/2+0]*sumi_d.x + sc_reg[i0/2+1]*sumi_d.y);
    }

    return d6 * sumf_d;
}

static __device__ __forceinline__ float vec_dot_q1_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q1_0 * bq1_0 = (const block_q1_0 *) vbq + kbx;

    // Q1_0: 128 elements with ONE scale
    // Q8_1: 32 elements per block with individual scales
    // iqs selects which of the 4 chunks of 32 elements to process (0-3)

    const float d1 = bq1_0->d;

    // Process only the chunk specified by iqs
    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;

    // Load 32 bits (4 bytes) for this chunk from Q1_0
    const int offset = iqs * 4;
    const int v = bq1_0->qs[offset + 0] | (bq1_0->qs[offset + 1] << 8) |
                  (bq1_0->qs[offset + 2] << 16) | (bq1_0->qs[offset + 3] << 24);

    // Unpack 32 bits into 32 signed values (-1 or +1)
    int vi_bytes[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int shift = j * 4;
        const int bits4 = (v >> shift) & 0x0F;
        const int b0 = (bits4 & 0x01) ? 1 : -1;
        const int b1 = (bits4 & 0x02) ? 1 : -1;
        const int b2 = (bits4 & 0x04) ? 1 : -1;
        const int b3 = (bits4 & 0x08) ? 1 : -1;
        vi_bytes[j] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }

    // Compute dot product for this 32-element chunk
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int u = get_int_b4(bq8_1_chunk->qs, j);
        sumi = ggml_cuda_dp4a(vi_bytes[j], u, sumi);
    }

    // Apply Q1_0's single scale and this chunk's Q8_1 scale
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d1 * d8 * sumi;
}

static __device__ __forceinline__ float vec_dot_q2_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_0 * bq2_0 = (const block_q2_0 *) vbq + kbx;

    // Q2_0: 128 elements with ONE scale, 2 bits per element (4 per byte). code c -> symbol s = c-1.
    // Use the identity dot(s,u) = dot(c,u) - sum(u): bit-spread the codes (no per-code subtract/borrow)
    // and apply the -sum(u) offset once via the q8_1 stored sum s8 = d8*sum(u).
    const float d2 = bq2_0->d;
    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;

    const int offset = iqs * 8;
    const int qs0 = bq2_0->qs[offset + 0] | (bq2_0->qs[offset + 1] << 8) |
                    (bq2_0->qs[offset + 2] << 16) | (bq2_0->qs[offset + 3] << 24);
    const int qs1 = bq2_0->qs[offset + 4] | (bq2_0->qs[offset + 5] << 8) |
                    (bq2_0->qs[offset + 6] << 16) | (bq2_0->qs[offset + 7] << 24);

    int sumi = 0;   // = dot(c, u), c in {0,1,2,3}
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int b0 = (qs0 >> (j*8)) & 0xFF;
        const int s0 = (b0 | (b0 << 6) | (b0 << 12) | (b0 << 18)) & 0x03030303; // 4 codes -> 4 bytes
        sumi = ggml_cuda_dp4a(s0, get_int_b4(bq8_1_chunk->qs, j), sumi);
        const int b1 = (qs1 >> (j*8)) & 0xFF;
        const int s1 = (b1 | (b1 << 6) | (b1 << 12) | (b1 << 18)) & 0x03030303;
        sumi = ggml_cuda_dp4a(s1, get_int_b4(bq8_1_chunk->qs, 4 + j), sumi);
    }

    const float d8 = __low2float(bq8_1_chunk->ds);
    const float s8 = __high2float(bq8_1_chunk->ds); // = d8 * sum(u)
    return d2 * (d8 * sumi - s8);
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq + kbx;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2*VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_b2(bq4_0->qs, iqs + i);
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}


static __device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_1 * bq4_1 = (const block_q4_1 *) vbq + kbx;

    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_b4(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq + kbx;

    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_b2(bq5_0->qs, iqs + i);
        vh[i]    = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
    }

    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_1 * bq5_1 = (const block_q5_1 *) vbq + kbx;

    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int  u[2*VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_b4(bq5_1->qs, iqs + i);
        vh[i]    = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
    }

    return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq + kbx;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_b2(bq8_0->qs, iqs + i);
        u[i] = get_int_b4(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d, __low2half(bq8_1->ds));
}

#define VDR_F8E4M3_Q8_1_MMVQ 2

// T73 (small-batch decode fix): a SECOND, wider VDR variant used only for
// batched decode (ncols_dst > 1, e.g. MTP verify's ~4-token batch). ISA
// inspection (roc-obj/llvm-objdump on the mmvq.cu code object) showed VDR=2
// already compiles to a single global_load_b64 per call (not two 32-bit
// loads), so raising VDR isn't about load width per se -- it's about fewer,
// bigger vec_dot calls (more independent decode+FMA chains in flight,
// fewer kbx-loop iterations). Benched (batched-bench proxy, BS 1-8 sweep):
// VDR=4 gains +4-17% at BS 2-8 but COSTS -3.7% at BS=1 (18.34 -> 17.66) --
// unacceptable given the no-BS=1-regression guardrail if applied uniformly.
// Since ncols_dst is a compile-time template parameter of the enclosing
// mul_mat_vec_q kernel, dispatch VDR=2 at ncols_dst==1 (preserves the T68
// win) and VDR=4 at ncols_dst>1 (gets the batch-scaling win) -- both are
// separate kernel instantiations already, so this is a free, zero-runtime-
// cost split, not a runtime branch.
#define VDR_F8E4M3_Q8_1_MMVQ_WIDE 4

// F8E4M3 (Path X, Phase 2a) mmvq decode dot product -- the bandwidth-bound
// bs=1 counterpart to the mmq/WMMA prefill path in mmq.cuh. block_f8e4m3 is
// byte-for-byte block_q8_0-shaped (1 fp16 scale + 32 packed bytes), but the
// weight bytes are signed e4m3 floats, not int8: they are NOT linear in the
// integer bit pattern, so unlike vec_dot_q8_0_q8_1 this cannot feed dp4a
// directly. Each weight byte is decoded to fp32 with the portable software
// e4m3 decoder (ggml_cuda_e4m3_to_fp32, common.cuh -- same decoder Phase 1a
// uses, no hardware fp8 dependency) and multiplied against the raw signed
// int8 activation byte; the q8_1 activation's own per-block scale is linear,
// so it factors out of the per-byte loop and is applied once at the end
// alongside the weight's per-block scale (mirrors vec_dot_q2_K_q8_1's
// pattern of a plain float accumulate against an int8-quantized activation).
//
// Templated on VDR so the BS=1 (vdr=2) and batched (vdr=4, T73) variants
// share one implementation -- only the loop trip count differs.
template <int vdr>
static __device__ __forceinline__ float vec_dot_f8e4m3_q8_1_impl(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_f8e4m3 * bq8 = (const block_f8e4m3 *) vbq + kbx;

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi = get_int_b2(bq8->qs, iqs + i);
        const int ui = get_int_b4(bq8_1->qs, iqs + i);

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t wv = (uint8_t) (vi >> (8*j));
            const int8_t  av = (int8_t)  (ui >> (8*j));
            sumf += ggml_cuda_e4m3_to_fp32(wv) * (float) av;
        }
    }

    return sumf * (float) bq8->d * __low2float(bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_f8e4m3_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_f8e4m3_q8_1_impl<VDR_F8E4M3_Q8_1_MMVQ>(vbq, bq8_1, kbx, iqs);
}

static __device__ __forceinline__ float vec_dot_f8e4m3_q8_1_wide(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_f8e4m3_q8_1_impl<VDR_F8E4M3_Q8_1_MMVQ_WIDE>(vbq, bq8_1, kbx, iqs);
}

// T77 (wide-SIMD decode): batched-decode (ncols_dst>1) variant that replaces
// the scalar ldexp/cndmask/bfe software e4m3 decode with the RDNA4 hardware
// path (ggml_cuda_dot2_e4m3_q8, common.cuh -- v_cvt_pk_f32_fp8 + v_cvt_pk_
// rtz_f16_f32 + v_dot2_f32_f16, see that comment for the ISA verification
// and the lossless-repack precision argument). Falls back to the T73 scalar
// impl (bit-identical result, just slower) when the hardware path isn't
// available (non-RDNA4 HIP arch, or a CUDA/MUSA build) so this stays
// portable. `vdr` keeps the same meaning as VDR_F8E4M3_Q8_1_MMVQ(_WIDE)
// above: number of int32 (4-byte/4-weight) chunks processed per call; each
// chunk is done as 2 dot2 calls (2 weights each).
template <int vdr>
static __device__ __forceinline__ float vec_dot_f8e4m3_q8_1_simd_impl(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_f8e4m3 * bq8 = (const block_f8e4m3 *) vbq + kbx;

#if defined(GGML_CUDA_F8E4M3_HAS_NATIVE_DOT2)
    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi = get_int_b2(bq8->qs, iqs + i);
        const int ui = get_int_b4(bq8_1->qs, iqs + i);

        const int8_t a0 = (int8_t) (ui >>  0);
        const int8_t a1 = (int8_t) (ui >>  8);
        const int8_t a2 = (int8_t) (ui >> 16);
        const int8_t a3 = (int8_t) (ui >> 24);

        sumf = ggml_cuda_dot2_e4m3_q8((uint32_t) vi        & 0xFFFF, a0, a1, sumf);
        sumf = ggml_cuda_dot2_e4m3_q8(((uint32_t) vi >> 16) & 0xFFFF, a2, a3, sumf);
    }

    return sumf * (float) bq8->d * __low2float(bq8_1->ds);
#else
    return vec_dot_f8e4m3_q8_1_impl<vdr>(vbq, bq8_1, kbx, iqs);
#endif
}

// Note: no fixed-VDR non-template wrapper here on purpose. mmvq.cu (cheap,
// ccache ~30-60s rebuild) instantiates vec_dot_f8e4m3_q8_1_simd_impl<N>
// directly for whichever N the sweep is testing, so the VDR/ILP sweep never
// needs to touch this file (vecdotq.cuh is widely included -> full template-
// recompile, minutes) after this one-time addition.

// T97: F8E5M2 (OCP bf8) decode dot products. Mirrors the F8E4M3 T73 (scalar,
// portable) and T77 (hardware-dot2) tiers above; deliberately does NOT mirror
// T79 (bf8xbf8 V_DOT4, requiring bf8-quantized activations) -- see the
// dispatch-site comment in mmvq.cu for the rationale. block_f8e5m2 is
// byte-for-byte block_q8_0-shaped, same as block_f8e4m3, so the same
// get_int_b2/get_int_b4 byte-packing applies unchanged.
#define VDR_F8E5M2_Q8_1_MMVQ 2

// T97 scalar/portable path: same structure as vec_dot_f8e4m3_q8_1_impl,
// decodes each weight byte with the software e5m2 decoder (no hardware
// dependency, correct on every backend/arch).
template <int vdr>
static __device__ __forceinline__ float vec_dot_f8e5m2_q8_1_impl(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_f8e5m2 * bq8 = (const block_f8e5m2 *) vbq + kbx;

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi = get_int_b2(bq8->qs, iqs + i);
        const int ui = get_int_b4(bq8_1->qs, iqs + i);

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t wv = (uint8_t) (vi >> (8*j));
            const int8_t  av = (int8_t)  (ui >> (8*j));
            sumf += ggml_cuda_e5m2_to_fp32(wv) * (float) av;
        }
    }

    return sumf * (float) bq8->d * __low2float(bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_f8e5m2_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_f8e5m2_q8_1_impl<VDR_F8E5M2_Q8_1_MMVQ>(vbq, bq8_1, kbx, iqs);
}

// T97 hardware-dot2 path (RDNA4 bf8 native decode, T77-equivalent). Falls
// back to the portable scalar impl (bit-identical result, just slower) when
// GGML_CUDA_F8E5M2_HAS_NATIVE_DOT2 isn't defined (non-RDNA4 HIP arch, or a
// CUDA/MUSA build), same portability contract as vec_dot_f8e4m3_q8_1_simd_impl.
template <int vdr>
static __device__ __forceinline__ float vec_dot_f8e5m2_q8_1_simd_impl(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_f8e5m2 * bq8 = (const block_f8e5m2 *) vbq + kbx;

#if defined(GGML_CUDA_F8E5M2_HAS_NATIVE_DOT2)
    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi = get_int_b2(bq8->qs, iqs + i);
        const int ui = get_int_b4(bq8_1->qs, iqs + i);

        const int8_t a0 = (int8_t) (ui >>  0);
        const int8_t a1 = (int8_t) (ui >>  8);
        const int8_t a2 = (int8_t) (ui >> 16);
        const int8_t a3 = (int8_t) (ui >> 24);

        sumf = ggml_cuda_dot2_bf8_q8((uint32_t) vi        & 0xFFFF, a0, a1, sumf);
        sumf = ggml_cuda_dot2_bf8_q8(((uint32_t) vi >> 16) & 0xFFFF, a2, a3, sumf);
    }

    return sumf * (float) bq8->d * __low2float(bq8_1->ds);
#else
    return vec_dot_f8e5m2_q8_1_impl<vdr>(vbq, bq8_1, kbx, iqs);
#endif
}

// MXFP8 (ROC8) decode dot product. Deliberately mirrors the ORIGINAL,
// portable F8E4M3 T73 tier (vec_dot_f8e4m3_q8_1_impl above) -- int8 q8_1
// activations, software e4m3 weight decode -- NOT F8E4M3's later T77/T79
// evolution (hardware dot2 SIMD / native e4m3xe4m3 V_DOT4, which requires a
// dedicated e4m3-activation quantizer swapped in at the call site). This is
// an intentional, explicitly-scoped-down first pass: get a correct,
// portable, always-available decode kernel in place first; porting the
// SIMD/native-dot4 speed optimizations to MXFP8's e8m0 scale is a follow-up
// (see ROC8 KB gotchas), not a correctness requirement. block_mxfp8 is
// byte-for-byte block_q8_0-shaped except the first field is 1 byte (uint8_t
// e8m0) instead of 2 (ggml_half) -- get_int_b2 packs 4 raw qs bytes per int
// exactly like every other 8-bit quant here, addressing is unaffected by the
// scale field's width since it lives outside `qs[]`.
#define VDR_MXFP8_Q8_1_MMVQ 2

template <int vdr>
static __device__ __forceinline__ float vec_dot_mxfp8_q8_1_impl(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_mxfp8 * bq8 = (const block_mxfp8 *) vbq + kbx;

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi = get_int_b2(bq8->qs, iqs + i);
        const int ui = get_int_b4(bq8_1->qs, iqs + i);

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t wv = (uint8_t) (vi >> (8*j));
            const int8_t  av = (int8_t)  (ui >> (8*j));
            sumf += ggml_cuda_e4m3_to_fp32(wv) * (float) av;
        }
    }

    // e8m0 (power-of-2) scale, NOT the "_HALF" MXFP4 variant -- qs bytes here
    // are raw (undoubled) e4m3, same convention as ggml_cuda_e8m0_to_fp32's
    // other direct caller, dequantize_mxfp8 (dequantize.cuh).
    return sumf * ggml_cuda_e8m0_to_fp32(bq8->e) * __low2float(bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_mxfp8_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_mxfp8_q8_1_impl<VDR_MXFP8_Q8_1_MMVQ>(vbq, bq8_1, kbx, iqs);
}

// T77-equivalent MXFP8 decode: RDNA4 hardware-dot2 path. MXFP8 values ARE e4m3
// (identical to block_f8e4m3), so the same ggml_cuda_dot2_e4m3_q8 hardware
// weight-decode applies verbatim -- ONLY the block scale differs (MXFP8's e8m0
// power-of-2 exponent vs F8E4M3's fp16 d). Mirrors vec_dot_f8e4m3_q8_1_simd_impl
// AND (deliberately) F8E5M2's T77 choice: activations stay native int8 q8_1
// (LOSSLESS, no activation-quantize swap), NOT F8E4M3's T79 pure-dot4 (which
// needs an e4m3 activation buffer + carries a disclosed ~1.2-1.5% accuracy hit).
// Falls back to the scalar T73 impl (bit-identical, slower) off RDNA4 / non-HIP.
template <int vdr>
static __device__ __forceinline__ float vec_dot_mxfp8_q8_1_simd_impl(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_mxfp8 * bq8 = (const block_mxfp8 *) vbq + kbx;

#if defined(GGML_CUDA_F8E4M3_HAS_NATIVE_DOT2)
    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi = get_int_b2(bq8->qs, iqs + i);
        const int ui = get_int_b4(bq8_1->qs, iqs + i);

        const int8_t a0 = (int8_t) (ui >>  0);
        const int8_t a1 = (int8_t) (ui >>  8);
        const int8_t a2 = (int8_t) (ui >> 16);
        const int8_t a3 = (int8_t) (ui >> 24);

        sumf = ggml_cuda_dot2_e4m3_q8((uint32_t) vi         & 0xFFFF, a0, a1, sumf);
        sumf = ggml_cuda_dot2_e4m3_q8(((uint32_t) vi >> 16) & 0xFFFF, a2, a3, sumf);
    }

    // e8m0 (power-of-2) scale, same convention as vec_dot_mxfp8_q8_1_impl above.
    return sumf * ggml_cuda_e8m0_to_fp32(bq8->e) * __low2float(bq8_1->ds);
#else
    return vec_dot_mxfp8_q8_1_impl<vdr>(vbq, bq8_1, kbx, iqs);
#endif
}

// T79: pure V_DOT4_F32_FP8_FP8 decode dot. Where T77's hardware-dot2 path
// still detours the ACTIVATION through int8 (q8_1) and only accelerates the
// WEIGHT-side decode (2 terms / 3 hardware instructions: cvt_pk_f32_fp8 +
// cvt_pkrtz + fdot2), this path requires BOTH operands to already be native
// e4m3 -- the activation buffer must come from quantize_row_f8e4m3_for_mmvq_cuda
// (quantize.cu, T79), not the standard int8 quantize_row_q8_1_cuda, so this
// is only reachable via the dedicated F8E4M3-only dispatch branch in mmvq.cu
// (both the vec_dot AND the activation-quantize call are swapped together;
// mixing this vec_dot with a q8_1-quantized activation buffer would silently
// misinterpret int8 bytes as e4m3 bit patterns -- a correctness bug, not
// just a slowdown, so the two call sites are co-located and commented
// accordingly in mmvq.cu).
//
// Per-term instruction cost: 1 hardware instruction covers 4 terms (vs
// T77's 1.5 instr/term) -- the ISA-maximal case for this data format, per
// the RDNA4 ISA audit (wiki T61-audit, finding 1b) and compile-verified
// (`__builtin_amdgcn_dot4_f32_fp8_fp8` -> exactly `v_dot4_f32_fp8_fp8`,
// llvm-objdump --mcpu=gfx1201, single VOP3P instruction, T79 session).
//
// ACCURACY DISCLOSURE (T79, mandatory -- see wiki T79): quantizing
// ACTIVATIONS to e4m3 is NOT lossless the way T77's int8->f16 repack is
// (e4m3 has only 3 mantissa bits vs int8's 8-bit linear code within the
// per-block dynamic range). Measured impact (Qwen3.6-27B F8E4M3, README.md
// corpus, 6 chunks, GPU0 fresh-isolated, perplexity is bit-deterministic so
// this is a REAL reproducible effect, not run noise -- resampled the
// baseline twice, identical to the bit): int8-activation baseline (T77,
// -ub 4 forcing mmvq) PPL 2.4423 +/- 0.1236 (matches prior-session 2.4340
// +/- 0.1229 within noise) vs e4m3-activation (-ub 512 forcing the existing
// Phase-1b WMMA activation quantizer, the same numerics this vec_dot
// consumes) PPL 2.4707 +/- 0.1261 -- a real, small, +1.2-1.5% relative
// increase. Judged within the directed 1-2% tolerance band; disclose this
// on every future accuracy claim for this path, do not silently drop it.
template <int vdr>
static __device__ __forceinline__ float vec_dot_f8e4m3_f8e4m3_impl(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_f8e4m3 * bq8 = (const block_f8e4m3 *) vbq + kbx;

#if defined(GGML_CUDA_F8E4M3_HAS_NATIVE_DOT2)
    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const uint32_t vi = (uint32_t) get_int_b2(bq8->qs, iqs + i);
        const uint32_t ui = (uint32_t) get_int_b4(bq8_1->qs, iqs + i);
        sumf = __builtin_amdgcn_dot4_f32_fp8_fp8(vi, ui, sumf);
    }

    return sumf * (float) bq8->d * __low2float(bq8_1->ds);
#else
    // Portable fallback: no native dot4 available off RDNA4/gfx12 -- reuse
    // the T77 dot2 (or T73 scalar) impl. NOTE this reads bq8_1 as int8 q8_1
    // activations, which is WRONG if the caller already swapped the
    // quantize function to the e4m3 buffer -- this branch is dead code on
    // any target this dispatch is actually reachable from (F8E4M3 decode is
    // gated GGML_CUDA_CC_IS_RDNA4 at the mmvq.cu call site, same gate this
    // macro expands under), kept only so the translation unit still
    // compiles portably.
    return vec_dot_f8e4m3_q8_1_impl<vdr>(vbq, bq8_1, kbx, iqs);
#endif
}

// Card 137 fix 2: pure V_DOT4_F32_BF8_BF8 decode dot -- the bf8 (F8E5M2)
// twin of vec_dot_f8e4m3_f8e4m3_impl (T79) above. RDNA4 has a dedicated bf8
// dot4 opcode right beside fp8's (ISA doc 70651 p.4744, V_DOT4_F32_BF8_BF8,
// opcode 39, vs fp8's V_DOT4_F32_FP8_FP8 opcode 38); ISA-verified this
// session (`__builtin_amdgcn_dot4_f32_bf8_bf8` -> exactly `v_dot4_f32_bf8_bf8`,
// llvm-objdump --mcpu=gfx1201 on a standalone HIP test kernel, single VOP3P
// instruction). Same requirement as T79: BOTH operands must already be
// native bf8 -- the activation buffer must come from
// quantize_row_f8e5m2_for_mmvq_cuda (quantize.cu, this card), not the
// standard int8 quantize_row_q8_1_cuda, so this is only reachable via the
// dedicated F8E5M2-only dispatch branch in mmvq.cu (both the vec_dot AND the
// activation-quantize call are swapped together, same correctness-coupling
// discipline as T79 -- see that function's comment for the full rationale).
//
// ACCURACY DISCLOSURE (mandatory, mirrors T79): quantizing ACTIVATIONS to
// bf8 (e5m2, only 2 mantissa bits vs e4m3's 3) is a strictly LARGER
// quantization step than T79's e4m3-activation swap, which itself already
// cost +1.2-1.5% PPL relative to the lossless int8-activation baseline. This
// path is therefore expected to cost MORE than T79's hit, on top of bf8's
// existing weight-side +2.5% PPL delta (T97, vs BF16) -- gate adoption on a
// fresh PPL re-measurement (see card 137 KB) against the T97 hardware-dot2
// (int8-activation) baseline this supersedes; do not assume the T79 numbers
// transfer.
template <int vdr>
static __device__ __forceinline__ float vec_dot_f8e5m2_f8e5m2_impl(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_f8e5m2 * bq8 = (const block_f8e5m2 *) vbq + kbx;

#if defined(GGML_CUDA_F8E5M2_HAS_NATIVE_DOT2)
    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const uint32_t vi = (uint32_t) get_int_b2(bq8->qs, iqs + i);
        const uint32_t ui = (uint32_t) get_int_b4(bq8_1->qs, iqs + i);
        sumf = __builtin_amdgcn_dot4_f32_bf8_bf8(vi, ui, sumf);
    }

    return sumf * (float) bq8->d * __low2float(bq8_1->ds);
#else
    // Portable fallback: no native dot4 available off RDNA4/gfx12 -- reuse
    // the T97 dot2 (or scalar) impl. NOTE this reads bq8_1 as int8 q8_1
    // activations, which is WRONG if the caller already swapped the
    // quantize function to the bf8 buffer -- this branch is dead code on any
    // target this dispatch is actually reachable from (F8E5M2 decode is
    // gated GGML_CUDA_CC_IS_RDNA4 at the mmvq.cu call site, same gate this
    // macro expands under), kept only so the translation unit still compiles
    // portably.
    return vec_dot_f8e5m2_q8_1_impl<vdr>(vbq, bq8_1, kbx, iqs);
#endif
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq + kbx;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_b4(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q3_K * bq3_K = (const block_q3_K *) vbq + kbx;

    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = bq3_K->d;

    const int vl = get_int_b2(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_b2(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    int    u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_K * bq5_K = (const block_q5_K *) vbq + kbx;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

#define VDR_IQ2_XXS_Q8_1_MMVQ 2
#define VDR_IQ2_XXS_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;

    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[k0/2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
    sumi = sumi * ls / 8;           // (sumi * scale + sumi / 2) / 4
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ2_XS_Q8_1_MMVQ 2
#define VDR_IQ2_XS_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;

    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l0/2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0/2] >> 9);

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ2_S_Q8_1_MMVQ 2
#define VDR_IQ2_S_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq2->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid_pos = (const int *)(iq2s_grid + (qs[l0/2] | ((qh << (8-l0)) & 0x300)));

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;

    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ3_XXS_Q8_1_MMVQ 2
#define VDR_IQ3_XXS_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ3_S_Q8_1_MMVQ 2
#define VDR_IQ3_S_Q8_1_MMQ  2

// TODO: don't use lookup table for signs
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ1_S_Q8_1_MMVQ 1
#define VDR_IQ1_S_Q8_1_MMQ  1

static __device__ __forceinline__ float vec_dot_iq1_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq1_s * bq1 = (const block_iq1_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq1->qh[iqs];

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = iq1s_grid_gpu[qs[l0/2] | (((qh >> 3*(l0/2)) & 0x07) << 8)];

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const float  d1q   = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);
    const float2 ds    = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x*sumi + ds.y*delta);
}

#define VDR_IQ1_M_Q8_1_MMVQ 1
#define VDR_IQ1_M_Q8_1_MMQ  1

static __device__ __forceinline__ float vec_dot_iq1_m_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq1_m * bq1 = (const block_iq1_m *) vbq + kbx;

    const int       qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    int   sumi[2] = {0};
    float sumf[2] = {0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2*iqs + l0/4] >> (4 * ((l0/2) % 2));

        const int grid = iq1s_grid_gpu[qs[l0/2] | ((qhl & 0x07) << 8)];

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);

        sumi[l0/4] = ggml_cuda_dp4a(grid0, u0, sumi[l0/4]);
        sumi[l0/4] = ggml_cuda_dp4a(grid1, u1, sumi[l0/4]);

        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f*IQ1M_DELTA/0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0/4] += delta*sumy;
    }

    const uint16_t * sc = (const uint16_t *) bq1->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);

    const int tmp = sc[iqs/2] >> (6*(iqs%2));
    const int sc0 = 2*((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2*((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

#define VDR_IQ4_NL_Q8_1_MMVQ 2
#define VDR_IQ4_NL_Q8_1_MMQ  4

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_nl * bq4 = (const block_iq4_nl *) vbq + kbx;

    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }

    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}

#define VDR_IQ4_XS_Q8_1_MMVQ 4
#define VDR_IQ4_XS_Q8_1_MMQ  4

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq + kbx;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

        const int u0 = get_int_b4(bq8_1[iqs/4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs/4].qs, j + 4);

        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }

    const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
    sumi *= ls - 32;

    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs/4].ds);
    return d * sumi;
}
