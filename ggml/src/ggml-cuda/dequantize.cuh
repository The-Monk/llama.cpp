#include "common.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q2_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q2_0 * x = (const block_q2_0 *) vx;

    const float d = x[ib].d;

    // Q2_0: 2 bits per element, 4 elements per byte. code c in {0,1,2,3} -> symbol c-1 in {-1,0,+1,+2}
    const int byte_index_0 = iqs       / 4;
    const int bit_offset_0 = (iqs      % 4) * 2;
    const int byte_index_1 = (iqs + 1) / 4;
    const int bit_offset_1 = ((iqs + 1) % 4) * 2;

    const int c0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 0x3;
    const int c1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 0x3;

    v.x = (c0 - 1) * d;
    v.y = (c1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

static __device__ __forceinline__ void dequantize_f8e4m3(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_f8e4m3 * x = (const block_f8e4m3 *) vx;

    const float d = x[ib].d;

    v.x = ggml_cuda_e4m3_to_fp32(x[ib].qs[iqs + 0]) * d;
    v.y = ggml_cuda_e4m3_to_fp32(x[ib].qs[iqs + 1]) * d;
}

static __device__ __forceinline__ void dequantize_f8e5m2(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_f8e5m2 * x = (const block_f8e5m2 *) vx;

    const float d = x[ib].d;

    v.x = ggml_cuda_e5m2_to_fp32(x[ib].qs[iqs + 0]) * d;
    v.y = ggml_cuda_e5m2_to_fp32(x[ib].qs[iqs + 1]) * d;
}

// MXFP8 (ROC8): mechanical mirror of dequantize_f8e4m3 above -- same e4m3 leaf
// decode, only the scale source differs (shared e8m0 byte, not a per-block
// fp16 half).
static __device__ __forceinline__ void dequantize_mxfp8(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_mxfp8 * x = (const block_mxfp8 *) vx;

    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);

    v.x = ggml_cuda_e4m3_to_fp32(x[ib].qs[iqs + 0]) * d;
    v.y = ggml_cuda_e4m3_to_fp32(x[ib].qs[iqs + 1]) * d;
}

// ROC8: MXFP6 -- same shared e8m0 scale as MXFP8 above, but the qs[] payload
// is 6-bit-packed (4 codes / 3 bytes), not byte-per-value. `iqs` here is
// always even (the dequantize_block_cont_cuda caller processes elements in
// pairs) and (iqs, iqs+1) always fall in the SAME 4-value group (group size
// 4 divides evenly into the stride-2 iteration), so one mxfp6_unpack4 call
// covers both -- decode via the same lossless e3m2->e4m3->fp32 chain the
// GPU vec_dot uses (ggml_cuda_e3m2_to_e4m3, mirrors ggml_e3m2_to_fp32 on the
// CPU side bit-for-bit).
static __device__ __forceinline__ void dequantize_mxfp6(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_mxfp6 * x = (const block_mxfp6 *) vx;

    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);

    const int g = iqs >> 2;
    const int r = iqs & 3;

    uint8_t codes[4];
    mxfp6_unpack4(x[ib].qs + 3*g, codes);

    v.x = ggml_cuda_e4m3_to_fp32(ggml_cuda_e3m2_to_e4m3(codes[r + 0])) * d;
    v.y = ggml_cuda_e4m3_to_fp32(ggml_cuda_e3m2_to_e4m3(codes[r + 1])) * d;
}
