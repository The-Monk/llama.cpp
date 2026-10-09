#pragma once

// [TAG_ACT_FUSE] T407: activation-path fusion set for prefill (MMQ consumers).
//
// on by default (GGML_ACT_FUSE=0 disables); GGML_ACT_FUSE_MASK selects parts:
//   1 = quantize dedup: sibling MUL_MATs that read the same src1 share one
//       MMQ q8_1 activation buffer (ctx.act_cache_*), quantized once.
//   2 = GLU -> quantize: a swiglu whose only consumer is the next MUL_MAT
//       (ffn_down) writes the down-projection's MMQ activation directly into
//       the cache and never materializes its fp32 output.
//   4 = [ADD ->] RMS_NORM -> MUL -> quantize: the norm (with the residual add
//       folded in when the graph has ADD right before it) writes its fp32
//       outputs as before AND the consumers' MMQ activation into the cache.
// GGML_ACT_FUSE_VERIFY=1 additionally runs the unfused reference for every
// fused producer and counts differing q8_1 bytes (slow; gate only).
//
// Every producer computes the same fp32 value in registers as the unfused
// kernels and quantizes with the code of quantize_mmq_q8_1 (quantize.cu), so
// the cached bytes equal the bytes the skipped quantize launch would write.
//
// Producers are split into a LOAD functor (which fp32 values) and a STORE
// functor (which quantized layout). mmq_q8_1_store<layout> below is the MMQ
// q8_1 store; another quantized activation layout (e.g. the N1 engine's) plugs
// in as another store functor with the same warp-collective contract:
// called by every lane with 4 consecutive values, lane L of a warp holding
// columns [4L, 4L+4) of a 128-aligned span.

#include "common.cuh"

enum ggml_cuda_act_fuse_bit {
    GGML_ACT_FUSE_DEDUP = 1,
    GGML_ACT_FUSE_GLU   = 2,
    GGML_ACT_FUSE_NORM  = 4,
};

// T412: activation layout id of the N1 int8 WMMA GEMM (n1_act.cuh: int8 X tiles + fp32 sx[K/128][Npad]).
// Outside the mmq_q8_1_ds_layout values, so an MMQ consumer can never hit an N1 entry (and vice versa).
// Producers need ncols % 128 == 0 for it; y must hold ggml_cuda_n1_act_bytes(dst) bytes.
#define GGML_CUDA_ACT_LAYOUT_N1 100

// T399: activation layout id of the single-copy decode / small-batch kernels (sc_act.cuh: int8 xq[NP][K] + half2
// ds[K/32][NP]). Producers need ncols % 128 == 0; y must hold ggml_cuda_sc_act_bytes(dst) bytes.
#define GGML_CUDA_ACT_LAYOUT_SC 101

int  ggml_cuda_act_fuse_mask();
bool ggml_cuda_act_fuse_verify();

// MMQ q8_1 layout a quantized src0 consumes (mmq_q8_1_ds_layout value), or -1
// when MMQ would feed it something else (fp4/fp8 activations). Mirrors the
// producer selection in ggml_cuda_mul_mat_q.
int ggml_cuda_mmq_act_layout(const ggml_tensor * src0, int cc);

// Bytes of the MMQ activation buffer for src1 (same formula as mmq.cu).
size_t ggml_cuda_mmq_act_bytes(const ggml_tensor * src1, int cc);

// Fused producers. y = MMQ activation buffer (layout `layout`), 2D only.
// GLU: dst = silu(gate) * up, swiglu_split semantics; dst values are NOT stored.
void ggml_cuda_act_glu_quant(const float * gate, const float * up, int64_t nc, int64_t nrows,
        int64_t s_gate, int64_t s_up, int64_t ne0_padded, int layout, void * y, cudaStream_t stream);

// [ADD ->] RMS_NORM -> MUL: x = a (+ b); stores x to dst_add when add_b != null;
// dst_mul = rsqrt(mean(x^2)+eps) * x * w; quantizes dst_mul into y.
// Returns false (nothing launched) when the shape is unsupported.
bool ggml_cuda_act_norm_quant(const float * a, const float * add_b, float * dst_add, float * dst_mul,
        const float * w, int64_t ncols, int64_t nrows, int64_t s_a, int64_t s_b,
        float eps, int64_t ne0_padded, int layout, void * y, cudaStream_t stream);

// Verify helper: number of differing bytes between two device buffers (syncs).
int64_t ggml_cuda_act_count_diff(const void * a, const void * b, size_t nbytes, cudaStream_t stream);
