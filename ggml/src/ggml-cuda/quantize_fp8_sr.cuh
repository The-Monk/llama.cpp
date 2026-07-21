// Stage 26: stochastic-rounding (SR) fp8 WEIGHT quantizer -- an ACCURACY
// lever, not a speed lever. `quantize_row_f8e4m3_ref`/`quantize_row_f8e5m2_ref`
// (ggml-quants.c) are round-to-nearest (RTN) only, which is a DETERMINISTIC,
// systematically-biased rounding rule (values consistently round toward
// whichever direction the format's representable grid happens to favor at
// that magnitude). Stochastic rounding instead rounds up/down
// PROBABILISTICALLY, weighted by the fractional distance to each neighbor,
// so E[SR(x)] == x -- unbiased in expectation, at the cost of higher
// per-element variance (the tradeoff this stage measures).
//
// Uses the REAL hardware SR convert instructions confirmed in the Stage 18
// capability sweep (`gfx1201_capability_matrix_2026-07-21.tsv`):
// __builtin_amdgcn_cvt_sr_fp8_f32 / __builtin_amdgcn_cvt_sr_bf8_f32
// (feature `fp8-conversion-insts`, same gate as the plain `cvt_f32_fp8`/
// `cvt_pk_fp8_f32` already exploited elsewhere in this driver -- NOT the
// gfx1250-exclusive `cvt_scalef32_sr_*` MX family, which is a DIFFERENT,
// absent-on-gfx1201 instruction, confirmed dead in Stage 18/25).
//
// RNG: mirrors the exact pattern already used in this ecosystem for SR
// quantizers (CK's mxf4_utils.hpp, `~/src/ck-ref/include/ck/utility/`):
// `__builtin_amdgcn_prng_b32(__builtin_readcyclecounter() * (gid + 1))`,
// one fresh draw per element per quantize call (not reused across blocks).
//
// Dormant, off by default -- see GGML_HIP_FP8_SR_QUANT_SELFTEST in
// ggml-cuda.cu. No live dispatch path currently calls this (there's no
// existing on-device weight-requantize entry point in this codebase to
// hook into -- weight quantization is normally a one-time CPU step,
// `llama-quantize`/`ggml_quantize_chunk`); this stage delivers the KERNEL,
// ISA-confirmed and bias/error-measured, as a ready-to-use utility +
// the accuracy verdict for a future integration decision, matching the
// "wire it, measure it, decide later" scope of this task.
#pragma once

#include "common.cuh"

// Quantize `n` (multiple of 32) float values at `d_src` into `n/32`
// block_f8e4m3 / block_f8e5m2 blocks at `d_dst`, using stochastic rounding.
// Same per-32-block amax/{448,57344} scale convention as
// quantize_row_f8e4m3_ref/quantize_row_f8e5m2_ref (ggml-quants.c) -- SR
// quantized tensors are drop-in dequantizable by the existing
// dequantize_row_f8e4m3/f8e5m2 CPU functions.
void ggml_cuda_quantize_weight_f8e4m3_sr(ggml_backend_cuda_context & ctx, const float * d_src, void * d_dst, int64_t n);
void ggml_cuda_quantize_weight_f8e5m2_sr(ggml_backend_cuda_context & ctx, const float * d_src, void * d_dst, int64_t n);

// The decisive test: quantizes a large random tensor with BOTH SR and RTN
// (CPU reference quantize_row_*_ref), over multiple independent random
// draws/seeds, and reports mean quantization BIAS (mean of dequant(q(x))-x)
// and per-element RMS error for both, both formats. Also confirms SR
// output values are valid/finite fp8 bytes that dequantize within the
// format's representable range. ISA emission (v_cvt_sr_fp8_f32/
// v_cvt_sr_bf8_f32, not an emulation) is confirmed separately via
// llvm-objdump on the built kernel, see FINDINGS.md Stage 26.
bool ggml_cuda_fp8_sr_quant_bias_test();
