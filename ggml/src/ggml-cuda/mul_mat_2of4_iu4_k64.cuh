// T187 (reopened): does the K64 int4 2:4-sparse SWMMAC
// (V_SWMMAC_I32_16X16X64_IU4) survive as a REAL, full, correctness-gated
// tiled GEMM kernel with genuine on-device gather of compressed-A + metadata
// from global memory, or does the 2:4 gather/pack tax eat the raw-instruction
// 3.90x ISA ceiling (bench_ilp.hip, register-resident, zero gather)?
//
// MEASURED VERDICT (standalone bench, ~/int4-research/pocs/t187-k64-full-gemm/
// gemm_bench.hip, correctness-gated max_abs_err=0 on every shape before any
// throughput number was trusted; GPU0-only): the full kernel captures
// **88-94% of the raw ISA ceiling at large K** (K=8192: 3.67x measured vs
// 3.90x ceiling against int8-K16-dense-WMMA = 94%; 1.72x measured vs 1.95x
// ceiling against int4-K32-dense-WMMA = 88%). The gather/pack tax is SMALL,
// not dominant -- this DOES NOT reproduce T162's ~1% capture, because T162's
// K32 path was repack-HEAVY (on-the-fly top-2-of-4 selection in the hot
// loop), whereas this kernel (like the already-shipped mul_mat_2of4_fp8 V3)
// streams a PRE-PACKED physical layout: byte-per-group values + nibble-per-
// group-pair metadata chosen to match V_SWMMAC_I32_16X16X64_IU4's own
// per-lane addressing exactly (see swmmac24_iu4_k64.cu/h for the hardware-
// validated formulas, independently cross-checked against
// ROCm/amd_matrix_instruction_calculator --architecture RDNA4
// --detail-instruction), so every per-lane operand load is a single
// contiguous memcpy -- zero on-the-fly bit-shuffling. The measured tax is
// real (a second global load for metadata + its own register + an extra
// loop-body dependency vs a dense kernel touching only one array), just
// small relative to the raw compute win.
//
// All three kernels below (int8-K16-dense-WMMA baseline, int4-K32-dense-WMMA
// baseline, int4-K64-2:4-sparse-SWMMAC target) share ONE WARPSxILP tiling
// skeleton mirroring ggml-cuda/mul_mat_2of4_fp8.cu's winning V3 shape (zero
// __syncthreads, every operand fully private per-lane, double-buffered
// global-load prefetch, c0=0-per-call + software int32 accumulate). Also
// fixed along the way: a genuine DRAM critical-stride/bank-conflict
// pathology (8-30x cliffs, reproduced cold/isolated, NOT thermal noise) at
// row byte-pitches landing on exact power-of-two boundaries -- fixed with
// the standard technique (pad the row pitch), same as any real production
// GEMM kernel.
//
// DORMANT: nothing here is registered in the default mul_mat dispatch.
// Opt-in only, via GGML_HIP_2OF4_IU4_K64_SELFTEST (correctness) and
// GGML_HIP_2OF4_IU4_K64_SHAPE_BENCH (throughput sweep + ratio vs both
// baselines), same discipline as swmmac24_iu4_k64.cu / mul_mat_2of4_fp8.cu's
// GGML_HIP_2OF4_FP8_SHAPE_BENCH hooks. GPU0-only development; fork-local,
// nothing upstreamed. Vikunja T187.
#pragma once

#include "common.cuh"

bool ggml_cuda_mul_mat_2of4_iu4_k64_selftest();
bool ggml_cuda_mul_mat_2of4_iu4_k64_shape_bench();
