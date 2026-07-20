// Phase-3(B) M-gated int4-weight dispatch: the M==1 (single-token decode)
// route. Originated as a port of the standalone microbench at
// ~/int4-research/lean-gemv/int4_gemv_v3.hip (that PoC reported 543 GB/s
// "= 96% of a 565 GB/s DRAM ceiling"); BOTH numbers in that framing were
// corrected during this integration (2026-07-20, coordinator-directed
// re-grade):
//   - The "565 GB/s ceiling" was CLOCK-RAMP LIMITED, not a DRAM wall. The
//     ceiling probe (and the PoC's own benchmark sweep) used discrete
//     per-config launch+hipDeviceSynchronize, which -- confirmed by
//     rocm-smi clock polling live during the run -- lets mclk fall back to
//     its 96 MHz idle/power-save state between reps instead of holding the
//     1258 MHz boosted steady state (DPM only partially ramps, e.g. to the
//     875 MHz level-3 step, before the next sync-triggered idle gap). A
//     SUSTAINED probe (long warm-up burst + back-to-back timed launches on
//     one stream, NO host sync in the middle) holds full boost (sclk
//     ~3250-3400 MHz, mclk 1258 MHz) and measures **639.4 GB/s** (DtoD
//     hipMemcpy 569.7, streaming read-only reduction steady-state 638-639)
//     -- essentially the ~640-650 GB/s datasheet figure. GRADE AGAINST 639
//     GB/s, NOT 565.
//   - The PoC's own "543 GB/s" GEMV number was measured with the SAME
//     discrete-launch-limited harness, so it is also an underestimate of
//     what the technique can do once clocks are held. Re-measured under
//     the sustained methodology (this kernel's V2 design, see the .cu
//     doctrine comment for the V1->V2 load-width fix): 464.8-566.1 GB/s
//     across representative decode shapes (N4096K4096 attn/FFN-ish,
//     N13824K5120 FFN-down-ish), i.e. 72.7-88.6% of the corrected 639 GB/s
//     ceiling -- comparable to or better than the PoC's figure, not worse,
//     once both are graded on the same clock footing.
//   - REMAINING gap to ceiling is real but shape-dependent, not a single
//     "vector width" knob: large transfers (N13824K5120, ~40 MB/launch)
//     reach ~88%; small transfers typical of a single attention
//     projection at real model sizes (N1024K4096, ~2.4 MB/launch) only
//     reach ~45% even with the fixed load-width kernel -- that gap is
//     PER-LAUNCH-OVERHEAD-bound (fixed kernel-launch latency amortized
//     over too few bytes), not further closeable by wider vector loads or
//     more warps/CTA (WARPS_PER_CTA 4/8/16/32 all land within a few % of
//     each other at a fixed shape, see the .cu file's launch config
//     comment). The lever for THAT gap is reducing launch-overhead-per-op
//     (HIP graphs -- already ON in the ship stack per house doctrine), not
//     a further GEMV kernel change; not attempted here (out of scope for
//     Phase-3(B)'s M-gated dispatch task).
//
// Into a real ggml-cuda kernel that consumes the GGML_TYPE_IU4 tensor
// (block_iu4, ggml-common.h) DIRECTLY --
// no repack, no separate buffer. This is "option (a)" from the task brief:
// the microbench's activation-transpose + 512-byte logical_k weight-slot
// swizzle exist to fix an M-SCALING problem (V1->V2) and to widen the
// vector-load burst at LARGE N sweeps; neither is needed for this kernel's
// actual dispatch scope (M==1 only -- M in [2,16] is the separate CK WMMA
// route, mul_mat_iu4_ck_wmma.cu). So this kernel keeps the microbench's
// core technique (warp-per-row, one lane per nibble, vectorized 128-bit
// weight-block loads, register accumulator, single warp-reduce at the end)
// but reads block_iu4's plain per-32-element block layout as-is:
//   - qs[16]: 32 signed int4 values, byte b's low nibble = element 2*b,
//     high nibble = element 2*b+1 (SAME convention block_iu4's own doctrine
//     comment + mul_mat_iu4.cu's pack_iu4_block describe -- i.e. the raw
//     nibble bit pattern already IS two's complement; decode is a sign
//     extend of the 4-bit field, NOT the microbench's "nibble - 8"
//     offset-binary convention, which used a DIFFERENT bespoke packing that
//     does not match GGML_TYPE_IU4).
//   - d: one fp16 scale per 32-element block (matches block_iu4 exactly).
//
// M is a compile-time template parameter (1/2/4/8/16 instantiated) purely
// so the correctness selftest (ggml_cuda_mul_mat_iu4_gemv_selftest) can
// validate every M the task brief asks for; the REAL dispatch path
// (ggml_cuda_op_mul_mat_iu4_gemv, hooked in ggml-cuda.cu behind
// GGML_HIP_IU4_MSWITCH) only ever launches M==1 -- for M>1 the per-token
// activation load is a strided scalar read (same shape as the microbench's
// V1 M-scaling collapse), which is fine for a correctness check but is NOT
// the fast path (that's CK, M in [2,16]).
//
// Opt-in via GGML_HIP_IU4_MSWITCH (see ggml-cuda.cu dispatch site); zero
// effect on any other type/path when unset.
#pragma once

#include "common.cuh"

bool ggml_cuda_op_mul_mat_iu4_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// In-scope check for the M-gated dispatch: GGML_TYPE_IU4 weight, F32
// activation/output, 2D tensors, and (checked by the caller) dst->ne[1]==1.
bool ggml_cuda_iu4_gemv_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// Correctness-gated selftest (house pattern, see mul_mat_iu4.cu /
// mul_mat_iu4_mmq.cu): hand-packs block_iu4 operands with the SAME nibble
// convention a real GGUF IU4 tensor would use and compares against a CPU
// int64 accumulate reference, across M in {1,2,4,8,16} and operand range
// [-8,7] (including the -8 asymmetric edge). A wrong-answer kernel is not
// benchmarked -- see GGML_HIP_MUL_MAT_IU4_GEMV_SELFTEST in ggml-cuda.cu.
bool ggml_cuda_mul_mat_iu4_gemv_selftest();
