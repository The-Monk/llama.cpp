// gfx1201 ISA-coverage follow-up (T186): V_DOT2_F32_BF16 driver-completeness
// self-test. Full sweep of the RDNA4 ISA (llvm-mc + BuiltinsAMDGPU.inc,
// gfx1201) found this instruction PRESENT in silicon but with ZERO uses
// across all roc9 compute kernels -- we already ride the sibling fp16 form
// (v_dot2_f32_f16 / __builtin_amdgcn_fdot2, common.cuh's
// ggml_cuda_dot2_e4m3_q8/ggml_cuda_dot2_bf8_q8, ~125k static uses) but never
// exercised the bf16 sibling (__builtin_amdgcn_fdot2_f32_bf16). This file
// closes that coverage gap the same way swmmac24.cu/iu4_w4a4.cu do for their
// instructions: a small, self-contained, correctness-gated probe kernel,
// off by default, NOT wired into any mul_mat dispatch path.
//
// Builtin signature (BuiltinsAMDGPU.inc, gfx1201-confirmed via compile +
// disasm): `float __builtin_amdgcn_fdot2_f32_bf16(short2 a, short2 b, float
// c, bool clamp)` -- a/b are 2x packed raw bf16 BIT PATTERNS (not a native
// bf16x2 vector type; short2 carries the 16-bit bf16 encodings), accumulate
// is fp32, matching the existing fp16 dot2's a.x*b.x + a.y*b.y + c contract.
// Emits exactly `v_dot2_f32_bf16` (confirmed via --cuda-device-only -S
// disasm on gfx1201, see task writeup / commit message for the grep proof).
//
// Zero effect on any existing model/quant/kernel path when GGML_HIP_
// DOT2_BF16_SELFTEST is unset. Runs at most once per process, same
// call-site pattern as GGML_HIP_MXFP8_SELFTEST (ggml-cuda.cu). Gating
// discipline: this whole TU is `GGML_USE_HIP`-only (host+device passes);
// the device kernel body alone is `RDNA4`-gated; the host entry point does a
// RUNTIME cc check (GGML_CUDA_CC_IS_RDNA4) before ever launching it.
#pragma once

#include "common.cuh"

bool ggml_cuda_dot2_bf16_selftest();
