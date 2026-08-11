// ROC8: MXFP8 (OCP Microscaling FP8) type-plumbing + kernel correctness
// selftest. Synthetic (no model weights, no GGUF, no big matmul) -- exercises
// the CPU reference codec, the CUDA/HIP dequant kernel, the mmvq decode dot
// (vec_dot_mxfp8_q8_1), and the mmq/WMMA prefill dot (vec_dot_mxfp8_mxfp8_mma)
// against a fp32 reference computed independently on the host, all with a
// KNOWN synthetic matrix -- exactly the Phase-1 correctness gate this type's
// design doc calls for before any converter/real-model work.
//
// Gate: `GGML_HIP_MXFP8_SELFTEST` env var (see ggml-cuda.cu). Zero effect on
// any existing model/quant/kernel path when unset. Same pattern/doctrine as
// GGML_HIP_IU4_W4A4_SELFTEST / GGML_HIP_SWMMAC24_SELFTEST -- runs at most
// once per process, at ggml_backend_cuda_init.
//
// T122 gating discipline (mandatory, see iu4_w4a4.cu header comment for the
// full incident writeup): this whole TU is gated on `GGML_USE_HIP` ONLY
// (true in both host and device compile passes). `RDNA4` (== `__GFX12__`) is
// a device-pass-only macro -- it must NEVER gate the host-callable entry
// point itself, only device kernel bodies / WMMA builtins. The RDNA4-only
// WMMA sub-test below is gated by a RUNTIME cc check
// (`ggml_cuda_info().devices[id].cc` + `GGML_CUDA_CC_IS_RDNA4()`) in the host
// wrapper, not a compile-time `#if defined(RDNA4)` around the whole function.
#pragma once

#include "common.cuh"

bool ggml_cuda_mxfp8_selftest();
