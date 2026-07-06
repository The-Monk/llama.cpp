// T89: native iu4 x iu4 (int4 weight x int4 activation, "W4A4") RDNA4 WMMA --
// driver-completeness self-test, mirroring the swmmac24.cu / T101 doctrine
// ("keep CORRECT + AVAILABLE but DORMANT" for a capability whose accuracy
// gate did not clear the bar for an ACTIVE default path -- see
// wiki/tech/int4-iu4-notes.md for the full accuracy-gate + disposition writeup).
//
// Today's shipped Q4_0/Q4_1/K-quant decode+prefill path expands int4 weight
// nibbles to int8 and rides the existing iu8 WMMA/dp4a path (confirmed,
// `wiki/tech/rdna4-hardware-optimal-config.md`, T88) -- i.e. weights are
// int4 but ACTIVATIONS stay int8 ("W4A8"). The native
// `V_WMMA_I32_16X16X32_IU4` instruction (int4 x int4 -> int32, confirmed via
// disasm on gfx1201, `wiki/tech/phase2-lever-validation.md`) is completely
// unused anywhere in ggml-cuda. Using it for real would require quantizing
// ACTIVATIONS to int4 too (true W4A4) -- the accuracy-risky part this task's
// STEP 1 gate measured. This header/`.cu` only proves the driver-level
// instruction support (`ggml_cuda_mma::mma_iu4`, see mma.cuh) is wired and
// numerically correct; it does NOT plug into any ggml op or quant type.
//
// Gate: `GGML_HIP_IU4_W4A4_SELFTEST` env var (see ggml-cuda.cu). Zero effect
// on any existing model/quant/kernel path when unset (checked once per
// process, at `ggml_backend_cuda_init`, same call site/pattern as
// swmmac24.cu's `GGML_HIP_SWMMAC24_SELFTEST`).
#pragma once

#include "common.cuh"

bool ggml_cuda_iu4_w4a4_selftest();
