#pragma once

#include "common.cuh"
#include "ggml.h"

// T491 GGML_CUDA_B2_ABPAIR: the two GDN gate projections (ssm_beta, ssm_alpha: BF16 [K, 48] each) read the same fp32 activation.
// Prefill runs them as fp32->bf16 conversion + two small hipBLASLt GEMMs (3 dependent launches, ~103 us per layer and
// ubatch at N=1024, of which the conversion alone is 65 us). This computes both projections in ONE launch with bf16 WMMA,
// converting the activation in registers (round-to-nearest-even, as convert_unary does).
bool ggml_cuda_gdn_ab_proj_supported(const ggml_tensor * a, const ggml_tensor * b, int cc);
void ggml_cuda_gdn_ab_proj(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b);
