// SWMMAC prefill kernel for GGML_TYPE_2OF4_T1 (2:4-structured-sparse ternary,
// q2.4): routes the batch>8 MUL_MAT path through the RDNA4 sparse tensor
// pipes (v_swmmac_i32_16x16x32_iu8) instead of the dequant-to-f16 fallback.
//
// Shape/plumbing adapts the validated T162 capstone kernel
// (mul_mat_2of4_fp8_mmq.cu): cooperative BM x BN output tile, NWARPS
// warps/block, register-blocked SWMMAC tiles/warp, LDS staging, one
// syncthreads pair per staged K-chunk. Differences, all forced by the type:
//
//  * One staged K-chunk = ONE block_2of4_t1 = K=128 = FOUR K32 SWMMAC
//    windows (fp8 blocks are K=32). Single-buffered (the KPS-style shape):
//    the 4-windows-per-sync granularity already amortizes barriers the way
//    the fp8 kernel needed KPS=4 for, and the LDS budget (21.5 KB at
//    128x128) keeps 2 blocks/CU resident where a double buffer would not.
//
//  * The A (weight) operand is assembled IN-KERNEL from the native block
//    storage -- NO repack pass, NO side buffer: the survivor codes
//    {+1,-1,0} come from sgn_lut[sign nibble] & msk_lut[meta byte] (the
//    SAME two LUTs the validated mmvq decode path, vec_dot_2of4_t1_q8_1,
//    uses -- including the v1.1 single/empty-group spare states, where the
//    masked slot's code is 0 so whatever position its idx field names
//    contributes exactly 0). The meta bytes themselves ARE the ISA
//    sparsity_idx field (same 2-bit-pair packing as block_2of4_fp8.meta,
//    which the fp8 kernel feeds to the hardware unchanged); each lane takes
//    its k-half's 16 bits, exactly like the fp8 kernel.
//
//  * B is the activation, online-quantized to per-32 int8 (q8_1-granularity
//    scale, no offset needed: ternary weights are symmetric, zero-mean
//    correction does not apply -- same reasoning as the decode path's
//    "the -sum(u) correction does not apply here" note).
//
//  * int32 SWMMAC result x (per-128 weight scale d) x (per-32 activation
//    scale) accumulated in fp32, c0=0 per call -- the fp8 kernel's exact
//    fixup pattern, with the weight-scale LDS read hoisted out of the
//    window loop (t1's d covers all four windows of the chunk).
//
// Dispatch: hooked at the top of ggml_cuda_mul_mat for
// src0->type == GGML_TYPE_2OF4_T1 && src1->ne[1] > MMVQ_MAX_BATCH_SIZE,
// default ON; GGML_HIP_2OF4_T1_MMQ=0 restores the dequant fallback (the
// PPL A/B lever). Decode (ne1 <= 8) keeps the existing mmvq path. MUL_MAT
// only (no MUL_MAT_ID), single-GPU (no split buffers), RDNA4 only.
#pragma once

#include "common.cuh"

bool ggml_cuda_op_mul_mat_2of4_t1_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
