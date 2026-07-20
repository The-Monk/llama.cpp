// Phase-3(B) M-gated int4-weight dispatch: the M in [2,16] (batched
// spec-decode VERIFY) route, meant to wrap Composable Kernel's int4-weight
// WMMA GEMM (device_gemm_b_scale_wmma_f16_i4_f16, source at
// ~/src/ck-ref/example/01_gemm/gemm_wmma_fp16_pk_i4_v3_b_scale.cpp).
//
// STATUS (2026-07-20): SCAFFOLDING ONLY, NOT VALIDATED, NOT ENABLED.
// ggml_cuda_iu4_ck_wmma_supports() unconditionally returns false, so the
// M-switch (ggml-cuda.cu, GGML_HIP_IU4_MSWITCH) always falls through to the
// existing default GGML_TYPE_IU4 path for M in [2,16] -- this file has ZERO
// effect on any dispatch today. It is kept because the CMake link/compile
// integration (the part with real, durable value for whoever picks this up
// next) IS proven working; only the operand packing is unresolved.
//
// What was verified this session (see mul_mat_iu4_gemv.cuh's doctrine
// comment for the companion M==1 route's story):
//   1. CK's classic (ck::, not ck_tile::) device_gemm_wmma_cshuffle_v3_b_scale
//      header compiles and LINKS cleanly against our HIP/gfx1201 toolchain
//      with nothing more than the two CK include dirs (ck-ref/include,
//      ck-ref/library/include) -- no CK cmake build system, no prebuilt
//      .so/.a needed (it's a header-only device-op template, same as every
//      CK "01_gemm" example). De-risks the integration mechanics.
//   2. The B-operand packing (ck::pk_i4_t, 2 int4 values/byte) could NOT be
//      reverse-engineered to bit-exact correctness within this session's
//      budget from the example source alone:
//        - PermuteB=true (the example's own config) applies a KPerBlock=64
//          host-side tile reshape (stage 1) followed by an 8-nibble group
//          permute "01234567->20643175" (stage 2). Stage 1 writes via
//          HostTensorDescriptor's single-index operator() (confirmed, by
//          reading GetOffsetFromMultiIndex's inner_product-over-shorter-
//          range behavior, to be a RAW FLAT byte offset into the
//          [K0,N,K1]-tiled layout); stage 2 reads/writes via the SAME
//          Tensor object's 2-index operator() (which resolves through the
//          ORIGINAL [K,N] col-major strides {1,K}). For N>1 these two
//          addressing schemes touch DIFFERENT physical flat positions --
//          i.e. stage 2, as transcribed directly from the example, does
//          NOT operate on the tile-reshaped data stage 1 produced. This
//          could be a genuine subtlety intended by the CK authors (e.g.
//          relying on specific loop-order/aliasing behavior not obvious
//          from a single static read) or a transcription blind spot on my
//          part -- either way, porting it byte-for-byte without a working
//          CK reference harness to diff against was assessed as too high
//          a correctness risk to ship silently.
//        - PermuteB=false (skips the host reshape entirely -- the
//          O(1)-effort escape hatch, since the DeviceGemm_BScale_Wmma_
//          CShuffleV3 template takes PermuteB as a compile-time bool and
//          the kernel is presumably self-consistent for either setting)
//          WAS tried as the cheaper alternative: compiles, links,
//          IsSupportedArgument() accepts the argument, kernel runs without
//          crashing -- but produces WRONG results (max_rel_err ~2e4) against
//          a CPU fp64 reference, for BOTH nibble-order conventions tried
//          (k_even=low/k_odd=high, and the reference-decode-implied
//          k_even=high/k_odd=low). So PermuteB=false is not simply "the
//          untiled path" for this pipeline version -- something else in
//          the operand contract (scale indexing, A-side, or the 4-nibble-
//          group instruction order the underlying WMMA builtin itself
//          expects) is still unaccounted for.
//   Both probes (probe2.cpp/probe3.cpp under the session scratchpad) are
//   preserved in this branch's commit message / session notes for whoever
//   picks this up; not checked into the tree (they are throwaway host-side
//   probes, not part of the ggml build).
//
// Per the task brief's explicit fallback clause ("if the packing
// reconciliation turns out to be multi-day, STOP at a correctness-gated
// GEMV-only M=1 route + a documented CK-wrapper stub") -- this IS that
// stop point. The M=1 GEMV route (mul_mat_iu4_gemv.cu) is fully wired,
// selftested, and real; this M in [2,16] CK route is not.
#pragma once

#include "common.cuh"

// Always returns false today (see doctrine comment above) -- kept as a
// real function (not deleted) so the M-switch's call site in ggml-cuda.cu
// needs no further edits once this route is completed by a future pass.
bool ggml_cuda_iu4_ck_wmma_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// Not reachable via the M-switch today (supports() gates it out first),
// kept for completeness/symmetry with the GEMV route and so the scaffold
// compiles as a real callable.
bool ggml_cuda_op_mul_mat_iu4_ck_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// Honesty-gated selftest (house pattern): actually runs the CK device op
// end-to-end against a CPU reference and reports the TRUE result --
// currently FAIL, by design (see doctrine comment). Opt-in via
// GGML_HIP_MUL_MAT_IU4_CK_WMMA_SELFTEST; a PASS here (after a future fix)
// is the correctness gate that must clear before ggml_cuda_iu4_ck_wmma_supports()
// may be changed to return true.
bool ggml_cuda_mul_mat_iu4_ck_wmma_selftest();
