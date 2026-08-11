#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>

// ROC9->runtime-toggle: experimental RDNA4 decode levers, shipped as ONE
// binary with env-var switches rather than separate builds. Each flag is
// read ONCE per process (static local, magic-statics init) and gates BOTH
// the kernel-side dispatch (which vec_dot/vdr template instantiation gets
// launched) and the matching host-side activation-quantizer call, so the
// two sides never drift out of sync. Default OFF on every flag = the
// ship-safe path (matches what every other quant type does). Mirrors the
// existing GGML_HIP_Q2_0_WMMA_DECODE pattern (ggml-cuda.cu, intercepts at
// the ggml_cuda_mul_mat call site) -- same idea, applied at the mmvq
// dispatch/launch site instead since this lever lives inside the generic
// mul_mat_vec_q template rather than a standalone kernel file.
//
//   GGML_HIP_F8E5M2_DOT4 (default: unset/OFF)
//     OFF (default) -> vec_dot_f8e5m2_q8_1_simd_dispatch + int8 q8_1
//                       activations (quantize_row_q8_1_cuda). Lossless,
//                       matches every other quant type's decode contract.
//     ON  (=1)       -> vec_dot_f8e5m2_f8e5m2_dispatch (native bf8xbf8
//                       V_DOT4) + bf8 activations (quantize_row_f8e5m2_for_
//                       mmvq_cuda). Measured +25% at npl8 verify-batch,
//                       +2.5% relative PPL increase (card 137 gate) --
//                       real speed for a real, disclosed accuracy cost.
//                       NOT the ship default; opt in for batched/verify
//                       workloads where the accuracy cost is acceptable.
//
// Both instantiations are compiled into every build; the flag only picks
// which one gets LAUNCHED, so there is no rebuild between the two.
//
//   GGML_RDNA4_PROFILE=default|batch|mtp|self-draft (default: "default")
//     A THIN resolver bundling the individual flags above -- not a new
//     lever, just a convenience name for a combination. Resolved once/
//     process, same magic-statics pattern.
//       default    -> every flag's own ship-safe default (F8E5M2_DOT4 off).
//       batch      -> F8E5M2_DOT4 on (the accuracy-tolerant batched-verify
//                     shape the flag's own doc already names as the
//                     intended opt-in use case).
//       mtp        -> F8E5M2_DOT4 on (MTP verify is also a batched, small-N,
//                     accuracy-tolerant shape).
//       self-draft -> F8E5M2_DOT4 on (same rationale as mtp).
//     An EXPLICITLY-SET individual flag always overrides the profile
//     (checked first in ggml_cuda_f8e5m2_dot4_enabled below). NOTE:
//     GGML_HIP_F8E5M2_DOT4 uses the same PRESENCE-check convention as every
//     other GGML_HIP_* flag in this file (set = on, regardless of value --
//     `=0` still counts as "set"), so the override is "is the var present
//     at all", not "what value does it hold".
enum class ggml_rdna4_profile {
    DEFAULT,
    BATCH,
    MTP,
    SELF_DRAFT,
};

static ggml_rdna4_profile ggml_cuda_rdna4_profile() {
    static const ggml_rdna4_profile profile = [] {
        const char * env = getenv("GGML_RDNA4_PROFILE");
        if (env == nullptr) {
            return ggml_rdna4_profile::DEFAULT;
        }
        if (strcmp(env, "batch") == 0) {
            return ggml_rdna4_profile::BATCH;
        }
        if (strcmp(env, "mtp") == 0) {
            return ggml_rdna4_profile::MTP;
        }
        if (strcmp(env, "self-draft") == 0) {
            return ggml_rdna4_profile::SELF_DRAFT;
        }
        return ggml_rdna4_profile::DEFAULT; // unrecognized value: safest fallback
    }();
    return profile;
}

static bool ggml_cuda_f8e5m2_dot4_enabled() {
    static const bool enabled = [] {
        // Individual flag always overrides the profile, if explicitly set
        // (presence-check convention, matches every other GGML_HIP_* flag
        // in this codebase -- the VALUE doesn't matter, only whether the
        // variable is present in the environment at all).
        if (getenv("GGML_HIP_F8E5M2_DOT4") != nullptr) {
            return true;
        }
        switch (ggml_cuda_rdna4_profile()) {
            case ggml_rdna4_profile::BATCH:
            case ggml_rdna4_profile::MTP:
            case ggml_rdna4_profile::SELF_DRAFT:
                return true;
            case ggml_rdna4_profile::DEFAULT:
            default:
                return false;
        }
    }();
    return enabled;
}

// GGML_HIP_FUSE_MMVQ_QUANT (default: unset/OFF, presence-check convention
// like every other GGML_HIP_* flag here): fuses the quantize_row_q8_1
// activation quantize into the mmvq decode kernel itself (one launch per
// matmul instead of two). Gated at the ggml_cuda_mul_mat_vec_q call site to
// the standard int8 q8_1 decode path only -- see the eligibility check
// there (ggml_cuda_mmvq_fuse_quant_eligible). Runtime kernel param, not a
// template bool, so it costs zero extra template instantiations.
static bool ggml_cuda_fuse_mmvq_quant_enabled() {
    static const bool enabled = getenv("GGML_HIP_FUSE_MMVQ_QUANT") != nullptr;
    return enabled;
}

// T180 (env GGML_HIP_DEDUP_MMVQ_QUANT, default OFF): dedup the
// quantize_row_q8_1 dispatch across sibling mmvq matmuls that share one
// input tensor, INSTEAD of fusing the quantize into the mmvq kernel itself
// (that's GGML_HIP_FUSE_MMVQ_QUANT above, measured -56 to -84% regression on
// 2026-07-23 -- redundant per-row-block quantize competes with dp4a on the
// SAME VALU pipe rather than overlapping it). This lever keeps quantize as a
// separate launch (preserves the packed dp4a path untouched) and only skips
// launches that would recompute byte-identical output for a sibling matmul
// reading the exact same activation tensor (e.g. wq/wk/wv, ffn_gate/ffn_up,
// wqkv/wqkv_gate, ssm_alpha/ssm_beta within one decoder layer).
static bool ggml_cuda_dedup_mmvq_quant_enabled() {
    static const bool enabled = getenv("GGML_HIP_DEDUP_MMVQ_QUANT") != nullptr;
    return enabled;
}

// T180-verify (env GGML_HIP_DEDUP_MMVQ_QUANT_BATCH, default OFF, REQUIRES the
// base GGML_HIP_DEDUP_MMVQ_QUANT flag too): widen the sibling quant-dedup
// above to ncols_dst>1 (ne11>1) -- i.e. a spec-decode VERIFY pass batching
// n_draft+1 candidate tokens through the target in one forward, not just the
// ne11==1 raw single-token decode path the base flag was scoped to. The
// underlying redundancy is IDENTICAL at any batch size: sibling matmuls
// (wq/wk/wv, wqkv/wqkv_gate, ssm_alpha/ssm_beta) still read the exact same
// activation tensor whether it holds 1 token or N -- confirmed reachable via
// rocprofv3 kernel-trace showing mul_mat_vec_q<type,3,...> (ncols_dst=3) as
// the dominant kernel during a Q2_0 MTP n_max=2 verify pass. Separate flag
// (not folded into the base) so the raw single-stream win (shipped,
// 28ec66b65) and this verify-pass extension can be A/B'd independently.
static bool ggml_cuda_dedup_mmvq_quant_batch_enabled() {
    static const bool enabled = getenv("GGML_HIP_DEDUP_MMVQ_QUANT_BATCH") != nullptr;
    return enabled;
}

// Cap on the dynamic shared memory the fused quantizer is allowed to
// request (one block_q8_1 per 32 activations). Conservative vs. RDNA4's
// 64KB LDS/block, leaving headroom for the kernel's existing static shared
// arrays (tmp_shared/tmp_shared_gate). Falls back to the unfused path above
// this size instead of risking an LDS overflow.
#define GGML_HIP_FUSE_MMVQ_QUANT_MAX_SHARED_BYTES 49152

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_F8E4M3:  return vec_dot_f8e4m3_q8_1;
        case GGML_TYPE_F8E5M2:  return vec_dot_f8e5m2_q8_1;
        case GGML_TYPE_MXFP8:   return vec_dot_mxfp8_q8_1;
        case GGML_TYPE_MXFP6:   return vec_dot_mxfp6_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_F8E4M3:  return VDR_F8E4M3_Q8_1_MMVQ;
        case GGML_TYPE_F8E5M2:  return VDR_F8E5M2_Q8_1_MMVQ;
        case GGML_TYPE_MXFP8:   return VDR_MXFP8_Q8_1_MMVQ;
        case GGML_TYPE_MXFP6:   return VDR_MXFP6_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

// T77 (wide-SIMD decode): VDR/ILP for the batched (ncols_dst>1) hardware-
// dot2 path (vec_dot_f8e4m3_q8_1_simd_impl, vecdotq.cuh). Sweep lever --
// intentionally a plain #define in mmvq.cu (NOT vecdotq.cuh) so the sweep
// only pays the cheap mmvq.cu-only rebuild (ccache ~30-60s), never the full
// template-instance recompile. JM-directed sweep: 2, 4, 6, 8 (T75 showed ILP
// is the real lever, not load width; T68 showed VDR/nwarps sweeps are NOT
// monotonic -- occupancy/register-pressure cliffs are real and reproducible,
// so every value here must be benched, not assumed).
#define VDR_F8E4M3_Q8_1_MMVQ_SIMD 8

static __device__ __forceinline__ float vec_dot_f8e4m3_q8_1_simd_dispatch(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_f8e4m3_q8_1_simd_impl<VDR_F8E4M3_Q8_1_MMVQ_SIMD>(vbq, bq8_1, kbx, iqs);
}

// T79: pure V_DOT4_F32_FP8_FP8 path (both operands native e4m3 -- see
// vec_dot_f8e4m3_f8e4m3_impl, vecdotq.cuh, for the full design/accuracy-
// disclosure comment). VDR sweep lever, same cheap-rebuild-only convention
// T77 established (plain #define here, NOT in vecdotq.cuh). JM-directed
// sweep: 2, 4, 6, 8.
#define VDR_F8E4M3_F8E4M3_MMVQ_DOT4 2

static __device__ __forceinline__ float vec_dot_f8e4m3_f8e4m3_dispatch(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_f8e4m3_f8e4m3_impl<VDR_F8E4M3_F8E4M3_MMVQ_DOT4>(vbq, bq8_1, kbx, iqs);
}

// T97: F8E5M2 decode dispatch. Deliberately mirrors T77 (hardware-dot2,
// activations stay native int8 q8_1), NOT T79 (pure bf8xbf8 V_DOT4) -- the
// dot4 path would require its own online bf8 activation quantizer (a new
// quantize.cu kernel) and its own accuracy re-validation (e5m2 activations
// have only 2 mantissa bits, so the quantization error would likely be
// larger than T79's already-disclosed +1.2-1.5% e4m3-activation hit). Given
// the doctrine for this type (correctness + completeness gate, not required
// to beat E4M3's speed), the lower-risk int8-activation hardware-dot2 path
// is the right default; a future bf8xbf8 dot4 path is a valid next lever
// (see wiki T97), not built this session.
// VDR value inherited from T77's e4m3 finding (8), NOT independently swept
// for bf8 -- flag this if it ever becomes the speed-critical path.
#define VDR_F8E5M2_Q8_1_MMVQ_SIMD 8

static __device__ __forceinline__ float vec_dot_f8e5m2_q8_1_simd_dispatch(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_f8e5m2_q8_1_simd_impl<VDR_F8E5M2_Q8_1_MMVQ_SIMD>(vbq, bq8_1, kbx, iqs);
}

// Card 137 fix 2: pure V_DOT4_F32_BF8_BF8 path (both operands native bf8 --
// see vec_dot_f8e5m2_f8e5m2_impl, vecdotq.cuh, for the full design/accuracy-
// disclosure comment). This is the T97 "next lever" the comment above
// flagged: RDNA4 has a native bf8xbf8 dot4 opcode right beside fp8's
// (ISA-confirmed, V_DOT4_F32_BF8_BF8 opcode 39). VDR sweep lever, same
// cheap-rebuild-only convention T77/T79 established (plain #define here,
// NOT in vecdotq.cuh).
#define VDR_F8E5M2_F8E5M2_MMVQ_DOT4 2

static __device__ __forceinline__ float vec_dot_f8e5m2_f8e5m2_dispatch(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_f8e5m2_f8e5m2_impl<VDR_F8E5M2_F8E5M2_MMVQ_DOT4>(vbq, bq8_1, kbx, iqs);
}

// MXFP8 decode dispatch: T77 hardware-dot2 (ggml_cuda_dot2_e4m3_q8), activations
// stay native int8 q8_1 -- mirrors F8E5M2 above (lossless, no activation swap,
// NOT F8E4M3's T79 dot4). MXFP8 values are e4m3 so the same hw weight-decode
// applies; only the e8m0 scale differs. VDR=8 (whole 32-value block per call),
// same as the F8E5M2/T77 finding; the fallback to the portable scalar impl lives
// INSIDE vec_dot_mxfp8_q8_1_simd_impl (#if GGML_CUDA_F8E4M3_HAS_NATIVE_DOT2/#else),
// so this stays correct on non-RDNA4 HIP/CUDA/MUSA builds.
#define VDR_MXFP8_Q8_1_MMVQ_SIMD 8

static __device__ __forceinline__ float vec_dot_mxfp8_q8_1_simd_dispatch(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_mxfp8_q8_1_simd_impl<VDR_MXFP8_Q8_1_MMVQ_SIMD>(vbq, bq8_1, kbx, iqs);
}

// ROC8/ROC9 v2: MXFP6 decode dispatch, mirrors MXFP8 exactly (T77
// hardware-dot2, int8 q8_1 activations unchanged -- only the weight-side
// CK-style whole-block upconvert differs, see vecdotq.cuh /
// mxfp6_load_block, common.cuh). VDR=8 (whole 32-value block per call),
// same value MXFP8/F8E5M2 settled on.
#define VDR_MXFP6_Q8_1_MMVQ_SIMD 8

static __device__ __forceinline__ float vec_dot_mxfp6_q8_1_simd_dispatch(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_mxfp6_q8_1_simd_impl<VDR_MXFP6_Q8_1_MMVQ_SIMD>(vbq, bq8_1, kbx, iqs);
}

// T73 (small-batch decode fix): F8E4M3-only, batch-size-aware VDR/vec_dot
// selection, used ONLY by the main mul_mat_vec_q kernel (ncols_dst is a real
// compile-time template parameter there, so this is a free, zero-runtime-
// cost dispatch -- NOT used by mul_mat_vec_q_moe or the should_use_small_k
// host-side heuristic, both of which still see the type-only VDR=2 view;
// MoE has no validated F8E4M3 model yet, per T68, and small_k's heuristic
// staleness here is a minor perf-only risk, not a correctness one). BS=1
// keeps VDR=2 (the T68-validated decode kernel, unchanged); BS>1 (e.g. MTP
// verify) now gets the T77 hardware-dot2 SIMD path (VDR_F8E4M3_Q8_1_MMVQ_SIMD
// above) instead of the T73 scalar-VDR=4 "wide" path -- see T77 KB for the
// A/B that justified the swap.
// T79: the pure dot4 path (both operands native e4m3) is not an ILP/batch
// lever the way T73/T77's split was -- it's a strictly cheaper way to do
// the SAME per-term work at every batch size (per the ISA audit, backlog
// item 1c: T77's technique should also win at BS=1). So F8E4M3 now always
// takes vec_dot_f8e4m3_f8e4m3_dispatch here, superseding the old
// ncols_dst>1-gated T77 hardware-dot2 (int8-activation) split entirely --
// that path is left defined above (dead code on this dispatch, still used
// nowhere else) rather than deleted, so a revert is a one-line diff if the
// BS=1 A/B below doesn't hold. REQUIRES the matching activation-quantize
// swap at the ggml_cuda_mul_mat_vec_q call site (this file) -- the two
// changes are correctness-coupled, see the accuracy-disclosure comment on
// vec_dot_f8e4m3_f8e4m3_impl (vecdotq.cuh).
static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda_decode(ggml_type type, int ncols_dst, bool f8e5m2_dot4 = false) {
    if (type == GGML_TYPE_F8E4M3) {
        return vec_dot_f8e4m3_f8e4m3_dispatch;
    }
    // ROC9->runtime-toggle (GGML_HIP_F8E5M2_DOT4, see file-top comment
    // block): built the pure bf8xbf8 V_DOT4 path
    // (vec_dot_f8e5m2_f8e5m2_dispatch below) mirroring T79's F8E4M3
    // supersede, and ISA-confirmed it emits the native v_dot4_f32_bf8_bf8
    // opcode with the predicted register win (mmvq decode kernel VGPR
    // dropped ~102->27, matching fp8's T79 profile). BUT the mandatory PPL
    // gate (card 137 KB, Qwen3.6-27B-F8E5M2, README.md corpus, 6 chunks,
    // `-ub 4` to force this exact mmvq path -- PPL is bit-deterministic so a
    // single fresh run is reproducible, no drift/resample caveat needed)
    // shows it is measurably WORSE on accuracy, not just noise: int8-
    // activation baseline PPL 2.5162 +/- 0.13044 vs bf8-activation dot4 PPL
    // 2.5799 +/- 0.13637 -- a real +2.5% relative increase, worse in 4/6
    // chunks, exactly the direction the accuracy-disclosure comment on
    // vec_dot_f8e5m2_f8e5m2_impl (vecdotq.cuh) predicted (bf8 activations
    // have only 2 mantissa bits, a strictly larger quantization step than
    // T79's already-costly e4m3-activation swap). It's ALSO measurably
    // FASTER at batched/verify shapes (+25% at npl8, see mmvq.cu top-of-file
    // GGML_HIP_F8E5M2_DOT4 doc). Speed-vs-accuracy tradeoff, not a strict
    // win -- so this is a ship-time CHOICE, not a fixed default: default
    // OFF (int8-activation hardware-dot2, lossless, matches every other
    // quant type's decode contract), opt-in ON for accuracy-tolerant
    // batched/verify/spec-decode workloads. Both instantiations compile into
    // every build (see mul_mat_vec_q_switch_type's F8E5M2 case) -- this
    // function only picks which one a given kernel INSTANCE embeds.
    if (type == GGML_TYPE_F8E5M2) {
        return f8e5m2_dot4 ? vec_dot_f8e5m2_f8e5m2_dispatch : vec_dot_f8e5m2_q8_1_simd_dispatch;
    }
    // ROC8: MXFP8 T77 hardware-dot2 decode (same lossless int8-activation path
    // as F8E5M2 above; fallback lives inside the simd_impl for non-RDNA4).
    if (type == GGML_TYPE_MXFP8) {
        return vec_dot_mxfp8_q8_1_simd_dispatch;
    }
    // ROC8: MXFP6 T77 hardware-dot2 decode (same lossless int8-activation path).
    if (type == GGML_TYPE_MXFP6) {
        return vec_dot_mxfp6_q8_1_simd_dispatch;
    }
    GGML_UNUSED(ncols_dst);
    return get_vec_dot_q_cuda(type);
}

static constexpr __host__ __device__ int get_vdr_mmvq_decode(ggml_type type, int ncols_dst, bool f8e5m2_dot4 = false) {
    if (type == GGML_TYPE_F8E4M3) {
        return VDR_F8E4M3_F8E4M3_MMVQ_DOT4;
    }
    if (type == GGML_TYPE_F8E5M2) {
        // ROC9->runtime-toggle: see get_vec_dot_q_cuda_decode above, same flag.
        return f8e5m2_dot4 ? VDR_F8E5M2_F8E5M2_MMVQ_DOT4 : VDR_F8E5M2_Q8_1_MMVQ_SIMD;
    }
    if (type == GGML_TYPE_MXFP8) {
        return VDR_MXFP8_Q8_1_MMVQ_SIMD;
    }
    if (type == GGML_TYPE_MXFP6) {
        return VDR_MXFP6_Q8_1_MMVQ_SIMD;
    }
    GGML_UNUSED(ncols_dst);
    return get_vdr_mmvq(type);
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // F8E4M3 MUL_MAT_ID (T81, fp8-moe branch, completes the deferred MoE
    // path T79 left open): mul_mat_vec_q_moe now dispatches F8E4M3 through
    // get_vec_dot_q_cuda_decode/get_vdr_mmvq_decode (see that kernel, above),
    // which correctly matches the native-e4m3 activation quantization
    // ggml_cuda_mul_mat_vec_q always performs for this type (regardless of
    // whether `ids` is set) -- fixed alongside this gate opening; before that
    // fix, opening this gate would have fed e4m3 activation bytes to the
    // int8-q8_1 dot (vec_dot_f8e4m3_q8_1) and produced silently wrong output.
    // RDNA4-gated explicitly here (unlike most entries in this function,
    // which gate by arch via the table dispatch below): this function's own
    // call site (ggml_cuda_mul_mat_id, ggml-cuda.cu) has no separate cc check
    // before calling ggml_cuda_mul_mat_vec_q, unlike the dense MUL_MAT path
    // where ggml_cuda_should_use_mmvq() itself gates on GGML_CUDA_CC_IS_RDNA4
    // before that function is ever invoked -- so the gate has to live here.
    // Cap = MMVQ_MAX_BATCH_SIZE. Phase-2 perf sweep DONE (2026-07-10): the flat
    // cap is CONFIRMED correct on RDNA4 (gfx1201) for F8E4M3 MoE -- drift-
    // controlled interleaved A/B on Qwen3.6-35B-A3B-F8E4M3, mmvq(dp4a) beats
    // mmq(WMMA) at M=8 by ~8% (257 vs 238 t/s, 4/4 pairs). Crossover is >8 (same
    // as the dense path, measured 1-16), so mmvq for M<=8 / mmq for M>8 is
    // optimal; no per-type lowering needed. NOTE: MoE tokens scatter across
    // experts under top-k routing at small M, so no cache reuse -> bandwidth-
    // bound like dense -> the crossover does NOT slide below 8 as cache-residency
    // alone would suggest. (Interleaving is required to measure this: separate
    // 2-GPU tensor-split invocations jitter +/-30% and gave a false WMMA-wins-at-8.)
    if (type == GGML_TYPE_F8E4M3) {
        return GGML_CUDA_CC_IS_AMD(cc) && GGML_CUDA_CC_IS_RDNA4(cc) ? MMVQ_MAX_BATCH_SIZE : 0;
    }
    // T97: same reasoning as F8E4M3 above -- no MoE F8E5M2 model exists to
    // validate against, keep MUL_MAT_ID on the dequant fallback.
    if (type == GGML_TYPE_F8E5M2) {
        return 0;
    }
    // ROC8: Qwen3.6-27B-MXFP8-MTP (the validated target model) is dense, not
    // MoE -- no MUL_MAT_ID path exists to test against, same conservative
    // "keep on the dequant fallback" stance as F8E5M2 above.
    if (type == GGML_TYPE_MXFP8) {
        return 0;
    }
    // ROC8: MXFP6 -- same rationale as MXFP8 above (dense-only target model,
    // no MUL_MAT_ID path exists to test/validate against).
    if (type == GGML_TYPE_MXFP6) {
        return 0;
    }
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    // F8E4M3 (Path X, Phase 2a): dedicated mmvq decode kernel now exists
    // (vec_dot_f8e4m3_q8_1, vecdotq.cuh) -- closes the batch-size gap left by
    // Phase 1b, where should_use_mmq() has no bs=1 split for this type and
    // decode was forced through the tile/WMMA prefill kernel. mul_mat_vec_q
    // is preferred over mul_mat_q whenever both are true (ggml-cuda.cu
    // dispatch order), so this is the actual decode/prefill split point --
    // mirrors Q8_0, which relies on the same precedence rather than an
    // explicit batch check in should_use_mmq. RDNA4-only: the vec_dot itself
    // is portable (software e4m3 decode, no HW dependency), but F8E4M3
    // tensors are only produced/loaded for gfx1201 today (Phase 1b's WMMA
    // prefill path is RDNA4-gated) -- keep other archs on the Phase 1a
    // dequant fallback until that's validated too.
    if (type == GGML_TYPE_F8E4M3) {
        return GGML_CUDA_CC_IS_RDNA4(cc) && ne11 <= MMVQ_MAX_BATCH_SIZE;
    }
    // T97: same RDNA4-only gate as F8E4M3 above -- no F8E5M2 tensors are
    // produced/loaded off gfx1201 today (this type has no WMMA/mmq prefill
    // kernel at all yet, unlike F8E4M3's Phase 1b -- prefill for F8E5M2
    // correctly falls back to the generic cuBLAS dequant path,
    // ggml_cuda_op_mul_mat_cublas via ggml_get_to_fp16_cuda, whenever mmvq
    // isn't applicable, e.g. ne11 > MMVQ_MAX_BATCH_SIZE).
    if (type == GGML_TYPE_F8E5M2) {
        return GGML_CUDA_CC_IS_RDNA4(cc) && ne11 <= MMVQ_MAX_BATCH_SIZE;
    }
    // ROC8: MXFP8 decode kernel (vec_dot_mxfp8_q8_1, vecdotq.cuh) is portable
    // (software e4m3/e8m0 decode, no HW dependency), but -- same rationale as
    // F8E4M3/F8E5M2 above -- MXFP8 tensors are only produced/loaded for
    // gfx1201 today (the WMMA prefill path below is RDNA4-gated), so keep
    // other archs on the dequant fallback until validated there too.
    if (type == GGML_TYPE_MXFP8) {
        return GGML_CUDA_CC_IS_RDNA4(cc) && ne11 <= MMVQ_MAX_BATCH_SIZE;
    }
    // ROC8: MXFP6 decode kernel (vec_dot_mxfp6_q8_1, vecdotq.cuh) is a lossless
    // upconvert onto the existing portable/hardware-dot2 e4m3 decode -- same
    // RDNA4-only gate as MXFP8 above (only produced/loaded for gfx1201 today;
    // no WMMA/mmq prefill kernel exists for this type, prefill correctly falls
    // back to the generic cuBLAS dequant path, ggml_cuda_op_mul_mat_cublas,
    // whenever mmvq isn't applicable).
    if (type == GGML_TYPE_MXFP6) {
        return GGML_CUDA_CC_IS_RDNA4(cc) && ne11 <= MMVQ_MAX_BATCH_SIZE;
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q1_0:
                case GGML_TYPE_Q2_0:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_F8E4M3:
                case GGML_TYPE_F8E5M2:
                case GGML_TYPE_MXFP8:
                case GGML_TYPE_MXFP6:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                case GGML_TYPE_MXFP4:
                    // Dispatch entry (arch=RDNA4, type=MXFP4): swept 1-8 on Qwen3.6-35B-A3B MoE
                    // decode. Confirmed on a clean paired re-test: baseline(1) 74.2 -> 3: 76.5
                    // t/s (+3.2%), tight variance, no pp512 impact.
                    return 3;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        if (ncols_dst == 1) {
            // Quant-aware rows/block for decode (gfx1201). Decode is memory-bound; rows/block sets
            // the number of independent weight-row load streams per block (memory-level parallelism),
            // but the concurrent rows are row_stride bytes apart, so the choice also interacts with
            // GDDR6 channel aliasing. Measured 2026-07-10:
            //   fp8 (8-bit, ~4x the per-row byte stride): rpb=3 = +6.5% over rpb=2 (UMC 75% vs 70%).
            //   Q2_0 (2-bit, tiny stride): rpb=2 optimal; rpb=4 channel-aliases (-13.4% pothole).
            //   That sweep was on the dense Bonsai-8B (Qwen3 arch).
            // Re-swept 2026-07-14 on Bonsai-27B (qwen35 hybrid-attn/GatedDeltaNet, different
            // matmul row shapes/strides than the 8B): rpb=3 reproducibly beats rpb=2 by ~2.3%
            // (5-build interleaved A/B: rpb2 50.31/49.82 vs rpb3 51.12/51.27/51.18 t/s tg128).
            // Re-checked the 8B on this same build: rpb2/3 still tied, no regression (160.8 vs
            // 161.9 t/s). PPL byte-identical between rpb=2/3 on the 27B (wikitext-2, 20 chunks:
            // 11.5576 +/- 0.47262 both) -- row-batching only changes grid parallelism, not the
            // math, so this is correctness-neutral. Moved Q2_0 into the rpb=3 bucket: shape-
            // dependent, not a flat per-quant constant -- ideally a per-(type,n_rows) dispatch
            // key, kept here pending that.
            // Re-swept 2026-07-16 (card 147) on GPU0 specifically (0000:04:00.0, the
            // previous 07-14 sweep didn't record which card) with the archive job
            // SIGSTOPped for a clean cold window, Bonsai-27B Q2_0, llama-bench tg128 -r5:
            //   rpb=1: 44.93+/-0.37  rpb=2: 47.26+/-0.24  rpb=3: 49.68+/-0.78  rpb=4: 48.62+/-0.43
            // rpb=3 confirmed still the GPU0 winner (monotonic 1<2<3>4), no regression
            // from the 07-14 result. Kept as-is.
            if (type == GGML_TYPE_F8E4M3 || type == GGML_TYPE_F8E5M2 || type == GGML_TYPE_Q2_0) {
                return 3;
            }
            // Q1_0 re-swept 2026-08-11 after the 2c-1 identity rewrite of its vec_dot
            // (the leaner kernel moved the optimum): Bonsai-27B tg128 -r3, GPU0:
            //   rpb=2 56.28  rpb=3 58.84  rpb=4 58.88  rpb=6 60.01  rpb=8 58.99  rpb=10 54.96
            // Monotonic rise to 6 then falloff; nwarps 1/2/4/8 all within noise at
            // rpb=6 (60.0-60.4), so only rpb changes. VDR=2 was also tried and lost
            // (~56 across geometries -- fewer lanes/block costs more than the extra
            // in-flight loads gain). Swept on the 27B only.
            if (type == GGML_TYPE_Q1_0) {
                return 6;
            }
            // Swept 2026-07-20 (Vulkan-decode-gap investigation) on GPU0, llama-bench
            // tg128 -r5, median-of-3+ reps per point:
            //   Devstral-13B (Q4_K_M): rpb=2 35.90/35.93/36.01 -> rpb=3 36.86/36.86/37.22
            //     (+2.9%, clearly outside the rpb=2 noise band).
            //   Qwen3-8B (Q4_K_M):     rpb=2 92.53/93.07/93.08 -> rpb=3 92.77/93.24/93.40/93.46
            //     (+0.3%, inside/at the edge of the rpb=2 noise band -- no regression).
            // pp512 unaffected on both (rpb only changes ncols_dst==1 decode grid shape).
            // Correctness: test-backend-ops MUL_MAT(q4_K) 43/43 + Paris check both models.
            // Narrows but does not close the ~13% Vulkan decode lead (isolated
            // test-backend-ops MUL_MAT perf is noise-flat, so the win is an
            // occupancy/grid-parallelism effect visible only at full-model scale).
            if (type == GGML_TYPE_Q4_K) {
                return 3;
            }
            return 2;
        }
        return 1;
    }
    return 1;
}

// ROC9->runtime-toggle: f8e5m2_dot4 (default false = ship-safe lossless) is
// a compile-time template bool, NOT a runtime branch inside this kernel --
// get_vdr_mmvq_decode/get_vec_dot_q_cuda_decode below must fold to a
// constexpr (vdr feeds __launch_bounds__-adjacent unroll/tile-size math).
// Both `true` and `false` instantiations of this kernel are compiled into
// the binary; mul_mat_vec_q_switch_type picks which one to LAUNCH based on
// the GGML_HIP_F8E5M2_DOT4 env var (see file-top comment block) -- exactly
// the "compile both, choose at runtime via the launch site" pattern
// GGML_HIP_Q2_0_WMMA_DECODE uses (ggml-cuda.cu), applied here at template-
// instantiation granularity since this lever lives inside the generic mmvq
// kernel rather than a standalone one. Meaningless (ignored) for every type
// other than F8E5M2.
template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool f8e5m2_dot4 = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride, const bool fuse_quant) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq_decode(type, ncols_dst, f8e5m2_dot4); // T73: F8E4M3 batch-aware (see def)
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda_decode(type, ncols_dst, f8e5m2_dot4); // T73

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    // Fused activation quantize (GGML_HIP_FUSE_MMVQ_QUANT): vy points at the
    // raw fp32 activation row instead of a pre-quantized block_q8_1 buffer.
    // Each block quantizes its own y row into dynamic shared memory before
    // the dot-product loop below, replacing the separate quantize_row_q8_1
    // kernel launch the host would otherwise issue. Same scale convention as
    // quantize_q8_1 (quantize.cu): d = amax/127, q = round(x/d), ds = (d, sum).
    // Only reachable when ncols_dst==1 and warp_size==QK8_1 (host-gated, see
    // ggml_cuda_mul_mat_vec_q); the `if constexpr` keeps this dead weight on
    // any build where physical warp_size != 32 (e.g. GCN/CDNA).
    extern __shared__ block_q8_1 mmvq_y_smem[];
    const block_q8_1 * y;
    if constexpr (ncols_dst == 1 && warp_size == QK8_1) {
        if (fuse_quant) {
            const float * GGML_CUDA_RESTRICT y_raw =
                ((const float *) vy) + (size_t) sample_y*stride_sample_y + (size_t) channel_y*stride_channel_y;
            const int nblocks_q8 = ncols_x / QK8_1;
            for (int ib = threadIdx.y; ib < nblocks_q8; ib += nwarps) {
                const float xi   = y_raw[ib*QK8_1 + threadIdx.x];
                float       amax = warp_reduce_max<QK8_1>(fabsf(xi));
                const float sum  = warp_reduce_sum<QK8_1>(xi);
                const float d    = amax / 127.0f;
                const int8_t q   = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
                mmvq_y_smem[ib].qs[threadIdx.x] = q;
                if (threadIdx.x == 0) {
                    mmvq_y_smem[ib].ds = make_half2(d, sum);
                }
            }
            __syncthreads();
            y = mmvq_y_smem;
        } else {
            y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
        }
    } else {
        y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
        GGML_UNUSED(fuse_quant);
    }
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    // T81 (MoE completeness, fp8-moe): use the *_decode dispatch here too, not
    // the bare get_vdr_mmvq/get_vec_dot_q_cuda. ggml_cuda_mul_mat_vec_q (the
    // sole caller that fills vy for this kernel) quantizes F8E4M3 activations
    // to native e4m3 UNCONDITIONALLY -- regardless of whether `ids` is set --
    // via quantize_row_f8e4m3_for_mmvq_cuda (see that function, this file).
    // The bare get_vdr_mmvq/get_vec_dot_q_cuda(F8E4M3) return the Phase-2a
    // int8-q8_1-activation dot (vec_dot_f8e4m3_q8_1), which would silently
    // reinterpret e4m3 activation bytes as int8 q8_1 -- garbage output. This
    // was a latent bug: unreachable today only because get_mmvq_mmid_max_batch
    // returns 0 for F8E4M3 (see below), so mul_mat_vec_q_moe is compiled but
    // never actually launched for F8E4M3 -- until that gate is opened for the
    // MoE decode path, at which point this mismatch would fire on real
    // output. The literal `1` passed as ncols_dst is intentionally a dummy:
    // get_vec_dot_q_cuda_decode/get_vdr_mmvq_decode for F8E4M3 ignore ncols_dst
    // entirely (always return the fp8xfp8 dot, per T79) and for every OTHER
    // type they fall straight back to get_vec_dot_q_cuda/get_vdr_mmvq
    // unchanged -- so this swap is a no-op for all non-F8E4M3 MoE types
    // (Q4_K, Q6_K, MXFP4, etc.) and only changes F8E4M3 behavior. Must stay a
    // compile-time literal (not a runtime ncols_dst) so `vdr` below can stay
    // constexpr (ncols_dst is a real runtime kernel arg in this kernel, unlike
    // the ncols_dst-templated mul_mat_vec_q above).
    constexpr int vdr = get_vdr_mmvq_decode(type, 1);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda_decode(type, 1);

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool f8e5m2_dot4 = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, const bool fuse_quant, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, f8e5m2_dot4>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, f8e5m2_dot4>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block>, launch_params,
        vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride);
}

template <ggml_type type, bool f8e5m2_dot4 = false>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, const bool fuse_quant, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
        constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
        constexpr int vdr                   = get_vdr_mmvq(type);
        const int     blocks_per_row_x      = ncols_x / qk;
        const int     blocks_per_iter_1warp = vdr * warp_size / qi;
        const int     nwarps                = calc_nwarps(type, c_ncols_dst, table_id);
        bool          use                   = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            constexpr int c_ncols_dst = 1;

            bool use_small_k = should_use_small_k(c_ncols_dst);

            // ncols_x is a multiple of QK8_1 here (GGML_ASSERT above via the
            // type's block size, which is itself a multiple of 32 for every
            // type this fuses -- see ggml_cuda_mmvq_fuse_quant_eligible).
            const int nbytes_shared_quant = fuse_quant ? (int) ((ncols_x / QK8_1) * sizeof(block_q8_1)) : 0;

            if (use_small_k) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id, true);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, true, f8e5m2_dot4>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, nbytes_shared_quant, ids_stride,
                    fuse_quant, stream);
            } else {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, f8e5m2_dot4>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, nbytes_shared_quant, ids_stride,
                    fuse_quant, stream);
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, f8e5m2_dot4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, false, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, f8e5m2_dot4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, false, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, f8e5m2_dot4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, false, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, f8e5m2_dot4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, false, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, f8e5m2_dot4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, false, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, f8e5m2_dot4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, false, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, f8e5m2_dot4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, false, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, const bool fuse_quant, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_F8E4M3:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_F8E4M3>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_F8E5M2:
            // ROC9->runtime-toggle (GGML_HIP_F8E5M2_DOT4, file-top comment
            // block): BOTH the dot4 and int8-dot2 template instantiations
            // are already compiled into this binary (mul_mat_vec_q<...,
            // f8e5m2_dot4=true|false>) -- this is the ONE place that picks
            // which gets launched, checked once/process, exactly mirroring
            // GGML_HIP_Q2_0_WMMA_DECODE's runtime intercept (ggml-cuda.cu).
            if (ggml_cuda_f8e5m2_dot4_enabled()) {
                mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_F8E5M2, true>
                    (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                     nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                     nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            } else {
                mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_F8E5M2, false>
                    (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                     nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                     nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            }
            break;
        case GGML_TYPE_MXFP8:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP8>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_MXFP6:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP6>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fuse_quant, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// GGML_HIP_TERNARY_FP16_GEMV (default: unset/OFF): full-precision ternary
// decode GEMV for GGML_TYPE_Q2_0. A Q2_0 code c encodes symbol s = c-1 in
// {-1,0,1} (see vec_dot_q2_0_q8_1 above); the weight is already exact, so
// dp4a's int8 multiply is spent entirely on a value the activation side
// still has to LOSE precision to reach (the q8_1 quantize). This kernel
// skips activation quantization entirely -- no q8_1, no dp4a -- and
// sign-select-accumulates the raw fp32 activation straight from src1,
// scaled once per 128-element block by that block's fp16 delta. Decode-
// only (ncols_dst==1, no MUL_MAT_ID): see ggml_cuda_ternary_fp16_gemv_
// enabled below for the eligibility gate.
static bool ggml_cuda_ternary_fp16_gemv_enabled() {
    static const bool enabled = getenv("GGML_HIP_TERNARY_FP16_GEMV") != nullptr;
    return enabled;
}

#define GGML_HIP_TERNARY_GEMV_ROWS_PER_BLOCK 3
#define GGML_HIP_TERNARY_GEMV_NWARPS 8

__launch_bounds__(GGML_HIP_TERNARY_GEMV_NWARPS*32, 1)
static __global__ void mul_mat_vec_ternary_fp16(
        const void * __restrict__ vx_ptr, const float * __restrict__ vy_ptr, float * __restrict__ dst_ptr,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t stride_row_x,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint3 sample_ratio, const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
    constexpr int rows_per_block = GGML_HIP_TERNARY_GEMV_ROWS_PER_BLOCK;
    constexpr int nwarps         = GGML_HIP_TERNARY_GEMV_NWARPS;

    const block_q2_0 * GGML_CUDA_RESTRICT vx = (const block_q2_0 *) vx_ptr;
    const float       * GGML_CUDA_RESTRICT vy = vy_ptr;
    float             * GGML_CUDA_RESTRICT dst = dst_ptr;

    const uint32_t row0 = rows_per_block*blockIdx.x;
    const uint32_t blocks_per_row_x = ncols_x / QK2_0;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);

    // Reads src1 directly (4B/elem, redundant once per output-row-group --
    // inherent to any GEMV grid, same pattern the dp4a baseline's q8_1
    // buffer already has). A fp16-staged variant (2B/elem) was tried and
    // measured WORSE on both speed and accuracy at 8B and 27B: the added
    // staging kernel's own launch + global-memory round-trip cost more than
    // the halved redundant-read bandwidth saved, and fp16 rounding ate most
    // of the accuracy gain at 27B. Reverted -- see wiki/tech/bonsai-ternary-
    // models.md for the numbers. Accumulate stays fp32 either way.
    const float * GGML_CUDA_RESTRICT y_row =
        vy + (size_t) sample_dst*stride_sample_y + (size_t) channel_dst*stride_channel_y;
    const int64_t kbx_offset =
        (int64_t) sample_x*stride_sample_x + (int64_t) channel_x*stride_channel_x + (int64_t) row0*stride_row_x;

    // One lane per qs byte (32 bytes/block = 4 codes/byte * 32 lanes = 128
    // elements/block = QK2_0). Each lane accumulates its byte-group's
    // contribution across all blocks this warp strides over, applying that
    // block's own delta before summing into the per-row running total --
    // only ONE warp_reduce per row is needed, after the loop, not per block.
    float row_tmp[rows_per_block] = {0.0f};
    for (uint32_t block = threadIdx.y; block < blocks_per_row_x; block += nwarps) {
        const float * y_blk = y_row + (size_t) block*QK2_0 + threadIdx.x*4;
        const float a0 = y_blk[0], a1 = y_blk[1], a2 = y_blk[2], a3 = y_blk[3];
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            const block_q2_0 * bq = vx + kbx_offset + (int64_t) r*stride_row_x + block;
            const uint8_t byte = bq->qs[threadIdx.x];
            const float s0 = (float) ((byte >> 0) & 0x3) - 1.0f;
            const float s1 = (float) ((byte >> 2) & 0x3) - 1.0f;
            const float s2 = (float) ((byte >> 4) & 0x3) - 1.0f;
            const float s3 = (float) ((byte >> 6) & 0x3) - 1.0f;
            const float part = s0*a0 + s1*a1 + s2*a2 + s3*a3;
            row_tmp[r] += part * __half2float(bq->d);
        }
    }

#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
        row_tmp[r] = warp_reduce_sum<32>(row_tmp[r]);
    }

    __shared__ float smem[nwarps-1][rows_per_block];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            smem[threadIdx.y-1][r] = row_tmp[r];
        }
    }
    __syncthreads();
    if (threadIdx.y != 0) {
        return;
    }
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
#pragma unroll
        for (int l = 0; l < nwarps-1; ++l) {
            row_tmp[r] += smem[l][r];
        }
    }

    if (threadIdx.x < rows_per_block && row0 + threadIdx.x < nrows_x) {
        dst[(size_t) sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0 + threadIdx.x] = row_tmp[threadIdx.x];
    }
}

static void ggml_cuda_mul_mat_vec_ternary_fp16(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;
    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    const int64_t s01 = src0->nb[1] / ts_src0; // blocks/row (block_q2_0 units)
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s2  = dst->nb[2] / ts_dst;
    const int64_t s3  = dst->nb[3] / ts_dst;
    GGML_UNUSED(s11);

    constexpr int rows_per_block = GGML_HIP_TERNARY_GEMV_ROWS_PER_BLOCK;
    const int64_t nblocks = (ne01 + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks, ne2, ne3);
    const dim3 block_dims(32, GGML_HIP_TERNARY_GEMV_NWARPS, 1);

    const uint3 channel_ratio_fd = init_fastdiv_values(ne2 / ne02);
    const uint3 sample_ratio_fd  = init_fastdiv_values(ne3 / ne03);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_ternary_fp16, launch_params,
        (const void *) src0->data, (const float *) src1->data, (float *) dst->data,
        (uint32_t) ne00, (uint32_t) ne01, (uint32_t) s01,
        channel_ratio_fd, (uint32_t) s02, (uint32_t) s12, (uint32_t) s2,
        sample_ratio_fd, (uint32_t) s03, (uint32_t) s13, (uint32_t) s3);
}

// Q1_0 port of the above, same flag (GGML_HIP_TERNARY_FP16_GEMV). Q1_0 is
// 1-bit/binary {-1,+1}, no zero case. Bit convention DERIVED from
// vec_dot_q1_0_q8_1 (vecdotq.cuh), not guessed: that impl reads
// v = qs[off+0] | qs[off+1]<<8 | qs[off+2]<<16 | qs[off+3]<<24 (standard
// little-endian byte packing), then for nibble j=0..7 of v (bits 4j..4j+3)
// extracts bit i=0..3 as b_i = bit(4j+i) ? +1 : -1 and dp4a's it against
// y bytes [4j+0..4j+3] -- i.e. weight element e (0-indexed within the
// 128-elem block) is bit (e%8) of byte qs[e/8], LSB-first, decoded as
// ((qs[e/8] >> (e%8)) & 1) ? +1 : -1. No "-sum(u)" correction term either
// (unlike Q2_0's code-1 identity) -- vec_dot_q1_0_q8_1 returns d1*d8*sumi
// directly, so this kernel's math is a straight sign-select accumulate.
#define GGML_HIP_BINARY_GEMV_ROWS_PER_BLOCK 2
#define GGML_HIP_BINARY_GEMV_NWARPS 8

__launch_bounds__(GGML_HIP_BINARY_GEMV_NWARPS*32, 1)
static __global__ void mul_mat_vec_binary_fp16(
        const void * __restrict__ vx_ptr, const float * __restrict__ vy_ptr, float * __restrict__ dst_ptr,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t stride_row_x,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint3 sample_ratio, const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
    constexpr int rows_per_block = GGML_HIP_BINARY_GEMV_ROWS_PER_BLOCK;
    constexpr int nwarps         = GGML_HIP_BINARY_GEMV_NWARPS;

    const block_q1_0 * GGML_CUDA_RESTRICT vx = (const block_q1_0 *) vx_ptr;
    const float       * GGML_CUDA_RESTRICT vy = vy_ptr;
    float             * GGML_CUDA_RESTRICT dst = dst_ptr;

    const uint32_t row0 = rows_per_block*blockIdx.x;
    const uint32_t blocks_per_row_x = ncols_x / QK1_0;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);

    const float * GGML_CUDA_RESTRICT y_row =
        vy + (size_t) sample_dst*stride_sample_y + (size_t) channel_dst*stride_channel_y;
    const int64_t kbx_offset =
        (int64_t) sample_x*stride_sample_x + (int64_t) channel_x*stride_channel_x + (int64_t) row0*stride_row_x;

    // One lane per nibble (16 qs bytes/block * 2 nibbles/byte = 32 lanes,
    // 4 elements/nibble = 128 elements/block = QK1_0) -- same per-lane
    // 4-element workload as the Q2_0 ternary kernel, just a 1-bit decode
    // instead of a 2-bit LUT.
    float row_tmp[rows_per_block] = {0.0f};
    for (uint32_t block = threadIdx.y; block < blocks_per_row_x; block += nwarps) {
        const float * y_blk = y_row + (size_t) block*QK1_0 + threadIdx.x*4;
        const float a0 = y_blk[0], a1 = y_blk[1], a2 = y_blk[2], a3 = y_blk[3];
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            const block_q1_0 * bq = vx + kbx_offset + (int64_t) r*stride_row_x + block;
            const uint8_t qbyte  = bq->qs[threadIdx.x/2];
            const uint8_t nibble = (threadIdx.x % 2 == 0) ? (qbyte & 0x0F) : (qbyte >> 4);
            const float s0 = (nibble & 0x1) ? 1.0f : -1.0f;
            const float s1 = (nibble & 0x2) ? 1.0f : -1.0f;
            const float s2 = (nibble & 0x4) ? 1.0f : -1.0f;
            const float s3 = (nibble & 0x8) ? 1.0f : -1.0f;
            const float part = s0*a0 + s1*a1 + s2*a2 + s3*a3;
            row_tmp[r] += part * __half2float(bq->d);
        }
    }

#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
        row_tmp[r] = warp_reduce_sum<32>(row_tmp[r]);
    }

    __shared__ float smem[nwarps-1][rows_per_block];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            smem[threadIdx.y-1][r] = row_tmp[r];
        }
    }
    __syncthreads();
    if (threadIdx.y != 0) {
        return;
    }
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
#pragma unroll
        for (int l = 0; l < nwarps-1; ++l) {
            row_tmp[r] += smem[l][r];
        }
    }

    if (threadIdx.x < rows_per_block && row0 + threadIdx.x < nrows_x) {
        dst[(size_t) sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0 + threadIdx.x] = row_tmp[threadIdx.x];
    }
}

static void ggml_cuda_mul_mat_vec_binary_fp16(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;
    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    const int64_t s01 = src0->nb[1] / ts_src0; // blocks/row (block_q1_0 units)
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s2  = dst->nb[2] / ts_dst;
    const int64_t s3  = dst->nb[3] / ts_dst;
    GGML_UNUSED(s11);

    constexpr int rows_per_block = GGML_HIP_BINARY_GEMV_ROWS_PER_BLOCK;
    const int64_t nblocks = (ne01 + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks, ne2, ne3);
    const dim3 block_dims(32, GGML_HIP_BINARY_GEMV_NWARPS, 1);

    const uint3 channel_ratio_fd = init_fastdiv_values(ne2 / ne02);
    const uint3 sample_ratio_fd  = init_fastdiv_values(ne3 / ne03);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_binary_fp16, launch_params,
        (const void *) src0->data, (const float *) src1->data, (float *) dst->data,
        (uint32_t) ne00, (uint32_t) ne01, (uint32_t) s01,
        channel_ratio_fd, (uint32_t) s02, (uint32_t) s12, (uint32_t) s2,
        sample_ratio_fd, (uint32_t) s03, (uint32_t) s13, (uint32_t) s3);
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    // GGML_HIP_TERNARY_FP16_GEMV: full-precision Q2_0/Q1_0 decode, mutually
    // exclusive with the fuse_quant path below (no q8_1 buffer involved at
    // all here). Decode-only (ne11==1), no MUL_MAT_ID, no gate/bias fusion.
    if (!ids && !fusion && ne11 == 1 && ggml_cuda_ternary_fp16_gemv_enabled()) {
        if (src0->type == GGML_TYPE_Q2_0) {
            ggml_cuda_mul_mat_vec_ternary_fp16(ctx, src0, src1, dst);
            return;
        }
        if (src0->type == GGML_TYPE_Q1_0) {
            ggml_cuda_mul_mat_vec_binary_fp16(ctx, src0, src1, dst);
            return;
        }
    }

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    // GGML_HIP_FUSE_MMVQ_QUANT (see mmvq.cu top-of-file doc): skip the
    // separate quantize_row_q8_1 launch and let the mmvq kernel quantize its
    // own y row into shared memory instead. Scoped to what mul_mat_vec_q's
    // fused branch actually implements: plain (non-MUL_MAT_ID) decode
    // (ne11==1) on the standard int8 q8_1 path -- F8E4M3/F8E5M2/MXFP8/MXFP6
    // keep using their own dedicated activation quantizers below, untouched.
    const int device = ggml_cuda_get_device();
    const bool fuse_quant_eligible_type =
        src0->type != GGML_TYPE_F8E4M3 && src0->type != GGML_TYPE_F8E5M2 &&
        src0->type != GGML_TYPE_MXFP8  && src0->type != GGML_TYPE_MXFP6;
    const bool fuse_quant =
        ggml_cuda_fuse_mmvq_quant_enabled() && !ids && ne11 == 1 && fuse_quant_eligible_type &&
        ggml_cuda_info().devices[device].warp_size == QK8_1 &&
        (size_t) (ne10 / QK8_1) * sizeof(block_q8_1) <= GGML_HIP_FUSE_MMVQ_QUANT_MAX_SHARED_BYTES;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    // T180-fp8 (env GGML_HIP_DEDUP_MMVQ_QUANT_FP8, default OFF, REQUIRES the
    // base flag too): extend dedup caching to the F8E4M3 native quantizer
    // path (quantize_row_f8e4m3_for_mmvq_cuda). Same principle as the q8_1
    // case: sibling matmuls sharing one input tensor (e.g. DFlash-target
    // wq/wk/wv, still fp8 F8E4M3 weights) still redundantly re-quantize that
    // SAME activation today. Separate opt-in from the q8_1 dedup because the
    // f8e4m3 quantizer is a DIFFERENT function -- verified it's safe to
    // reuse the SAME cache-buffer size formula (block_q8_1-sized) because
    // the existing (pre-T180) code already shares that exact allocation
    // (src1_q8_1) across all three quantizer variants (q8_1/f8e4m3/f8e5m2)
    // at this call site, so the size invariant is already proven, not new.
    const bool dedup_quant_fp8_enabled = getenv("GGML_HIP_DEDUP_MMVQ_QUANT_FP8") != nullptr;
    const bool dedup_quant_eligible_type =
        (src0->type != GGML_TYPE_F8E4M3 || dedup_quant_fp8_enabled) &&
        !(src0->type == GGML_TYPE_F8E5M2 && ggml_cuda_f8e5m2_dot4_enabled());
    const bool dedup_quant_batch_ok =
        ne11 == 1 || (ne11 > 1 && ggml_cuda_dedup_mmvq_quant_batch_enabled());
    const bool dedup_quant =
        ggml_cuda_dedup_mmvq_quant_enabled() && !fuse_quant && !ids && dedup_quant_batch_ok && dedup_quant_eligible_type;
    const bool dedup_hit = dedup_quant &&
        ctx.mmvq_quant_cache_tensor == src1 && ctx.mmvq_quant_cache_buf;

    // dedup_quant (hit OR miss-that-populates) ALWAYS routes data through
    // ctx.mmvq_quant_cache_buf, never through the local src1_q8_1 -- so the
    // local buffer must be skipped (size 0) whenever dedup_quant is true,
    // not just on a hit. (Bug found in first correctness pass: gating this
    // on dedup_hit alone left the first/miss occurrence's vy_ptr pointing at
    // an allocated-but-never-written src1_q8_1 buffer -- uninitialized
    // memory -- while the real quantized data went into the cache buffer
    // instead. Immediate garbage-token output, not a race.)
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), (fuse_quant || dedup_quant) ? 0 : ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    if (!fuse_quant && !dedup_hit) {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        // T79: F8E4M3 decode activations are quantized to native e4m3 (not
        // int8 q8_1) so vec_dot_f8e4m3_f8e4m3_dispatch (mmvq.cu dispatch
        // table above) can feed both operands directly to
        // __builtin_amdgcn_dot4_f32_fp8_fp8. Correctness-coupled with the
        // get_vec_dot_q_cuda_decode swap above -- see the accuracy-
        // disclosure comment on vec_dot_f8e4m3_f8e4m3_impl (vecdotq.cuh).
        // Reachable only when ggml_cuda_should_use_mmvq already gated
        // F8E4M3 to RDNA4 (mmvq.cu, should_use_mmvq), so no separate cc
        // check is needed here.
        // ROC9->runtime-toggle (GGML_HIP_F8E5M2_DOT4, file-top comment
        // block): F8E5M2 activations are int8 q8_1 by DEFAULT (the T97
        // lossless path; failed the PPL gate as a fixed default -- see
        // get_vec_dot_q_cuda_decode above) and swap to native bf8 ONLY when
        // the flag is on, matching the SAME runtime check the dispatch
        // table (mul_mat_vec_q_switch_type) uses. This is the 3rd of 3
        // coupled points the flag must gate consistently -- an unconditional
        // swap here without matching the dispatch swap would silently feed
        // bf8 bytes to a vec_dot that expects int8, or vice versa.
        if (src0->type == GGML_TYPE_F8E4M3) {
            if (dedup_quant) {
                ctx.mmvq_quant_cache_buf = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
                ctx.mmvq_quant_cache_tensor = src1;
                quantize_row_f8e4m3_for_mmvq_cuda(src1_d, nullptr, ctx.mmvq_quant_cache_buf->get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
            } else {
                quantize_row_f8e4m3_for_mmvq_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
            }
        } else if (src0->type == GGML_TYPE_F8E5M2 && ggml_cuda_f8e5m2_dot4_enabled()) {
            quantize_row_f8e5m2_for_mmvq_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        } else if (dedup_quant) {
            ctx.mmvq_quant_cache_buf = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
            ctx.mmvq_quant_cache_tensor = src1;
            quantize_row_q8_1_cuda(src1_d, nullptr, ctx.mmvq_quant_cache_buf->get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        } else {
            quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        }
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    // fuse_quant: vy is the raw fp32 src1 tensor, so s11/s12/s13 must be its
    // real element strides, not the padded q8_1-block-unit strides the
    // unfused path derives from ne10_padded.
    const int64_t s11 = fuse_quant ? src1->nb[1] / ts_src1 : ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = fuse_quant ? src1->nb[2] / ts_src1 : ne11*s11;
    const int64_t s13 = fuse_quant ? src1->nb[3] / ts_src1 : ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    const void * vy_ptr = fuse_quant ? (const void *) src1_d :
        dedup_quant ? (const void *) ctx.mmvq_quant_cache_buf->get() : (const void *) src1_q8_1.get();

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, vy_ptr, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, fuse_quant, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, false, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
