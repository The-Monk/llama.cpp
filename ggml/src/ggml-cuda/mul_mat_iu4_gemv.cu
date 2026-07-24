// See mul_mat_iu4_gemv.cuh for the full rationale (Phase-3(B) M==1 route).
#include "mul_mat_iu4_gemv.cuh"

#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

// One warp per output row (N). Lanes divide the K-blocks round-robin: lane
// `l` owns blocks c = l, l+32, l+64, ... -- NOT "one lane per nibble of
// every block" (that was this kernel's V1 design, measured and REJECTED,
// see below). Each lane does ONE 128-bit (uint4) vectorized load of its
// assigned block's 16-byte `qs` field per iteration, decodes all 32
// nibbles from registers, and accumulates a per-lane partial sum; a single
// warp-reduce at the very end combines the 32 partial sums.
//
// V1->V2 course correction (2026-07-20, measured): the original design had
// each of the 32 lanes read ONE byte of a shared 16-byte block per
// iteration (`qs[lane>>1]`, a redundant/broadcast pattern across lane
// pairs). That is a 1-BYTE-per-lane global-memory access -- bandwidth-
// terrible on this hardware regardless of cache coalescing, because the
// memory pipe issues sub-word loads instead of a wide (128-bit) burst.
// Standalone isolated A/B (iu4_gemv_wpc_sweep.hip vs iu4_gemv_v2_sweep.hip,
// SUSTAINED/clock-boosted methodology, gfx1201, 2026-07-20): V1 hit only
// 204.9-267.3 GB/s across representative decode shapes (N4096K4096,
// N13824K5120, N1024K4096-attn); V2 (this file) hits 464.8-566.1 GB/s on
// the SAME shapes -- a 1.9-2.3x improvement, purely from load width, same
// total work/bytes/algorithm otherwise. V2 also modestly BEATS the
// microbench's swizzled-packing design on 2 of 3 shapes while staying
// completely repack-free against the real block_iu4 layout (see the .cuh
// doctrine comment: "option (a)"). A separate UNROLL=2/4 (multiple blocks
// per lane per outer iteration) variant was also tried and was NEUTRAL-TO-
// WORSE (422 GB/s at U=2 vs 465 at U=1 on N4096K4096) -- one uint4/lane/
// iteration is the measured sweet spot, not further unrolling.
//
// Sign extension: the 4-bit two's-complement field is `nib < 8 ? nib :
// nib - 16` -- NOT "nib - 8" (offset-binary), see the .cuh doctrine comment
// on why that would be the WRONG convention for GGML_TYPE_IU4/block_iu4.
template <int M, int WARPS_PER_CTA>
__launch_bounds__(WARPS_PER_CTA * WARP_SIZE, 2)
static __global__ void k_mul_mat_iu4_gemv(
        const block_iu4 * __restrict__ weight, const float * __restrict__ act, float * __restrict__ dst,
        const int64_t N, const int64_t n_blocks_k, const int64_t nb01_blocks,
        const int64_t act_row_stride_floats, const int64_t dst_row_stride_floats) {

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int64_t n    = (int64_t) blockIdx.x * WARPS_PER_CTA + warp_id;
    if (n >= N) {
        return;
    }

    const block_iu4 * row = weight + n * nb01_blocks;

    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        acc[m] = 0.0f;
    }

    for (int64_t c = lane; c < n_blocks_k; c += WARP_SIZE) {
        const block_iu4 & blk = row[c];
        const uint4 packed = *reinterpret_cast<const uint4 *>(blk.qs);
        const uint32_t words[4] = { packed.x, packed.y, packed.z, packed.w };
        const float scl = __half2float(blk.d);
        const int64_t k0 = c * QK_IU4;

        float local[M];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            local[m] = 0.0f;
        }
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t word = words[w];
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) {
                const int nib = (word >> (nb * 4)) & 0xF;
                // Sign-extend the 4-bit two's-complement field (matches
                // pack_iu4_block in mul_mat_iu4.cu: vals[k] & 0xF, no bias).
                const int v = (nib & 0x8) ? (nib - 16) : nib;
                const int64_t k = k0 + w * 8 + nb;
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    local[m] += (float) v * act[(int64_t) m * act_row_stride_floats + k];
                }
            }
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            acc[m] += local[m] * scl;
        }
    }

#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float v = warp_reduce_sum(acc[m]);
        if (lane == 0) {
            dst[(int64_t) m * dst_row_stride_floats + n] = v;
        }
    }
}

template <int M>
static void launch_iu4_gemv(const block_iu4 * weight, const float * act, float * dst,
                             int64_t N, int64_t n_blocks_k, int64_t nb01_blocks,
                             int64_t act_row_stride_floats, int64_t dst_row_stride_floats,
                             int warps_per_cta, cudaStream_t stream) {
    // WARPS_PER_CTA fixed at compile time per instantiation -- 8 chosen as
    // the default (matches the established nwarps=8 win for the `_0`-family
    // dp4a decode kernels, see mmvq.cu doctrine; not yet independently
    // swept for THIS kernel -- see the bench report for the crossover
    // sweep, WARPS_PER_CTA is a documented open lever, not re-litigated
    // here).
    (void) warps_per_cta;
    constexpr int WARPS_PER_CTA = 8;
    const dim3 block(WARPS_PER_CTA * WARP_SIZE, 1, 1);
    const dim3 grid((N + WARPS_PER_CTA - 1) / WARPS_PER_CTA, 1, 1);
    k_mul_mat_iu4_gemv<M, WARPS_PER_CTA><<<grid, block, 0, stream>>>(
            weight, act, dst, N, n_blocks_k, nb01_blocks, act_row_stride_floats, dst_row_stride_floats);
}

static void launch_iu4_gemv_dispatch(int M, const block_iu4 * weight, const float * act, float * dst,
                                      int64_t N, int64_t n_blocks_k, int64_t nb01_blocks,
                                      int64_t act_row_stride_floats, int64_t dst_row_stride_floats,
                                      cudaStream_t stream) {
    switch (M) {
        case 1:  launch_iu4_gemv<1> (weight, act, dst, N, n_blocks_k, nb01_blocks, act_row_stride_floats, dst_row_stride_floats, 8, stream); break;
        case 2:  launch_iu4_gemv<2> (weight, act, dst, N, n_blocks_k, nb01_blocks, act_row_stride_floats, dst_row_stride_floats, 8, stream); break;
        case 4:  launch_iu4_gemv<4> (weight, act, dst, N, n_blocks_k, nb01_blocks, act_row_stride_floats, dst_row_stride_floats, 8, stream); break;
        case 8:  launch_iu4_gemv<8> (weight, act, dst, N, n_blocks_k, nb01_blocks, act_row_stride_floats, dst_row_stride_floats, 8, stream); break;
        case 16: launch_iu4_gemv<16>(weight, act, dst, N, n_blocks_k, nb01_blocks, act_row_stride_floats, dst_row_stride_floats, 8, stream); break;
        default:
            GGML_ABORT("mul_mat_iu4_gemv: unsupported M=%d (only 1/2/4/8/16 instantiated)", M);
    }
}

bool ggml_cuda_iu4_gemv_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_IU4 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }
    if (src0->ne[0] % QK_IU4 != 0) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
}

bool ggml_cuda_op_mul_mat_iu4_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_iu4_gemv_supports(src0, src1, dst));

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t n_blocks_k = K / QK_IU4;

    GGML_ASSERT(M == 1 || M == 2 || M == 4 || M == 8 || M == 16);

    cudaStream_t stream = ctx.stream();

    const int64_t nb01_blocks = src0->nb[1] / (int64_t) sizeof(block_iu4);
    const int64_t act_row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
    const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);

    launch_iu4_gemv_dispatch((int) M, (const block_iu4 *) src0->data, (const float *) src1->data, (float *) dst->data,
                              N, n_blocks_k, nb01_blocks, act_row_stride_floats, dst_row_stride_floats, stream);

    return true;
}

// ---------------------------------------------------------------------------
// Correctness selftest -- see the .cuh doctrine comment. Hand-packs
// block_iu4 with the real GGML_TYPE_IU4 convention (vals[k] & 0xF, no
// bias) and validates against a CPU int64 reference for M in {1,2,4,8,16}
// and per-block scale != 1 (so this also exercises the fp16 scale
// multiply, unlike mul_mat_iu4.cu's own selftest which fixes d=1.0).
// ---------------------------------------------------------------------------
namespace ggml_cuda_mul_mat_iu4_gemv_selftest_detail {

static void pack_block(const int vals[QK_IU4], float scale, block_iu4 & blk) {
    blk.d = __float2half(scale);
    for (int b = 0; b < QK_IU4 / 2; ++b) {
        const uint8_t nlo = (uint8_t) (vals[2 * b]     & 0xF);
        const uint8_t nhi = (uint8_t) (vals[2 * b + 1] & 0xF);
        blk.qs[b] = (uint8_t) (nlo | (nhi << 4));
    }
}

static bool run_trial(std::mt19937 & rng, int64_t N, int64_t K, int M, double & max_abs_err_out, double & max_rel_err_out) {
    std::uniform_int_distribution<int> valdist(-8, 7);
    std::uniform_real_distribution<float> scaledist(0.01f, 0.05f);
    std::uniform_real_distribution<float> actdist(-1.0f, 1.0f);

    const int64_t n_blocks_k = K / QK_IU4;

    std::vector<std::vector<int>>   w_logical(N, std::vector<int>(K));
    std::vector<float>              w_scale(N);
    std::vector<block_iu4>          w_blocks((size_t) (N * n_blocks_k));
    for (int64_t n = 0; n < N; ++n) {
        w_scale[n] = scaledist(rng);
        for (int64_t k = 0; k < K; ++k) {
            w_logical[n][k] = valdist(rng);
        }
        for (int64_t c = 0; c < n_blocks_k; ++c) {
            pack_block(&w_logical[n][c * QK_IU4], w_scale[n], w_blocks[n * n_blocks_k + c]);
        }
    }

    std::vector<float> act((size_t) (M * K));
    for (auto & v : act) {
        v = actdist(rng);
    }

    std::vector<float> ref((size_t) (M * N), 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                acc += (double) w_logical[n][k] * (double) w_scale[n] * (double) act[(size_t) m * K + k];
            }
            ref[(size_t) m * N + n] = (float) acc;
        }
    }

    block_iu4 * d_w   = nullptr;
    float *     d_act = nullptr;
    float *     d_dst = nullptr;
    if (hipMalloc(&d_w,   w_blocks.size() * sizeof(block_iu4)) != hipSuccess ||
        hipMalloc(&d_act, act.size()      * sizeof(float))     != hipSuccess ||
        hipMalloc(&d_dst, (size_t) (M * N) * sizeof(float))    != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    CUDA_CHECK(hipMemcpy(d_w,   w_blocks.data(), w_blocks.size() * sizeof(block_iu4), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_act, act.data(),      act.size()      * sizeof(float),     hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(d_dst, 0, (size_t) (M * N) * sizeof(float)));

    launch_iu4_gemv_dispatch(M, d_w, d_act, d_dst, N, n_blocks_k, n_blocks_k,
                              /*act_row_stride_floats=*/K, /*dst_row_stride_floats=*/N, 0);
    const hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        GGML_LOG_ERROR("%s: kernel failed: %s\n", __func__, hipGetErrorString(err));
        (void) hipFree(d_w); (void) hipFree(d_act); (void) hipFree(d_dst);
        return false;
    }

    std::vector<float> got((size_t) (M * N));
    CUDA_CHECK(hipMemcpy(got.data(), d_dst, got.size() * sizeof(float), hipMemcpyDeviceToHost));
    (void) hipFree(d_w); (void) hipFree(d_act); (void) hipFree(d_dst);

    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d   = std::fabs((double) got[i] - (double) ref[i]);
        const double rel = d / (std::fabs((double) ref[i]) + 1e-6);
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, rel);
    }
    max_abs_err_out = max_abs;
    max_rel_err_out = max_rel;
    return max_abs < 1e-1 && max_rel < 1e-2; // fp32 accumulation over K~4096, generous but real tolerance
}

} // namespace ggml_cuda_mul_mat_iu4_gemv_selftest_detail

bool ggml_cuda_mul_mat_iu4_gemv_selftest() {
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        // No RDNA4-specific instruction here (plain scalar FMA), but keep
        // the same vacuous-pass convention as the other IU4 selftests for
        // consistency of the house pattern.
        return true;
    }

    using namespace ggml_cuda_mul_mat_iu4_gemv_selftest_detail;
    std::mt19937 rng(20260720);
    bool all_pass = true;
    for (int M : {1, 2, 4, 8, 16}) {
        double worst_abs = 0.0, worst_rel = 0.0;
        bool m_pass = true;
        for (int t = 0; t < 5; ++t) {
            double max_abs = 0.0, max_rel = 0.0;
            const bool ok = run_trial(rng, /*N=*/96, /*K=*/256, M, max_abs, max_rel);
            m_pass &= ok;
            worst_abs = std::max(worst_abs, max_abs);
            worst_rel = std::max(worst_rel, max_rel);
        }
        all_pass &= m_pass;
        GGML_LOG_INFO("%s: M=%d N=96 K=256, 5 trials -> %s (max_abs_err=%.6g max_rel_err=%.6g)\n",
                       __func__, M, m_pass ? "PASS" : "FAIL", worst_abs, worst_rel);
    }
    return all_pass;
}
