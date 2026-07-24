// See mul_mat_q2_0_gemv.cuh for the full rationale.
#include "mul_mat_q2_0_gemv.cuh"

#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

// One warp per output row (N), lanes divide the QK2_0=128-element blocks
// round-robin (lane l owns blocks c = l, l+32, ...), same technique as
// mul_mat_iu4_gemv's V2 kernel. Q2_0's block is 34 bytes (2 scale + 32 qs)
// -- TWO 128-bit (uint4) vectorized loads/lane/block instead of IU4's one,
// since the qs field is 32 bytes not 16.
//
// Decode: qs global byte index gb (0..31), bit-pair e (0..3) -> code =
// (byte>>2e)&3, logical value = code-1 (codes: 0->-1, 1->0, 2->+1; code 3
// unused), global element k = 4*gb+e. Verified algebraically equivalent to
// mul_mat_iu4_mmq.cu's unpack_q2_0_chunk_to_iu4_words (which expresses the
// SAME mapping in terms of a per-32-elem subchunk-local byte index) --
// global-byte-index gb = subc*8+b, and subc*32+4b+e collapses to 4*gb+e
// for all subc/b -- so this is the SAME convention, not independently
// re-derived, just applied over the whole 32-byte qs array in one pass
// instead of 4 separate 8-byte subchunk calls.
template <int M, int WARPS_PER_CTA>
__launch_bounds__(WARPS_PER_CTA * WARP_SIZE, 2)
static __global__ void k_mul_mat_q2_0_gemv(
        const block_q2_0 * __restrict__ weight, const float * __restrict__ act, float * __restrict__ dst,
        const int64_t N, const int64_t n_blocks_k, const int64_t act_row_stride_floats, const int64_t dst_row_stride_floats) {

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int64_t n    = (int64_t) blockIdx.x * WARPS_PER_CTA + warp_id;
    if (n >= N) {
        return;
    }

    const block_q2_0 * row = weight + n * n_blocks_k;

    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        acc[m] = 0.0f;
    }

    for (int64_t c = lane; c < n_blocks_k; c += WARP_SIZE) {
        const block_q2_0 & blk = row[c];
        const uint4 qs0 = *reinterpret_cast<const uint4 *>(blk.qs + 0);
        const uint4 qs1 = *reinterpret_cast<const uint4 *>(blk.qs + 16);
        const uint32_t bytes32[8] = { qs0.x, qs0.y, qs0.z, qs0.w, qs1.x, qs1.y, qs1.z, qs1.w };
        const float scl = __half2float(blk.d);
        const int64_t k0 = c * QK2_0;

        float local[M];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            local[m] = 0.0f;
        }
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t word = bytes32[w];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int byte = (word >> (b * 8)) & 0xFF;
                const int gb   = w * 4 + b; // global byte index within the 32-byte qs array
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int code = (byte >> (e * 2)) & 0x3;
                    const int v    = code - 1; // 0/1/2 -> -1/0/+1
                    const int64_t k = k0 + 4 * gb + e;
#pragma unroll
                    for (int m = 0; m < M; ++m) {
                        local[m] += (float) v * act[(int64_t) m * act_row_stride_floats + k];
                    }
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
static void launch_q2_0_gemv(const block_q2_0 * weight, const float * act, float * dst,
                              int64_t N, int64_t n_blocks_k, int64_t act_row_stride_floats, int64_t dst_row_stride_floats,
                              cudaStream_t stream) {
    constexpr int WARPS_PER_CTA = 8; // matches the IU4 GEMV default; not independently re-swept this pass
    const dim3 block(WARPS_PER_CTA * WARP_SIZE, 1, 1);
    const dim3 grid((N + WARPS_PER_CTA - 1) / WARPS_PER_CTA, 1, 1);
    k_mul_mat_q2_0_gemv<M, WARPS_PER_CTA><<<grid, block, 0, stream>>>(
            weight, act, dst, N, n_blocks_k, act_row_stride_floats, dst_row_stride_floats);
}

static void launch_q2_0_gemv_dispatch(int M, const block_q2_0 * weight, const float * act, float * dst,
                                       int64_t N, int64_t n_blocks_k, int64_t act_row_stride_floats, int64_t dst_row_stride_floats,
                                       cudaStream_t stream) {
    switch (M) {
        case 1:  launch_q2_0_gemv<1> (weight, act, dst, N, n_blocks_k, act_row_stride_floats, dst_row_stride_floats, stream); break;
        case 2:  launch_q2_0_gemv<2> (weight, act, dst, N, n_blocks_k, act_row_stride_floats, dst_row_stride_floats, stream); break;
        case 4:  launch_q2_0_gemv<4> (weight, act, dst, N, n_blocks_k, act_row_stride_floats, dst_row_stride_floats, stream); break;
        case 8:  launch_q2_0_gemv<8> (weight, act, dst, N, n_blocks_k, act_row_stride_floats, dst_row_stride_floats, stream); break;
        case 16: launch_q2_0_gemv<16>(weight, act, dst, N, n_blocks_k, act_row_stride_floats, dst_row_stride_floats, stream); break;
        default:
            GGML_ABORT("mul_mat_q2_0_gemv: unsupported M=%d (only 1/2/4/8/16 instantiated)", M);
    }
}

bool ggml_cuda_q2_0_gemv_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_Q2_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }
    if (src0->ne[0] % QK2_0 != 0) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
}

bool ggml_cuda_op_mul_mat_q2_0_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_q2_0_gemv_supports(src0, src1, dst));

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t n_blocks_k = K / QK2_0;

    GGML_ASSERT(M == 1 || M == 2 || M == 4 || M == 8 || M == 16);

    cudaStream_t stream = ctx.stream();

    const int64_t nb01_blocks = src0->nb[1] / (int64_t) sizeof(block_q2_0);
    GGML_ASSERT(nb01_blocks == n_blocks_k); // contiguous, checked by _supports()
    const int64_t act_row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
    const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);

    launch_q2_0_gemv_dispatch((int) M, (const block_q2_0 *) src0->data, (const float *) src1->data, (float *) dst->data,
                               N, n_blocks_k, act_row_stride_floats, dst_row_stride_floats, stream);

    return true;
}

// ---------------------------------------------------------------------------
// Correctness selftest.
// ---------------------------------------------------------------------------
namespace ggml_cuda_mul_mat_q2_0_gemv_selftest_detail {

static void pack_block(const int vals[QK2_0] /* each in {-1,0,1} */, float scale, block_q2_0 & blk) {
    blk.d = __float2half(scale);
    for (int gb = 0; gb < QK2_0 / 4; ++gb) {
        uint8_t byte = 0;
        for (int e = 0; e < 4; ++e) {
            const int code = vals[4 * gb + e] + 1; // -1/0/1 -> 0/1/2
            byte |= (uint8_t) (code & 0x3) << (2 * e);
        }
        blk.qs[gb] = byte;
    }
}

static bool run_trial(std::mt19937 & rng, int64_t N, int64_t K, int M, double & max_abs_err_out, double & max_rel_err_out) {
    std::uniform_int_distribution<int> valdist(-1, 1);
    std::uniform_real_distribution<float> scaledist(0.01f, 0.05f);
    std::uniform_real_distribution<float> actdist(-1.0f, 1.0f);

    const int64_t n_blocks_k = K / QK2_0;

    std::vector<std::vector<int>>   w_logical(N, std::vector<int>(K));
    std::vector<float>              w_scale(N);
    std::vector<block_q2_0>         w_blocks((size_t) (N * n_blocks_k));
    for (int64_t n = 0; n < N; ++n) {
        w_scale[n] = scaledist(rng);
        for (int64_t k = 0; k < K; ++k) {
            w_logical[n][k] = valdist(rng);
        }
        for (int64_t c = 0; c < n_blocks_k; ++c) {
            pack_block(&w_logical[n][c * QK2_0], w_scale[n], w_blocks[n * n_blocks_k + c]);
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

    block_q2_0 * d_w   = nullptr;
    float *      d_act = nullptr;
    float *      d_dst = nullptr;
    if (hipMalloc(&d_w,   w_blocks.size() * sizeof(block_q2_0)) != hipSuccess ||
        hipMalloc(&d_act, act.size()      * sizeof(float))      != hipSuccess ||
        hipMalloc(&d_dst, (size_t) (M * N) * sizeof(float))     != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    CUDA_CHECK(hipMemcpy(d_w,   w_blocks.data(), w_blocks.size() * sizeof(block_q2_0), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_act, act.data(),      act.size()      * sizeof(float),      hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(d_dst, 0, (size_t) (M * N) * sizeof(float)));

    launch_q2_0_gemv_dispatch(M, d_w, d_act, d_dst, N, n_blocks_k,
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
    return max_abs < 1e-1 && max_rel < 1e-2;
}

} // namespace ggml_cuda_mul_mat_q2_0_gemv_selftest_detail

bool ggml_cuda_mul_mat_q2_0_gemv_selftest() {
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        return true;
    }

    using namespace ggml_cuda_mul_mat_q2_0_gemv_selftest_detail;
    std::mt19937 rng(20260720);
    bool all_pass = true;
    for (int M : {1, 2, 4, 8, 16}) {
        double worst_abs = 0.0, worst_rel = 0.0;
        bool m_pass = true;
        for (int t = 0; t < 5; ++t) {
            double max_abs = 0.0, max_rel = 0.0;
            const bool ok = run_trial(rng, /*N=*/96, /*K=*/QK2_0 * 2, M, max_abs, max_rel);
            m_pass &= ok;
            worst_abs = std::max(worst_abs, max_abs);
            worst_rel = std::max(worst_rel, max_rel);
        }
        all_pass &= m_pass;
        GGML_LOG_INFO("%s: M=%d N=96 K=%d, 5 trials -> %s (max_abs_err=%.6g max_rel_err=%.6g)\n",
                       __func__, M, QK2_0 * 2, m_pass ? "PASS" : "FAIL", worst_abs, worst_rel);
    }
    return all_pass;
}

// ===========================================================================
// Q1_0 companion (binary {-1,+1}, QK1_0=128, 16 qs bytes/block). Same
// warp-per-row / lane-owns-blocks-round-robin technique; ONE 128-bit
// (uint4) load/lane/block since qs is only 16 bytes (vs Q2_0's 32).
// ===========================================================================

template <int M, int WARPS_PER_CTA>
__launch_bounds__(WARPS_PER_CTA * WARP_SIZE, 2)
static __global__ void k_mul_mat_q1_0_gemv(
        const block_q1_0 * __restrict__ weight, const float * __restrict__ act, float * __restrict__ dst,
        const int64_t N, const int64_t n_blocks_k, const int64_t act_row_stride_floats, const int64_t dst_row_stride_floats) {

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int64_t n    = (int64_t) blockIdx.x * WARPS_PER_CTA + warp_id;
    if (n >= N) {
        return;
    }

    const block_q1_0 * row = weight + n * n_blocks_k;

    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        acc[m] = 0.0f;
    }

    for (int64_t c = lane; c < n_blocks_k; c += WARP_SIZE) {
        const block_q1_0 & blk = row[c];
        const uint4 qs = *reinterpret_cast<const uint4 *>(blk.qs);
        const uint32_t words4[4] = { qs.x, qs.y, qs.z, qs.w };
        const float scl = __half2float(blk.d);
        const int64_t k0 = c * QK1_0;

        float local[M];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            local[m] = 0.0f;
        }
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t word = words4[w];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int byte = (word >> (b * 8)) & 0xFF;
                const int gb   = w * 4 + b; // global byte index within the 16-byte qs array
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int v = ((byte >> i) & 1) ? 1 : -1;
                    const int64_t k = k0 + 8 * gb + i;
#pragma unroll
                    for (int m = 0; m < M; ++m) {
                        local[m] += (float) v * act[(int64_t) m * act_row_stride_floats + k];
                    }
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
static void launch_q1_0_gemv(const block_q1_0 * weight, const float * act, float * dst,
                              int64_t N, int64_t n_blocks_k, int64_t act_row_stride_floats, int64_t dst_row_stride_floats,
                              cudaStream_t stream) {
    constexpr int WARPS_PER_CTA = 8;
    const dim3 block(WARPS_PER_CTA * WARP_SIZE, 1, 1);
    const dim3 grid((N + WARPS_PER_CTA - 1) / WARPS_PER_CTA, 1, 1);
    k_mul_mat_q1_0_gemv<M, WARPS_PER_CTA><<<grid, block, 0, stream>>>(
            weight, act, dst, N, n_blocks_k, act_row_stride_floats, dst_row_stride_floats);
}

static void launch_q1_0_gemv_dispatch(int M, const block_q1_0 * weight, const float * act, float * dst,
                                       int64_t N, int64_t n_blocks_k, int64_t act_row_stride_floats, int64_t dst_row_stride_floats,
                                       cudaStream_t stream) {
    switch (M) {
        case 1:  launch_q1_0_gemv<1> (weight, act, dst, N, n_blocks_k, act_row_stride_floats, dst_row_stride_floats, stream); break;
        case 2:  launch_q1_0_gemv<2> (weight, act, dst, N, n_blocks_k, act_row_stride_floats, dst_row_stride_floats, stream); break;
        case 4:  launch_q1_0_gemv<4> (weight, act, dst, N, n_blocks_k, act_row_stride_floats, dst_row_stride_floats, stream); break;
        case 8:  launch_q1_0_gemv<8> (weight, act, dst, N, n_blocks_k, act_row_stride_floats, dst_row_stride_floats, stream); break;
        case 16: launch_q1_0_gemv<16>(weight, act, dst, N, n_blocks_k, act_row_stride_floats, dst_row_stride_floats, stream); break;
        default:
            GGML_ABORT("mul_mat_q1_0_gemv: unsupported M=%d (only 1/2/4/8/16 instantiated)", M);
    }
}

bool ggml_cuda_q1_0_gemv_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_Q1_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }
    if (src0->ne[0] % QK1_0 != 0) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
}

bool ggml_cuda_op_mul_mat_q1_0_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_q1_0_gemv_supports(src0, src1, dst));

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t n_blocks_k = K / QK1_0;

    GGML_ASSERT(M == 1 || M == 2 || M == 4 || M == 8 || M == 16);

    cudaStream_t stream = ctx.stream();

    const int64_t nb01_blocks = src0->nb[1] / (int64_t) sizeof(block_q1_0);
    GGML_ASSERT(nb01_blocks == n_blocks_k);
    const int64_t act_row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
    const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);

    launch_q1_0_gemv_dispatch((int) M, (const block_q1_0 *) src0->data, (const float *) src1->data, (float *) dst->data,
                               N, n_blocks_k, act_row_stride_floats, dst_row_stride_floats, stream);

    return true;
}

namespace ggml_cuda_mul_mat_q1_0_gemv_selftest_detail {

static void pack_block(const int vals[QK1_0] /* each in {-1,+1} */, float scale, block_q1_0 & blk) {
    blk.d = __float2half(scale);
    for (int gb = 0; gb < QK1_0 / 8; ++gb) {
        uint8_t byte = 0;
        for (int i = 0; i < 8; ++i) {
            if (vals[8 * gb + i] > 0) {
                byte |= (uint8_t) (1u << i);
            }
        }
        blk.qs[gb] = byte;
    }
}

static bool run_trial(std::mt19937 & rng, int64_t N, int64_t K, int M, double & max_abs_err_out, double & max_rel_err_out) {
    std::uniform_int_distribution<int> bindist(0, 1);
    std::uniform_real_distribution<float> scaledist(0.01f, 0.05f);
    std::uniform_real_distribution<float> actdist(-1.0f, 1.0f);

    const int64_t n_blocks_k = K / QK1_0;

    std::vector<std::vector<int>>   w_logical(N, std::vector<int>(K));
    std::vector<float>              w_scale(N);
    std::vector<block_q1_0>         w_blocks((size_t) (N * n_blocks_k));
    for (int64_t n = 0; n < N; ++n) {
        w_scale[n] = scaledist(rng);
        for (int64_t k = 0; k < K; ++k) {
            w_logical[n][k] = bindist(rng) ? 1 : -1;
        }
        for (int64_t c = 0; c < n_blocks_k; ++c) {
            pack_block(&w_logical[n][c * QK1_0], w_scale[n], w_blocks[n * n_blocks_k + c]);
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

    block_q1_0 * d_w   = nullptr;
    float *      d_act = nullptr;
    float *      d_dst = nullptr;
    if (hipMalloc(&d_w,   w_blocks.size() * sizeof(block_q1_0)) != hipSuccess ||
        hipMalloc(&d_act, act.size()      * sizeof(float))      != hipSuccess ||
        hipMalloc(&d_dst, (size_t) (M * N) * sizeof(float))     != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    CUDA_CHECK(hipMemcpy(d_w,   w_blocks.data(), w_blocks.size() * sizeof(block_q1_0), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_act, act.data(),      act.size()      * sizeof(float),      hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(d_dst, 0, (size_t) (M * N) * sizeof(float)));

    launch_q1_0_gemv_dispatch(M, d_w, d_act, d_dst, N, n_blocks_k,
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
    return max_abs < 1e-1 && max_rel < 1e-2;
}

} // namespace ggml_cuda_mul_mat_q1_0_gemv_selftest_detail

bool ggml_cuda_mul_mat_q1_0_gemv_selftest() {
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        return true;
    }

    using namespace ggml_cuda_mul_mat_q1_0_gemv_selftest_detail;
    std::mt19937 rng(20260720);
    bool all_pass = true;
    for (int M : {1, 2, 4, 8, 16}) {
        double worst_abs = 0.0, worst_rel = 0.0;
        bool m_pass = true;
        for (int t = 0; t < 5; ++t) {
            double max_abs = 0.0, max_rel = 0.0;
            const bool ok = run_trial(rng, /*N=*/96, /*K=*/QK1_0 * 2, M, max_abs, max_rel);
            m_pass &= ok;
            worst_abs = std::max(worst_abs, max_abs);
            worst_rel = std::max(worst_rel, max_rel);
        }
        all_pass &= m_pass;
        GGML_LOG_INFO("%s: M=%d N=96 K=%d, 5 trials -> %s (max_abs_err=%.6g max_rel_err=%.6g)\n",
                       __func__, M, QK1_0 * 2, m_pass ? "PASS" : "FAIL", worst_abs, worst_rel);
    }
    return all_pass;
}
