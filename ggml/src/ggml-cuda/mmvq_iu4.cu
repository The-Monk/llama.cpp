// T170: dot8 int4-drafter decode path implementation.
// See mmvq_iu4.cuh for the full rationale/scope.
//
// Kernel math ported verbatim from the correctness- and perf-validated
// isolated PoC at ~/int4-research/pocs/dot8-int4-decode/decode_poc.hip
// (Stage 20), generalized from that PoC's fixed K=4096/BLOCK=128 to an
// arbitrary-K grid-stride loop so it's correct for any real model's hidden
// size (a multiple of QK_IU4=32), not just the PoC's one tested shape.
//
// IMPORTANT (the Stage-20 bug, do not re-introduce): block_iu4 uses the
// INTERLEAVED-PAIRS nibble convention (byte b: low nibble = element 2b,
// high nibble = element 2b+1) -- NOT block_q4_0's SPLIT-HALF convention
// (byte b: low = element b, high = element b+16). The dot8 kernel below
// needs no unpack at all (nibbles are consumed exactly as packed), so this
// only matters for anyone tempted to add an int8-unpack fallback here --
// see mmvq.cu's vec_dot_q4_0_q8_1_impl for what NOT to copy verbatim.

#include "mmvq_iu4.cuh"

#include <cstring>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

#define IU4_MMVQ_BLOCK 256

// Verbatim (file-local) port of mul_mat_iu4.cu's k_quantize_act_iu4: online
// per-block (32-elem) symmetric int4 (amax/7) activation quantization into
// block_iu4. Kept as an intentional duplicate rather than exposing that
// file's internal symbol -- keeps this new decode path fully independent
// of the WMMA file (zero risk of a change here affecting the M>1 path).
static __global__ void k_quantize_act_iu4_mmvq(
        const float * __restrict__ x, block_iu4 * __restrict__ y) {
    const int64_t c   = blockIdx.x;
    const int     tid = threadIdx.x;

    __shared__ float sh_val[32];
    __shared__ float sh_scale;
    __shared__ int   sh_q[32];

    const float v = x[c * 32 + tid];
    sh_val[tid] = fabsf(v);
    __syncthreads();

    if (tid == 0) {
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            amax = fmaxf(amax, sh_val[i]);
        }
        const float d = amax / 7.0f;
        sh_scale = d;
        y[c].d = __float2half(d);
    }
    __syncthreads();

    const float d  = sh_scale;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q = (int) rintf(v * id);
    q = q < -8 ? -8 : (q > 7 ? 7 : q);
    sh_q[tid] = q;
    __syncthreads();

    if ((tid & 1) == 0) {
        const int q0 = sh_q[tid];
        const int q1 = sh_q[tid + 1];
        y[c].qs[tid / 2] = (uint8_t) ((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }
}

// dot8 GEMV: one threadblock per output row, launched with a RUNTIME block
// size (dynamic shared memory) chosen by the host launcher to match
// Kblocks exactly when it fits in one block (the common decode case, e.g.
// K=4096 -> Kblocks=128) so every launched thread does real work -- the
// first version of this kernel hardcoded a hard-coded 256-thread launch
// unconditionally (matching neither of the tested K=4096 shapes' 128
// blocks), which measurably cost ~20-35% of the Stage-20 PoC's throughput
// (half the threads did zero work every launch, plus extra no-op
// reduction rounds) -- caught by directly diffing in-tree perf against
// the PoC's absolute numbers, not assumed. Grid-stride loop still present
// for Kblocks larger than the chosen launch size, so this stays correct
// (just not maximally fast) for arbitrarily large K.
static __global__ void k_mmvq_dot8_iu4(
        const char * __restrict__ vweight, const block_iu4 * __restrict__ act,
        float * __restrict__ dst, int64_t Kblocks, int64_t row_stride_bytes) {
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;

    float partial = 0.0f;
    const block_iu4 * wrow = (const block_iu4 *) (vweight + row * row_stride_bytes);
    for (int64_t c = tid; c < Kblocks; c += blockDim.x) {
        uint32_t ww[4], xw[4];
        memcpy(ww, wrow[c].qs, 16);
        memcpy(xw, act[c].qs, 16);
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            // sign-select convention (Stage 19 finding, also documented at
            // mma.cuh's mma_iu4 for the WMMA sibling): true=signed for both
            // operands is the ONLY combination that reproduces the correct
            // signed dot product for this two's-complement [-8,7] packing.
            sumi = __builtin_amdgcn_sudot8(true, (int) ww[i], true, xw[i], sumi, false);
        }
        partial += (float) sumi * __half2float(wrow[c].d) * __half2float(act[c].d);
    }

    extern __shared__ float sdata[];
    sdata[tid] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        dst[row] = sdata[0];
    }
}

// Pick the launch block size. Empirically swept 32/64/128/256/512/1024
// against both PoC-matching shapes (K=4096, N in {4096,14336}) -- occupancy
// (more concurrent partially-idle waves hiding DRAM latency) mattered more
// than "exact fit, zero idle threads": a small fixed block beat a
// Kblocks-exact-fit block on BOTH shapes. 64 was consistently at or near
// the best of the sweep for both N=4096 and N=14336 (see
// ~/int4-research/FINDINGS.md Stage 23 for the full sweep table); kept
// simple/fixed rather than shape-conditional since the win from further
// per-shape tuning was within run-to-run noise. Still correct for any
// Kblocks via the grid-stride loop in k_mmvq_dot8_iu4 regardless of this
// choice.
static inline int iu4_mmvq_pick_block(int64_t Kblocks) {
    GGML_UNUSED(Kblocks);
    return 64;
}

bool ggml_cuda_mmvq_iu4_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_IU4)  return false;
    if (src1->type != GGML_TYPE_F32)  return false;
    if (dst->type  != GGML_TYPE_F32)  return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src1->ne[1] != 1) return false; // M=1 (decode/drafter) only -- M>1 stays on the WMMA path
    if (src0->ne[0] % QK_IU4 != 0) return false;
    if (src0->ne[0] != src1->ne[0]) return false;
    return true;
}

bool ggml_cuda_op_mul_mat_vec_iu4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_mmvq_iu4_supports(src0, src1, dst) && "ggml_cuda_op_mul_mat_vec_iu4 does not support this tensor shape (caller must gate on ggml_cuda_mmvq_iu4_supports)");

    // T170 / Stage 23 dispatch-routing verifiability: one-shot log so the
    // person running this can CONFIRM (not just trust) that an M=1 call
    // actually took this new dot8 path rather than silently falling
    // through to the WMMA intercept below it in ggml_cuda_mul_mat(). Only
    // ever reachable when GGML_HIP_IU4_MMVQ_DECODE is set AND M==1 (see
    // ggml_cuda_mmvq_iu4_supports) -- M>1 NEVER calls this function.
    {
        static bool logged_once = false;
        if (!logged_once) {
            logged_once = true;
            GGML_LOG_INFO("%s: dot8 int4-drafter decode path ACTIVE (M=1 IU4 MUL_MAT routed here, not WMMA)\n", __func__);
        }
    }

    const int64_t K       = src0->ne[0];
    const int64_t N       = src0->ne[1];
    const int64_t Kblocks = K / QK_IU4;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_iu4> act_q(ctx.pool(), (size_t) Kblocks);
    k_quantize_act_iu4_mmvq<<<Kblocks, 32, 0, stream>>>((const float *) src1->data, act_q.get());

    const int64_t row_stride_bytes = src0->nb[1];
    const dim3 grid((unsigned) N, 1, 1);
    const int  block_size = iu4_mmvq_pick_block(Kblocks);
    const dim3 block(block_size, 1, 1);
    k_mmvq_dot8_iu4<<<grid, block, block_size * sizeof(float), stream>>>(
            (const char *) src0->data, act_q.get(), (float *) dst->data, Kblocks, row_stride_bytes);

    return true;
}

// ---------------------------------------------------------------------
// Correctness self-test -- same style as ggml_cuda_mul_mat_iu4_selftest()
// (mul_mat_iu4.cu): hand-packs int4 weight/activation operands with a
// fixed unit scale (so the int32 dot IS the answer, no fp rounding
// anywhere), bypasses the ggml_tensor machinery entirely, launches
// k_mmvq_dot8_iu4 directly, compares against a CPU int4 x int4 reference.
// ---------------------------------------------------------------------
namespace ggml_cuda_mmvq_iu4_selftest_detail {

static void pack_iu4_block(const int vals[QK_IU4], block_iu4 & blk) {
    blk.d = __float2half(1.0f);
    for (int b = 0; b < QK_IU4 / 2; ++b) {
        const int q0 = vals[2*b] & 0xF;
        const int q1 = vals[2*b+1] & 0xF;
        blk.qs[b] = (uint8_t) (q0 | (q1 << 4));
    }
}

static bool run_trial(std::mt19937 & rng, int64_t N, int64_t K, long & max_abs_err_out) {
    std::uniform_int_distribution<int> valdist(-8, 7);
    const int64_t n_blocks_k = K / QK_IU4;

    std::vector<int> act_logical(K);
    std::vector<std::vector<int>> w_logical(N, std::vector<int>(K));
    for (int64_t k = 0; k < K; ++k) act_logical[k] = valdist(rng);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) w_logical[n][k] = valdist(rng);

    std::vector<block_iu4> act_blocks(n_blocks_k);
    std::vector<block_iu4> w_blocks((size_t) (N * n_blocks_k));
    for (int64_t c = 0; c < n_blocks_k; ++c) pack_iu4_block(&act_logical[c * QK_IU4], act_blocks[c]);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < n_blocks_k; ++c)
            pack_iu4_block(&w_logical[n][c * QK_IU4], w_blocks[n * n_blocks_k + c]);

    block_iu4 * d_act = nullptr;
    block_iu4 * d_w   = nullptr;
    float *     d_dst = nullptr;
    if (hipMalloc(&d_act, act_blocks.size() * sizeof(block_iu4)) != hipSuccess ||
        hipMalloc(&d_w,   w_blocks.size()   * sizeof(block_iu4)) != hipSuccess ||
        hipMalloc(&d_dst, (size_t) N * sizeof(float)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    CUDA_CHECK(hipMemcpy(d_act, act_blocks.data(), act_blocks.size() * sizeof(block_iu4), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_w,   w_blocks.data(),   w_blocks.size()   * sizeof(block_iu4), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(d_dst, 0, (size_t) N * sizeof(float)));

    const int64_t row_stride_bytes = n_blocks_k * (int64_t) sizeof(block_iu4);
    const dim3 grid((unsigned) N, 1, 1);
    const int  block_size = iu4_mmvq_pick_block(n_blocks_k);
    const dim3 block(block_size, 1, 1);
    k_mmvq_dot8_iu4<<<grid, block, block_size * sizeof(float), 0>>>((const char *) d_w, d_act, d_dst, n_blocks_k, row_stride_bytes);
    const hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        GGML_LOG_ERROR("%s: k_mmvq_dot8_iu4 failed: %s\n", __func__, hipGetErrorString(err));
        (void) hipFree(d_act); (void) hipFree(d_w); (void) hipFree(d_dst);
        return false;
    }

    std::vector<float> out(N);
    CUDA_CHECK(hipMemcpy(out.data(), d_dst, out.size() * sizeof(float), hipMemcpyDeviceToHost));
    (void) hipFree(d_act); (void) hipFree(d_w); (void) hipFree(d_dst);

    long max_abs = 0;
    for (int64_t n = 0; n < N; ++n) {
        long ref = 0;
        for (int64_t k = 0; k < K; ++k) ref += (long) act_logical[k] * (long) w_logical[n][k];
        const long got  = (long) out[n];
        const long diff = std::labs(got - ref);
        max_abs = std::max(max_abs, diff);
    }
    max_abs_err_out = max_abs;
    return max_abs == 0;
}

} // namespace ggml_cuda_mmvq_iu4_selftest_detail

bool ggml_cuda_mul_mat_vec_iu4_selftest() {
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        return true; // vacuously true off RDNA4, v_dot8_i32_iu4 is gfx1201-specific
    }

    using namespace ggml_cuda_mmvq_iu4_selftest_detail;
    std::mt19937 rng(170170); // T170
    bool all_pass = true;
    long worst = 0;
    const int n_trials = 20;
    // N deliberately not a multiple of anything special (no tile-boundary
    // constraint here, unlike the WMMA kernel); K spans several blocks
    // (K=4096 -> 128 blocks) AND a non-round K (4128 -> not a "nice" 4096
    // multiple relationship) to exercise the grid-stride loop generality.
    for (int t = 0; t < n_trials; ++t) {
        long max_abs_err = 0;
        int64_t K = (t % 2 == 0) ? 4096 : 4128; // 128 blocks vs 129 blocks
        if (!run_trial(rng, /*N=*/37, K, max_abs_err)) {
            all_pass = false;
        }
        worst = std::max(worst, max_abs_err);
    }
    GGML_LOG_INFO("%s: k_mmvq_dot8_iu4 real-model wrapper, %d random trials (N=37,K=4096/4128) -> %s (max_abs_err=%ld)\n",
                   __func__, n_trials, all_pass ? "PASS" : "FAIL", worst);
    return all_pass;
}
