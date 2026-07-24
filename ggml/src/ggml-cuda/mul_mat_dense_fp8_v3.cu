// See mul_mat_dense_fp8_v3.cuh for the full rationale (T162 follow-up,
// coordinator directive "the missing measurement" -- a dense-fp8 twin of
// mul_mat_2of4_fp8.cu's V3 kernel, same shape, dense WMMA math instead of
// sparse SWMMAC math, isolating kernel-quality from the sparsity question).
#include "mul_mat_dense_fp8_v3.cuh"
#include "ggml-quants.h"
#include "ggml-impl.h" // ggml_fp32_to_e4m3 (host RTN ref, matches k_quantize_act_f8e4m3's device scheme)

#include <cstring>
#include <random>
#include <vector>

static __device__ __forceinline__ int32_t pack4_dv3(const uint8_t * p) {
    int32_t w;
    memcpy(&w, p, 4);
    return w;
}

typedef int   v4i_dv3 __attribute__((ext_vector_type(4)));
typedef float v8f_dv3 __attribute__((ext_vector_type(8)));

// Identical online fp8 activation quantizer to mul_mat_2of4_fp8.cu's
// k_quantize_act_f8e4m3 (duplicated locally -- this file's own kernels are
// self-contained, matching the established one-quantizer-per-kernel-file
// convention in this codebase, e.g. mul_mat_iu4_mmq.cu's
// k_quantize_act_iu4_mmq).
static __global__ void k_quantize_act_f8e4m3_dv3(
        const float * __restrict__ x, block_f8e4m3 * __restrict__ y,
        const int64_t n_blocks_k, const int64_t row_stride_floats) {
    const int64_t c   = blockIdx.x;
    const int64_t m   = blockIdx.y;
    const int     tid = threadIdx.x;

    __shared__ float sh_val[32];
    __shared__ float sh_scale;

    const float v = x[m * row_stride_floats + c * 32 + tid];
    sh_val[tid] = fabsf(v);
    __syncthreads();

    if (tid == 0) {
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            amax = fmaxf(amax, sh_val[i]);
        }
        const float d = amax / 448.0f;
        sh_scale = d;
        y[m * n_blocks_k + c].d = __float2half(d);
    }
    __syncthreads();

    const float d  = sh_scale;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    y[m * n_blocks_k + c].qs[tid] = ggml_cuda_fp32_to_e4m3(v * id);
}

// Dense-fp8 twin of k_mul_mat_2of4_fp8_v3<ILP,need_check> -- IDENTICAL
// shape (private registers, zero LDS, zero __syncthreads(), 2-deep register
// software pipeline, WARPS via blockDim.y, ILP register-blocked tiles/warp)
// -- only the per-lane operand width/compute call differ:
//   - weight is dense block_f8e4m3 (32 bytes/row), not compressed
//     block_2of4_fp8 (16 bytes/row) -- so each lane's own k_half needs 4
//     ints (16 bytes) instead of 2 (8 bytes), matching the SAME per-lane
//     tile<16,8,int> layout the production mma.cuh fp8 WMMA overload uses
//     (row=lane%16, k_half=lane/16, ne=4 ints/lane).
//   - compute is TWO __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12
//     calls per K=32 chunk (copied verbatim from mma.cuh's validated fp8
//     mma() overload: a_vec[0]=ints[0:1], a_vec[1]=ints[2:3]), not ONE
//     SWMMAC call -- no sparsity_idx operand, no compression.
// D-matrix layout (out_row_base+l / lane%16) is IDENTICAL to the SWMMAC
// kernel's (both are 16x16 wave32 WMMA-family output tiles, same hardware
// convention -- confirmed via mma.cuh's DATA_LAYOUT_J_MAJOR transpose of
// the SAME I_MAJOR get_i/get_j formulas the SWMMAC D-matrix uses).
template <int ILP, bool need_check>
__launch_bounds__(1024, 1)
static __global__ void k_mul_mat_dense_fp8_v3(
        const block_f8e4m3 * __restrict__ weight, const block_f8e4m3 * __restrict__ act, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t n_blocks_k,
        const int64_t dst_row_stride_floats) {
#if defined(RDNA4)
    using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;

    const int lane      = threadIdx.x; // 0..31
    const int warp_id_u = __builtin_amdgcn_readfirstlane((int) threadIdx.y);
    const int n_warps   = blockDim.y;

    const int k_half       = (lane < 16) ? 0 : 1;
    const int local_idx    = (lane < 16) ? lane : (lane - 16);
    const int out_row_base = (lane >= 16) ? 8 : 0;

    const int64_t n0      = (int64_t) blockIdx.x * ((int64_t) n_warps * ILP * 16) + (int64_t) warp_id_u * ILP * 16;
    const int64_t m0      = (int64_t) blockIdx.y * 16;
    const int64_t act_col = m0 + local_idx;

    v4i_dv3 a_cur[ILP], a_nxt[ILP];
    float   dw_cur[ILP], dw_nxt[ILP];
    v4i_dv3 b_cur, b_nxt;
    float   da_cur, da_nxt;

    auto load_chunk = [&] (int64_t c, v4i_dv3 (&a)[ILP], float (&dw)[ILP], v4i_dv3 & b, float & da) {
        if (!need_check || act_col < M) {
            const block_f8e4m3 & blka = act[act_col * n_blocks_k + c];
            b.x = pack4_dv3(blka.qs + k_half*16 + 0);
            b.y = pack4_dv3(blka.qs + k_half*16 + 4);
            b.z = pack4_dv3(blka.qs + k_half*16 + 8);
            b.w = pack4_dv3(blka.qs + k_half*16 + 12);
            da  = __half2float(blka.d);
        } else {
            b  = v4i_dv3{0, 0, 0, 0};
            da = 0.0f;
        }
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
            const int64_t weight_row = n0 + (int64_t) t*16 + local_idx;
            if (!need_check || weight_row < N) {
                const block_f8e4m3 & blkw = weight[weight_row * n_blocks_k + c];
                a[t].x = pack4_dv3(blkw.qs + k_half*16 + 0);
                a[t].y = pack4_dv3(blkw.qs + k_half*16 + 4);
                a[t].z = pack4_dv3(blkw.qs + k_half*16 + 8);
                a[t].w = pack4_dv3(blkw.qs + k_half*16 + 12);
                dw[t]  = __half2float(blkw.d);
            } else {
                a[t]  = v4i_dv3{0, 0, 0, 0};
                dw[t] = 0.0f;
            }
        }
    };

    float acc[ILP][8];
#pragma unroll
    for (int t = 0; t < ILP; ++t) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            acc[t][l] = 0.0f;
        }
    }

    if (n_blocks_k > 0) {
        load_chunk(0, a_cur, dw_cur, b_cur, da_cur);
    }

    for (int64_t c = 0; c < n_blocks_k; ++c) {
        const bool have_next = (c + 1 < n_blocks_k);
        if (have_next) {
            load_chunk(c + 1, a_nxt, dw_nxt, b_nxt, da_nxt);
        }

        v8f_dv3 raw[ILP];
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
            const int32x2_t a_vec0 = {a_cur[t].x, a_cur[t].y};
            const int32x2_t a_vec1 = {a_cur[t].z, a_cur[t].w};
            const int32x2_t b_vec0 = {b_cur.x, b_cur.y};
            const int32x2_t b_vec1 = {b_cur.z, b_cur.w};
            v8f_dv3 accf = {0, 0, 0, 0, 0, 0, 0, 0};
            accf = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec0, b_vec0, accf);
            accf = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec1, b_vec1, accf);
            raw[t] = accf;
        }
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float dw_row = __shfl_sync(0xFFFFFFFFu, dw_cur[t], out_row_base + l, WARP_SIZE);
                acc[t][l] += raw[t][l] * dw_row * da_cur;
            }
        }

        if (have_next) {
#pragma unroll
            for (int t = 0; t < ILP; ++t) {
                a_cur[t]  = a_nxt[t];
                dw_cur[t] = dw_nxt[t];
            }
            b_cur  = b_nxt;
            da_cur = da_nxt;
        }
    }

    const int out_col = local_idx;
#pragma unroll
    for (int t = 0; t < ILP; ++t) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int64_t n = n0 + (int64_t) t*16 + out_row_base + l;
            const int64_t m = m0 + out_col;
            if (!need_check || (m < M && n < N)) {
                dst[m * dst_row_stride_floats + n] = acc[t][l];
            }
        }
    }
#else
    GGML_UNUSED(weight);
    GGML_UNUSED(act);
    GGML_UNUSED(dst);
    GGML_UNUSED(M);
    GGML_UNUSED(N);
    GGML_UNUSED(n_blocks_k);
    GGML_UNUSED(dst_row_stride_floats);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

static int ggml_cuda_dense_fp8_v3_warps() {
    static const int w = [] {
        const char * env = getenv("GGML_HIP_DENSE_FP8_V3_WARPS");
        const int v = env ? atoi(env) : 32;
        return (v >= 1 && v <= 32) ? v : 32;
    }();
    return w;
}

// T162 apples-to-apples fairness fix: the 2:4-V3 kernel got a measured,
// shape-adaptive WARPS rule (N<=4096 && K<=4096 -> WARPS=4); leaving this
// dense twin on a flat WARPS=32 default would have made the "dense-V3 vs
// 2:4-V3" comparison unfair in the OTHER direction. The isolated per-shape
// sweep (GGML_HIP_DENSE_FP8_V3_SHAPE_BENCH) found dense-V3's own optimum
// differs from 2:4-V3's: q/o-proj (N=4096,K=4096) wants WARPS=4 (986k vs
// flat-32's 792k pp-equiv, a real 24% left on the table), down-proj
// (N=4096,K=14336) ALSO wants WARPS=4 (239k vs 228k), and gate/up-proj
// (N=14336,K=4096) wants WARPS=32 (matches the flat default). So dense-V3's
// rule is simpler than 2:4-V3's: N<=4096 -> WARPS=4 regardless of K (not
// N<=4096 && K<=4096 -- the two kernels' occupancy sweet spots are NOT the
// same function of shape, a real, measured, non-obvious difference between
// dense WMMA (double-issue K=16x2) and sparse SWMMAC (single-issue K=32)
// register/occupancy profiles). Opt-in via GGML_HIP_DENSE_FP8_V3_ADAPTIVE
// (default off, same convention as the 2:4-V3 side) so it can be A/B'd.
static int ggml_cuda_dense_fp8_v3_warps_for_shape(int64_t N, int64_t /*K*/) {
    if (getenv("GGML_HIP_DENSE_FP8_V3_ADAPTIVE") == nullptr) {
        return ggml_cuda_dense_fp8_v3_warps();
    }
    return (N <= 4096) ? 4 : 32;
}
static int ggml_cuda_dense_fp8_v3_ilp() {
    static const int v = [] {
        const char * env = getenv("GGML_HIP_DENSE_FP8_V3_ILP");
        const int x = env ? atoi(env) : 4;
        return (x == 1 || x == 2 || x == 3 || x == 4 || x == 5 || x == 6 || x == 8) ? x : 4;
    }();
    return v;
}

template <int ILP>
static void launch_dense_fp8_v3(const dim3 & grid, const dim3 & block, cudaStream_t stream, bool exact,
        const block_f8e4m3 * weight, const block_f8e4m3 * act, float * dst,
        int64_t M, int64_t N, int64_t n_blocks_k, int64_t dst_row_stride_floats) {
    if (exact) {
        k_mul_mat_dense_fp8_v3<ILP, false><<<grid, block, 0, stream>>>(weight, act, dst, M, N, n_blocks_k, dst_row_stride_floats);
    } else {
        k_mul_mat_dense_fp8_v3<ILP, true><<<grid, block, 0, stream>>>(weight, act, dst, M, N, n_blocks_k, dst_row_stride_floats);
    }
}

bool ggml_cuda_op_mul_mat_dense_fp8_v3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F8E4M3);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(src0->ne[0] % QK_F8E4M3 == 0);

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    GGML_ASSERT(GGML_CUDA_CC_IS_RDNA4(cc) && "GGML_HIP_DENSE_FP8_V3 requires RDNA4");

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t n_blocks_k = K / QK_F8E4M3;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_f8e4m3> act_q(ctx.pool(), (size_t) (M * n_blocks_k));
    {
        const int64_t row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
        const dim3 grid(n_blocks_k, M, 1);
        const dim3 block(32, 1, 1);
        k_quantize_act_f8e4m3_dv3<<<grid, block, 0, stream>>>((const float *) src1->data, act_q.get(), n_blocks_k, row_stride_floats);
    }

    const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
    const int n_warps = ggml_cuda_dense_fp8_v3_warps_for_shape(N, K);
    const int ilp     = ggml_cuda_dense_fp8_v3_ilp();
    const dim3 block(32, n_warps, 1);
    const int64_t bn = (int64_t) n_warps * ilp * 16;
    const dim3 grid((N + bn - 1) / bn, (M + 15) / 16, 1);
    const bool exact = (M % 16 == 0) && (N % bn == 0);

    const block_f8e4m3 * d_w = (const block_f8e4m3 *) src0->data;

    switch (ilp) {
        case 1: launch_dense_fp8_v3<1>(grid, block, stream, exact, d_w, act_q.get(), (float *) dst->data, M, N, n_blocks_k, dst_row_stride_floats); break;
        case 2: launch_dense_fp8_v3<2>(grid, block, stream, exact, d_w, act_q.get(), (float *) dst->data, M, N, n_blocks_k, dst_row_stride_floats); break;
        case 3: launch_dense_fp8_v3<3>(grid, block, stream, exact, d_w, act_q.get(), (float *) dst->data, M, N, n_blocks_k, dst_row_stride_floats); break;
        case 5: launch_dense_fp8_v3<5>(grid, block, stream, exact, d_w, act_q.get(), (float *) dst->data, M, N, n_blocks_k, dst_row_stride_floats); break;
        case 6: launch_dense_fp8_v3<6>(grid, block, stream, exact, d_w, act_q.get(), (float *) dst->data, M, N, n_blocks_k, dst_row_stride_floats); break;
        case 8: launch_dense_fp8_v3<8>(grid, block, stream, exact, d_w, act_q.get(), (float *) dst->data, M, N, n_blocks_k, dst_row_stride_floats); break;
        default: launch_dense_fp8_v3<4>(grid, block, stream, exact, d_w, act_q.get(), (float *) dst->data, M, N, n_blocks_k, dst_row_stride_floats); break;
    }

    return true;
}

// Isolated per-shape microbench -- mirrors run_2of4_fp8_shape_bench in
// mul_mat_2of4_fp8.cu exactly (same shapes, same WARPS/ILP sweep, same
// hipEvent timing convention) so the two print side by side and can be
// diffed directly.
static double run_dense_fp8_v3_shape_bench(int64_t M, int64_t N, int64_t K, int n_warps, int ilp) {
    std::mt19937 rng(162163); // T162 dense-twin
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int64_t n_blocks_k = K / QK_F8E4M3;

    std::vector<float> w_f32((size_t) (N * K));
    for (auto & v : w_f32) { v = dist(rng); }
    std::vector<block_f8e4m3> w_blocks((size_t) (N * n_blocks_k));
    // Host-side quantize matching k_quantize_act_f8e4m3_dv3's device scheme
    // (per-32-block symmetric RTN to e4m3, amax/448.0 scale).
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < n_blocks_k; ++c) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) { amax = fmaxf(amax, fabsf(w_f32[n * K + c * 32 + i])); }
            const float d  = amax / 448.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            w_blocks[n * n_blocks_k + c].d = GGML_FP32_TO_FP16(d);
            for (int i = 0; i < 32; ++i) {
                w_blocks[n * n_blocks_k + c].qs[i] = ggml_fp32_to_e4m3(w_f32[n * K + c * 32 + i] * id);
            }
        }
    }

    std::vector<float> act_f32((size_t) (M * K));
    for (auto & v : act_f32) { v = dist(rng); }

    block_f8e4m3 * d_w = nullptr;
    float * d_act_f32 = nullptr;
    block_f8e4m3 * d_act_q = nullptr;
    float * d_dst = nullptr;
    if (hipMalloc(&d_w, w_blocks.size() * sizeof(block_f8e4m3)) != hipSuccess ||
        hipMalloc(&d_act_f32, act_f32.size() * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_act_q, (size_t) (M * n_blocks_k) * sizeof(block_f8e4m3)) != hipSuccess ||
        hipMalloc(&d_dst, (size_t) (M * N) * sizeof(float)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed (M=%ld N=%ld K=%ld)\n", __func__, (long) M, (long) N, (long) K);
        return -1.0;
    }
    CUDA_CHECK(hipMemcpy(d_w, w_blocks.data(), w_blocks.size() * sizeof(block_f8e4m3), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_act_f32, act_f32.data(), act_f32.size() * sizeof(float), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(d_dst, 0, (size_t) (M * N) * sizeof(float)));

    {
        const dim3 grid(n_blocks_k, M, 1);
        const dim3 block(32, 1, 1);
        k_quantize_act_f8e4m3_dv3<<<grid, block>>>(d_act_f32, d_act_q, n_blocks_k, K);
        CUDA_CHECK(hipDeviceSynchronize());
    }

    const dim3 block(32, n_warps, 1);
    const int64_t bn = (int64_t) n_warps * ilp * 16;
    const dim3 grid((N + bn - 1) / bn, (M + 15) / 16, 1);
    const bool exact = (M % 16 == 0) && (N % bn == 0);

    auto launch = [&] () {
        switch (ilp) {
            case 2: launch_dense_fp8_v3<2>(grid, block, 0, exact, d_w, d_act_q, d_dst, M, N, n_blocks_k, N); break;
            case 8: launch_dense_fp8_v3<8>(grid, block, 0, exact, d_w, d_act_q, d_dst, M, N, n_blocks_k, N); break;
            default: launch_dense_fp8_v3<4>(grid, block, 0, exact, d_w, d_act_q, d_dst, M, N, n_blocks_k, N); break;
        }
    };

    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);

    for (int r = 0; r < 3; ++r) { launch(); }
    CUDA_CHECK(hipDeviceSynchronize());

    const int n_reps = 10;
    hipEventRecord(ev0);
    for (int r = 0; r < n_reps; ++r) { launch(); }
    hipEventRecord(ev1);
    CUDA_CHECK(hipDeviceSynchronize());

    float t_ms = 0.0f;
    hipEventElapsedTime(&t_ms, ev0, ev1);
    hipEventDestroy(ev0);
    hipEventDestroy(ev1);
    (void) hipFree(d_w);
    (void) hipFree(d_act_f32);
    (void) hipFree(d_act_q);
    (void) hipFree(d_dst);

    return (double) t_ms / n_reps;
}

bool ggml_cuda_mul_mat_dense_fp8_v3_shape_bench() {
    struct Shape { const char * name; int64_t M, N, K; };
    const Shape shapes[] = {
        { "N=4096/K=4096 (q/o-proj)",   512, 4096,  4096 },
        { "N=4096/K=14336 (down-proj)", 512, 4096, 14336 },
        { "N=14336/K=4096 (gate/up-proj)", 512, 14336, 4096 },
    };
    const int warps_sweep[] = { 4, 8, 16, 32 };
    const int ilp_sweep[]   = { 2, 4, 8 };
    const int cu_count = 64;

    bool ok = true;
    for (const Shape & s : shapes) {
        double best_ms = 1e300;
        int    best_w = 0, best_i = 0;
        GGML_LOG_INFO("%s: --- shape %s (M=%ld N=%ld K=%ld) ---\n", __func__, s.name, (long) s.M, (long) s.N, (long) s.K);
        for (int w : warps_sweep) {
            for (int i : ilp_sweep) {
                const int64_t bn = (int64_t) w * i * 16;
                const int64_t grid_x = (s.N + bn - 1) / bn;
                const int64_t grid_y = (s.M + 15) / 16;
                const int64_t nblocks = grid_x * grid_y;
                const double  blocks_per_cu = (double) nblocks / cu_count;
                const double  ms = run_dense_fp8_v3_shape_bench(s.M, s.N, s.K, w, i);
                if (ms < 0.0) { ok = false; continue; }
                const double pp_equiv = ms > 0.0 ? (double) s.M / (ms / 1000.0) : 0.0;
                if (ms < best_ms) { best_ms = ms; best_w = w; best_i = i; }
                GGML_LOG_INFO("%s:   WARPS=%2d ILP=%d BN=%4ld blocks=%3ld (%.2f/CU) -> %.4f ms/call, pp-equiv=%.1f t/s\n",
                              __func__, w, i, (long) bn, (long) nblocks, blocks_per_cu, ms, pp_equiv);
            }
        }
        const double best_pp = best_ms > 0.0 ? (double) s.M / (best_ms / 1000.0) : 0.0;
        GGML_LOG_INFO("%s: BEST for %s -> WARPS=%d ILP=%d, %.4f ms/call, pp-equiv=%.1f t/s\n",
                      __func__, s.name, best_w, best_i, best_ms, best_pp);
    }
    return ok;
}
