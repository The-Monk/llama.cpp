#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

// T368: single-token conv with the conv state updated in place (vLLM causal_conv1d_update pattern).
// One thread per channel, exactly like ssm_conv_f32's i == 0 iteration: the window [state, x_new] is loaded
// into registers, the tap sum uses the same products in the same order as the compiled ssm_conv_f32 (see the
// d_conv == 4 note below), the same silu; then the thread writes the shifted window tail back over its own state row. Each
// thread reads its row before writing it and no other thread touches that row, so there is no race.
template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_update_f32(float * state_ptr, const float * x_ptr, const float * w_ptr,
                                           const float * bias_ptr,
                                           const int state_nb1, const int state_nb2, const int x_nb2, const int w_nb1,
                                           float * dst_ptr, const int dst_nb2) {
    ggml_cuda_pdl_lc();
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const int ch = bidy * split_d_inner + tid;

    float *       s_row = (float *) ((char *) state_ptr + bidx * state_nb2 + ch * state_nb1);
    const float * x_row = (const float *) ((const char *) x_ptr + bidx * x_nb2);
    const float * w_row = (const float *) ((const char *) w_ptr + ch * w_nb1);
    float *       y_row = (float *) ((char *) dst_ptr + bidx * dst_nb2);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_row[j];
    }

    float b = bias_ptr != nullptr ? bias_ptr[ch] : 0.0f;

#pragma unroll
    for (size_t j = 0; j < d_conv - 1; j++) {
        x[j] = s_row[j];
    }
    x[d_conv - 1] = x_row[ch];

    float sumf;
    if constexpr (d_conv == 4) {
        // Bit-exactness with the graph path: ggml-hip builds with -funsafe-math-optimizations, and for
        // ssm_conv_f32<*, 128, 4>'s i == 0 iteration (the only one a decode token runs) the compiler
        // reassociates `0 + x0*w0 + x1*w1 + x2*w2 + x3*w3 + b` into fma(x3,w3,b) -> +x1*w1 -> +x2*w2 ->
        // +x0*w0 (read off the gfx1201 ISA, T368). Explicit fmaf calls are not reassociated, so this pins
        // the same order here. Re-check the ISA if the compiler changes.
        sumf = fmaf(x[3], w[3], b);
        sumf = fmaf(x[1], w[1], sumf);
        sumf = fmaf(x[2], w[2], sumf);
        sumf = fmaf(x[0], w[0], sumf);
    } else {
        sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[j] * w[j];
        }
        sumf += b;
    }
    y_row[ch] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;

#pragma unroll
    for (size_t j = 0; j < d_conv - 1; j++) {
        s_row[j] = x[j + 1];
    }
}

// T409: keeps the compiler from reassociating a float sum across this point (same trick as gated_delta_net.cu's
// gdn_opaque, T361); ggml-hip builds with -funsafe-math-optimizations.
static __device__ __forceinline__ float ssm_conv_opaque(float x) {
#if defined(GGML_USE_HIP)
    asm volatile("" : "+v"(x));
#endif
    return x;
}

// T409: multi-token conv for prefill with the conv state read from and written back to the recurrent cache row in
// place. Replaces the graph's get_rows gather + concat([state | x^T]) + ssm_conv(+silu) + write-back cpy, and
// optionally the two L2_NORM ops on the q/k rows. Structure and compute loop are ssm_conv_long_token_f32's (grid z
// over split_n_t-token tiles, a [split_d_inner][d_conv-1+split_n_t] smem window); only the smem load differs: the
// window is read straight from the state row (tile 0's halo) and from x in its native token-major layout
// (128 consecutive channels per token = coalesced) instead of from a materialized transposed concat.
// State write-back: the new state is the last d_conv-1 window columns, which come from x (an input) or, when
// n_t < d_conv-1, from the old state. Only tile bidz == 0 reads the old state and only it writes the new one, and
// each thread reads and writes only its own channel row, so there is no race.
// L2 (l2_n > 0): blocks whose 128 channels lie in [0, l2_n) are one q or k head; their outputs go through smem and
// each wave normalizes whole tokens with norm.cu l2_norm_f32<32>'s exact arithmetic (lane l accumulates channels
// l, l+32, l+64, l+96 in that order, warp_reduce_sum, scale = rsqrtf(fmaxf(sum, eps^2)), out = scale * y).
template <bool apply_silu, bool do_l2, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_prefill_f32(float * __restrict__ state, const float * __restrict__ x,
                                            const float * __restrict__ src1, const float * __restrict__ bias,
                                            const int state_nb1, const int state_nb2, const int x_nb1,
                                            const int x_nb2, const int src1_nb1, float * __restrict__ dst,
                                            const int dst_nb1, const int dst_nb2, const int64_t n_t,
                                            const int state_is_zero, const int l2_n, const float l2_eps) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const int ch = bidy * split_d_inner + tid;

    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * sizeof(float));
    const float * x_seq = (const float *) ((const char *) x + bidx * x_nb2);
    float *       s_row = (float *) ((char *) state + bidx * state_nb2 + ch * state_nb1);

    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);
    const int stride_x = x_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    // window column p of this tile = global window position gp = bidz*split_n_t + p; positions < d_conv-1 are
    // the old state (only tile 0 has them), position d_conv-1+t is token t (zero past the last token, never used)
#pragma unroll
    for (int p = 0; p < (int) (d_conv - 1 + split_n_t); p++) {
        const int64_t gp = bidz * split_n_t + p;
        float v;
        if (gp < (int64_t) (d_conv - 1)) {
            v = state_is_zero ? 0.0f : s_row[gp];
        } else {
            const int64_t t = gp - (int64_t) (d_conv - 1);
            v = t < n_t ? x_seq[t * stride_x + ch] : 0.0f;
        }
        smem[tid * n_cols + p] = v;
    }
    __syncthreads();

    if (bidz == 0) {
        // new state = window positions n_t .. n_t+d_conv-2 (own row; the old values were read above)
#pragma unroll
        for (int j = 0; j < (int) (d_conv - 1); j++) {
            const int64_t gp = n_t + j;
            s_row[j] = gp < (int64_t) (d_conv - 1) ? smem[tid * n_cols + gp] : x_seq[(gp - (int64_t) (d_conv - 1)) * stride_x + ch];
        }
    }

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    const bool l2_block = do_l2 && (int) ((bidy + 1) * split_d_inner) <= l2_n;
    float * ys = smem + split_d_inner * n_cols; // [split_n_t][split_d_inner], do_l2 only

    // Compute from shared memory (ssm_conv_long_token_f32's loop)
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf;
        if constexpr (d_conv == 4) {
            // Pin the tap order ssm_conv_long_token_f32<*, 128, 4, 32> compiles to under -funsafe-math (read off
            // the gfx1201 ISA, T409; same order T368 found for ssm_conv_f32): fma(x3,w3,b) -> x1*w1 -> x2*w2 ->
            // x0*w0. Without the pin the do_l2 instantiation reassociated to w0..w3 and was not bit-identical.
            const float * xs = smem + tid * n_cols + i;
            sumf = fmaf(xs[3], w[3], b);
            sumf = fmaf(xs[1], w[1], sumf);
            sumf = fmaf(xs[2], w[2], sumf);
            sumf = fmaf(xs[0], w[0], sumf);
        } else {
            sumf = 0.0f;
#pragma unroll
            for (size_t j = 0; j < d_conv; j++) {
                sumf += smem[tid * n_cols + i + j] * w[j];
            }
            sumf += b;
        }
        const float yv = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
        if (l2_block) {
            ys[i * split_d_inner + tid] = yv;
        } else {
            y_block[i * stride_y + tid] = yv;
        }
    }

    if constexpr (do_l2) {
        static_assert(split_d_inner == 128, "L2 epilogue assumes one 128-wide head per block");
        if (l2_block) {
            constexpr int ws = 32;
            __syncthreads();
            const int wave = tid / ws;
            const int lane = tid % ws;
            for (int64_t i = wave; i < local_n_t; i += split_d_inner / ws) {
                const float * yr = ys + i * split_d_inner;
                float acc = 0.0f;
#pragma unroll
                for (int r = 0; r < (int) split_d_inner / ws; r++) {
                    acc = ssm_conv_opaque(acc + yr[r * ws + lane] * yr[r * ws + lane]);
                }
                const float sum   = warp_reduce_sum<ws>(acc);
                const float scale = rsqrtf(fmaxf(sum, l2_eps * l2_eps));
#pragma unroll
                for (int r = 0; r < (int) split_d_inner / ws; r++) {
                    y_block[i * stride_y + r * ws + lane] = scale * yr[r * ws + lane];
                }
            }
        }
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// T409 host side of ssm_conv_prefill_f32 (multi-token GGML_OP_SSM_CONV_UPDATE).
static void ssm_conv_prefill_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool apply_silu) {
    ggml_tensor *              state = dst->src[0];
    const struct ggml_tensor * w     = dst->src[1];
    const struct ggml_tensor * x     = dst->src[2];

    const bool  state_is_zero = ggml_get_op_params_i32(dst, 1) != 0;
    const int   l2_n          = ggml_get_op_params_i32(dst, 2);
    const float l2_eps        = ggml_get_op_params_f32(dst, 3);

    const int64_t nc  = w->ne[0];
    const int64_t nr  = state->ne[1];
    const int64_t n_s = state->ne[2];
    const int64_t n_t = x->ne[1];

    GGML_ASSERT(state->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(state->ne[0] == nc - 1);
    GGML_ASSERT(state->nb[0] == sizeof(float) && x->nb[0] == sizeof(float) && w->nb[0] == sizeof(float));
    GGML_ASSERT(dst->ne[0] == nr && dst->ne[1] == n_t && dst->nb[0] == sizeof(float));

    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);
    GGML_ASSERT(l2_n % threads == 0 && l2_n <= nr);
    GGML_ASSERT(l2_n == 0 || ggml_cuda_info().devices[ctx.device].warp_size == 32);

    const int64_t split_n_t = 32;
    const dim3    blocks(n_s, nr / threads, (n_t + split_n_t - 1) / split_n_t);
    cudaStream_t  stream = ctx.stream();

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        const size_t smem_win = threads * (kNC - 1 + split_n_t) * sizeof(float);
        const size_t smem_l2  = threads * split_n_t * sizeof(float);
#define T409_LAUNCH(SILU, L2)                                                                                    \
        ssm_conv_prefill_f32<SILU, L2, threads, kNC, split_n_t><<<blocks, threads, smem_win + ((L2) ? smem_l2 : 0), stream>>>( \
            (float *) state->data, (const float *) x->data, (const float *) w->data, (const float *) nullptr,     \
            (int) state->nb[1], (int) state->nb[2], (int) x->nb[1], (int) x->nb[2], (int) w->nb[1],                \
            (float *) dst->data, (int) dst->nb[1], (int) dst->nb[2], n_t, state_is_zero ? 1 : 0, l2_n, l2_eps)
        if (apply_silu) {
            if (l2_n > 0) { T409_LAUNCH(true, true); } else { T409_LAUNCH(true, false); }
        } else {
            if (l2_n > 0) { T409_LAUNCH(false, true); } else { T409_LAUNCH(false, false); }
        }
#undef T409_LAUNCH
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv_update(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor *              state = dst->src[0];  // conv state view, updated in place
    const struct ggml_tensor * w     = dst->src[1];  // conv1d.weight
    const struct ggml_tensor * x     = dst->src[2];  // new token

    const bool apply_silu = ggml_get_op_params_i32(dst, 0) != 0;

    const int64_t nc  = w->ne[0];       // d_conv
    const int64_t nr  = state->ne[1];   // d_inner
    const int64_t n_s = state->ne[2];

    if (x->ne[1] > 1) {
        ssm_conv_prefill_cuda(ctx, dst, apply_silu);
        return;
    }

    GGML_ASSERT(state->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(state->ne[0] == nc - 1);
    GGML_ASSERT(state->nb[0] == sizeof(float) && x->nb[0] == sizeof(float) && w->nb[0] == sizeof(float));
    GGML_ASSERT(dst->ne[0] == nr && dst->nb[0] == sizeof(float));

    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    float *       state_d = (float *) state->data;
    const float * x_d     = (const float *) x->data;
    const float * w_d     = (const float *) w->data;
    float *       dst_d   = (float *) dst->data;
    cudaStream_t  stream  = ctx.stream();

    const dim3 blocks(n_s, nr / threads, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (apply_silu) {
            ggml_cuda_kernel_launch(ssm_conv_update_f32<true, threads, kNC>, launch_params, state_d, x_d, w_d,
                                    (const float *) nullptr, (int) state->nb[1], (int) state->nb[2], (int) x->nb[2],
                                    (int) w->nb[1], dst_d, (int) dst->nb[2]);
        } else {
            ggml_cuda_kernel_launch(ssm_conv_update_f32<false, threads, kNC>, launch_params, state_d, x_d, w_d,
                                    (const float *) nullptr, (int) state->nb[1], (int) state->nb[2], (int) x->nb[2],
                                    (int) w->nb[1], dst_d, (int) dst->nb[2]);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}
