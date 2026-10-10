#include "common.cuh"
#include "fwht.cuh"
#include "n1_act.cuh"
#include "unary.cuh"

#include <climits>
#include <cstdlib>

// The butterfly network, shared by fwht_cuda and fwht_quant_n1t (T434). Under -funsafe-math-optimizations the compiler may
// reassociate or contract these adds differently in every kernel it is inlined into (the fused producer's rotated fp32
// differed from the standalone kernel's in ~13% of bytes); one strict-IEEE body makes both produce the same values.
template <int N, int WS>
static __device__ __forceinline__ void fwht_net(float (&reg)[N / WS], const int lane) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    constexpr int el_w = N / WS;
#pragma unroll
    for (int h = 1; h < WS; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, WS);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = WS; h < N; h *= 2) {
        const int step = h / WS;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

template <int N, bool has_signs = false>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const float * src, float * dst, const int64_t n_rows, const float scale,
                          const float * signs = nullptr, const int n_blk = 1) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    // prism.hadamard folds an explicit +/-1 sign vector into the transform. Applying it
    // here costs one multiply on a value already in a register; as a separate ggml_mul it
    // costs a kernel launch plus a full read/write pass over the activation.
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int idx = i * warp_size + lane;
        float v = src[idx] * scale;
        if (has_signs) {
            v *= signs_row[idx];
        }
        reg[i] = v;
    }

    fwht_net<N, warp_size>(reg, lane);

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

// T434 M3: signed FWHT (1024 blocks, the arithmetic of fwht_cuda<1024, true> verbatim) + per-token int8 quantize into the N1
// layout (n1_act.cuh) in one kernel. One workgroup per token, one warp per 1024-block of the row; the rotated values stay in
// registers until the token's amax (block reduce) is known, then are quantized exactly like k_n1_quant_act (G = K): d =
// amax/127, q = roundf(v / d); sx[token] = d. The int8 bytes are staged through LDS so every lane stores 4 contiguous
// bytes (u32) at n1_off. write_fp32: also store the rotated fp32 (the unfused FWHT output) for consumers that need it.
// glu: the FWHT input is silu(src) * up (a swiglu_split node whose only consumer is this transform), read straight from the
// gate (src) and up tensors with row strides s_src / s_up (elements); the same float expression as the GLU producers.
template <bool write_fp32, bool glu>
__launch_bounds__(1024, 1)
__global__ void fwht_quant_n1t(const float * __restrict__ src, const float * __restrict__ up, const int64_t s_src, const int64_t s_up,
                               float * __restrict__ dst, const float scale,
                               const float * __restrict__ signs, int8_t * __restrict__ X, float * __restrict__ sx,
                               const int S) {
    constexpr int N = 1024, warp_size = 32, el_w = N / warp_size;
    const int tok = blockIdx.x, w = threadIdx.y, lane = threadIdx.x;
    const int K = blockDim.y * N;
    extern __shared__ __align__(16) char smem[];
    float  * red = (float *) smem;
    int8_t * qs  = (int8_t *) (smem + 128);

    const float * s  = src + (size_t) tok * s_src + (size_t) w * N;
    const float * u  = glu ? up + (size_t) tok * s_up + (size_t) w * N : nullptr;
    const float * sg = signs + (size_t) w * N;
    float reg[el_w];
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int idx = i * warp_size + lane;
        float v = glu ? ggml_cuda_op_silu_single(s[idx]) * u[idx] : s[idx];
        v *= scale;
        v *= sg[idx];
        reg[i] = v;
    }
    fwht_net<N, warp_size>(reg, lane);
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        asm volatile("" : "+v"(reg[i]));   // the rotated value is a rounded fp32 from here on (as when the unfused kernel stores it)
    }
    if (write_fp32) {   // dst is the FWHT node: contiguous rows of K
        float * d = dst + (size_t) tok * K + (size_t) w * N;
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            d[i * warp_size + lane] = reg[i];
        }
    }
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        amax = fmaxf(amax, fabsf(reg[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, o, 32));
    }
    if (lane == 0) {
        red[w] = amax;
    }
    __syncthreads();
    amax = 0.f;
    for (int i = 0; i < (int) blockDim.y; ++i) {
        amax = fmaxf(amax, red[i]);
    }
    const float d  = n1_scale(amax);
    const float id = n1_recip(d);   // d == 0 (an all-zero token): the quantized bytes are forced to 0 below
    if (w == 0 && lane == 0) {
        sx[tok] = d;
    }
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        qs[w * N + i * warp_size + lane] = (int8_t) (d > 0.f ? (uint8_t) n1_q1(reg[i], id) : 0);
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < N / 4 / warp_size; ++m) {
        const int j = lane + warp_size * m;
        *(uint32_t *) (X + n1_off(tok, w * N + 4 * j, S)) = *(const uint32_t *) (qs + w * N + 4 * j);
    }
}

static bool fwht_dispatch(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                          const ggml_tensor * signs_t) {
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // the transform width is dst's row length, not src's: in the fused sign-flip form the
    // source is the whole activation (e.g. 5120) while each transform covers one block (1024)
    const int     n    = dst->ne[0];
    const int64_t rows = ggml_nelements(dst) / n;

    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const float * signs_d = nullptr;
    int           n_blk   = 1;
    if (signs_t) {
        if (signs_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs_t) || signs_t->ne[0] % n != 0) {
            return false;
        }
        signs_d = (const float *) signs_t->data;
        n_blk   = signs_t->ne[0] / n;
    }

    const float * src_d = (const float *) src->data;
    float *       dst_d = (float *) dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    const float scale = 1 / sqrtf(n);

    // A/B escape hatch: GGML_CUDA_FWHT_MAX_N caps the accepted transform width, so the
    // dense NxN f16 GEMM fallback can be measured against this kernel without a rebuild.
    static const int fwht_max_n = []() {
        const char * e = getenv("GGML_CUDA_FWHT_MAX_N");
        return e ? atoi(e) : INT_MAX;
    }();
    if (n > fwht_max_n) {
        return false;
    }

    switch (n) {
        case 64:
            if (signs_d) {
                ggml_cuda_kernel_launch(fwht_cuda<64, true>,  launch_params, src_d, dst_d, rows, scale, signs_d, n_blk);
            } else {
                ggml_cuda_kernel_launch(fwht_cuda<64, false>, launch_params, src_d, dst_d, rows, scale, (const float *) nullptr, 1);
            }
            return true;
        case 128:
            if (signs_d) {
                ggml_cuda_kernel_launch(fwht_cuda<128, true>,  launch_params, src_d, dst_d, rows, scale, signs_d, n_blk);
            } else {
                ggml_cuda_kernel_launch(fwht_cuda<128, false>, launch_params, src_d, dst_d, rows, scale, (const float *) nullptr, 1);
            }
            return true;
        case 256:
            if (signs_d) {
                ggml_cuda_kernel_launch(fwht_cuda<256, true>,  launch_params, src_d, dst_d, rows, scale, signs_d, n_blk);
            } else {
                ggml_cuda_kernel_launch(fwht_cuda<256, false>, launch_params, src_d, dst_d, rows, scale, (const float *) nullptr, 1);
            }
            return true;
        case 512:
            if (signs_d) {
                ggml_cuda_kernel_launch(fwht_cuda<512, true>,  launch_params, src_d, dst_d, rows, scale, signs_d, n_blk);
            } else {
                ggml_cuda_kernel_launch(fwht_cuda<512, false>, launch_params, src_d, dst_d, rows, scale, (const float *) nullptr, 1);
            }
            return true;
        // N=1024/2048 keep the register path: el_w = N/warp_size is 32/64 floats per
        // thread, still inside the 96-VGPR occupancy knee measured on gfx1201. Without
        // these the whole transform falls back to a dense NxN f16 GEMM -- which is what
        // block_size=1024 models (Ternary-Bonsai-2-27B) hit: 54.7 -> 33.9 t/s.
        case 1024:
            if (signs_d) {
                ggml_cuda_kernel_launch(fwht_cuda<1024, true>,  launch_params, src_d, dst_d, rows, scale, signs_d, n_blk);
            } else {
                ggml_cuda_kernel_launch(fwht_cuda<1024, false>, launch_params, src_d, dst_d, rows, scale, (const float *) nullptr, 1);
            }
            return true;
        case 2048:
            if (signs_d) {
                ggml_cuda_kernel_launch(fwht_cuda<2048, true>,  launch_params, src_d, dst_d, rows, scale, signs_d, n_blk);
            } else {
                ggml_cuda_kernel_launch(fwht_cuda<2048, false>, launch_params, src_d, dst_d, rows, scale, (const float *) nullptr, 1);
            }
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    return fwht_dispatch(ctx, src, dst, nullptr);
}

bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst) {
    return fwht_dispatch(ctx, src, dst, signs);
}

bool ggml_cuda_op_fwht_signed_quant_n1t(ggml_backend_cuda_context & ctx, const ggml_tensor * src, const ggml_tensor * up,
                                        const ggml_tensor * signs, ggml_tensor * dst, bool write_fp32, void * y) {
    const bool glu = up != nullptr;
    if (!ggml_is_contiguous(dst) || !ggml_is_contiguous(signs) ||
        src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || signs->type != GGML_TYPE_F32) {
        return false;
    }
    if (glu ? (!ggml_is_contiguous_1(src) || !ggml_is_contiguous_1(up) || up->type != GGML_TYPE_F32 ||
               src->nb[0] != sizeof(float) || up->nb[0] != sizeof(float) || !ggml_are_same_shape(src, up) ||
               ggml_nrows(src) != src->ne[1])
            : !ggml_is_contiguous(src)) {
        return false;
    }
    const int64_t K = src->ne[0];
    if (dst->ne[0] != 1024 || K % 1024 != 0 || K > 32 * 1024 || signs->ne[0] != K || K % 128 != 0) {
        return false;
    }
    const int64_t ntok = ggml_nelements(src) / K;
    if (ggml_nelements(dst) != ntok * K || ntok > (1 << 20)) {
        return false;
    }
    const int64_t s_src = glu ? (int64_t) (src->nb[1] / sizeof(float)) : K;
    const int64_t s_up  = glu ? (int64_t) (up->nb[1] / sizeof(float)) : K;
    const int nblk = (int) (K / 1024);
    const int64_t Npad = n1_npad(ntok);
    int8_t * X  = (int8_t *) y;
    float *  sx = (float *) ((char *) y + (size_t) Npad * K);

    const dim3   grid((unsigned) ntok, 1, 1), block(32, nblk, 1);
    const size_t shmem = 128 + (size_t) nblk * 1024;
    const ggml_cuda_kernel_launch_params lp(grid, block, shmem, ctx.stream());
    const float scale = 1 / sqrtf(1024.f);
    const float * a = (const float *) src->data;
    const float * b = glu ? (const float *) up->data : a;
    float *       d = (float *) dst->data;
    const float * g = (const float *) signs->data;
    const int     S = (int) (K / 128);
#define FQ(WF, GL) ggml_cuda_kernel_launch(fwht_quant_n1t<WF, GL>, lp, a, b, s_src, s_up, d, scale, g, X, sx, S)
    if (glu) {
        if (write_fp32) FQ(true, true); else FQ(false, true);
    } else {
        if (write_fp32) FQ(true, false); else FQ(false, false);
    }
#undef FQ
    return true;
}
