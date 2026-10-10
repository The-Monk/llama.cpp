#include "common.cuh"
#include "fwht.cuh"
#include "fwht-dev.cuh"

#include <climits>
#include <cstdlib>

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

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
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

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

// T430 GGML_CUDA_FWHT_QUANT: N=1024 signed transform that also emits the Q2_FIELD-layout q8_1 form of its output
// (the MMVQ decode activation for Q2_0 g128 weights; the bytes quantize_q8_1<Q8_1_LAYOUT_Q2_FIELD> writes reading
// dst back). Lane j holds the 32 contiguous elements j*32..j*32+31 = exactly one q8_1 block, so the quantizer is
// in-lane (no shuffles). The butterfly network is the one fwht_cuda runs, stage for stage (h = 1, 2, 4, ...); only
// the element->thread mapping differs (low 5 index bits in registers, high 5 across lanes), so dst is
// bit-identical, and the in-lane sum uses warp_reduce_sum's xor-tree association so ds.y is too.
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_q8_1_cuda(const float * src, float * dst, const int64_t n_rows, const float scale,
                               const float * signs, const int n_blk, block_q8_1 * qy,
                               const int perm_hd, const int perm_nk, const int perm_rep) {
    constexpr int N = 1024;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == QK8_1 && N == warp_size*QK8_1, "one q8_1 block per lane");

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n_rows) {
        return;
    }
    const int lane = threadIdx.x;

    // perm_rep > 0: src is the TILED [hd, nk, rep] layout of the GDN output and the transform input is the grouped
    // [hd, rep, nk] reorder (build_lora_mm's reshape/permute/cont, folded into this load). 32 contiguous output
    // elements stay inside one hd-run (32 | hd), so the gather is still one contiguous 32-float read per lane.
    int64_t so = r * N + lane * QK8_1;
    if (perm_rep > 0) {
        const int64_t a0 = so % perm_hd, t = so / perm_hd;
        so = a0 + (int64_t) perm_hd * (t / perm_rep + (int64_t) perm_nk * (t % perm_rep));
    }
    const float4 * s4 = (const float4 *) (src + so);
    const float4 * g4 = (const float4 *) (signs + (r % n_blk) * N + lane * QK8_1);
    float reg[QK8_1];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int k = 0; k < QK8_1/4; ++k) {
        const float4 v = s4[k];
        const float4 g = g4[k];
        reg[4*k + 0] = (v.x * scale) * g.x;
        reg[4*k + 1] = (v.y * scale) * g.y;
        reg[4*k + 2] = (v.z * scale) * g.z;
        reg[4*k + 3] = (v.w * scale) * g.w;
    }

    ggml_cuda_fwht1024_q8_1_lane(reg, lane, dst + r * N + lane * QK8_1, qy + r * (N / QK8_1) + lane);
}

static bool fwht_dispatch(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                          const ggml_tensor * signs_t, block_q8_1 * qy = nullptr,
                          const int perm_hd = 0, const int perm_nk = 0, const int perm_rep = 0) {
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

    if (qy && (n != 1024 || n % QK8_1 != 0)) {
        return false;
    }
    if (perm_rep > 0 && (!qy || perm_hd % QK8_1 != 0 || (int64_t) perm_hd * perm_nk * perm_rep != ggml_nelements(dst) ||
                         ggml_nelements(src) != ggml_nelements(dst))) {
        return false;   // the gather form is single-token decode only (one run of hd*nk*rep elements)
    }

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
            if (qy) {
                if (!signs_d || ((uintptr_t) src_d | (uintptr_t) dst_d | (uintptr_t) signs_d) % 16 != 0) {
                    return false;
                }
                ggml_cuda_kernel_launch(fwht_q8_1_cuda, launch_params, src_d, dst_d, rows, scale, signs_d, n_blk, qy, perm_hd, perm_nk, perm_rep);
                return true;
            }
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

bool ggml_cuda_op_fwht_signed_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                                   const ggml_tensor * signs, ggml_tensor * dst, void * qy) {
    return fwht_dispatch(ctx, src, dst, signs, (block_q8_1 *) qy);
}

bool ggml_cuda_op_fwht_signed_q8_1_perm(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                                        const ggml_tensor * signs, ggml_tensor * dst, void * qy,
                                        int perm_hd, int perm_nk, int perm_rep) {
    return fwht_dispatch(ctx, src, dst, signs, (block_q8_1 *) qy, perm_hd, perm_nk, perm_rep);
}
