// [TAG_ACT_FUSE] T407: see act-fuse.cuh.

#include "act-fuse.cuh"
#include "n1_act.cuh"
#include "sc_act.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "unary.cuh"

#include <cstdlib>
#include <cstring>
#include <type_traits>

int ggml_cuda_act_fuse_mask() {
    static const int mask = [] {
        const char * e = getenv("GGML_ACT_FUSE");
        if (e != nullptr && atoi(e) == 0) {
            return 0;
        }
        const char * m = getenv("GGML_ACT_FUSE_MASK");
        const int v = m ? atoi(m) : (GGML_ACT_FUSE_DEDUP | GGML_ACT_FUSE_GLU | GGML_ACT_FUSE_NORM);
        fprintf(stderr, "%s: T407 activation-path fusion on, mask %d (1 dedup, 2 glu, 4 norm/add)%s\n",
                      __func__, v, getenv("GGML_ACT_FUSE_VERIFY") ? ", VERIFY" : "");
        return v;
    }();
    return mask;
}

bool ggml_cuda_act_fuse_verify() {
    static const bool v = getenv("GGML_ACT_FUSE_VERIFY") != nullptr && atoi(getenv("GGML_ACT_FUSE_VERIFY")) != 0;
    return v;
}

// ---------------------------------------------------------------------------
// Store functor: MMQ q8_1 (block_q8_1_mmq), 2D activation (channel 0).
// The body is quantize_mmq_q8_1 (quantize.cu) from the amax load on, so the
// bytes are identical for the same fp32 inputs.

template <mmq_q8_1_ds_layout ds_layout>
struct mmq_q8_1_store {
    void *  vy;
    int64_t ne1; // rows (tokens) of the activation

    __device__ __forceinline__ void operator()(const int64_t i1, const int64_t i0, const float4 xi) const {
        constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 :
                                       ds_layout == MMQ_Q8_1_DS_LAYOUT_D128 ? 128 : 32;
        constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

        block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

        const int64_t ib  = (i0 / (4*QK8_1))*ne1 + i1; // block index in channel
        const int64_t iqs = i0 % (4*QK8_1);            // quant index in block

        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
        for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
        }

        float sum;
        if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4 && ds_layout != MMQ_Q8_1_DS_LAYOUT_D128) {
            sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
            for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
            }
        }

        const float d_inv = 127.0f / amax;
        char4 q;
        q.x = roundf(xi.x*d_inv);
        q.y = roundf(xi.y*d_inv);
        q.z = roundf(xi.z*d_inv);
        q.w = roundf(xi.w*d_inv);

        char4 * yqs4 = (char4 *) y[ib].qs;
        yqs4[iqs/4] = q;

        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            if (iqs % 16 != 0 || iqs >= 96) {
                return;
            }
            y[ib].d2s6[2 + iqs/16] = sum;
            if (iqs % 64 != 0) {
                return;
            }
            const float d = 1.0f / d_inv;
            y[ib].d2s6[iqs/64] = d;
            return;
        }

        if (iqs % 32 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs/32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs/32]  = d;
        }
    }
};

// ---------------------------------------------------------------------------
// GLU (swiglu_split) -> quantize. Thread/block map = quantize_mmq_q8_1.

template <class store_t>
static __global__ void act_glu_quant_kernel(
        const float * __restrict__ gate, const float * __restrict__ up,
        const int64_t nc, const int64_t s_gate, const int64_t s_up, const int64_t ne0, const store_t store) {
    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;
    if (i0 >= ne0) {
        return;
    }
    const int64_t i1 = blockIdx.x;

    ggml_cuda_pdl_sync();
    const float * g = gate + i1*s_gate;
    const float * u = up   + i1*s_up;

    // same expression as unary_gated_op_kernel<op_silu>: op((float)x[j0]) * (float)g[j1]
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int64_t c = i0 + k;
        v[k] = c < nc ? ggml_cuda_op_silu_single(g[c]) * u[c] : 0.0f;
    }
    if constexpr (std::is_same_v<store_t, n1_act_store256>) {
        // T422: the other half of this 256-group (column ^ 128), same expression
        float p[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int64_t c = (i0 + k) ^ 128;
            p[k] = c < nc ? ggml_cuda_op_silu_single(g[c]) * u[c] : 0.0f;
        }
        store(i1, i0, make_float4(v[0], v[1], v[2], v[3]), make_float4(p[0], p[1], p[2], p[3]));
    } else {
        store(i1, i0, make_float4(v[0], v[1], v[2], v[3]));
    }
}

template <class store_t>
static void act_glu_quant_launch(const float * gate, const float * up, int64_t nc, int64_t nrows,
        int64_t s_gate, int64_t s_up, int64_t ne0_padded, const store_t & store, cudaStream_t stream) {
    const int64_t block_num_y = (ne0_padded + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(nrows, block_num_y, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    act_glu_quant_kernel<<<num_blocks, block_size, 0, stream>>>(gate, up, nc, s_gate, s_up, ne0_padded, store);
}

void ggml_cuda_act_glu_quant(const float * gate, const float * up, int64_t nc, int64_t nrows,
        int64_t s_gate, int64_t s_up, int64_t ne0_padded, int layout, void * y, cudaStream_t stream) {
    GGML_ASSERT(ne0_padded % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0);
    switch (layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            act_glu_quant_launch(gate, up, nc, nrows, s_gate, s_up, ne0_padded, mmq_q8_1_store<MMQ_Q8_1_DS_LAYOUT_D4>{y, nrows}, stream);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            act_glu_quant_launch(gate, up, nc, nrows, s_gate, s_up, ne0_padded, mmq_q8_1_store<MMQ_Q8_1_DS_LAYOUT_DS4>{y, nrows}, stream);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            act_glu_quant_launch(gate, up, nc, nrows, s_gate, s_up, ne0_padded, mmq_q8_1_store<MMQ_Q8_1_DS_LAYOUT_D2S6>{y, nrows}, stream);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D128:
            act_glu_quant_launch(gate, up, nc, nrows, s_gate, s_up, ne0_padded, mmq_q8_1_store<MMQ_Q8_1_DS_LAYOUT_D128>{y, nrows}, stream);
            break;
        case GGML_CUDA_ACT_LAYOUT_N1:
            GGML_ASSERT(nc % 128 == 0);
            act_glu_quant_launch(gate, up, nc, nrows, s_gate, s_up, ne0_padded, n1_act_store_make(y, nc, nrows), stream);
            break;
        case GGML_CUDA_ACT_LAYOUT_N1G256:
            GGML_ASSERT(nc % 256 == 0);
            act_glu_quant_launch(gate, up, nc, nrows, s_gate, s_up, ne0_padded, n1_act_store256_make(y, nc, nrows), stream);
            break;
        case GGML_CUDA_ACT_LAYOUT_SC:
            GGML_ASSERT(nc % 128 == 0);
            act_glu_quant_launch(gate, up, nc, nrows, s_gate, s_up, ne0_padded, sc_act_store_make(y, nc, nrows), stream);
            break;
        default:
            GGML_ABORT("act-fuse: unsupported layout");
    }
}

// ---------------------------------------------------------------------------
// [ADD ->] RMS_NORM -> MUL -> quantize. One 1024-thread block per row, the
// reduction is rms_norm_f32<1024, ...> (norm.cu) verbatim; the normalized row
// is staged in LDS and re-read in the quantizer's 4-values-per-lane map.

static constexpr int ACT_NORM_BLOCK     = 1024;
static constexpr int ACT_NORM_MAX_NCOLS = 12288; // 48 KiB of LDS staging

template <bool do_add, class store_t>
static __global__ void __launch_bounds__(ACT_NORM_BLOCK)
act_norm_quant_kernel(
        const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ dst_add,
        float * __restrict__ dst_mul, const float * __restrict__ w,
        const int ncols, const int64_t s_a, const int64_t s_b, const float eps, const int64_t ne0, const store_t store) {
    extern __shared__ float act_smem[];
    float * s_sum = act_smem;      // 32 floats, block_reduce scratch
    float * xs    = act_smem + 32; // ncols floats

    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;

    a       += row*s_a;
    dst_mul += row*ncols;
    if constexpr (do_add) {
        b       += row*s_b;
        dst_add += row*ncols;
    }

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += ACT_NORM_BLOCK) {
        float xi;
        if constexpr (do_add) {
            xi = a[col] + b[col];
            dst_add[col] = xi;
        } else {
            xi = a[col];
        }
        xs[col] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, ACT_NORM_BLOCK>(tmp, s_sum);
    __syncthreads(); // xs fully written

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int64_t i0 = 4*(int64_t)tid; i0 < ne0; i0 += 4*ACT_NORM_BLOCK) {
        float v[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int64_t c = i0 + k;
            if (c < ncols) {
                v[k] = scale * xs[c] * w[c];
                dst_mul[c] = v[k];
            } else {
                v[k] = 0.0f;
            }
        }
        if constexpr (std::is_same_v<store_t, n1_act_store256>) {
            // T422: the other half of this 256-group (column ^ 128) from the staged row, same expression
            float p[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int64_t c = (i0 + k) ^ 128;
                p[k] = c < ncols ? scale * xs[c] * w[c] : 0.0f;
            }
            store(row, i0, make_float4(v[0], v[1], v[2], v[3]), make_float4(p[0], p[1], p[2], p[3]));
        } else {
            store(row, i0, make_float4(v[0], v[1], v[2], v[3]));
        }
    }
}

template <class store_t>
static void act_norm_quant_launch(const float * a, const float * b, float * dst_add, float * dst_mul, const float * w,
        int64_t ncols, int64_t nrows, int64_t s_a, int64_t s_b, float eps, int64_t ne0_padded,
        const store_t & store, cudaStream_t stream) {
    const size_t smem = (32 + ncols)*sizeof(float);
    const dim3 num_blocks(nrows, 1, 1);
    const dim3 block_size(ACT_NORM_BLOCK, 1, 1);
    if (b) {
        act_norm_quant_kernel<true><<<num_blocks, block_size, smem, stream>>>(a, b, dst_add, dst_mul, w, (int) ncols, s_a, s_b, eps, ne0_padded, store);
    } else {
        act_norm_quant_kernel<false><<<num_blocks, block_size, smem, stream>>>(a, b, dst_add, dst_mul, w, (int) ncols, s_a, s_b, eps, ne0_padded, store);
    }
}

bool ggml_cuda_act_norm_quant(const float * a, const float * add_b, float * dst_add, float * dst_mul,
        const float * w, int64_t ncols, int64_t nrows, int64_t s_a, int64_t s_b,
        float eps, int64_t ne0_padded, int layout, void * y, cudaStream_t stream) {
    // rms_norm_f32 uses a 1024-thread block only for ncols >= 1024; below that
    // its reduction order differs, so the fusion declines.
    if (ncols < 1024 || ncols > ACT_NORM_MAX_NCOLS || ne0_padded % (4*WARP_SIZE) != 0) {
        return false;
    }
    switch (layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            act_norm_quant_launch(a, add_b, dst_add, dst_mul, w, ncols, nrows, s_a, s_b, eps, ne0_padded, mmq_q8_1_store<MMQ_Q8_1_DS_LAYOUT_D4>{y, nrows}, stream);
            return true;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            act_norm_quant_launch(a, add_b, dst_add, dst_mul, w, ncols, nrows, s_a, s_b, eps, ne0_padded, mmq_q8_1_store<MMQ_Q8_1_DS_LAYOUT_DS4>{y, nrows}, stream);
            return true;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            act_norm_quant_launch(a, add_b, dst_add, dst_mul, w, ncols, nrows, s_a, s_b, eps, ne0_padded, mmq_q8_1_store<MMQ_Q8_1_DS_LAYOUT_D2S6>{y, nrows}, stream);
            return true;
        case MMQ_Q8_1_DS_LAYOUT_D128:
            act_norm_quant_launch(a, add_b, dst_add, dst_mul, w, ncols, nrows, s_a, s_b, eps, ne0_padded, mmq_q8_1_store<MMQ_Q8_1_DS_LAYOUT_D128>{y, nrows}, stream);
            return true;
        case GGML_CUDA_ACT_LAYOUT_N1:
            if (ncols % 128 != 0) {
                return false;
            }
            act_norm_quant_launch(a, add_b, dst_add, dst_mul, w, ncols, nrows, s_a, s_b, eps, ne0_padded, n1_act_store_make(y, ncols, nrows), stream);
            return true;
        case GGML_CUDA_ACT_LAYOUT_N1G256:
            if (ncols % 256 != 0) {
                return false;
            }
            act_norm_quant_launch(a, add_b, dst_add, dst_mul, w, ncols, nrows, s_a, s_b, eps, ne0_padded, n1_act_store256_make(y, ncols, nrows), stream);
            return true;
        case GGML_CUDA_ACT_LAYOUT_SC:
            if (ncols % 128 != 0) {
                return false;
            }
            act_norm_quant_launch(a, add_b, dst_add, dst_mul, w, ncols, nrows, s_a, s_b, eps, ne0_padded, sc_act_store_make(y, ncols, nrows), stream);
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------

static __global__ void act_count_diff_kernel(const uint8_t * a, const uint8_t * b, size_t n, unsigned long long * cnt) {
    unsigned long long local = 0;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x*blockDim.x) {
        local += a[i] != b[i];
    }
    if (local) {
        atomicAdd(cnt, local);
    }
}

int64_t ggml_cuda_act_count_diff(const void * a, const void * b, size_t nbytes, cudaStream_t stream) {
    unsigned long long * d_cnt = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(*d_cnt)));
    CUDA_CHECK(cudaMemsetAsync(d_cnt, 0, sizeof(*d_cnt), stream));
    act_count_diff_kernel<<<256, 256, 0, stream>>>((const uint8_t *) a, (const uint8_t *) b, nbytes, d_cnt);
    unsigned long long h = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h, d_cnt, sizeof(h), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_cnt));
    return (int64_t) h;
}
