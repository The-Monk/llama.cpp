#include "common.cuh"
#include "fwht.cuh"
#include "n1_act.cuh"
#include "unary.cuh"
#include "fwht-dev.cuh"

#include <algorithm>
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
// perm (T439 GGML_CUDA_B2_PERM_FUSE): src rows are the TILED [hd, nk, rep] layout of the GDN output and the transform input is
// the grouped [hd, rep, nk] reorder (build_lora_mm's reshape/permute/cont of ssm_out, folded into the load). 32 | hd, so each
// warp-wide load stays one contiguous run; the values are copied unchanged, so the result equals the cont + plain producer.
template <bool write_fp32, bool glu, bool perm = false>
__launch_bounds__(1024, 1)
__global__ void fwht_quant_n1t(const float * __restrict__ src, const float * __restrict__ up, const int64_t s_src, const int64_t s_up,
                               float * __restrict__ dst, const float scale,
                               const float * __restrict__ signs, int8_t * __restrict__ X, float * __restrict__ sx,
                               const int S, const int perm_hd = 0, const int perm_nk = 0, const int perm_rep = 0) {
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
        int sidx = idx;
        if (perm) {
            const int ob = w * N + i * warp_size;   // warp-uniform start of 32 consecutive grouped elements
            const int a0 = ob % perm_hd, t = ob / perm_hd;
            sidx = (a0 + lane + perm_hd * (t / perm_rep + perm_nk * (t % perm_rep))) - w * N;
        }
        float v = glu ? ggml_cuda_op_silu_single(s[idx]) * u[idx] : s[sidx];
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
        if (perm) {
            __syncthreads();   // dst may alias src: every warp gathers from the whole row before any warp overwrites its chunk
        }
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

// T439 GGML_CUDA_B2_GLUPF: fwht_quant_n1t<write_fp32, glu> as a persistent kernel. One block per token leaves the memory pipe idle while
// the block transforms and quantizes (the producer reads 2 x K floats per token, mostly from DRAM); here a block walks the tokens
// tok, tok + gridDim.x, ... and issues the next token's gate/up loads before it rotates the current one. Same arithmetic as
// fwht_quant_n1t, so the output bytes are identical.
template <bool write_fp32>
__launch_bounds__(1024, 1)
__global__ void fwht_quant_n1t_glu_pf(const float * __restrict__ src, const float * __restrict__ up, const int64_t s_src, const int64_t s_up,
                                      float * __restrict__ dst, const float scale,
                                      const float * __restrict__ signs, int8_t * __restrict__ X, float * __restrict__ sx,
                                      const int S, const int ntok) {
    constexpr int N = 1024, warp_size = 32, el_w = N / warp_size;
    const int w = threadIdx.y, lane = threadIdx.x;
    const int K = blockDim.y * N;
    extern __shared__ __align__(16) char smem[];
    float  * red = (float *) smem;
    int8_t * qs  = (int8_t *) (smem + 128);
    const float * sg = signs + (size_t) w * N;

    float cs[el_w], cu[el_w];
    int tok = blockIdx.x;
    if (tok < ntok) {
        const float * s = src + (size_t) tok * s_src + (size_t) w * N;
        const float * u = up + (size_t) tok * s_up + (size_t) w * N;
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            cs[i] = s[i * warp_size + lane];
            cu[i] = u[i * warp_size + lane];
        }
    }
    for (; tok < ntok; tok += gridDim.x) {
        float reg[el_w];
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            float v = ggml_cuda_op_silu_single(cs[i]) * cu[i];
            v *= scale;
            v *= sg[i * warp_size + lane];
            reg[i] = v;
        }
        const int tn = tok + gridDim.x;
        if (tn < ntok) {
            const float * s = src + (size_t) tn * s_src + (size_t) w * N;
            const float * u = up + (size_t) tn * s_up + (size_t) w * N;
#pragma unroll
            for (int i = 0; i < el_w; ++i) {
                cs[i] = s[i * warp_size + lane];
                cu[i] = u[i * warp_size + lane];
            }
        }
        fwht_net<N, warp_size>(reg, lane);
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            asm volatile("" : "+v"(reg[i]));
        }
        if (write_fp32) {
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
        const float id = n1_recip(d);
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
}

// T439 GGML_CUDA_B2_ADDNORM_FUSE: the residual ADD + RMS_NORM + MUL(weight) in front of a Hadamard-folded matmul fused with the
// M3 producer: one launch per token instead of k_bin_bcast add, rms_norm_f32<1024,true> and fwht_quant_n1t. Stage 1 is
// rms_norm_f32<1024, true> verbatim (strided partial sums over the same columns in the same order, the same block_reduce, the
// same `scale * x * w` expression), the sum row is stored as the ADD node's value; the normalized row is staged in LDS (and
// stored as the MUL node's value when another node reads it); stage 2 is fwht_quant_n1t's tail on the staged values, one warp
// per 1024-block (warps beyond the row's blocks only take part in the barriers).
template <int NB, bool write_norm, bool write_fp32>
__launch_bounds__(256, 1)
__global__ void add_norm_fwht_quant_n1t(const float * __restrict__ xa, const float * __restrict__ xb, float * __restrict__ sum_dst,
                                        const int ntok, const float eps, const float * __restrict__ mulw,
                                        const float * __restrict__ signs, const float fscale,
                                        float * __restrict__ norm_dst, float * __restrict__ fwht_dst,
                                        int8_t * __restrict__ X, float * __restrict__ sx, const int S) {
    constexpr int N = 1024, warp_size = 32, el_w = N / warp_size, NT = 256, NJ = 1024 / NT, ncols = NB * N;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    extern __shared__ __align__(16) char smem[];
    float  * s_sum = (float *) smem;
    float  * red   = (float *) (smem + 128);
    int8_t * qs    = (int8_t *) (smem + 256);
    float  * nvs   = (float *) (smem + 256 + (size_t) NB * N);

    // Persistent over tokens (tok, tok + gridDim.x, ...): the next token's two input rows are loaded into registers before the
    // current token's reduction / transform / quantize, so the memory pipe stays busy while the block computes.
    float ca[NJ][NB], cb[NJ][NB];
    int tok = blockIdx.x;
    if (tok < ntok) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
#pragma unroll
            for (int k = 0; k < NB; ++k) {
                const size_t o = (size_t) tok * ncols + tid + NT * j + k * 1024;
                ca[j][k] = xa[o];
                cb[j][k] = xb[o];
            }
        }
    }
    for (; tok < ntok; tok += gridDim.x) {
        const size_t row = (size_t) tok * ncols;
        // This block has NT = 256 threads where rms_norm_f32<1024> has 1024: thread `tid` carries the partial sums of the
        // reference threads t = tid + NT * j (j < NJ), each over its own strided columns in ascending order, so every partial
        // sum and, below, the per-warp tree and the final 32-way tree are the reference's (reference warp w + 8 * j lives in
        // warp w).
        float p[NJ];
        float sv[NJ][NB];
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            p[j] = 0.0f;
#pragma unroll
            for (int k = 0; k < NB; ++k) {
                const float s = ca[j][k] + cb[j][k];
                sv[j][k] = s;
                sum_dst[row + tid + NT * j + k * 1024] = s;
                p[j] += s * s;
            }
        }
        const int tn = tok + gridDim.x;
        if (tn < ntok) {   // prefetch the next token's rows
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
#pragma unroll
                for (int k = 0; k < NB; ++k) {
                    const size_t o = (size_t) tn * ncols + tid + NT * j + k * 1024;
                    ca[j][k] = xa[o];
                    cb[j][k] = xb[o];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            p[j] = block_reduce_policy<block_reduce_method::SUM, float>::reduce(p[j]);
            if (lane == 0) {
                s_sum[w + (NT / warp_size) * j] = p[j];
            }
        }
        __syncthreads();
        const float tmp   = block_reduce_policy<block_reduce_method::SUM, float>::reduce(s_sum[lane]);
        const float mean  = tmp / ncols;
        const float scale = rsqrtf(mean + eps);
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
#pragma unroll
            for (int k = 0; k < NB; ++k) {
                const int col = tid + NT * j + k * 1024;
                const float nv = scale * sv[j][k] * mulw[col];
                nvs[col] = nv;
                if (write_norm) {
                    norm_dst[row + col] = nv;
                }
            }
        }
        __syncthreads();

        float reg[el_w];
        float amax = 0.f;
        if (w < NB) {
            const float * sg = signs + (size_t) w * N;
#pragma unroll
            for (int i = 0; i < el_w; ++i) {
                const int idx = i * warp_size + lane;
                float v = nvs[w * N + idx];
                v *= fscale;
                v *= sg[idx];
                reg[i] = v;
            }
            fwht_net<N, warp_size>(reg, lane);
#pragma unroll
            for (int i = 0; i < el_w; ++i) {
                asm volatile("" : "+v"(reg[i]));
            }
            if (write_fp32) {
                float * d = fwht_dst + row + (size_t) w * N;
#pragma unroll
                for (int i = 0; i < el_w; ++i) {
                    d[i * warp_size + lane] = reg[i];
                }
            }
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
        }
        __syncthreads();
        amax = 0.f;
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            amax = fmaxf(amax, red[i]);
        }
        const float d  = n1_scale(amax);
        const float id = n1_recip(d);
        if (tid == 0) {
            sx[tok] = d;
        }
        if (w < NB) {
#pragma unroll
            for (int i = 0; i < el_w; ++i) {
                qs[w * N + i * warp_size + lane] = (int8_t) (d > 0.f ? (uint8_t) n1_q1(reg[i], id) : 0);
            }
        }
        __syncthreads();
        if (w < NB) {
#pragma unroll
            for (int m = 0; m < N / 4 / warp_size; ++m) {
                const int jj = lane + warp_size * m;
                *(uint32_t *) (X + n1_off(tok, w * N + 4 * jj, S)) = *(const uint32_t *) (qs + w * N + 4 * jj);
            }
        }
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

bool ggml_cuda_op_fwht_signed_quant_n1t(ggml_backend_cuda_context & ctx, const ggml_tensor * src, const ggml_tensor * up,
                                        const ggml_tensor * signs, ggml_tensor * dst, bool write_fp32, void * y,
                                        int perm_hd, int perm_nk, int perm_rep) {
    const bool glu = up != nullptr;
    const bool perm = perm_rep > 0;
    if (perm && (glu || perm_hd % 32 != 0 || perm_hd <= 0 || perm_nk <= 0 || (int64_t) perm_hd * perm_nk * perm_rep != src->ne[0])) {
        return false;
    }
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
#define FQ(WF, GL) ggml_cuda_kernel_launch(fwht_quant_n1t<WF, GL>, lp, a, b, s_src, s_up, d, scale, g, X, sx, S, 0, 0, 0)
#define FQP(WF) ggml_cuda_kernel_launch(fwht_quant_n1t<WF, false, true>, lp, a, b, s_src, s_up, d, scale, g, X, sx, S, perm_hd, perm_nk, perm_rep)
    static const bool glu_pf = [] { const char * v = getenv("GGML_CUDA_B2_GLUPF"); return v && atoi(v) != 0; }();
    if (glu && glu_pf) {
        static const int pf_bps = [] { const char * v = getenv("GGML_CUDA_B2_GLUPF_BPS"); return v ? atoi(v) : 2; }();
        const int sms = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        const dim3 pgrid((unsigned) std::min<int64_t>(ntok, (int64_t) sms * pf_bps), 1, 1);
        const ggml_cuda_kernel_launch_params plp(pgrid, block, shmem, ctx.stream());
        if (write_fp32) {
            ggml_cuda_kernel_launch(fwht_quant_n1t_glu_pf<true>, plp, a, b, s_src, s_up, d, scale, g, X, sx, S, (int) ntok);
        } else {
            ggml_cuda_kernel_launch(fwht_quant_n1t_glu_pf<false>, plp, a, b, s_src, s_up, d, scale, g, X, sx, S, (int) ntok);
        }
    } else if (perm) {
        if (write_fp32) FQP(true); else FQP(false);
    } else if (glu) {
        if (write_fp32) FQ(true, true); else FQ(false, true);
    } else {
        if (write_fp32) FQ(true, false); else FQ(false, false);
    }
#undef FQ
#undef FQP
    return true;
}

bool ggml_cuda_op_add_norm_fwht_quant_n1t(ggml_backend_cuda_context & ctx, const ggml_tensor * xa, const ggml_tensor * xb,
                                          ggml_tensor * sum, const ggml_tensor * mulw, const ggml_tensor * signs,
                                          ggml_tensor * norm_dst, ggml_tensor * mm, float eps, bool write_fp32, void * y) {
    const int64_t ncols = xa->ne[0];
    const int64_t ntok  = ggml_nelements(xa) / ncols;
    if (ncols % 1024 != 0 || ncols > 8192 || ntok > (1 << 20) || ncols % 128 != 0 ||
        xa->type != GGML_TYPE_F32 || xb->type != GGML_TYPE_F32 || sum->type != GGML_TYPE_F32 || mulw->type != GGML_TYPE_F32 ||
        signs->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 || (norm_dst && norm_dst->type != GGML_TYPE_F32) ||
        !ggml_is_contiguous(xa) || !ggml_is_contiguous(xb) || !ggml_is_contiguous(sum) || !ggml_is_contiguous(mulw) ||
        !ggml_is_contiguous(signs) || !ggml_is_contiguous(mm) || (norm_dst && !ggml_is_contiguous(norm_dst)) ||
        xb->ne[0] != ncols || ggml_nelements(xb) != ggml_nelements(xa) || ggml_nelements(sum) != ggml_nelements(xa) ||
        ggml_nelements(mulw) != ncols || ggml_nelements(signs) != ncols || ggml_nelements(mm) != ggml_nelements(xa) ||
        (norm_dst && ggml_nelements(norm_dst) != ggml_nelements(xa)) || mm->ne[0] != 1024) {
        return false;
    }
    const int nblk = (int) (ncols / 1024);
    if (nblk != 4 && nblk != 5 && nblk != 6 && nblk != 8) {
        return false;
    }
    const int64_t Npad = n1_npad(ntok);
    int8_t * X  = (int8_t *) y;
    float *  sx = (float *) ((char *) y + (size_t) Npad * ncols);
    static const int blocks_per_sm = [] { const char * v = getenv("GGML_CUDA_B2_ADDNORM_BPS"); return v ? atoi(v) : 8; }();
    const int sms = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const dim3   grid((unsigned) std::min<int64_t>(ntok, (int64_t) sms * blocks_per_sm), 1, 1), block(256, 1, 1);
    const size_t shmem = 256 + (size_t) nblk * 1024 + (size_t) ncols * sizeof(float);
    const ggml_cuda_kernel_launch_params lp(grid, block, shmem, ctx.stream());
    const float fscale = 1 / sqrtf(1024.f);
    const int   S      = (int) (ncols / 128);
#define AQ(NB, WN, WF) ggml_cuda_kernel_launch(add_norm_fwht_quant_n1t<NB, WN, WF>, lp, (const float *) xa->data, (const float *) xb->data, \
        (float *) sum->data, (int) ntok, eps, (const float *) mulw->data, (const float *) signs->data, fscale, \
        norm_dst ? (float *) norm_dst->data : (float *) nullptr, (float *) mm->data, X, sx, S)
#define AQN(NB) \
    if (norm_dst) { if (write_fp32) AQ(NB, true, true); else AQ(NB, true, false); } \
    else          { if (write_fp32) AQ(NB, false, true); else AQ(NB, false, false); }
    switch (nblk) {
        case 4: AQN(4) break;
        case 5: AQN(5) break;
        case 6: AQN(6) break;
        default: AQN(8) break;
    }
#undef AQN
#undef AQ
    return true;
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
