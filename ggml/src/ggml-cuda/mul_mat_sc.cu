// T399 single-copy layout, phase 2: MUL_MAT on GGML_TYPE_NK_Q2_0_W2ONLY (ENC_W2 tile stream + fp16 sw[K/128][F],
// 34 B per 128 weights = Q2_0's bytes, the model's ONLY copy of the weight).
//
// Kernels = native-kernels single_copy/sc_kernels.cuh (phase 1, T399-SINGLE-COPY-RESULT.md), integrated:
//   k_sc_gemv<NC,NW,HALF>  N = 1..7: matrix table in kernel args (up to 4 matrices sharing one activation = one
//                          horizontal-fusion group like T395), item = (matrix, 128-row tile[, row half]), dp4a on the
//                          W2 words against natural-order int8 activations, in-block reduction, deterministic.
//   k_sc_wmma<NT>          N = 8..48: v_wmma_i32_16x16x16_iu8 straight from the W2 registers; same matrix table.
//   k_sc_quant             the activation quantizer (ggml quantize_q8_1 arithmetic) into the SC layout (sc_act.cuh).
//   N >= GGML_SC_N1_MIN_N (default 49): the N1 int8 WMMA GEMM reading the tensor in place (mul_mat_n1.cu).
// Math at N <= 48 = Q2_0 mmvq's exactly: per 32-chunk d2 * (d8 * sum(code*q) - s8), integer chunk sums exact.
//
// The activation goes through the act-fuse cache (GGML_CUDA_ACT_LAYOUT_SC): a fused producer (RMS_NORM->MUL, ADD->
// RMS_NORM->MUL, SWIGLU) writes it directly and the separate quantize launch disappears; siblings share it.
//
// Env: GGML_SC_N1_MIN_N=<n>  smallest batch routed to N1 (default 49, range 9..65; below it the WMMA kernel).
//      GGML_SC_STATS=1       per-route call counts at exit.

#include "mul_mat_sc.cuh"
#include "mul_mat_n1.cuh"
#include "act-fuse.cuh"
#include "sc_act.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(GGML_USE_HIP)

namespace {

struct sc_env {
    int64_t n1_min = 49;
    bool    stats  = false;
    bool    n1_act_tok = false;   // GGML_N1_ACT=token (read once: this is on the per-node host path)
    sc_env() {
        const char * t = getenv("GGML_N1_ACT");
        n1_act_tok = t && strcmp(t, "token") == 0;
        const char * m = getenv("GGML_SC_N1_MIN_N");
        if (m) {
            n1_min = atoll(m);
            if (n1_min < SC_GEMV_MAX_N + 2) n1_min = SC_GEMV_MAX_N + 2;
            if (n1_min > 65) n1_min = 65;   // 50..65: batches 49..64 on the NT=64 WMMA kernel (measured option)
        }
        const char * s = getenv("GGML_SC_STATS");
        stats = s && strcmp(s, "0") != 0;
    }
};
const sc_env & env() { static sc_env e; return e; }

struct sc_stats {
    std::atomic<uint64_t> gemv{0}, wmma{0}, n1{0}, mats{0}, quant{0}, hit{0};
    ~sc_stats() {
        if (!env().stats) return;
        fprintf(stderr, "[SC] stats: launches gemv %llu wmma %llu n1 %llu (matrices %llu); activation quantized %llu, from act-fuse cache %llu\n",
                (unsigned long long) gemv.load(), (unsigned long long) wmma.load(), (unsigned long long) n1.load(),
                (unsigned long long) mats.load(), (unsigned long long) quant.load(), (unsigned long long) hit.load());
    }
};
sc_stats g_stats;

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
using i32x2 = __attribute__((__vector_size__(8))) int;
using i32x8 = __attribute__((__vector_size__(32))) int;

__device__ __forceinline__ int sc_dp4a(int a, int b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }
__device__ __forceinline__ uint4 sc_ldnt(const uint4 * p) {     // streaming weight load (read once per token)
    const u32x4 v = __builtin_nontemporal_load((const u32x4 *) p);
    return make_uint4(v.x, v.y, v.z, v.w);
}
__device__ __forceinline__ uint32_t sc_comp(const uint4 & v, int d) { return d == 0 ? v.x : d == 1 ? v.y : d == 2 ? v.z : v.w; }

struct sc_mat  { const uint4 * W; const __half * sw; float * y; int F; int pad; };
struct sc_args { sc_mat m[SC_MAX_GROUP]; int item_end[SC_MAX_GROUP]; int nm, K, N, NP; };

__device__ __forceinline__ int sc_pick(const sc_args & a) {
    int mi = 0;
#pragma unroll
    for (int i = 0; i < SC_MAX_GROUP - 1; ++i) mi += (i < a.nm - 1 && (int) blockIdx.x >= a.item_end[i]);
    return mi;
}

// ---- activation quantiser: ggml quantize_q8_1 arithmetic (32-lane xor butterfly 16, 8, 4, 2, 1) into the SC layout.
// grid (K/256, NP), block 256; src row n at x + n*s1 (floats); rows n >= N are zero.
__global__ void __launch_bounds__(256) k_sc_quant(const float * __restrict__ x, int64_t s1, int K, int N, int NP,
        int8_t * __restrict__ xq, __half2 * __restrict__ ds) {
    const int n = blockIdx.y, k = blockIdx.x * 256 + threadIdx.x;
    const float xi = (n < N && k < K) ? x[(int64_t) n * s1 + k] : 0.0f;
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, o, 32));
        sum  = sc_pin(sum + __shfl_xor(sum, o, 32));
    }
    const float d  = sc_scale(amax);
    const float id = amax == 0.0f ? 0.0f : sc_inv(d);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) sc_q(xi, id);
    if (k < K) {
        xq[(size_t) n * K + k] = q;
        if ((threadIdx.x & 31) == 0) ds[(size_t) (k >> 5) * NP + n] = __halves2half2(__float2half(d), __float2half(sum));
    }
}

// ---- decode GEMV (N = NC = 1..7). Item = (matrix, tile[, row half]); NW waves stride the K stages of the item;
// lane L = 16h + l16 of a wave reads uint4 kb*32 + L of each row half of a stage: word d of it holds the 16 codes of
// row (2d + r)*16 + l16 at k = kb*32 + 16h + [0,16), dword k4 = (word >> 2*k4) & 0x03030303 = elements 4k4..4k4+3.
template <int NC, int NW, bool HALF>
__global__ void __launch_bounds__(NW * 32) k_sc_gemv(const sc_args a, const int8_t * __restrict__ xq, const __half2 * __restrict__ ds) {
    constexpr int RH = HALF ? 1 : 2;           // row halves per block
    const int mi = sc_pick(a);
    const sc_mat m = a.m[mi];
    const int item = blockIdx.x - (mi ? a.item_end[mi - 1] : 0);
    const int tile = HALF ? item >> 1 : item, rh0 = HALF ? item & 1 : 0;
    const int K = a.K, S = K >> 7, NP = a.NP;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, h = lane >> 4, l16 = lane & 15;
    const uint4 * Wt = m.W + (size_t) tile * S * 256 + rh0 * 128 + lane;
    const __half * swt = m.sw + tile * 128 + rh0 * 16 + l16;
    float acc[RH][4][NC];
#pragma unroll
    for (int r = 0; r < RH; ++r)
#pragma unroll
        for (int d = 0; d < 4; ++d)
#pragma unroll
            for (int n = 0; n < NC; ++n) acc[r][d][n] = 0.f;
    uint4 nwv[RH][4]; __half nsw[RH][4];
    auto fetch = [&](int s) {
#pragma unroll
        for (int r = 0; r < RH; ++r)
#pragma unroll
            for (int kb = 0; kb < 4; ++kb) nwv[r][kb] = sc_ldnt(Wt + (size_t) s * 256 + r * 128 + kb * 32);
#pragma unroll
        for (int r = 0; r < RH; ++r)
#pragma unroll
            for (int d = 0; d < 4; ++d) nsw[r][d] = swt[(size_t) s * m.F + r * 16 + d * 32];
    };
    if (w < S) fetch(w);
    for (int s = w; s < S; s += NW) {
        uint4 wv[RH][4]; float swv[RH][4];
#pragma unroll
        for (int r = 0; r < RH; ++r)
#pragma unroll
            for (int kb = 0; kb < 4; ++kb) { wv[r][kb] = nwv[r][kb]; swv[r][kb] = __half2float(nsw[r][kb]); }
        if (s + NW < S) fetch(s + NW);
#pragma unroll
        for (int n = 0; n < NC; ++n) {
            const int8_t * xs = xq + (size_t) n * K + s * 128 + h * 16;
            int4 xv[4];
            float d8[4], S8 = 0.f;
#pragma unroll
            for (int kb = 0; kb < 4; ++kb) {
                xv[kb] = *(const int4 *) (xs + kb * 32);
                const float2 f = __half22float2(ds[(size_t) (s * 4 + kb) * NP + n]);
                d8[kb] = f.x; S8 += f.y;
            }
#pragma unroll
            for (int r = 0; r < RH; ++r)
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                float t = 0.f;
#pragma unroll
                for (int kb = 0; kb < 4; ++kb) {
                    const uint32_t wd = sc_comp(wv[r][kb], d);
                    int sum = 0;
                    sum = sc_dp4a((int) (wd & 0x03030303u), xv[kb].x, sum);
                    sum = sc_dp4a((int) ((wd >> 2) & 0x03030303u), xv[kb].y, sum);
                    sum = sc_dp4a((int) ((wd >> 4) & 0x03030303u), xv[kb].z, sum);
                    sum = sc_dp4a((int) ((wd >> 6) & 0x03030303u), xv[kb].w, sum);
                    t += d8[kb] * (float) sum;
                }
                acc[r][d][n] += swv[r][d] * (h ? t : t - S8);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RH; ++r)
#pragma unroll
        for (int d = 0; d < 4; ++d)
#pragma unroll
            for (int n = 0; n < NC; ++n) acc[r][d][n] += __shfl_xor(acc[r][d][n], 16, 32);
    __shared__ float red[NW][NC][RH * 64];
    if (h == 0) {
#pragma unroll
        for (int r = 0; r < RH; ++r)
#pragma unroll
            for (int d = 0; d < 4; ++d)
#pragma unroll
                for (int n = 0; n < NC; ++n) red[w][n][(r * 4 + d) * 16 + l16] = acc[r][d][n];
    }
    __syncthreads();
    for (int o = threadIdx.x; o < RH * 64 * NC; o += NW * 32) {
        const int n = o / (RH * 64), q = o % (RH * 64), r = q >> 6, d = (q >> 4) & 3, l = q & 15;
        float v = 0.f;
#pragma unroll
        for (int ww = 0; ww < NW; ++ww) v += red[ww][n][q];
        m.y[(size_t) n * m.F + tile * 128 + (2 * d + rh0 + r) * 16 + l] = v;
    }
}

// ---- small batch (N = 9..48 in NT = 16/32/48 token blocks): v_wmma_i32_16x16x16_iu8 from the W2 registers.
// Weights = B (lane%16 = row), activations = A (lane%16 = token). Lane (h,l16)'s W2 word holds k = 16h + [0,16) of
// one 32-chunk; elements 0..7 feed WMMA #1 and 8..15 WMMA #2 against the same 16 natural-order activation bytes.
// Block = 8 waves: r2 = w&1 = row half, ph = w>>1 strides stages by 4. Item = (matrix, tile).
template <int NT>
__global__ void __launch_bounds__(256) k_sc_wmma(const sc_args a, const int8_t * __restrict__ xq, const __half2 * __restrict__ ds) {
    constexpr int TB = NT / 16;
    const int mi = sc_pick(a);
    const sc_mat m = a.m[mi];
    const int tile = blockIdx.x - (mi ? a.item_end[mi - 1] : 0);
    const int K = a.K, S = K >> 7, NP = a.NP;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, h = lane >> 4, l16 = lane & 15;
    const int r2 = w & 1, ph = w >> 1;
    const uint4 * Wt = m.W + (size_t) tile * S * 256 + r2 * 128 + lane;
    const __half * swt = m.sw + tile * 128 + r2 * 16 + l16;
    float acc[4][TB][8];
#pragma unroll
    for (int d = 0; d < 4; ++d)
#pragma unroll
        for (int tb = 0; tb < TB; ++tb)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[d][tb][e] = 0.f;

    uint4 nwv[4]; __half nsw[4];
    auto fetch = [&](int s) {
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) nwv[kb] = sc_ldnt(Wt + (size_t) s * 256 + kb * 32);
#pragma unroll
        for (int d = 0; d < 4; ++d) nsw[d] = swt[(size_t) s * m.F + d * 32];
    };
    if (ph < S) fetch(ph);
    for (int s = ph; s < S; s += 4) {
        uint4 wv[4];
        float swv[4];
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) wv[kb] = nwv[kb];
#pragma unroll
        for (int d = 0; d < 4; ++d) swv[d] = __half2float(nsw[d]);
        if (s + 4 < S) fetch(s + 4);
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
#pragma unroll
            for (int tb = 0; tb < TB; ++tb) {
                const int4 xa = *(const int4 *) (xq + (size_t) (tb * 16 + l16) * K + s * 128 + kb * 32 + h * 16);
                const uint4 * dsp = (const uint4 *) (ds + (size_t) (s * 4 + kb) * NP + tb * 16 + 8 * h);
                const uint4 dA = dsp[0], dB = dsp[1];
                const uint32_t dw[8] = {dA.x, dA.y, dA.z, dA.w, dB.x, dB.y, dB.z, dB.w};
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    const uint32_t wd = sc_comp(wv[kb], d);
                    const i32x2 blo = {(int) (wd & 0x03030303u), (int) ((wd >> 2) & 0x03030303u)};
                    const i32x2 bhi = {(int) ((wd >> 4) & 0x03030303u), (int) ((wd >> 6) & 0x03030303u)};
                    i32x8 c = {0, 0, 0, 0, 0, 0, 0, 0};
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, i32x2{xa.x, xa.y}, true, blo, c, false);
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, i32x2{xa.z, xa.w}, true, bhi, c, false);
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const float2 f = __half22float2(*(const __half2 *) &dw[e]);
                        acc[d][tb][e] += swv[d] * (f.x * (float) c[e] - f.y);
                    }
                }
                asm volatile("" ::: "memory");   // keep the (kb,tb) loads from being hoisted (phase 1: fewer spills)
            }
            asm volatile("" ::: "memory");
        }
    }
    // reduce the 4 stage phases through LDS, deterministic order ((p3 + p2) + p1) + p0
    __shared__ float red[NT][128];
    for (int p = 3; p >= 0; --p) {
        if (ph == p) {
#pragma unroll
            for (int d = 0; d < 4; ++d)
#pragma unroll
                for (int tb = 0; tb < TB; ++tb)
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        float & r = red[tb * 16 + 8 * h + e][(2 * d + r2) * 16 + l16];
                        r = (p == 3) ? acc[d][tb][e] : r + acc[d][tb][e];
                    }
        }
        __syncthreads();
    }
    for (int o = threadIdx.x; o < 128 * a.N; o += 256) {
        m.y[(size_t) (o >> 7) * m.F + tile * 128 + (o & 127)] = red[o >> 7][o & 127];
    }
}

// ---- host side
bool shape_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    const bool collapse = src1->ne[2] * src1->ne[3] == 1 ||
        (src1->nb[2] == src1->nb[1] * src1->ne[1] && src1->nb[3] == src1->nb[2] * src1->ne[2] &&
         dst->nb[2] == dst->nb[1] * dst->ne[1] && dst->nb[3] == dst->nb[2] * dst->ne[2]);
    return src0->type == GGML_TYPE_NK_Q2_0_W2ONLY && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && ggml_is_contiguous(src0) &&
        src0->ne[0] % 128 == 0 && src0->ne[1] % 128 == 0 && src0->ne[0] == src1->ne[0] &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 && ((uintptr_t) src1->data % 16) == 0 &&
        ggml_is_contiguous(dst) && collapse && src1->ne[1] * src1->ne[2] * src1->ne[3] <= (1 << 20);
}

int64_t ncols(const ggml_tensor * src1) { return src1->ne[1] * src1->ne[2] * src1->ne[3]; }

template <int NC>
void launch_gemv_nc(const sc_args & a, int nitems, bool half, const int8_t * xq, const __half2 * ds, cudaStream_t st) {
    if (half) k_sc_gemv<NC, 4, true ><<<nitems, 128, 0, st>>>(a, xq, ds);
    else      k_sc_gemv<NC, 4, false><<<nitems, 128, 0, st>>>(a, xq, ds);
}

void launch_gemv(const sc_args & a, int nitems, bool half, const int8_t * xq, const __half2 * ds, cudaStream_t st) {
    switch (a.N) {
        case 1: launch_gemv_nc<1>(a, nitems, half, xq, ds, st); break;
        case 2: launch_gemv_nc<2>(a, nitems, half, xq, ds, st); break;
        case 3: launch_gemv_nc<3>(a, nitems, half, xq, ds, st); break;
        case 4: launch_gemv_nc<4>(a, nitems, half, xq, ds, st); break;
        case 5: launch_gemv_nc<5>(a, nitems, true, xq, ds, st); break;   // whole tile spills at NC >= 5 (phase 1)
        case 6: launch_gemv_nc<6>(a, nitems, true, xq, ds, st); break;
        case 7: launch_gemv_nc<7>(a, nitems, true, xq, ds, st); break;
        default: GGML_ABORT("SC: bad GEMV batch %d", a.N);
    }
}

void launch_wmma(const sc_args & a, int nitems, const int8_t * xq, const __half2 * ds, cudaStream_t st) {
    switch (a.NP) {
        case 16: k_sc_wmma<16><<<nitems, 256, 0, st>>>(a, xq, ds); break;
        case 32: k_sc_wmma<32><<<nitems, 256, 0, st>>>(a, xq, ds); break;
        case 48: k_sc_wmma<48><<<nitems, 256, 0, st>>>(a, xq, ds); break;
        case 64: k_sc_wmma<64><<<nitems, 256, 0, st>>>(a, xq, ds); break;   // only with GGML_SC_N1_MIN_N > 49
        default: GGML_ABORT("SC: bad WMMA block %d", a.NP);
    }
}

} // namespace

bool ggml_cuda_sc_supports_op(const ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_NK_Q2_0_W2ONLY) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_SRC; ++i) {
        if (op->src[i] && op->src[i]->type == GGML_TYPE_NK_Q2_0_W2ONLY) return false;
    }
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || !GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[dev].cc)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0], * src1 = op->src[1];
    // strides / alignment of the activation are checked at run time (shape_ok); these are the static parts
    return src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src0->ne[0] % 128 == 0 && src0->ne[1] % 128 == 0 && src1->nb[0] == sizeof(float) &&
        (src1->ne[2] * src1->ne[3] == 1 || (ggml_is_contiguous_rows(src1) &&
            src1->nb[2] == src1->nb[1] * src1->ne[1] && src1->nb[3] == src1->nb[2] * src1->ne[2]));
}

int64_t ggml_cuda_sc_group_max_n() { return env().n1_min - 1; }

size_t ggml_cuda_sc_act_bytes(const ggml_tensor * src1) {
    return sc_act_bytes(src1->ne[0], ncols(src1));
}

int ggml_cuda_sc_act_layout(const ggml_tensor * mm) {
    const ggml_tensor * src0 = mm->src[0], * src1 = mm->src[1];
    if (src0->type != GGML_TYPE_NK_Q2_0_W2ONLY || !shape_ok(src0, src1, mm)) {
        return -1;
    }
    if (ncols(src1) < env().n1_min) {
        return GGML_CUDA_ACT_LAYOUT_SC;
    }
    // N1 reads the weight in place, so it is always "converted"; its token arm cannot read a fused activation
    return env().n1_act_tok ? -1 : ggml_cuda_n1_w2only_act_layout(src0);   // T422: N1G256 under GGML_N1_M2=1
}

void ggml_cuda_sc_act_ref_quant(const float * x, int64_t s11, int64_t K, int64_t N, void * y, cudaStream_t stream) {
    const int64_t NP = sc_np(N);
    k_sc_quant<<<dim3((unsigned) (K / 256 + (K % 256 != 0)), (unsigned) NP), 256, 0, stream>>>(
        x, s11, (int) K, (int) N, (int) NP, (int8_t *) y, (__half2 *) ((char *) y + (size_t) NP * K));
}

void ggml_cuda_sc_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, int n) {
    GGML_ASSERT(n >= 1 && n <= SC_MAX_GROUP);
    const ggml_tensor * src1 = dsts[0]->src[1];
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * src0 = dsts[i]->src[0];
        if (!shape_ok(src0, dsts[i]->src[1], dsts[i]) || dsts[i]->src[1] != src1 || src0->ne[0] != dsts[0]->src[0]->ne[0]) {
            GGML_ABORT("SC: unsupported MUL_MAT %s (type %s, src1 [%lld,%lld,%lld] nb1 %zu): single-copy weights have no other path",
                       dsts[i]->name, ggml_type_name(src0->type), (long long) dsts[i]->src[1]->ne[0],
                       (long long) dsts[i]->src[1]->ne[1], (long long) dsts[i]->src[1]->ne[2], dsts[i]->src[1]->nb[1]);
        }
    }
    const int64_t N = ncols(src1);
    if (N >= env().n1_min) {
        for (int i = 0; i < n; ++i) {
            const bool ok = ggml_cuda_n1_mul_mat_w2only(ctx, dsts[i]->src[0], src1, dsts[i]);
            GGML_ASSERT(ok && "SC: N1 refused a single-copy weight");
            g_stats.n1++; g_stats.mats++;
        }
        return;
    }
    const int K = (int) src1->ne[0];
    cudaStream_t st = ctx.stream();

    // [TAG_ACT_FUSE] activation through the act-fuse cache (SC layout), as N1 / MMQ do
    const size_t nbytes   = sc_act_bytes(K, N);
    const int    NP       = (int) sc_np(N);
    const int    mask     = ggml_cuda_act_fuse_mask();
    const bool   act_hit  = mask && ctx.act_cache_tensor == src1 && ctx.act_cache_buf &&
        ctx.act_cache_layout == GGML_CUDA_ACT_LAYOUT_SC && ctx.act_cache_bytes == nbytes;
    const bool   act_cache = act_hit || (mask & GGML_ACT_FUSE_DEDUP);
    if (ctx.act_pending == src1) {
        GGML_ASSERT(act_hit && "act-fuse: pending GLU consumer missed the SC cache");
        ctx.act_pending = nullptr;
    }
    if (act_cache) {
        act_hit ? ctx.act_stat_hit++ : ctx.act_stat_miss++;
    }
    ggml_cuda_pool_alloc<char> local(ctx.pool(), act_cache ? 0 : nbytes);
    if (act_cache && !act_hit) {
        ctx.act_cache_buf.reset();
        ctx.act_cache_buf    = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), nbytes);
        ctx.act_cache_tensor = src1;
        ctx.act_cache_layout = GGML_CUDA_ACT_LAYOUT_SC;
        ctx.act_cache_bytes  = nbytes;
    }
    char * buf = act_cache ? ctx.act_cache_buf->get() : local.get();
    int8_t  * xq = (int8_t *) buf;
    __half2 * ds = (__half2 *) (buf + (size_t) NP * K);
    if (!act_hit) {
        k_sc_quant<<<dim3((unsigned) (K / 256 + (K % 256 != 0)), (unsigned) NP), 256, 0, st>>>(
            (const float *) src1->data, (int64_t) (src1->nb[1] / sizeof(float)), K, (int) N, NP, xq, ds);
        g_stats.quant++;
    } else {
        g_stats.hit++;
    }

    sc_args a{};
    a.nm = n; a.K = K; a.N = (int) N; a.NP = NP;
    const bool gemv = N <= SC_GEMV_MAX_N;
    // whole-tile GEMV items for the ffn gate+up pair at N <= 4 (phase 1 per-kind config); row halves elsewhere
    const bool half = !(gemv && N <= 4 && n == 2 && dsts[0]->src[0]->ne[1] >= 16384 && dsts[1]->src[0]->ne[1] >= 16384);
    int items = 0;
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * src0 = dsts[i]->src[0];
        const int F = (int) src0->ne[1];
        a.m[i].W  = (const uint4 *) src0->data;
        a.m[i].sw = (const __half *) ((const char *) src0->data + (size_t) F * K / 4);
        a.m[i].y  = (float *) dsts[i]->data;
        a.m[i].F  = F;
        items += (gemv && half) ? 2 * (F / 128) : F / 128;
        a.item_end[i] = items;
    }
    for (int i = n; i < SC_MAX_GROUP; ++i) a.item_end[i] = items;
    if (gemv) {
        launch_gemv(a, items, half, xq, ds, st);
        g_stats.gemv++;
    } else {
        launch_wmma(a, items, xq, ds, st);
        g_stats.wmma++;
    }
    g_stats.mats += n;
    CUDA_CHECK(cudaGetLastError());
}

#else

bool ggml_cuda_sc_supports_op(const ggml_tensor *) { return false; }
int64_t ggml_cuda_sc_group_max_n() { return 0; }
void ggml_cuda_sc_mul_mat(ggml_backend_cuda_context &, ggml_tensor * const *, int) { GGML_ABORT("SC: HIP only"); }
int ggml_cuda_sc_act_layout(const ggml_tensor *) { return -1; }
size_t ggml_cuda_sc_act_bytes(const ggml_tensor *) { return 0; }
void ggml_cuda_sc_act_ref_quant(const float *, int64_t, int64_t, int64_t, void *, cudaStream_t) {}

#endif
