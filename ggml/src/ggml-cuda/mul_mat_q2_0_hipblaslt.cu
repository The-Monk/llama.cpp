// See mul_mat_q2_0_hipblaslt.cuh for the full rationale.
// Stage 3 of ~/hipblaslt-prefill-scope.md: wire the tuned hipBLASLt int8 GEMM
// as an opt-in prefill route for Q2_0. Correctness-first; algo/handle caching
// and the fp8 front-end are Stage 4.

#include "mul_mat_q2_0_hipblaslt.cuh"

// Real implementation only on the AMD/HIP build (hipBLASLt is a ROCm lib).
// On a CUDA build these become stubs so the globbed source is a no-op.
#if defined(__HIP_PLATFORM_AMD__) && !defined(GGML_HIP_NO_HIPBLASLT)

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_fp16.h>
#include <map>
#include <tuple>
#include <mutex>

namespace {

constexpr int   Q2K   = QK2_0;          // 128 elements per Q2_0 block
constexpr float I8MAX = 127.0f;

#define LT_OK(x) do { hipblasStatus_t s_ = (x); if (s_ != HIPBLAS_STATUS_SUCCESS) { \
    GGML_LOG_ERROR("%s: hipBLASLt error %d at %s:%d\n", __func__, (int)s_, __FILE__, __LINE__); \
    return false; } } while(0)

// ---- device helpers --------------------------------------------------------

__device__ __forceinline__ float q2_half2float(ggml_half h) {
    return __half2float(*reinterpret_cast<const __half *>(&h));
}

// Requant a Q2_0 weight row -> int8 with ONE symmetric scale per output channel.
// One block per output row n. Weight row layout: n_blocks contiguous block_q2_0.
// Output int8 is row-major [N x K]  (== col-major [K x N], the TN A-operand).
__global__ void k_requant_q2_0_to_int8_perchannel(
        const char * __restrict__ wdata, int64_t nb01,
        int8_t * __restrict__ q8, float * __restrict__ wscale,
        int64_t K, int64_t n_blocks) {
    const int64_t n = blockIdx.x;                 // output channel
    const block_q2_0 * row = (const block_q2_0 *)(wdata + n * nb01);

    // pass 1: block-wide amax over the K weights of this row
    float amax = 0.0f;
    for (int64_t l = threadIdx.x; l < K; l += blockDim.x) {
        const int64_t b  = l / Q2K;
        const int     t  = (int)(l % Q2K);
        const float   d  = q2_half2float(row[b].d);
        const uint8_t q  = (row[b].qs[t >> 2] >> ((t & 3) * 2)) & 0x3;
        const float   w  = ((int)q - 1) * d;      // {-d,0,+d,+2d}
        amax = fmaxf(amax, fabsf(w));
    }
    __shared__ float sred[1024];
    sred[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sred[threadIdx.x] = fmaxf(sred[threadIdx.x], sred[threadIdx.x + s]);
        __syncthreads();
    }
    const float scale = (sred[0] > 0.0f) ? sred[0] / I8MAX : 1.0f;
    if (threadIdx.x == 0) wscale[n] = scale;
    const float inv = 1.0f / scale;

    // pass 2: quantize
    int8_t * out = q8 + n * K;
    for (int64_t l = threadIdx.x; l < K; l += blockDim.x) {
        const int64_t b = l / Q2K;
        const int     t = (int)(l % Q2K);
        const float   d = q2_half2float(row[b].d);
        const uint8_t q = (row[b].qs[t >> 2] >> ((t & 3) * 2)) & 0x3;
        const float   w = ((int)q - 1) * d;
        int v = __float2int_rn(w * inv);
        v = max(-127, min(127, v));
        out[l] = (int8_t)v;
    }
}

// Quantize activations -> int8 with ONE symmetric scale per token (column).
// src1 fp32, column j at (src1 + j*nb1), K contiguous floats. Output col-major
// [K x M] (ld=K) + per-token scale.
__global__ void k_quantize_act_int8_percol(
        const char * __restrict__ src1, int64_t nb1,
        int8_t * __restrict__ x8, float * __restrict__ ascale, int64_t K) {
    const int64_t j = blockIdx.x;                 // token / column
    const float * col = (const float *)(src1 + j * nb1);

    float amax = 0.0f;
    for (int64_t l = threadIdx.x; l < K; l += blockDim.x) amax = fmaxf(amax, fabsf(col[l]));
    __shared__ float sred[1024];
    sred[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sred[threadIdx.x] = fmaxf(sred[threadIdx.x], sred[threadIdx.x + s]);
        __syncthreads();
    }
    const float scale = (sred[0] > 0.0f) ? sred[0] / I8MAX : 1.0f;
    if (threadIdx.x == 0) ascale[j] = scale;
    const float inv = 1.0f / scale;

    int8_t * out = x8 + j * K;
    for (int64_t l = threadIdx.x; l < K; l += blockDim.x) {
        int v = __float2int_rn(col[l] * inv);
        v = max(-127, min(127, v));
        out[l] = (int8_t)v;
    }
}

// Dequant the int32 GEMM result: dst[n,j] = i32[n,j] * wscale[n] * ascale[j].
// Both dst and i32 are col-major [N x M] (dst ld from nb1, i32 ld=N).
__global__ void k_apply_scales(
        const int32_t * __restrict__ i32, char * __restrict__ dst, int64_t nb1,
        const float * __restrict__ wscale, const float * __restrict__ ascale,
        int64_t N, int64_t M) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= N * M) return;
    const int64_t n = idx % N;
    const int64_t j = idx / N;
    float * out = (float *)(dst + j * nb1);
    out[n] = (float)i32[j * N + n] * wscale[n] * ascale[j];
}

// ============================ fp8 (E4M3) variant ============================
// Same v1 scheme, but encode to OCP e4m3 and use hipBLASLt's fp8 GEMM (202
// tuned gfx1201 kernels vs 16 int8). e4m3 max = 448; scale to fill the range.
constexpr float F8MAX = 448.0f;

__global__ void k_requant_q2_0_to_e4m3_perchannel(
        const char * __restrict__ wdata, int64_t nb01,
        uint8_t * __restrict__ q8, float * __restrict__ wscale,
        int64_t K, int64_t n_blocks) {
    const int64_t n = blockIdx.x;
    const block_q2_0 * row = (const block_q2_0 *)(wdata + n * nb01);
    float amax = 0.0f;
    for (int64_t l = threadIdx.x; l < K; l += blockDim.x) {
        const int64_t b = l / Q2K; const int t = (int)(l % Q2K);
        const float d = q2_half2float(row[b].d);
        const uint8_t q = (row[b].qs[t >> 2] >> ((t & 3) * 2)) & 0x3;
        amax = fmaxf(amax, fabsf(((int)q - 1) * d));
    }
    __shared__ float sred[1024];
    sred[threadIdx.x] = amax; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sred[threadIdx.x] = fmaxf(sred[threadIdx.x], sred[threadIdx.x + s]);
        __syncthreads();
    }
    const float scale = (sred[0] > 0.0f) ? sred[0] / F8MAX : 1.0f;
    if (threadIdx.x == 0) wscale[n] = scale;
    const float inv = 1.0f / scale;
    uint8_t * out = q8 + n * K;
    for (int64_t l = threadIdx.x; l < K; l += blockDim.x) {
        const int64_t b = l / Q2K; const int t = (int)(l % Q2K);
        const float d = q2_half2float(row[b].d);
        const uint8_t q = (row[b].qs[t >> 2] >> ((t & 3) * 2)) & 0x3;
        out[l] = ggml_cuda_fp32_to_e4m3(((int)q - 1) * d * inv);
    }
}

__global__ void k_quantize_act_e4m3_percol(
        const char * __restrict__ src1, int64_t nb1,
        uint8_t * __restrict__ x8, float * __restrict__ ascale, int64_t K) {
    const int64_t j = blockIdx.x;
    const float * col = (const float *)(src1 + j * nb1);
    float amax = 0.0f;
    for (int64_t l = threadIdx.x; l < K; l += blockDim.x) amax = fmaxf(amax, fabsf(col[l]));
    __shared__ float sred[1024];
    sred[threadIdx.x] = amax; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sred[threadIdx.x] = fmaxf(sred[threadIdx.x], sred[threadIdx.x + s]);
        __syncthreads();
    }
    const float scale = (sred[0] > 0.0f) ? sred[0] / F8MAX : 1.0f;
    if (threadIdx.x == 0) ascale[j] = scale;
    const float inv = 1.0f / scale;
    uint8_t * out = x8 + j * K;
    for (int64_t l = threadIdx.x; l < K; l += blockDim.x) out[l] = ggml_cuda_fp32_to_e4m3(col[l] * inv);
}

// fp8 GEMM emits f32 accumulate directly -> apply scales from a float buffer.
__global__ void k_apply_scales_f32(
        const float * __restrict__ acc, char * __restrict__ dst, int64_t nb1,
        const float * __restrict__ wscale, const float * __restrict__ ascale,
        int64_t N, int64_t M) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= N * M) return;
    const int64_t n = idx % N, j = idx / N;
    float * out = (float *)(dst + j * nb1);
    out[n] = acc[j * N + n] * wscale[n] * ascale[j];
}

// One hipBLASLt handle per process (single-GPU prefill use). Thread-safe lazy init.
hipblasLtHandle_t get_lt_handle() {
    static hipblasLtHandle_t h = [](){ hipblasLtHandle_t t; hipblasLtCreate(&t); return t; }();
    return h;
}

// ---- per-shape plan cache -------------------------------------------------
// The dominant integration cost is the per-call heuristic search + descriptor
// churn (Stage 4: isolated GEMM = 190-338 TOPS but integrated = 55 because this
// ran EVERY matmul). A prefill touches only a handful of distinct (M,N,K)
// shapes (ubatch M x each weight's N,K), so build the descriptors + run the
// heuristic ONCE per shape and reuse the plan. hipblasLtMatmul takes fresh data
// pointers each call; the desc/layouts/algo are shape-only and safely shared.
constexpr size_t LT_WS_BYTES = 32ull << 20;   // fixed workspace budget for algo selection

struct lt_plan {
    hipblasLtMatmulDesc_t             op   = nullptr;
    hipblasLtMatrixLayout_t           lA   = nullptr, lB = nullptr, lD = nullptr;
    hipblasLtMatmulHeuristicResult_t  heur{};
    bool                              ok   = false;
};

enum gemm_mode { MODE_I8 = 0, MODE_F8 = 1 };   // int8 (i8->i32) or fp8 (e4m3->f32)

std::map<std::tuple<int64_t,int64_t,int64_t,int>, lt_plan> g_plan_cache;
std::mutex g_plan_mtx;

// Returns a cached (or freshly built) plan for D(NxM)=op(A=W)[NxK]*B(X)[KxM], TN.
// plan.ok == false means the heuristic found no algo for this shape/mode.
const lt_plan & get_plan(int64_t N, int64_t M, int64_t K, int mode) {
    std::lock_guard<std::mutex> lk(g_plan_mtx);
    auto key = std::make_tuple(N, M, K, mode);
    auto it = g_plan_cache.find(key);
    if (it != g_plan_cache.end()) {
        return it->second;
    }
    const hipblasComputeType_t compute = (mode == MODE_F8) ? HIPBLAS_COMPUTE_32F : HIPBLAS_COMPUTE_32I;
    const hipDataType scaleT = (mode == MODE_F8) ? HIP_R_32F : HIP_R_32I;
    const hipDataType abT    = (mode == MODE_F8) ? HIP_R_8F_E4M3 : HIP_R_8I;
    const hipDataType dT     = (mode == MODE_F8) ? HIP_R_32F : HIP_R_32I;
    lt_plan p;
    hipblasLtHandle_t h = get_lt_handle();
    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    bool built =
        hipblasLtMatmulDescCreate(&p.op, compute, scaleT) == HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)) == HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)) == HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatrixLayoutCreate(&p.lA, abT, K, N, K) == HIPBLAS_STATUS_SUCCESS &&   // stored KxN, op=T -> NxK
        hipblasLtMatrixLayoutCreate(&p.lB, abT, K, M, K) == HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatrixLayoutCreate(&p.lD, dT,  N, M, N) == HIPBLAS_STATUS_SUCCESS;
    if (built) {
        hipblasLtMatmulPreference_t pref = nullptr;
        size_t ws = LT_WS_BYTES;
        int nAlgo = 0;
        if (hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS &&
            hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)) == HIPBLAS_STATUS_SUCCESS &&
            hipblasLtMatmulAlgoGetHeuristic(h, p.op, p.lA, p.lB, p.lD, p.lD, pref, 1, &p.heur, &nAlgo) == HIPBLAS_STATUS_SUCCESS &&
            nAlgo > 0) {
            p.ok = true;
        }
        if (pref) hipblasLtMatmulPreferenceDestroy(pref);
    }
    auto res = g_plan_cache.emplace(key, p);
    return res.first->second;
}

// ---- bounded int8 weight cache ---------------------------------------------
// The residual prefill loss after the plan cache is the per-call requant pass
// (Q2_0 -> int8, a full memory pass that dp4a fuses into its kernel). Weights
// are constant, so cache the int8 copy + per-channel scale keyed by the weight
// pointer. A full-model int8 copy (~27 GB on 27B) won't fit beside the resident
// Q2_0 on a 32 GB card, so cap the cache at a VRAM budget: weights that fit are
// cached (requant paid once), the rest fall back to on-the-fly pool requant.
// hipMalloc failure also falls back -- never OOM-crash (the Stage-3 lesson).
struct cached_w { int8_t * q8 = nullptr; float * wscale = nullptr; size_t bytes = 0; };
std::map<const void *, cached_w> g_wcache;
size_t g_wcache_bytes = 0;
std::mutex g_wcache_mtx;

size_t wcache_budget_bytes() {
    static size_t b = [](){
        const char * e = getenv("GGML_HIP_Q2_0_HIPBLASLT_WCACHE_MB");
        size_t mb = e ? (size_t)atoll(e) : (size_t)12000;   // ~12 GB default: leaves headroom
                                                            // for model + llama.cpp compute bufs
        return mb << 20;
    }();
    return b;
}

// Returns cached int8 weight (building it on first miss if within budget), or
// nullptr -> caller must requant on-the-fly. Build + all uses share the stream,
// so the one-time requant is correctly ordered before any GEMM that reads it.
const cached_w * try_cache_weight(const void * key, const char * wdata, int64_t nb01,
                                  int64_t N, int64_t K, int64_t n_blocks, int mode, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_wcache_mtx);
    auto it = g_wcache.find(key);
    if (it != g_wcache.end()) return &it->second;   // mode is fixed per process (env-checked once)

    const size_t need = (size_t)N * K + (size_t)N * sizeof(float);
    if (g_wcache_bytes + need > wcache_budget_bytes()) return nullptr;   // budget hit

    cached_w c;
    if (hipMalloc(&c.q8, (size_t)N * K) != hipSuccess) return nullptr;
    if (hipMalloc(&c.wscale, (size_t)N * sizeof(float)) != hipSuccess) { hipFree(c.q8); return nullptr; }
    c.bytes = need;
    const dim3 grid((unsigned)N), block(256);
    if (mode == 1 /*MODE_F8*/) {
        k_requant_q2_0_to_e4m3_perchannel<<<grid, block, 0, stream>>>(wdata, nb01, (uint8_t *)c.q8, c.wscale, K, n_blocks);
    } else {
        k_requant_q2_0_to_int8_perchannel<<<grid, block, 0, stream>>>(wdata, nb01, c.q8, c.wscale, K, n_blocks);
    }
    g_wcache_bytes += need;
    auto res = g_wcache.emplace(key, c);
    return &res.first->second;
}

} // namespace

bool ggml_cuda_q2_0_hipblaslt_prefill_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_Q2_0)                       return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1)               return false;
    if (src1->ne[2] != 1 || src1->ne[3] != 1)               return false;
    if (src0->ne[0] != src1->ne[0] || src0->ne[0] % Q2K != 0) return false;

    // Prefill only: M must clear the threshold (decode stays on dp4a). Tunable.
    static const int64_t M_THRESH = [](){
        const char * e = getenv("GGML_HIP_Q2_0_HIPBLASLT_MTHRESH");
        return e ? (int64_t)atoll(e) : (int64_t)32;
    }();
    if (src1->ne[1] <= M_THRESH) return false;

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    return GGML_CUDA_CC_IS_RDNA4(cc);
}

bool ggml_cuda_op_mul_mat_q2_0_hipblaslt(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_Q2_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t n_blocks = K / Q2K;
    cudaStream_t  stream = ctx.stream();

    // int8 (default) or fp8/e4m3 GEMM -- env-checked once. fp8 rides the 202
    // tuned gfx1201 fp8 kernels (vs 16 int8); see the fp8 kernels above.
    static const int mode = (getenv("GGML_HIP_Q2_0_HIPBLASLT_FP8") != nullptr) ? MODE_F8 : MODE_I8;

    // ---- weight -> int8/e4m3 (per-output-channel): bounded cache, pool fallback ----
    int8_t * wq8_ptr = nullptr;
    float  * wsc_ptr = nullptr;
    ggml_cuda_pool_alloc<int8_t> wq8_pool;   // lazily allocated only on cache miss
    ggml_cuda_pool_alloc<float>  wsc_pool;
    const cached_w * cw = try_cache_weight(src0->data, (const char *)src0->data, src0->nb[1],
                                           N, K, n_blocks, mode, stream);
    if (cw) {
        wq8_ptr = cw->q8;
        wsc_ptr = cw->wscale;
    } else {
        wq8_ptr = wq8_pool.alloc(ctx.pool(), (size_t)N * K);
        wsc_ptr = wsc_pool.alloc(ctx.pool(), (size_t)N);
        const dim3 grid((unsigned)N), block(256);
        if (mode == MODE_F8) {
            k_requant_q2_0_to_e4m3_perchannel<<<grid, block, 0, stream>>>(
                (const char *)src0->data, src0->nb[1], (uint8_t *)wq8_ptr, wsc_ptr, K, n_blocks);
        } else {
            k_requant_q2_0_to_int8_perchannel<<<grid, block, 0, stream>>>(
                (const char *)src0->data, src0->nb[1], wq8_ptr, wsc_ptr, K, n_blocks);
        }
    }

    // ---- activation int8/e4m3 (per-token) + 4-byte accumulator, from the pool ----
    ggml_cuda_pool_alloc<int8_t>  x8   (ctx.pool(), (size_t)K * M);
    ggml_cuda_pool_alloc<float>   asc  (ctx.pool(), (size_t)M);
    ggml_cuda_pool_alloc<int32_t> acc  (ctx.pool(), (size_t)N * M);   // i32 (int8) or reinterpreted f32 (fp8)
    {
        const dim3 grid((unsigned)M), block(256);
        if (mode == MODE_F8) {
            k_quantize_act_e4m3_percol<<<grid, block, 0, stream>>>(
                (const char *)src1->data, src1->nb[1], (uint8_t *)x8.get(), asc.get(), K);
        } else {
            k_quantize_act_int8_percol<<<grid, block, 0, stream>>>(
                (const char *)src1->data, src1->nb[1], x8.get(), asc.get(), K);
        }
    }

    // ---- hipBLASLt GEMM: D(NxM) = op(A=W)[NxK] * B(X)[KxM], TN. Plan cached per (N,M,K,mode). ----
    hipblasLtHandle_t h = get_lt_handle();
    const lt_plan & plan = get_plan(N, M, K, mode);
    if (!plan.ok) {
        GGML_LOG_ERROR("%s: no hipBLASLt algo for %ldx%ldx%ld mode=%d\n", __func__, N, M, K, mode);
        return false;
    }

    ggml_cuda_pool_alloc<char> ws(ctx.pool(), LT_WS_BYTES);
    if (mode == MODE_F8) {
        const float alpha = 1.0f, beta = 0.0f;
        LT_OK(hipblasLtMatmul(h, plan.op, &alpha, wq8_ptr, plan.lA, x8.get(), plan.lB, &beta,
                              acc.get(), plan.lD, acc.get(), plan.lD,
                              &plan.heur.algo, ws.get(), LT_WS_BYTES, stream));
    } else {
        const int32_t alpha = 1, beta = 0;
        LT_OK(hipblasLtMatmul(h, plan.op, &alpha, wq8_ptr, plan.lA, x8.get(), plan.lB, &beta,
                              acc.get(), plan.lD, acc.get(), plan.lD,
                              &plan.heur.algo, ws.get(), LT_WS_BYTES, stream));
    }

    // ---- dequant: dst = acc * wscale[row] * ascale[col] (acc is i32 or f32) ----
    {
        const int64_t total = N * M;
        const dim3 block(256), grid((unsigned)((total + 255) / 256));
        if (mode == MODE_F8) {
            k_apply_scales_f32<<<grid, block, 0, stream>>>(
                (const float *)acc.get(), (char *)dst->data, dst->nb[1], wsc_ptr, asc.get(), N, M);
        } else {
            k_apply_scales<<<grid, block, 0, stream>>>(
                acc.get(), (char *)dst->data, dst->nb[1], wsc_ptr, asc.get(), N, M);
        }
    }
    return true;
}

#else  // ---- non-HIP / disabled: stubs so the globbed source is a no-op ----

bool ggml_cuda_q2_0_hipblaslt_prefill_supports(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) {
    return false;
}
bool ggml_cuda_op_mul_mat_q2_0_hipblaslt(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    return false;
}

#endif
