// See mul_mat_f16_hipblaslt.cuh for the full rationale + measured-verdict
// pointer. Structure mirrors mul_mat_q2_0_hipblaslt.cu's plan cache + disk-
// persisted algo tuner (Stage 3/4 there); this file skips the weight-requant
// and dequant-epilogue kernels entirely since F16/BF16 weights need no
// conversion and HH_SH/BB_SB emit fp32 D directly.

#include <cstdint>
#include "mul_mat_f16_hipblaslt.cuh"

#if defined(__HIP_PLATFORM_AMD__) && !defined(GGML_HIP_NO_HIPBLASLT)

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-version.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <map>
#include <tuple>
#include <mutex>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <sys/stat.h>

namespace {

#define LT_OK(x) do { hipblasStatus_t s_ = (x); if (s_ != HIPBLAS_STATUS_SUCCESS) { \
    GGML_LOG_ERROR("%s: hipBLASLt error %d at %s:%d\n", __func__, (int)s_, __FILE__, __LINE__); \
    return false; } } while(0)

enum gemm_mode { MODE_F16 = 0, MODE_BF16 = 1, MODE_F32 = 2 };

// ---- activation cast kernels: F32 -> F16/BF16, per column (token) ----------
__global__ void k_cast_act_f16(const char * __restrict__ src1, int64_t nb1, __half * __restrict__ out, int64_t K) {
    const int64_t j = blockIdx.x;
    const float * col = (const float *)(src1 + j * nb1);
    __half * o = out + j * K;
    for (int64_t l = threadIdx.x; l < K; l += blockDim.x) o[l] = __float2half(col[l]);
}
__global__ void k_cast_act_bf16(const char * __restrict__ src1, int64_t nb1, __hip_bfloat16 * __restrict__ out, int64_t K) {
    const int64_t j = blockIdx.x;
    const float * col = (const float *)(src1 + j * nb1);
    __hip_bfloat16 * o = out + j * K;
    for (int64_t l = threadIdx.x; l < K; l += blockDim.x) o[l] = __hip_bfloat16(col[l]);
}

// D (fp32) is column-major [N x M] with ld=N (matches ggml dst nb[1] IF dst
// is contiguous fp32 with nb1 == N*4 -- true for a freshly-allocated 2D dst).
// If dst has non-default stride we still need to scatter row-by-row, so copy
// through a staging buffer when nb1 != N*sizeof(float).
__global__ void k_scatter_dst(const float * __restrict__ acc, char * __restrict__ dst, int64_t nb1, int64_t N, int64_t M) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= N * M) return;
    const int64_t n = idx % N;
    const int64_t j = idx / N;
    float * out = (float *)(dst + j * nb1);
    out[n] = acc[j * N + n];
}

hipblasLtHandle_t get_lt_handle() {
    static hipblasLtHandle_t h = [](){ hipblasLtHandle_t t; hipblasLtCreate(&t); return t; }();
    return h;
}

// ---- per-shape plan cache (same design as the Q2_0 route) -----------------
constexpr size_t LT_WS_BYTES = 32ull << 20;

struct lt_plan {
    hipblasLtMatmulDesc_t             op   = nullptr;
    hipblasLtMatrixLayout_t           lA   = nullptr, lB = nullptr, lD = nullptr;
    hipblasLtMatmulHeuristicResult_t  heur{};
    bool                              ok   = false;
};

std::map<std::tuple<int64_t,int64_t,int64_t,int>, lt_plan> g_plan_cache;
std::mutex g_plan_mtx;

#define TUNE_S2(x) #x
#define TUNE_S(x) TUNE_S2(x)
constexpr char TUNE_MAGIC[8] = {'R','D','N','4','G','T','4','\0'};   // distinct tag from the Q2_0 cache

std::map<std::tuple<int64_t,int64_t,int64_t,int>, int32_t> g_disk_best;
bool g_disk_loaded = false;

const char * tune_cache_path() {
    static std::string path = [](){
        if (const char * e = getenv("GGML_HIP_F16_HIPBLASLT_TUNE_CACHE")) return std::string(e);
        const char * home = getenv("HOME");
        return std::string(home ? home : "/tmp") + "/.cache/ggml-rdna4-gemm-f16-tune.bin";
    }();
    return path.c_str();
}
std::string tune_version_tag() {
    return std::string("hipblaslt-") + TUNE_S(HIPBLASLT_VERSION_MAJOR) "." TUNE_S(HIPBLASLT_VERSION_MINOR)
         "." TUNE_S(HIPBLASLT_VERSION_PATCH) "-" TUNE_S(HIPBLASLT_VERSION_TWEAK);
}
struct TuneRec { int64_t N, M, K; int32_t mode; int32_t best_index; };

void load_disk_algos() {
    if (g_disk_loaded) return;
    g_disk_loaded = true;
    FILE * f = fopen(tune_cache_path(), "rb");
    if (!f) return;
    char magic[8] = {0};
    uint32_t vlen = 0;
    std::string want = tune_version_tag();
    std::string ver;
    if (fread(magic, 1, 8, f) == 8 && memcmp(magic, TUNE_MAGIC, 8) == 0 &&
        fread(&vlen, 4, 1, f) == 1 && vlen <= 256) {
        ver.resize(vlen);
        if (fread(&ver[0], 1, vlen, f) == vlen && ver == want) {
            TuneRec r;
            while (fread(&r, sizeof(TuneRec), 1, f) == 1) {
                g_disk_best[std::make_tuple(r.N, r.M, r.K, (int)r.mode)] = r.best_index;
            }
        }
    }
    fclose(f);
}
void save_disk_algos() {
    std::string p = tune_cache_path();
    auto slash = p.find_last_of('/');
    if (slash != std::string::npos) mkdir(p.substr(0, slash).c_str(), 0755);
    FILE * f = fopen(p.c_str(), "wb");
    if (!f) return;
    std::string ver = tune_version_tag();
    uint32_t vlen = (uint32_t)ver.size();
    fwrite(TUNE_MAGIC, 1, 8, f);
    fwrite(&vlen, 4, 1, f);
    fwrite(ver.data(), 1, vlen, f);
    for (auto & kv : g_disk_best) {
        TuneRec r{ std::get<0>(kv.first), std::get<1>(kv.first), std::get<2>(kv.first),
                   (int32_t)std::get<3>(kv.first), kv.second };
        fwrite(&r, sizeof(TuneRec), 1, f);
    }
    fclose(f);
}

const lt_plan & get_plan(int64_t N, int64_t M, int64_t K, int mode) {
    std::lock_guard<std::mutex> lk(g_plan_mtx);
    auto key = std::make_tuple(N, M, K, mode);
    auto it = g_plan_cache.find(key);
    if (it != g_plan_cache.end()) {
        return it->second;
    }
    const hipblasComputeType_t compute = HIPBLAS_COMPUTE_32F;
    const hipDataType abT = (mode == MODE_F32) ? HIP_R_32F : (mode == MODE_BF16) ? HIP_R_16BF : HIP_R_16F;
    const hipDataType dT  = HIP_R_32F;   // HH_SH / BB_SB / SS_SS: fp32 D directly
    lt_plan p;
    hipblasLtHandle_t h = get_lt_handle();
    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    bool built =
        hipblasLtMatmulDescCreate(&p.op, compute, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)) == HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)) == HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatrixLayoutCreate(&p.lA, abT, K, N, K) == HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatrixLayoutCreate(&p.lB, abT, K, M, K) == HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatrixLayoutCreate(&p.lD, dT,  N, M, N) == HIPBLAS_STATUS_SUCCESS;
    if (built) {
        load_disk_algos();
        const auto dkey = std::make_tuple(N, M, K, mode);

        hipblasLtMatmulPreference_t pref = nullptr;
        size_t ws = LT_WS_BYTES; int nAlgo = 0;
        constexpr int REQ = 64;
        std::vector<hipblasLtMatmulHeuristicResult_t> cand(REQ);
        if (hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS &&
            hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)) == HIPBLAS_STATUS_SUCCESS &&
            hipblasLtMatmulAlgoGetHeuristic(h, p.op, p.lA, p.lB, p.lD, p.lD, pref, REQ, cand.data(), &nAlgo) == HIPBLAS_STATUS_SUCCESS &&
            nAlgo > 0) {
            static const bool notune = (getenv("GGML_HIP_F16_HIPBLASLT_NOTUNE") != nullptr);
            int best = 0;
            auto di = g_disk_best.find(dkey);
            if (di != g_disk_best.end() && di->second >= 0 && di->second < nAlgo) {
                best = di->second;
            } else if (!notune && nAlgo > 1) {
                void *sA=nullptr,*sB=nullptr,*sD=nullptr,*sW=nullptr;
                const size_t esz = (mode == MODE_F32) ? sizeof(float) : (mode == MODE_BF16) ? sizeof(__hip_bfloat16) : sizeof(__half);
                if (hipMalloc(&sA,(size_t)K*N*esz)==hipSuccess && hipMalloc(&sB,(size_t)K*M*esz)==hipSuccess &&
                    hipMalloc(&sD,(size_t)N*M*4)==hipSuccess && hipMalloc(&sW,LT_WS_BYTES)==hipSuccess) {
                    hipMemset(sA,0x3c,(size_t)K*N*esz); hipMemset(sB,0x3c,(size_t)K*M*esz);
                    const float alpha=1.f, beta=0.f;
                    auto run=[&](hipblasLtMatmulAlgo_t &a){ return hipblasLtMatmul(h,p.op,&alpha,sA,p.lA,sB,p.lB,&beta,sD,p.lD,sD,p.lD,&a,sW,LT_WS_BYTES,0); };
                    hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1);
                    double bestMs = 1e30;
                    for (int i=0;i<nAlgo;i++){
                        if (run(cand[i].algo) != HIPBLAS_STATUS_SUCCESS) continue;
                        for(int w=0;w<2;w++) run(cand[i].algo);
                        hipDeviceSynchronize(); hipEventRecord(e0,0);
                        for(int r=0;r<10;r++) run(cand[i].algo);
                        hipEventRecord(e1,0); hipEventSynchronize(e1);
                        float ms=0; hipEventElapsedTime(&ms,e0,e1);
                        if (ms>0 && ms<bestMs){ bestMs=ms; best=i; }
                    }
                    hipEventDestroy(e0); hipEventDestroy(e1);
                }
                if (sA) hipFree(sA); if (sB) hipFree(sB); if (sD) hipFree(sD); if (sW) hipFree(sW);
                g_disk_best[dkey] = best;
                save_disk_algos();
            }
            p.heur = cand[best];
            p.ok = true;
        }
        if (pref) hipblasLtMatmulPreferenceDestroy(pref);
    }
    auto res = g_plan_cache.emplace(key, p);
    return res.first->second;
}

} // namespace

bool ggml_cuda_f16_hipblaslt_prefill_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16 && src0->type != GGML_TYPE_F32) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)   return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1)                        return false;
    if (src1->ne[2] != 1 || src1->ne[3] != 1)                        return false;
    if (src0->ne[0] != src1->ne[0])                                  return false;
    if (src0->nb[0] != ggml_type_size(src0->type))                   return false;   // contiguous rows required

    static const int64_t M_THRESH = [](){
        const char * e = getenv("GGML_HIP_F16_HIPBLASLT_MTHRESH");
        // DISABLED BY DEFAULT (measured 2026-08-05). This route wins per-matmul at
        // M=64..256 (+19% to +26%) but is a NET REGRESSION end to end: -2.3% to
        // -3.7% on pp1024 across four resamples, because at the production ubatch
        // (M=512) the gain is +0.6% -- inside noise -- while the per-call conversion
        // and dispatch overhead is not. There is no shape where it demonstrably wins
        // on a real model, so the default threshold is set above any realistic M.
        // Set GGML_HIP_F16_HIPBLASLT_MTHRESH explicitly to experiment with it.
        return e ? (int64_t)atoll(e) : (int64_t)INT64_MAX;   // was 32
    }();
    if (src1->ne[1] <= M_THRESH) return false;

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    return GGML_CUDA_CC_IS_RDNA4(cc);
}

bool ggml_cuda_op_mul_mat_f16_hipblaslt(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int mode = (src0->type == GGML_TYPE_F32) ? MODE_F32 : (src0->type == GGML_TYPE_BF16) ? MODE_BF16 : MODE_F16;
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    cudaStream_t  stream = ctx.stream();

    // Weight needs no conversion -- GGML_TYPE_F16/BF16 storage already IS the
    // hipBLASLt A-operand dtype (half / __hip_bfloat16 bit-compatible).
    const void * wptr = src0->data;
    // If the weight row isn't tightly packed (nb[1] != K*elsz) we can't feed
    // it directly as a [K x N] ld=K matrix -- bail to the caller's fallback.
    const size_t esz = (mode == MODE_F32) ? sizeof(float) : (mode == MODE_BF16) ? sizeof(__hip_bfloat16) : sizeof(__half);
    if ((size_t)src0->nb[1] != (size_t)K * esz) {
        return false;
    }

    // ---- B-operand: F16/BF16 need an F32->half cast+repack; F32 rides src1
    // directly when its rows are already [K x M] ld=K packed. ----
    const void * bptr;
    ggml_cuda_pool_alloc<char> x16;
    if (mode == MODE_F32) {
        if ((size_t)src1->nb[1] != (size_t)K * sizeof(float)) {
            return false;
        }
        bptr = src1->data;
    } else {
        x16.alloc(ctx.pool(), (size_t)K * M * esz);
        const dim3 grid((unsigned)M), block(256);
        if (mode == MODE_BF16) {
            k_cast_act_bf16<<<grid, block, 0, stream>>>((const char *)src1->data, src1->nb[1], (__hip_bfloat16 *)x16.get(), K);
        } else {
            k_cast_act_f16<<<grid, block, 0, stream>>>((const char *)src1->data, src1->nb[1], (__half *)x16.get(), K);
        }
        bptr = x16.get();
    }

    // ---- hipBLASLt GEMM: D(NxM, fp32) = op(A=W)[NxK] * B(X)[KxM], TN ----
    hipblasLtHandle_t h = get_lt_handle();
    const lt_plan & plan = get_plan(N, M, K, mode);
    if (!plan.ok) {
        GGML_LOG_ERROR("%s: no hipBLASLt algo for %ldx%ldx%ld mode=%d\n", __func__, N, M, K, mode);
        return false;
    }

    // Fast path: if dst is contiguous fp32 [N x M] (nb1 == N*4), write the
    // GEMM output directly into dst->data -- no staging/scatter needed.
    const bool dst_contig = ((size_t)dst->nb[1] == (size_t)N * sizeof(float));
    ggml_cuda_pool_alloc<float> acc_pool;
    float * acc_ptr;
    if (dst_contig) {
        acc_ptr = (float *)dst->data;
    } else {
        acc_ptr = acc_pool.alloc(ctx.pool(), (size_t)N * M);
    }

    ggml_cuda_pool_alloc<char> ws(ctx.pool(), LT_WS_BYTES);
    const float alpha = 1.0f, beta = 0.0f;
    LT_OK(hipblasLtMatmul(h, plan.op, &alpha, wptr, plan.lA, bptr, plan.lB, &beta,
                          acc_ptr, plan.lD, acc_ptr, plan.lD,
                          &plan.heur.algo, ws.get(), LT_WS_BYTES, stream));

    if (!dst_contig) {
        const int64_t total = N * M;
        const dim3 block(256), grid((unsigned)((total + 255) / 256));
        k_scatter_dst<<<grid, block, 0, stream>>>(acc_ptr, (char *)dst->data, dst->nb[1], N, M);
    }
    return true;
}

#else  // ---- non-HIP / disabled: stubs ----

bool ggml_cuda_f16_hipblaslt_prefill_supports(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) {
    return false;
}
bool ggml_cuda_op_mul_mat_f16_hipblaslt(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    return false;
}

#endif
