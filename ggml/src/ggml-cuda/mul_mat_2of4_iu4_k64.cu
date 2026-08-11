// T187 (reopened): full, correctness-gated K64 int4 2:4-sparse SWMMAC GEMM
// vs int8-K16-dense-WMMA / int4-K32-dense-WMMA baselines. See the .cuh
// header for the full rationale, provenance, and the measured verdict.
// Ported from the standalone bench (~/int4-research/pocs/t187-k64-full-gemm/
// gemm_bench.hip) where the correctness gate and perf sweep were originally
// run and validated (max_abs_err=0, GPU0-only).
#include "mul_mat_2of4_iu4_k64.cuh"

#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

#if defined(GGML_USE_HIP)

namespace ggml_cuda_2of4_iu4_k64 {

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

// Compressed 2:4 int4 K64 block (mirrors block_2of4_fp8's convention,
// ggml-common.h, scaled from QK=32/8-groups to QK=64/16-groups): qs[16] =
// 16 groups, byte g = lo_survivor(4b) | hi_survivor(4b)<<4; meta[8] = 8
// bytes, byte b covers groups (2b,2b+1), low/high nibble = that group's
// (idx0 | idx1<<2). Natural (unswapped) idx convention, matching
// swmmac24_iu4_k64.cu's already-hardware-validated K64 form.
struct block_iu4_24_k64 {
    uint8_t qs[16];
    uint8_t meta[8];
};
static constexpr int QK_IU4_24_K64 = 64;

// Row-pitch padding (found empirically this session): a naive row stride of
// EXACTLY K bytes (iu8) or K/2 bytes (iu4 nibble-packed) hits a real DRAM
// channel/bank critical-stride collision whenever it lands on a power-of-two
// boundary -- measured 8-30x cliffs, reproducible cold/isolated (not thermal
// noise). Every real production GEMM kernel pads its row pitch for exactly
// this reason.
static inline int64_t padded_row_stride_bytes(int64_t natural_bytes) {
    return natural_bytes + 256;
}

// ---- Kernel 1: dense int8, V_WMMA_I32_16X16X16_IU8 (baseline A) ----------
template <int WARPS, int ILP>
__launch_bounds__(WARPS * 32, 1)
static __global__ void k_gemm_iu8_k16(
        const int8_t * __restrict__ W, const int8_t * __restrict__ Act, int32_t * __restrict__ D,
        int64_t M, int64_t N, int64_t K, int64_t row_stride_a, int64_t row_stride_b) {
#if defined(RDNA4)
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int half = lane >> 4;
    const int li   = lane & 0xF;
    const int out_row_base = (lane >= 16) ? 8 : 0;

    const int64_t n0 = (int64_t) blockIdx.x * (WARPS * ILP * 16) + (int64_t) warp * ILP * 16;
    const int64_t m0 = (int64_t) blockIdx.y * 16;
    const int64_t act_col = m0 + li;
    const int64_t n_chunks = K / 16;

    auto load_a = [&] (int64_t c, v2i (&a)[ILP]) {
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
            a[t] = v2i{0, 0};
            const int64_t row = n0 + (int64_t) t * 16 + li;
            if (row < N) { const int8_t * p = W + row * row_stride_a + c * 16 + half * 8; memcpy(&a[t], p, 8); }
        }
    };
    auto load_b = [&] (int64_t c, v2i & b) {
        b = v2i{0, 0};
        if (act_col < M) { const int8_t * p = Act + act_col * row_stride_b + c * 16 + half * 8; memcpy(&b, p, 8); }
    };

    v2i a_cur[ILP], a_nxt[ILP];
    v2i b_cur, b_nxt;
    int32_t acc[ILP][8];
#pragma unroll
    for (int t = 0; t < ILP; ++t) { for (int l = 0; l < 8; ++l) acc[t][l] = 0; }
    if (n_chunks > 0) { load_a(0, a_cur); load_b(0, b_cur); }
    for (int64_t c = 0; c < n_chunks; ++c) {
        const bool have_next = (c + 1 < n_chunks);
        if (have_next) { load_a(c + 1, a_nxt); load_b(c + 1, b_nxt); }
        v8i raw[ILP];
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
            v8i c0 = {0,0,0,0,0,0,0,0};
            raw[t] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_cur[t], true, b_cur, c0, true);
        }
#pragma unroll
        for (int t = 0; t < ILP; ++t) for (int l = 0; l < 8; ++l) acc[t][l] += raw[t][l];
        if (have_next) { for (int t = 0; t < ILP; ++t) a_cur[t] = a_nxt[t]; b_cur = b_nxt; }
    }
    const int out_col = li;
#pragma unroll
    for (int t = 0; t < ILP; ++t) for (int l = 0; l < 8; ++l) {
        const int64_t n = n0 + (int64_t) t * 16 + out_row_base + l;
        const int64_t m = m0 + out_col;
        if (m < M && n < N) D[m * N + n] = acc[t][l];
    }
#else
    GGML_UNUSED(W); GGML_UNUSED(Act); GGML_UNUSED(D);
    GGML_UNUSED(M); GGML_UNUSED(N); GGML_UNUSED(K);
    GGML_UNUSED(row_stride_a); GGML_UNUSED(row_stride_b);
    NO_DEVICE_CODE;
#endif
}

// ---- Kernel 2: dense int4, V_WMMA_I32_16X16X32_IU4 (baseline B) ----------
template <int WARPS, int ILP>
__launch_bounds__(WARPS * 32, 1)
static __global__ void k_gemm_iu4_k32(
        const uint8_t * __restrict__ W, const uint8_t * __restrict__ Act, int32_t * __restrict__ D,
        int64_t M, int64_t N, int64_t K, int64_t row_stride_a, int64_t row_stride_b) {
#if defined(RDNA4)
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int half = lane >> 4;
    const int li   = lane & 0xF;
    const int out_row_base = (lane >= 16) ? 8 : 0;

    const int64_t n0 = (int64_t) blockIdx.x * (WARPS * ILP * 16) + (int64_t) warp * ILP * 16;
    const int64_t m0 = (int64_t) blockIdx.y * 16;
    const int64_t act_col = m0 + li;
    const int64_t n_chunks = K / 32;

    auto load_a = [&] (int64_t c, v2i (&a)[ILP]) {
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
            a[t] = v2i{0, 0};
            const int64_t row = n0 + (int64_t) t * 16 + li;
            if (row < N) { const uint8_t * p = W + row * row_stride_a + c * 16 + half * 8; memcpy(&a[t], p, 8); }
        }
    };
    auto load_b = [&] (int64_t c, v2i & b) {
        b = v2i{0, 0};
        if (act_col < M) { const uint8_t * p = Act + act_col * row_stride_b + c * 16 + half * 8; memcpy(&b, p, 8); }
    };

    v2i a_cur[ILP], a_nxt[ILP];
    v2i b_cur, b_nxt;
    int32_t acc[ILP][8];
#pragma unroll
    for (int t = 0; t < ILP; ++t) { for (int l = 0; l < 8; ++l) acc[t][l] = 0; }
    if (n_chunks > 0) { load_a(0, a_cur); load_b(0, b_cur); }
    for (int64_t c = 0; c < n_chunks; ++c) {
        const bool have_next = (c + 1 < n_chunks);
        if (have_next) { load_a(c + 1, a_nxt); load_b(c + 1, b_nxt); }
        v8i raw[ILP];
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
            v8i c0 = {0,0,0,0,0,0,0,0};
            raw[t] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a_cur[t], true, b_cur, c0, true);
        }
#pragma unroll
        for (int t = 0; t < ILP; ++t) for (int l = 0; l < 8; ++l) acc[t][l] += raw[t][l];
        if (have_next) { for (int t = 0; t < ILP; ++t) a_cur[t] = a_nxt[t]; b_cur = b_nxt; }
    }
    const int out_col = li;
#pragma unroll
    for (int t = 0; t < ILP; ++t) for (int l = 0; l < 8; ++l) {
        const int64_t n = n0 + (int64_t) t * 16 + out_row_base + l;
        const int64_t m = m0 + out_col;
        if (m < M && n < N) D[m * N + n] = acc[t][l];
    }
#else
    GGML_UNUSED(W); GGML_UNUSED(Act); GGML_UNUSED(D);
    GGML_UNUSED(M); GGML_UNUSED(N); GGML_UNUSED(K);
    GGML_UNUSED(row_stride_a); GGML_UNUSED(row_stride_b);
    NO_DEVICE_CODE;
#endif
}

// ---- Kernel 3 (TARGET): 2:4-sparse int4 K64, V_SWMMAC_I32_16X16X64_IU4 --
// Real gather from compressed A (values + metadata, two separate global
// arrays -- the actual 2:4 tax this bench measures) + dense B.
template <int WARPS, int ILP>
__launch_bounds__(WARPS * 32, 1)
static __global__ void k_gemm_iu4_k64s(
        const uint8_t * __restrict__ WcRaw, const uint8_t * __restrict__ Act, int32_t * __restrict__ D,
        int64_t M, int64_t N, int64_t K, int64_t nblk_row_stride_bytes, int64_t row_stride_b) {
#if defined(RDNA4)
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int half = lane >> 4;
    const int li   = lane & 0xF;
    const int out_row_base = (lane >= 16) ? 8 : 0;

    const int64_t n0 = (int64_t) blockIdx.x * (WARPS * ILP * 16) + (int64_t) warp * ILP * 16;
    const int64_t m0 = (int64_t) blockIdx.y * 16;
    const int64_t act_col = m0 + li;
    const int64_t nblk_k = K / QK_IU4_24_K64;

    auto load_a = [&] (int64_t c, v2i (&a)[ILP], unsigned (&idx)[ILP]) {
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
            a[t] = v2i{0, 0}; idx[t] = 0;
            const int64_t row = n0 + (int64_t) t * 16 + li;
            if (row < N) {
                const block_iu4_24_k64 & blk = *reinterpret_cast<const block_iu4_24_k64 *>(
                        WcRaw + row * nblk_row_stride_bytes + c * (int64_t) sizeof(block_iu4_24_k64));
                int ax = 0, ay = 0;
                memcpy(&ax, blk.qs + half * 8,     4);
                memcpy(&ay, blk.qs + half * 8 + 4, 4);
                a[t] = v2i{ ax, ay };
                unsigned m4 = 0;
                memcpy(&m4, blk.meta + half * 4, 4);
                idx[t] = m4;
            }
        }
    };
    auto load_b = [&] (int64_t c, v4i & b) {
        b = v4i{0, 0, 0, 0};
        if (act_col < M) { const uint8_t * p = Act + act_col * row_stride_b + c * 32 + half * 16; memcpy(&b, p, 16); }
    };

    v2i a_cur[ILP], a_nxt[ILP];
    unsigned idx_cur[ILP], idx_nxt[ILP];
    v4i b_cur, b_nxt;
    int32_t acc[ILP][8];
#pragma unroll
    for (int t = 0; t < ILP; ++t) { for (int l = 0; l < 8; ++l) acc[t][l] = 0; }
    if (nblk_k > 0) { load_a(0, a_cur, idx_cur); load_b(0, b_cur); }
    for (int64_t c = 0; c < nblk_k; ++c) {
        const bool have_next = (c + 1 < nblk_k);
        if (have_next) { load_a(c + 1, a_nxt, idx_nxt); load_b(c + 1, b_nxt); }
        v8i raw[ILP];
#pragma unroll
        for (int t = 0; t < ILP; ++t) {
            v8i c0 = {0,0,0,0,0,0,0,0};
            raw[t] = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(1, a_cur[t], 1, b_cur, c0, idx_cur[t], 0);
        }
#pragma unroll
        for (int t = 0; t < ILP; ++t) for (int l = 0; l < 8; ++l) acc[t][l] += raw[t][l];
        if (have_next) {
            for (int t = 0; t < ILP; ++t) { a_cur[t] = a_nxt[t]; idx_cur[t] = idx_nxt[t]; }
            b_cur = b_nxt;
        }
    }
    const int out_col = li;
#pragma unroll
    for (int t = 0; t < ILP; ++t) for (int l = 0; l < 8; ++l) {
        const int64_t n = n0 + (int64_t) t * 16 + out_row_base + l;
        const int64_t m = m0 + out_col;
        if (m < M && n < N) D[m * N + n] = acc[t][l];
    }
#else
    GGML_UNUSED(WcRaw); GGML_UNUSED(Act); GGML_UNUSED(D);
    GGML_UNUSED(M); GGML_UNUSED(N); GGML_UNUSED(K);
    GGML_UNUSED(nblk_row_stride_bytes); GGML_UNUSED(row_stride_b);
    NO_DEVICE_CODE;
#endif
}

// ---- Host-side data gen + CPU int64-accumulate reference ------------------
static void gen_dense_iu8(std::mt19937 & rng, int64_t rows, int64_t K, int64_t row_stride_bytes,
                           std::vector<int8_t> & packed, std::vector<int> & logical) {
    std::uniform_int_distribution<int> d(-100, 100);
    packed.assign((size_t)(rows * row_stride_bytes), 0);
    logical.assign((size_t)(rows * K), 0);
    for (int64_t r = 0; r < rows; ++r) for (int64_t k = 0; k < K; ++k) {
        int v = d(rng);
        packed[r*row_stride_bytes + k] = (int8_t) v;
        logical[r*K + k] = v;
    }
}

static void gen_dense_iu4(std::mt19937 & rng, int64_t rows, int64_t K, int64_t row_stride_bytes,
                           std::vector<uint8_t> & packed, std::vector<int> & logical) {
    std::uniform_int_distribution<int> d(-8, 7);
    packed.assign((size_t)(rows * row_stride_bytes), 0);
    logical.assign((size_t)(rows * K), 0);
    for (int64_t r = 0; r < rows; ++r) for (int64_t k = 0; k < K; k += 2) {
        int v0 = d(rng), v1 = d(rng);
        logical[r*K + k] = v0; logical[r*K + k + 1] = v1;
        packed[r*row_stride_bytes + k/2] = (uint8_t) ((v0 & 0xF) | ((v1 & 0xF) << 4));
    }
}

static void gen_sparse_iu4_24_k64(std::mt19937 & rng, int64_t rows, int64_t K, int64_t nblk_row_stride_bytes,
                                   std::vector<uint8_t> & blocksRaw, std::vector<int> & logical) {
    static const int choices[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    std::uniform_int_distribution<int> pick(0, 5);
    std::uniform_int_distribution<int> valdist(-8, 7);
    const int64_t nblk_k = K / QK_IU4_24_K64;
    blocksRaw.assign((size_t)(rows * nblk_row_stride_bytes), 0);
    logical.assign((size_t)(rows * K), 0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t bc = 0; bc < nblk_k; ++bc) {
            block_iu4_24_k64 blk{};
            for (int g = 0; g < 16; ++g) {
                const int c = pick(rng);
                const int lo_pos = choices[c][0], hi_pos = choices[c][1];
                const int lo_val = valdist(rng), hi_val = valdist(rng);
                const int64_t kbase = bc * 64 + g * 4;
                logical[r*K + kbase + lo_pos] = lo_val;
                logical[r*K + kbase + hi_pos] = hi_val;
                blk.qs[g] = (uint8_t) ((lo_val & 0xF) | ((hi_val & 0xF) << 4));
                const uint8_t nib = (uint8_t) ((lo_pos & 0x3) | ((hi_pos & 0x3) << 2));
                if (g % 2 == 0) blk.meta[g/2] = (blk.meta[g/2] & 0xF0) | nib;
                else            blk.meta[g/2] = (blk.meta[g/2] & 0x0F) | (nib << 4);
            }
            memcpy(blocksRaw.data() + r*nblk_row_stride_bytes + bc*(int64_t)sizeof(block_iu4_24_k64), &blk, sizeof(block_iu4_24_k64));
        }
    }
}

static long ref_gemm_max_abs_err(const std::vector<int> & Wlog, const std::vector<int> & Actlog,
                                  const std::vector<int32_t> & Dgot, int64_t M, int64_t N, int64_t K) {
    long max_err = 0;
    for (int64_t m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) {
        long long ref = 0;
        const int * wrow = &Wlog[n*K];
        const int * arow = &Actlog[m*K];
        for (int64_t k = 0; k < K; ++k) ref += (long long) wrow[k] * (long long) arow[k];
        max_err = std::max(max_err, (long) llabs((long long) Dgot[m*N + n] - ref));
    }
    return max_err;
}

// ---- Correctness gate -----------------------------------------------------
template <int WARPS, int ILP>
static bool check_iu4_k64s(int64_t M, int64_t N, int64_t K, unsigned seed) {
    std::mt19937 rng(seed);
    const int64_t nblk_k = K / QK_IU4_24_K64;
    const int64_t rs_a = padded_row_stride_bytes(nblk_k * (int64_t) sizeof(block_iu4_24_k64));
    const int64_t rs_b = padded_row_stride_bytes(K/2);
    std::vector<uint8_t> Wc, Ap; std::vector<int> Wlog, Alog;
    gen_sparse_iu4_24_k64(rng, N, K, rs_a, Wc, Wlog);
    gen_dense_iu4(rng, M, K, rs_b, Ap, Alog);

    uint8_t *dW = nullptr, *dA = nullptr; int32_t *dD = nullptr;
    if (hipMalloc(&dW, Wc.size()) != hipSuccess || hipMalloc(&dA, Ap.size()) != hipSuccess ||
        hipMalloc(&dD, (size_t)(M*N)*sizeof(int32_t)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    CUDA_CHECK(hipMemcpy(dW, Wc.data(), Wc.size(), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(dA, Ap.data(), Ap.size(), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(dD, 0, (size_t)(M*N)*sizeof(int32_t)));

    const dim3 block(32, WARPS, 1);
    const int64_t bn = (int64_t) WARPS * ILP * 16;
    const dim3 grid((N + bn - 1)/bn, (M+15)/16, 1);
    k_gemm_iu4_k64s<WARPS,ILP><<<grid, block>>>(dW, dA, dD, M, N, K, rs_a, rs_b);
    const hipError_t err0 = hipDeviceSynchronize();
    if (err0 != hipSuccess) {
        GGML_LOG_ERROR("%s: kernel launch failed: %s\n", __func__, hipGetErrorString(err0));
        (void) hipFree(dW); (void) hipFree(dA); (void) hipFree(dD);
        return false;
    }

    std::vector<int32_t> Dgot((size_t)(M*N));
    CUDA_CHECK(hipMemcpy(Dgot.data(), dD, Dgot.size()*sizeof(int32_t), hipMemcpyDeviceToHost));
    (void) hipFree(dW); (void) hipFree(dA); (void) hipFree(dD);

    const long err = ref_gemm_max_abs_err(Wlog, Alog, Dgot, M, N, K);
    GGML_LOG_INFO("%s: [iu4 K64 SWMMAC 2:4] M=%ld N=%ld K=%ld max_abs_err=%ld -> %s\n",
                   __func__, (long)M, (long)N, (long)K, err, err==0 ? "PASS" : "FAIL");
    return err == 0;
}

// ---- Perf sweep (TOPS convention: 2*M*N*K_logical/time, matching
// bench_ilp.hip's K_eff=64 SWMMAC convention -- directly comparable to the
// raw-instruction ISA ceiling, 3.90x vs K16 / 1.95x vs K32) --------------
template <int WARPS, int ILP>
static double perf_iu8_k16_ms(int64_t M, int64_t N, int64_t K) {
    std::mt19937 rng(7001);
    const int64_t rs = padded_row_stride_bytes(K);
    std::vector<int8_t> Wp, Ap; std::vector<int> dummy;
    gen_dense_iu8(rng, N, K, rs, Wp, dummy); dummy.clear();
    gen_dense_iu8(rng, M, K, rs, Ap, dummy);
    int8_t *dW = nullptr, *dA = nullptr; int32_t *dD = nullptr;
    (void) hipMalloc(&dW, Wp.size()); (void) hipMalloc(&dA, Ap.size());
    (void) hipMalloc(&dD, (size_t)(M*N)*sizeof(int32_t));
    (void) hipMemcpy(dW, Wp.data(), Wp.size(), hipMemcpyHostToDevice);
    (void) hipMemcpy(dA, Ap.data(), Ap.size(), hipMemcpyHostToDevice);
    const dim3 block(32, WARPS, 1);
    const int64_t bn = (int64_t) WARPS * ILP * 16;
    const dim3 grid((N + bn - 1)/bn, (M+15)/16, 1);
    auto launch = [&] () { k_gemm_iu8_k16<WARPS,ILP><<<grid, block>>>(dW, dA, dD, M, N, K, rs, rs); };
    for (int r=0;r<3;++r) launch();
    (void) hipDeviceSynchronize();
    std::vector<double> times;
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    for (int r=0;r<6;++r) { hipEventRecord(s); launch(); hipEventRecord(e); hipEventSynchronize(e);
        float ms; hipEventElapsedTime(&ms,s,e); times.push_back(ms); }
    std::sort(times.begin(), times.end());
    const double ms = times[times.size()/2];
    (void) hipFree(dW); (void) hipFree(dA); (void) hipFree(dD);
    return ms;
}

template <int WARPS, int ILP>
static double perf_iu4_k32_ms(int64_t M, int64_t N, int64_t K) {
    std::mt19937 rng(7002);
    const int64_t rs = padded_row_stride_bytes(K/2);
    std::vector<uint8_t> Wp, Ap; std::vector<int> dummy;
    gen_dense_iu4(rng, N, K, rs, Wp, dummy); dummy.clear();
    gen_dense_iu4(rng, M, K, rs, Ap, dummy);
    uint8_t *dW = nullptr, *dA = nullptr; int32_t *dD = nullptr;
    (void) hipMalloc(&dW, Wp.size()); (void) hipMalloc(&dA, Ap.size());
    (void) hipMalloc(&dD, (size_t)(M*N)*sizeof(int32_t));
    (void) hipMemcpy(dW, Wp.data(), Wp.size(), hipMemcpyHostToDevice);
    (void) hipMemcpy(dA, Ap.data(), Ap.size(), hipMemcpyHostToDevice);
    const dim3 block(32, WARPS, 1);
    const int64_t bn = (int64_t) WARPS * ILP * 16;
    const dim3 grid((N + bn - 1)/bn, (M+15)/16, 1);
    auto launch = [&] () { k_gemm_iu4_k32<WARPS,ILP><<<grid, block>>>(dW, dA, dD, M, N, K, rs, rs); };
    for (int r=0;r<3;++r) launch();
    (void) hipDeviceSynchronize();
    std::vector<double> times;
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    for (int r=0;r<6;++r) { hipEventRecord(s); launch(); hipEventRecord(e); hipEventSynchronize(e);
        float ms; hipEventElapsedTime(&ms,s,e); times.push_back(ms); }
    std::sort(times.begin(), times.end());
    const double ms = times[times.size()/2];
    (void) hipFree(dW); (void) hipFree(dA); (void) hipFree(dD);
    return ms;
}

template <int WARPS, int ILP>
static double perf_iu4_k64s_ms(int64_t M, int64_t N, int64_t K) {
    std::mt19937 rng(7003);
    const int64_t nblk_k = K / QK_IU4_24_K64;
    const int64_t rs_a = padded_row_stride_bytes(nblk_k * (int64_t) sizeof(block_iu4_24_k64));
    const int64_t rs_b = padded_row_stride_bytes(K/2);
    std::vector<uint8_t> Wc, Ap; std::vector<int> dummy;
    gen_sparse_iu4_24_k64(rng, N, K, rs_a, Wc, dummy); dummy.clear();
    gen_dense_iu4(rng, M, K, rs_b, Ap, dummy);
    uint8_t *dW = nullptr, *dA = nullptr; int32_t *dD = nullptr;
    (void) hipMalloc(&dW, Wc.size()); (void) hipMalloc(&dA, Ap.size());
    (void) hipMalloc(&dD, (size_t)(M*N)*sizeof(int32_t));
    (void) hipMemcpy(dW, Wc.data(), Wc.size(), hipMemcpyHostToDevice);
    (void) hipMemcpy(dA, Ap.data(), Ap.size(), hipMemcpyHostToDevice);
    const dim3 block(32, WARPS, 1);
    const int64_t bn = (int64_t) WARPS * ILP * 16;
    const dim3 grid((N + bn - 1)/bn, (M+15)/16, 1);
    auto launch = [&] () { k_gemm_iu4_k64s<WARPS,ILP><<<grid, block>>>(dW, dA, dD, M, N, K, rs_a, rs_b); };
    for (int r=0;r<3;++r) launch();
    (void) hipDeviceSynchronize();
    std::vector<double> times;
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    for (int r=0;r<6;++r) { hipEventRecord(s); launch(); hipEventRecord(e); hipEventSynchronize(e);
        float ms; hipEventElapsedTime(&ms,s,e); times.push_back(ms); }
    std::sort(times.begin(), times.end());
    const double ms = times[times.size()/2];
    (void) hipFree(dW); (void) hipFree(dA); (void) hipFree(dD);
    return ms;
}

} // namespace ggml_cuda_2of4_iu4_k64

using namespace ggml_cuda_2of4_iu4_k64;

bool ggml_cuda_mul_mat_2of4_iu4_k64_selftest() {
    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        return true; // not RDNA4: nothing to test, vacuously true.
    }
    constexpr int WARPS = 32, ILP = 4;
    bool ok = true;
    const struct { int64_t M,N,K; unsigned seed; } shapes[] = {
        { 32, 32, 256, 1 }, { 48, 80, 512, 2 }, { 17, 33, 128, 3 }, { 64, 64, 1024, 4 },
    };
    for (auto & s : shapes) {
        ok &= check_iu4_k64s<WARPS,ILP>(s.M, s.N, s.K, s.seed);
    }
    return ok;
}

bool ggml_cuda_mul_mat_2of4_iu4_k64_shape_bench() {
    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        return true;
    }
    // Correctness FIRST -- a fast-but-wrong kernel is a FAIL, refuse to
    // report throughput otherwise (T187 directive).
    if (!ggml_cuda_mul_mat_2of4_iu4_k64_selftest()) {
        GGML_LOG_ERROR("%s: correctness gate FAILED -- refusing to report throughput\n", __func__);
        return false;
    }
    constexpr int WARPS = 32, ILP = 4;
    const int64_t M = 4096, N = 4096;
    const int64_t Ksweep[] = { 2048, 4096, 8192 };
    GGML_LOG_INFO("%s: M=N=4096, WARPS=%d ILP=%d, median-of-6 (TOPS conv: 2*M*N*K_logical/time, "
                   "K_eff=64 for SWMMAC -- matches bench_ilp.hip's raw ISA ceiling convention)\n",
                   __func__, WARPS, ILP);
    for (int64_t K : Ksweep) {
        const double ms16  = perf_iu8_k16_ms<WARPS,ILP>(M, N, K);
        const double ms32  = perf_iu4_k32_ms<WARPS,ILP>(M, N, K);
        const double ms64s = perf_iu4_k64s_ms<WARPS,ILP>(M, N, K);
        const double t16  = 2.0 * M * N * K / (ms16 /1000.0) / 1e12;
        const double t32  = 2.0 * M * N * K / (ms32 /1000.0) / 1e12;
        const double t64s = 2.0 * M * N * K / (ms64s/1000.0) / 1e12;
        GGML_LOG_INFO("%s: K=%-5ld iu8-K16=%.3f TOPS  iu4-K32=%.3f TOPS  iu4-K64-2:4=%.3f TOPS  "
                       "(K64/K16=%.3fx, K64/K32=%.3fx)\n",
                       __func__, (long) K, t16, t32, t64s, t64s/t16, t64s/t32);
    }
    return true;
}

#else // !defined(GGML_USE_HIP)

bool ggml_cuda_mul_mat_2of4_iu4_k64_selftest()    { return true; }
bool ggml_cuda_mul_mat_2of4_iu4_k64_shape_bench()  { return true; }

#endif // defined(GGML_USE_HIP)
