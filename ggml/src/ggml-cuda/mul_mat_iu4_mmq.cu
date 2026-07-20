// T162: see mul_mat_iu4_mmq.cuh for the full rationale. This replaces the
// one-warp-per-tile PoC (mul_mat_iu4.cu / mul_mat_q2_0_wmma.cu) with an
// MMQ-style tiled GEMM: multi-warp block, register blocking over both M and
// N (each warp owns several 16x16 WMMA tiles), double-buffered shared
// staging (next K-chunk's global loads are issued before the current
// chunk's compute, so the compiler/hardware can hide load latency behind
// the WMMA+FMA work instead of serializing every chunk behind a sync), and
// ONE kernel body shared by the native IU4 weight layout and the Q2_0
// ternary layout via a loader trait template parameter.
//
// Toolchain note (same lesson as mul_mat_iu4.cu / mul_mat_q2_0_wmma.cu): the
// host launcher functions must NOT be wrapped in `#if defined(RDNA4)` --
// that macro is only defined during the device-code compilation pass of a
// .cu TU, never the host pass, so a host-side #if silently strips the real
// symbol and leaves only a stub. Gate __device__ code internally instead
// (mma_iu4() already does this).
#include "mul_mat_iu4_mmq.cuh"

#include "mma.cuh"
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <mutex>

using namespace ggml_cuda_mma;

namespace ggml_cuda_mul_mat_iu4_mmq_detail {

// --- tile geometry (V1) -----------------------------------------------------
// BM x BN output tile per block, NWARPS warps/block (32 lanes each).
// tiles_total = (BM/16)*(BN/16) must be an exact multiple of NWARPS so every
// warp owns the same number (ntx) of 16x16 minitiles -- register blocking.
// BM+BN <= NWARPS*32 so the (very small, ~5KB double-buffered) shared
// staging load can use one thread per weight/activation row with no loop.
// T163 iterate-A: NWARPS 8->4 (BM/BN unchanged). This doubles NTX (tiles,
// hence independent mma_iu4 calls, per warp per K-chunk) from 2 to 4 --
// coordinator directive T163: the diagnostic floor (Expt A, no-WMMA XOR
// swap) already beats MMQ, and dropping only the rescale (Expt B) recovers
// most of the rest, so the bottleneck is WMMA issue/dependency LATENCY, not
// occupancy -- give the warp scheduler more INDEPENDENT WMMA ops in flight
// per iteration to hide it. Side effect: BM+BN(128) == NWARPS*32(128)
// exactly now, so ALL 4 warps do staging (0 idle, vs 4-of-8 idle before).
static constexpr int MMQ_IU4_BM     = 64;
static constexpr int MMQ_IU4_BN     = 64;
static constexpr int MMQ_IU4_NWARPS = 4;
static constexpr int MMQ_IU4_NTILES = (MMQ_IU4_BM / 16) * (MMQ_IU4_BN / 16);
static constexpr int MMQ_IU4_NTX    = MMQ_IU4_NTILES / MMQ_IU4_NWARPS;
static_assert(MMQ_IU4_BM % 16 == 0 && MMQ_IU4_BN % 16 == 0, "tile size must be a multiple of 16");
static_assert(MMQ_IU4_NTILES % MMQ_IU4_NWARPS == 0, "tiles must divide evenly across warps");
static_assert(MMQ_IU4_BM + MMQ_IU4_BN <= MMQ_IU4_NWARPS * 32, "staging load assumes 1 thread/row");

static __global__ void k_quantize_act_iu4_mmq(
        const float * __restrict__ x, block_iu4 * __restrict__ y,
        const int64_t n_blocks_k, const int64_t row_stride_floats) {
    const int64_t c   = blockIdx.x; // which 32-elem block along K
    const int64_t m   = blockIdx.y; // which token/row
    const int     tid = threadIdx.x; // 0..31, element index within the block

    __shared__ float sh_val[32];
    __shared__ float sh_scale;
    __shared__ int   sh_q[32];

    const float v = x[m * row_stride_floats + c * 32 + tid];
    sh_val[tid] = fabsf(v);
    __syncthreads();

    if (tid == 0) {
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            amax = fmaxf(amax, sh_val[i]);
        }
        const float d = amax / 7.0f;
        sh_scale = d;
        y[m * n_blocks_k + c].d = __float2half(d);
    }
    __syncthreads();

    const float d  = sh_scale;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q = (int) rintf(v * id);
    q = q < -8 ? -8 : (q > 7 ? 7 : q);
    sh_q[tid] = q;
    __syncthreads();

    if ((tid & 1) == 0) {
        const int q0 = sh_q[tid];
        const int q1 = sh_q[tid + 1];
        y[m * n_blocks_k + c].qs[tid / 2] = (uint8_t) ((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }
}

static __device__ __forceinline__ void load_iu4_words(const block_iu4 & blk, int (&w)[4]) {
    memcpy(&w[0], blk.qs + 0,  4);
    memcpy(&w[1], blk.qs + 4,  4);
    memcpy(&w[2], blk.qs + 8,  4);
    memcpy(&w[3], blk.qs + 12, 4);
}

// Same unpack convention as mul_mat_q2_0_wmma.cu's
// unpack_q2_0_chunk_to_iu4_words: 8 consecutive Q2_0 qs bytes (32
// chunk-local ternary codes {0,1,2}, 4/byte at 2 bits each) -> the 4 int32
// words the validated iu4_w4a4.cu/mul_mat_iu4.cu packing convention expects.
static __device__ __forceinline__ void unpack_q2_0_chunk_to_iu4_words(const uint8_t * __restrict__ qs8, int (&w)[4]) {
    int vals[32];
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const int byte = qs8[b];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int code = (byte >> (e * 2)) & 0x3;
            vals[4*b + e] = code - 1;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t word = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int      k_lo = 8*j + 2*b;
            const int      k_hi = 8*j + 2*b + 1;
            const uint32_t nlo  = (uint32_t) (vals[k_lo] & 0xF);
            const uint32_t nhi  = (uint32_t) (vals[k_hi] & 0xF);
            word |= (nlo | (nhi << 4)) << (8*b);
        }
        w[j] = (int) word;
    }
}

// Q1_0 is binary {-1,+1}: QK1_0==128 elements (4 of our 32-wide `c` chunks)
// behind ONE fp16 scale, packed 1 bit/element (16 qs bytes/block, see
// ggml-common.h). Bit convention mirrors vec_dot_q1_0_q8_1 (vecdotq.cuh) --
// the ONLY other place this format is read on this backend, so we must
// match it exactly for correctness against real GGUF Q1_0 tensors: for the
// 4-byte (32-bit) sub-chunk, byte b's LOW nibble bit e (0..3) encodes
// element 8b+e, HIGH nibble bit e encodes element 8b+4+e; bit set -> +1,
// clear -> -1.
static __device__ __forceinline__ void unpack_q1_0_chunk_to_iu4_words(const uint8_t * __restrict__ qs4, int (&w)[4]) {
    int vals[32];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int byte = qs4[b];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            vals[8*b + e]     = ((byte >> e)       & 1) ? 1 : -1;
            vals[8*b + 4 + e] = ((byte >> (4 + e))  & 1) ? 1 : -1;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t word = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int      k_lo = 8*j + 2*b;
            const int      k_hi = 8*j + 2*b + 1;
            const uint32_t nlo  = (uint32_t) (vals[k_lo] & 0xF);
            const uint32_t nhi  = (uint32_t) (vals[k_hi] & 0xF);
            word |= (nlo | (nhi << 4)) << (8*b);
        }
        w[j] = (int) word;
    }
}

// --- weight loader traits ----------------------------------------------------
// Each loader reads ONE 32-element (QK_IU4) chunk `c` of weight row `n` (row
// stride `nb01` bytes, as stored in the ggml tensor) and produces the 4 iu4
// packed int32 words + the block's float scale. This is the ONLY thing that
// differs between the native IU4 tensor type and the Q2_0/Q1_0 ternary/
// binary types -- the rest of the kernel (tiling, staging, WMMA,
// accumulation) is shared.
struct IU4Loader {
    static __device__ __forceinline__ void load(
            const char * __restrict__ vweight, int64_t n, int64_t nb01, int64_t c,
            int (&w)[4], float & d) {
        const block_iu4 * blk = (const block_iu4 *) (vweight + n * nb01) + c;
        load_iu4_words(*blk, w);
        d = __half2float(blk->d);
    }
};

struct Q2_0Loader {
    static __device__ __forceinline__ void load(
            const char * __restrict__ vweight, int64_t n, int64_t nb01, int64_t c,
            int (&w)[4], float & d) {
        const int64_t      q2blk = c / 4;
        const int           subc  = (int) (c % 4);
        const block_q2_0 * bq2   = (const block_q2_0 *) (vweight + n * nb01) + q2blk;
        unpack_q2_0_chunk_to_iu4_words(bq2->qs + subc * 8, w);
        d = __half2float(bq2->d);
    }
};

struct Q1_0Loader {
    static __device__ __forceinline__ void load(
            const char * __restrict__ vweight, int64_t n, int64_t nb01, int64_t c,
            int (&w)[4], float & d) {
        const int64_t      q1blk = c / 4;
        const int           subc  = (int) (c % 4);
        const block_q1_0 * bq1   = (const block_q1_0 *) (vweight + n * nb01) + q1blk;
        unpack_q1_0_chunk_to_iu4_words(bq1->qs + subc * 4, w);
        d = __half2float(bq1->d);
    }
};

// --- T164 iterate-2: per-channel weight re-quantization ----------------------
// Re-quantizes a REAL Q2_0 (native per-128) weight tensor to a per-CHANNEL
// (whole-row) scale, producing a block_iu4-formatted buffer that
// k_mul_mat_iu4_mmq<IU4Loader, ..., defer_dw=true> reads via IU4Loader
// UNCHANGED (same redundant-per-block `d` layout, just now holding the
// SAME value for every block of a row instead of the native per-128 value).
// Mirrors scripts/int4-accuracy-gate/w4a4_gate_bonsai_weight_grid.py's
// ternary_weight_requant() EXACTLY (same round-to-nearest-of-{-1,0,1}
// against a new absmax-derived scale) -- that Python function is what the
// T164-1 accuracy grid (w=channel/a=32 -> +12.66% vs floor, PASS) actually
// validated, so the kernel must perform the identical operation, not just
// relabel the existing codes with a new scale (which would silently corrupt
// the ternary rounding and NOT match the accuracy-gated combo).
// One warp (32 lanes) per output row/channel; pass 1 finds the row-wide
// absmax over the DEQUANTIZED native-ternary values (code*native_block_scale);
// pass 2 re-rounds every element against the new channel scale and re-packs.
static __global__ void k_requant_q2_0_to_iu4_perchannel(
        const char * __restrict__ vweight, block_iu4 * __restrict__ out,
        const int64_t nb01, const int64_t n_blocks_k) {
    const int64_t n   = blockIdx.x; // one warp per output row/channel
    const int     tid = threadIdx.x; // 0..31, element index within a 32-chunk

    __shared__ float sh_val[32];
    __shared__ float sh_scale;
    __shared__ int   sh_code[32];

    auto read_dequant = [&] (int64_t c) -> float {
        const int64_t      q2blk = c / 4;
        const int           subc  = (int) (c % 4);
        const block_q2_0 * bq2   = (const block_q2_0 *) (vweight + n * nb01) + q2blk;
        const int byte = bq2->qs[subc * 8 + tid / 4];
        const int code = (byte >> ((tid % 4) * 2)) & 0x3;
        return (float) (code - 1) * __half2float(bq2->d);
    };

    // Pass 1: absmax over the WHOLE row (dequantized native-ternary values).
    float local_amax = 0.0f;
    for (int64_t c = 0; c < n_blocks_k; ++c) {
        local_amax = fmaxf(local_amax, fabsf(read_dequant(c)));
    }
    sh_val[tid] = local_amax;
    __syncthreads();
    if (tid == 0) {
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            amax = fmaxf(amax, sh_val[i]);
        }
        sh_scale = fmaxf(amax, 1e-9f);
    }
    __syncthreads();
    const float new_scale = sh_scale;

    // Pass 2: re-round every element against new_scale, pack into block_iu4
    // (same word/nibble convention as unpack_q2_0_chunk_to_iu4_words's output).
    for (int64_t c = 0; c < n_blocks_k; ++c) {
        const float val = read_dequant(c);
        int new_code = (int) rintf(val / new_scale);
        new_code = new_code < -1 ? -1 : (new_code > 1 ? 1 : new_code);
        sh_code[tid] = new_code;
        __syncthreads();

        if (tid < 4) {
            const int j = tid;
            uint32_t word = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int      k_lo = 8*j + 2*b;
                const int      k_hi = 8*j + 2*b + 1;
                const uint32_t nlo  = (uint32_t) (sh_code[k_lo] & 0xF);
                const uint32_t nhi  = (uint32_t) (sh_code[k_hi] & 0xF);
                word |= (nlo | (nhi << 4)) << (8*b);
            }
            std::memcpy(out[n * n_blocks_k + c].qs + 4*j, &word, 4);
        }
        if (tid == 0) {
            out[n * n_blocks_k + c].d = __float2half(new_scale);
        }
        __syncthreads(); // sh_code about to be reused by the next chunk
    }
}

// Per-(weight-tensor-pointer) cache so the re-quantization above runs ONCE
// (first use) rather than every mul_mat call -- this is the honest
// steady-state proxy for what a real re-quantized GGUF (Phase 3) would give
// for free; weight tensors are immutable for the lifetime of a loaded model,
// so keying on `src0->data` is safe. Never freed (process-lifetime cache) --
// acceptable for this experimental, env-gated, dev-only path.
static std::unordered_map<const void *, block_iu4 *> g_perchannel_weight_cache;
static std::mutex g_perchannel_weight_cache_mutex;

static block_iu4 * get_or_build_perchannel_weight(
        const ggml_tensor * src0, int64_t N, int64_t n_blocks_k, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_perchannel_weight_cache_mutex);
    auto it = g_perchannel_weight_cache.find(src0->data);
    if (it != g_perchannel_weight_cache.end()) {
        return it->second;
    }
    block_iu4 * buf = nullptr;
    CUDA_CHECK(hipMalloc(&buf, (size_t) (N * n_blocks_k) * sizeof(block_iu4)));
    const int64_t nb01 = src0->nb[1];
    const dim3 grid(N, 1, 1);
    const dim3 block(32, 1, 1);
    k_requant_q2_0_to_iu4_perchannel<<<grid, block, 0, stream>>>(
            (const char *) src0->data, buf, nb01, n_blocks_k);
    CUDA_CHECK(hipStreamSynchronize(stream)); // only on cache-miss (first use of this tensor)
    g_perchannel_weight_cache[src0->data] = buf;
    return buf;
}

// --- the MMQ-grade kernel ----------------------------------------------------
// Grid: ((N+BN-1)/BN, (M+BM-1)/BM). Block: (32, NWARPS, 1) -- threadIdx.x is
// the WMMA lane id (tile<>::get_i/get_j assume a bare 0..31 lane, so the
// warp dimension MUST live in threadIdx.y, not be folded into threadIdx.x).
// T162 iterate-1: `need_check` mirrors mmq.cuh's `load_tiles_q*<mmq_y, need_check>`
// pattern exactly (grep confirms every legacy-quant MMQ loader is templated on
// this bool). ISA disasm of the ungated kernel showed the m0+row<M / n0+row<N
// guard recompiled into a FULL save/restore-exec + branch + zero-fill sequence
// EVERY k-chunk iteration, even though m0/n0/M/N/row are loop-invariant -- the
// compiler cannot hoist this across the __syncthreads() in the loop body
// (barriers force wavefront reconvergence, blocking code motion across them
// on AMDGPU). The fix used everywhere else in this codebase is to not branch
// at all: pick this at the HOST launch call (need_check = false whenever
// M%BM==0 && N%BN==0, true otherwise) so the interior-tile (overwhelmingly
// the common case -- every real Bonsai-27B Q2_0 weight tensor's N is a
// multiple of 64, M=pp512's token count is too) instantiation has ZERO
// per-chunk bounds-check code at all, not just a cheaper one.
// T162 iterate-4 (tried, reverted -- MIN_BLOCKS=2 instead of 1, see
// wiki/tech/bonsai-ternary-models.md): measured NEUTRAL on this kernel body
// (1003-1008 t/s, indistinguishable from the 1004-1013 noise band with
// MIN_BLOCKS=1) -- the unmodified kernel's 47 VGPR/thread was already low
// enough that occupancy wasn't the limiter here, so there was no headroom
// for this lever to recover. Left at MIN_BLOCKS=1.
// T163 iterate-F: `ceiling_mode` (defaulted false -- EVERY existing
// production call site is byte-for-byte unaffected) prototypes the
// external-kernel-intel RC2 attack #1 (APEX4 per-channel weight scale +
// per-token activation scale -> ONE combined rescale multiply at
// write-out, ZERO per-chunk rescale at all) purely as a SELFTEST-HARNESS
// speed-ceiling measurement. This is NOT wired into any production path --
// it requires the weight scale to be genuinely constant across all of K
// (per-channel), which our real Q2_0/IU4 GGUF formats are not (group-32/
// group-128 respectively); a synthetic per-channel-scaled tensor is used
// to drive it in ggml_cuda_mul_mat_iu4_mmq_ceiling_bench() below.
// T164 iterate-2: `defer_dw` (defaulted false, zero risk to existing call
// sites) is the PRODUCTION-TARGETED half of ceiling_mode -- coordinator
// directive T164, following the Phase 1 accuracy grid
// (w4a4_gate_bonsai_weight_grid.py): per-CHANNEL weight scale is
// loop-invariant down K and factors to ONE epilogue multiply with NO
// accumulator array (unlike T162/T163-C's per-block deferrals, which needed
// a live scratch accumulator and blew VGPR/LDS budget). Activation scale
// (sh_da) MUST stay per-chunk -- Phase 1 proved per-token activation
// coarsening is not accuracy-safe (matches T163-E's real-model Paris
// failure) -- so only the weight-scale multiply is removed from the hot
// loop here, applied once at write-out instead.
template <typename WeightLoader, int BM, int BN, int NWARPS, bool need_check, bool ceiling_mode = false, bool defer_dw = false>
__launch_bounds__(NWARPS * 32, 1)
static __global__ void k_mul_mat_iu4_mmq(
        const char * __restrict__ vweight, const block_iu4 * __restrict__ act, float * __restrict__ dst,
        const int64_t M, const int64_t N, const int64_t nb01, const int64_t n_blocks_k,
        const int64_t dst_row_stride_floats) {
    constexpr int NTILES_N = BN / 16;
    constexpr int NTX      = (BM / 16) * (BN / 16) / NWARPS;

    const int64_t n0      = (int64_t) blockIdx.x * BN;
    const int64_t m0      = (int64_t) blockIdx.y * BM;
    const int     lane    = threadIdx.x;
    const int     warp_id = threadIdx.y;
    // T162 iterate-2: `threadIdx.y` (warp_id) is PHYSICALLY uniform across a
    // wave (block=(32,NWARPS,1), wave size 32 == blockDim.x, so a whole
    // wavefront always shares one threadIdx.y), but LLVM's AMDGPU divergence
    // analysis does not know that invariant from the builtin alone and was
    // compiling the `lin_tid < BM` / `lin_tid < BM+BN` warp-dispatch branches
    // below as if they were per-LANE divergent -- full exec-mask save/xor/
    // restore + s_cbranch_execz every single K-chunk iteration (confirmed in
    // ISA disasm even after the need_check=false fix eliminated the M/N
    // bounds-check branch: the remaining s_and_saveexec_b32/s_cbranch_execz
    // pairs per iteration were THIS dispatch, not bounds checking).
    // __builtin_amdgcn_readfirstlane forces the value into an SGPR, proving
    // to the compiler it is wave-uniform, so the branch below compiles to a
    // plain scalar s_cmp/s_cbranch with zero per-lane predication machinery.
    const int warp_id_u = __builtin_amdgcn_readfirstlane(warp_id);
    static_assert(BM % 32 == 0 && BN % 32 == 0, "warp-uniform dispatch assumes BM/BN are multiples of the wave size");
    constexpr int WARPS_A = BM / 32;
    constexpr int WARPS_B = BN / 32;

    __shared__ int   sh_A[2][BM][4];
    __shared__ float sh_da[2][BM];
    __shared__ int   sh_B[2][BN][4];
    __shared__ float sh_dw[2][BN];

    // Stage chunk `c` of both operands into shared buffer `buf`. One thread
    // per row (BM+BN <= NWARPS*32 by construction) -- trivially small
    // transfer (4 words + 1 scale/row), not worth a fancier mapping. Dispatch
    // is by warp_id_u (see above), not lin_tid, so the branch is warp-uniform.
    auto load_chunk = [&] (int64_t c, int buf) {
        if (warp_id_u < WARPS_A) {
            const int     row = warp_id_u * 32 + lane;
            const int64_t m   = m0 + row;
            if (!need_check || m < M) {
                const block_iu4 & blk = act[m * n_blocks_k + c];
                int w[4];
                load_iu4_words(blk, w);
                sh_A[buf][row][0] = w[0]; sh_A[buf][row][1] = w[1];
                sh_A[buf][row][2] = w[2]; sh_A[buf][row][3] = w[3];
                sh_da[buf][row] = __half2float(blk.d);
            } else {
                sh_A[buf][row][0] = sh_A[buf][row][1] = sh_A[buf][row][2] = sh_A[buf][row][3] = 0;
                sh_da[buf][row] = 0.0f;
            }
        } else if (warp_id_u < WARPS_A + WARPS_B) {
            const int     row = (warp_id_u - WARPS_A) * 32 + lane;
            const int64_t n   = n0 + row;
            if (!need_check || n < N) {
                int w[4];
                float d;
                WeightLoader::load(vweight, n, nb01, c, w, d);
                sh_B[buf][row][0] = w[0]; sh_B[buf][row][1] = w[1];
                sh_B[buf][row][2] = w[2]; sh_B[buf][row][3] = w[3];
                sh_dw[buf][row] = d;
            } else {
                sh_B[buf][row][0] = sh_B[buf][row][1] = sh_B[buf][row][2] = sh_B[buf][row][3] = 0;
                sh_dw[buf][row] = 0.0f;
            }
        }
    };

    float acc[NTX][8];
#pragma unroll
    for (int s = 0; s < NTX; ++s) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            acc[s][l] = 0.0f;
        }
    }

    load_chunk(0, 0);
    __syncthreads();

    int cur = 0;
    for (int64_t c = 0; c < n_blocks_k; ++c) {
        const int nxt = cur ^ 1;
        if (c + 1 < n_blocks_k) {
            // Issued before this chunk's compute below with no intervening
            // sync -- independent shared-memory regions (sh_*[nxt] vs
            // sh_*[cur]), so the compiler/hardware is free to interleave
            // these global loads with the WMMA/FMA work that follows,
            // hiding load latency behind compute instead of serializing.
            load_chunk(c + 1, nxt);
        }

#pragma unroll
        for (int s = 0; s < NTX; ++s) {
            const int t  = warp_id + s * NWARPS;
            const int mi = t / NTILES_N;
            const int ni = t % NTILES_N;

            tile<16, 4, int> A;
            tile<16, 4, int> B;
            load_generic(A, &sh_A[cur][mi * 16][0], 4);
            load_generic(B, &sh_B[cur][ni * 16][0], 4);

            // DATA_LAYOUT_J_MAJOR readback fix (same as mul_mat_iu4.cu /
            // mul_mat_q2_0_wmma.cu): the WMMA accumulator's physical VGPR
            // layout is the transpose of the A/B input layout on RDNA4.
            tile<16, 16, int, DATA_LAYOUT_J_MAJOR> D;
#pragma unroll
            for (int l = 0; l < D.ne; ++l) {
                D.x[l] = 0;
            }

            mma_iu4(D, A, B);

#pragma unroll
            for (int l = 0; l < D.ne; ++l) {
                const int i = D.get_i(l);
                const int j = D.get_j(l);
                if constexpr (ceiling_mode) {
                    // RC2 ceiling: NO rescale at all in the hot loop --
                    // valid only because the ceiling-bench harness feeds a
                    // synthetic per-channel-weight / per-row-activation
                    // tensor where both scales are provably chunk-invariant.
                    acc[s][l] += (float) D.x[l];
                } else if constexpr (defer_dw) {
                    // T164: weight scale (dw) is per-channel/loop-invariant
                    // in the re-quantized data this path is fed -- only the
                    // (still-per-chunk) activation scale applies here.
                    acc[s][l] += (float) D.x[l] * sh_da[cur][mi * 16 + i];
                } else {
                    acc[s][l] += (float) D.x[l] * sh_da[cur][mi * 16 + i] * sh_dw[cur][ni * 16 + j];
                }
            }
        }

        __syncthreads();
        cur = nxt;
    }

#pragma unroll
    for (int s = 0; s < NTX; ++s) {
        const int t  = warp_id + s * NWARPS;
        const int mi = t / NTILES_N;
        const int ni = t % NTILES_N;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int     i = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>::get_i(l);
            const int     j = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>::get_j(l);
            const int64_t m = m0 + mi * 16 + i;
            const int64_t n = n0 + ni * 16 + j;
            if (m < M && n < N) {
                if constexpr (ceiling_mode) {
                    // Apply BOTH scales exactly once here -- correct iff
                    // they are chunk-invariant (true for the synthetic
                    // ceiling-bench tensor; NOT true for real Q2_0/IU4 GGUF
                    // data, hence this path is never used in production).
                    dst[m * dst_row_stride_floats + n] = acc[s][l] * sh_da[cur][mi * 16 + i] * sh_dw[cur][ni * 16 + j];
                } else if constexpr (defer_dw) {
                    // dw is chunk-invariant (per-channel) here, so reading
                    // it from whichever `cur` buffer is current is correct.
                    dst[m * dst_row_stride_floats + n] = acc[s][l] * sh_dw[cur][ni * 16 + j];
                } else {
                    dst[m * dst_row_stride_floats + n] = acc[s][l];
                }
            }
        }
    }
}

template <typename WeightLoader>
static bool launch(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t n_blocks_k = K / QK_IU4;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_iu4> act_q(ctx.pool(), (size_t) (M * n_blocks_k));

    {
        const int64_t row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
        const dim3 grid(n_blocks_k, M, 1);
        const dim3 block(32, 1, 1);
        k_quantize_act_iu4_mmq<<<grid, block, 0, stream>>>((const float *) src1->data, act_q.get(), n_blocks_k, row_stride_floats);
    }

    {
        constexpr int BM = MMQ_IU4_BM;
        constexpr int BN = MMQ_IU4_BN;
        constexpr int NWARPS = MMQ_IU4_NWARPS;
        const int64_t nb01 = src0->nb[1];
        const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
        const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
        const dim3 block(32, NWARPS, 1);
        // need_check=false whenever every block in the grid is fully interior
        // (M,N exact multiples of BM,BN) -- see the k_mul_mat_iu4_mmq comment.
        // Real GGUF weight tensors' N (output features) are virtually always
        // a multiple of 64 and pp512's M (token count) is too, so the
        // production benchmark path takes the zero-overhead instantiation.
        const bool need_check = (M % BM != 0) || (N % BN != 0);
        if (need_check) {
            k_mul_mat_iu4_mmq<WeightLoader, BM, BN, NWARPS, true><<<grid, block, 0, stream>>>(
                    (const char *) src0->data, act_q.get(), (float *) dst->data,
                    M, N, nb01, n_blocks_k, dst_row_stride_floats);
        } else {
            k_mul_mat_iu4_mmq<WeightLoader, BM, BN, NWARPS, false><<<grid, block, 0, stream>>>(
                    (const char *) src0->data, act_q.get(), (float *) dst->data,
                    M, N, nb01, n_blocks_k, dst_row_stride_floats);
        }
    }

    return true;
}

// T164 iterate-2: PER-CHANNEL weight-scale production path (opt-in,
// GGML_HIP_IU4_MMQ_PERCHANNEL, see ggml_cuda_op_mul_mat_q2_0_iu4_mmq_perchannel
// below). Only wired for GGML_TYPE_Q2_0 (the model this was accuracy-gated
// against) -- re-quantizes the real weight tensor once (cached), then runs
// the SAME kernel with defer_dw=true (weight-scale multiply hoisted out of
// the K-loop, activation scale stays per-chunk exactly as in production).
static bool launch_perchannel(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t n_blocks_k = K / QK_IU4;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_iu4> act_q(ctx.pool(), (size_t) (M * n_blocks_k));
    {
        // Bug fix: this must match k_quantize_act_iu4_mmq's CURRENT (T163-E
        // reverted, production) grid shape -- one block per (chunk,row), NOT
        // one block per row (that was the T163-E per-row-quant variant,
        // reverted after failing the Paris gate). Copy-paste from that
        // reverted code path caused blockIdx.x to be misinterpreted as `m`
        // instead of `c`, corrupting every activation quantization call and
        // producing garbage ("???") output.
        const int64_t row_stride_floats = src1->nb[1] / (int64_t) sizeof(float);
        const dim3 grid(n_blocks_k, M, 1);
        const dim3 block(32, 1, 1);
        k_quantize_act_iu4_mmq<<<grid, block, 0, stream>>>((const float *) src1->data, act_q.get(), n_blocks_k, row_stride_floats);
    }

    block_iu4 * w_perchannel = get_or_build_perchannel_weight(src0, N, n_blocks_k, stream);

    {
        constexpr int BM = MMQ_IU4_BM;
        constexpr int BN = MMQ_IU4_BN;
        constexpr int NWARPS = MMQ_IU4_NWARPS;
        const int64_t nb01 = n_blocks_k * (int64_t) sizeof(block_iu4); // re-quantized buffer is IU4Loader-shaped, not Q2_0-shaped
        const int64_t dst_row_stride_floats = dst->nb[1] / (int64_t) sizeof(float);
        const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
        const dim3 block(32, NWARPS, 1);
        const bool need_check = (M % BM != 0) || (N % BN != 0);
        if (need_check) {
            k_mul_mat_iu4_mmq<IU4Loader, BM, BN, NWARPS, true, false, true><<<grid, block, 0, stream>>>(
                    (const char *) w_perchannel, act_q.get(), (float *) dst->data,
                    M, N, nb01, n_blocks_k, dst_row_stride_floats);
        } else {
            k_mul_mat_iu4_mmq<IU4Loader, BM, BN, NWARPS, false, false, true><<<grid, block, 0, stream>>>(
                    (const char *) w_perchannel, act_q.get(), (float *) dst->data,
                    M, N, nb01, n_blocks_k, dst_row_stride_floats);
        }
    }

    return true;
}

} // namespace ggml_cuda_mul_mat_iu4_mmq_detail

bool ggml_cuda_iu4_mmq_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_IU4 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q1_0) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0] || src0->ne[0] % QK_IU4 != 0) {
        return false;
    }
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    return GGML_CUDA_CC_IS_RDNA4(cc);
}

bool ggml_cuda_op_mul_mat_iu4_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using namespace ggml_cuda_mul_mat_iu4_mmq_detail;
    GGML_ASSERT(src0->type == GGML_TYPE_IU4);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(src0->ne[0] % QK_IU4 == 0);
    return launch<IU4Loader>(ctx, src0, src1, dst);
}

bool ggml_cuda_op_mul_mat_q2_0_iu4_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using namespace ggml_cuda_mul_mat_iu4_mmq_detail;
    GGML_ASSERT(src0->type == GGML_TYPE_Q2_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(src0->ne[0] % QK_IU4 == 0);
    return launch<Q2_0Loader>(ctx, src0, src1, dst);
}

bool ggml_cuda_op_mul_mat_q1_0_iu4_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using namespace ggml_cuda_mul_mat_iu4_mmq_detail;
    GGML_ASSERT(src0->type == GGML_TYPE_Q1_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(src0->ne[0] % QK_IU4 == 0);
    return launch<Q1_0Loader>(ctx, src0, src1, dst);
}

// T164 iterate-2: opt-in PER-CHANNEL weight-scale path, GGML_TYPE_Q2_0 only
// (gated separately from the T162/T163 production GGML_HIP_IU4_MMQ path by
// its own env var, GGML_HIP_IU4_MMQ_PERCHANNEL, in ggml-cuda.cu -- zero risk
// to the existing production path). See launch_perchannel()'s comment.
bool ggml_cuda_op_mul_mat_q2_0_iu4_mmq_perchannel(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using namespace ggml_cuda_mul_mat_iu4_mmq_detail;
    GGML_ASSERT(src0->type == GGML_TYPE_Q2_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(src0->ne[0] % QK_IU4 == 0);
    return launch_perchannel(ctx, src0, src1, dst);
}

// --- self-test ---------------------------------------------------------------
// Same style/CPU reference as ggml_cuda_mul_mat_iu4_selftest() in
// mul_mat_iu4.cu, but exercising THIS kernel (multi-warp, double-buffered)
// through both loader traits: IU4 (full int4 range both operands) and Q2_0
// (ternary {-1,0,+1} weight range, matching what the real quant produces).
namespace ggml_cuda_mul_mat_iu4_mmq_selftest_detail {
using namespace ggml_cuda_mul_mat_iu4_mmq_detail;

static void pack_iu4_block(const int vals[QK_IU4], block_iu4 & blk) {
    blk.d = __float2half(1.0f); // fixed unit scale: accumulator == answer, no rounding
    for (int j = 0; j < 4; ++j) {
        uint32_t word = 0;
        for (int b = 0; b < 4; ++b) {
            const int      k_lo = 8*j + 2*b;
            const int      k_hi = 8*j + 2*b + 1;
            const uint32_t nlo  = (uint32_t) (vals[k_lo] & 0xF);
            const uint32_t nhi  = (uint32_t) (vals[k_hi] & 0xF);
            word |= (nlo | (nhi << 4)) << (8*b);
        }
        std::memcpy(blk.qs + 4*j, &word, 4);
    }
}

// Packs 32 ternary {-1,0,+1} values into a Q2_0 block's qs bytes (unit scale
// via blk.d == 1, so blk.d isn't 1-per-32 like a real Q2_0 tensor -- the
// caller only ever fills subc==0..3 of ONE block per trial so this is fine
// for a synthetic single-block-per-row test).
static void pack_q2_0_subc(const int vals[32], uint8_t * qs8) {
    for (int b = 0; b < 8; ++b) {
        uint8_t byte = 0;
        for (int e = 0; e < 4; ++e) {
            const int code = vals[4*b + e] + 1; // {-1,0,1} -> {0,1,2}
            byte |= (uint8_t) (code & 0x3) << (e * 2);
        }
        qs8[b] = byte;
    }
}

// Packs 32 binary {-1,+1} values into a Q1_0 sub-chunk's 4 qs bytes, using
// the SAME bit convention as unpack_q1_0_chunk_to_iu4_words / the real
// vec_dot_q1_0_q8_1 reader (byte b low nibble bit e -> element 8b+e, high
// nibble bit e -> element 8b+4+e; +1 -> bit set).
static void pack_q1_0_subc(const int vals[32], uint8_t * qs4) {
    for (int b = 0; b < 4; ++b) {
        uint8_t byte = 0;
        for (int e = 0; e < 4; ++e) {
            if (vals[8*b + e] > 0)     { byte |= (uint8_t) (1u << e); }
            if (vals[8*b + 4 + e] > 0) { byte |= (uint8_t) (1u << (4 + e)); }
        }
        qs4[b] = byte;
    }
}

enum class LoaderKind { IU4, Q2_0, Q1_0 };

template <LoaderKind kind, bool need_check>
static bool run_trial(std::mt19937 & rng, int64_t M, int64_t N, int64_t K, long & max_abs_err_out) {
    const int64_t n_blocks_k = K / QK_IU4;
    std::uniform_int_distribution<int> actdist(-8, 7);
    const int wlo = kind == LoaderKind::IU4 ? -8 : -1;
    const int whi = kind == LoaderKind::IU4 ?  7 :  1;
    std::uniform_int_distribution<int> wdist_full(wlo, whi);
    // Q1_0 is strictly binary {-1,+1} -- no zero -- so a plain [-1,1]
    // uniform int distribution (which WOULD include 0) is wrong for it.
    std::uniform_int_distribution<int> wdist_bin(0, 1); // 0 -> -1, 1 -> +1

    std::vector<std::vector<int>> act_logical(M, std::vector<int>(K));
    std::vector<std::vector<int>> w_logical(N, std::vector<int>(K));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            act_logical[m][k] = actdist(rng);
        }
    }
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            if constexpr (kind == LoaderKind::Q1_0) {
                w_logical[n][k] = wdist_bin(rng) == 0 ? -1 : 1;
            } else {
                w_logical[n][k] = wdist_full(rng);
            }
        }
    }

    std::vector<block_iu4> act_blocks((size_t) (M * n_blocks_k));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t c = 0; c < n_blocks_k; ++c) {
            pack_iu4_block(&act_logical[m][c * QK_IU4], act_blocks[m * n_blocks_k + c]);
        }
    }

    void * d_w_raw = nullptr;
    int64_t nb01 = 0;
    std::vector<block_iu4>  w_blocks_iu4;
    std::vector<block_q2_0> w_blocks_q2_0;
    std::vector<block_q1_0> w_blocks_q1_0;
    if constexpr (kind == LoaderKind::Q2_0) {
        const int64_t n_q2_blocks = (n_blocks_k + 3) / 4;
        nb01 = n_q2_blocks * (int64_t) sizeof(block_q2_0);
        w_blocks_q2_0.assign((size_t) (N * n_q2_blocks), block_q2_0{});
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < n_blocks_k; ++c) {
                block_q2_0 & bq = w_blocks_q2_0[n * n_q2_blocks + c / 4];
                bq.d = __float2half(1.0f);
                pack_q2_0_subc(&w_logical[n][c * QK_IU4], bq.qs + (c % 4) * 8);
            }
        }
        if (hipMalloc(&d_w_raw, w_blocks_q2_0.size() * sizeof(block_q2_0)) != hipSuccess) {
            GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
            return false;
        }
        CUDA_CHECK(hipMemcpy(d_w_raw, w_blocks_q2_0.data(), w_blocks_q2_0.size() * sizeof(block_q2_0), hipMemcpyHostToDevice));
    } else if constexpr (kind == LoaderKind::Q1_0) {
        const int64_t n_q1_blocks = (n_blocks_k + 3) / 4;
        nb01 = n_q1_blocks * (int64_t) sizeof(block_q1_0);
        w_blocks_q1_0.assign((size_t) (N * n_q1_blocks), block_q1_0{});
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < n_blocks_k; ++c) {
                block_q1_0 & bq = w_blocks_q1_0[n * n_q1_blocks + c / 4];
                bq.d = __float2half(1.0f);
                pack_q1_0_subc(&w_logical[n][c * QK_IU4], bq.qs + (c % 4) * 4);
            }
        }
        if (hipMalloc(&d_w_raw, w_blocks_q1_0.size() * sizeof(block_q1_0)) != hipSuccess) {
            GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
            return false;
        }
        CUDA_CHECK(hipMemcpy(d_w_raw, w_blocks_q1_0.data(), w_blocks_q1_0.size() * sizeof(block_q1_0), hipMemcpyHostToDevice));
    } else {
        nb01 = n_blocks_k * (int64_t) sizeof(block_iu4);
        w_blocks_iu4.assign((size_t) (N * n_blocks_k), block_iu4{});
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < n_blocks_k; ++c) {
                pack_iu4_block(&w_logical[n][c * QK_IU4], w_blocks_iu4[n * n_blocks_k + c]);
            }
        }
        if (hipMalloc(&d_w_raw, w_blocks_iu4.size() * sizeof(block_iu4)) != hipSuccess) {
            GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
            return false;
        }
        CUDA_CHECK(hipMemcpy(d_w_raw, w_blocks_iu4.data(), w_blocks_iu4.size() * sizeof(block_iu4), hipMemcpyHostToDevice));
    }

    block_iu4 * d_act = nullptr;
    float *     d_dst = nullptr;
    if (hipMalloc(&d_act, act_blocks.size() * sizeof(block_iu4)) != hipSuccess ||
        hipMalloc(&d_dst, (size_t) (M * N) * sizeof(float)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        (void) hipFree(d_w_raw);
        return false;
    }
    CUDA_CHECK(hipMemcpy(d_act, act_blocks.data(), act_blocks.size() * sizeof(block_iu4), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(d_dst, 0, (size_t) (M * N) * sizeof(float)));

    constexpr int BM = MMQ_IU4_BM;
    constexpr int BN = MMQ_IU4_BN;
    constexpr int NWARPS = MMQ_IU4_NWARPS;
    const int64_t dst_row_stride_floats = N;
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
    const dim3 block(32, NWARPS, 1);

    using Loader = std::conditional_t<kind == LoaderKind::Q2_0, Q2_0Loader,
                   std::conditional_t<kind == LoaderKind::Q1_0, Q1_0Loader, IU4Loader>>;
    k_mul_mat_iu4_mmq<Loader, BM, BN, NWARPS, need_check><<<grid, block, 0, 0>>>(
            (const char *) d_w_raw, d_act, d_dst, M, N, nb01, n_blocks_k, dst_row_stride_floats);
    const hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        GGML_LOG_ERROR("%s: k_mul_mat_iu4_mmq failed: %s\n", __func__, hipGetErrorString(err));
        (void) hipFree(d_act); (void) hipFree(d_w_raw); (void) hipFree(d_dst);
        return false;
    }

    std::vector<float> out((size_t) (M * N));
    CUDA_CHECK(hipMemcpy(out.data(), d_dst, out.size() * sizeof(float), hipMemcpyDeviceToHost));
    (void) hipFree(d_act); (void) hipFree(d_w_raw); (void) hipFree(d_dst);

    long max_abs = 0;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            long ref = 0;
            for (int64_t k = 0; k < K; ++k) {
                ref += (long) act_logical[m][k] * (long) w_logical[n][k];
            }
            const long got = (long) out[m * N + n];
            max_abs = std::max(max_abs, std::labs(got - ref));
        }
    }
    max_abs_err_out = max_abs;
    return max_abs == 0;
}

// T163 iterate-F: RC2 speed-ceiling prototype (coordinator directive, see
// w4a4-external-intel.md APEX4 attack #1). IU4Loader's existing synthetic
// packer (pack_iu4_block) ALREADY gives a fixed unit scale (d=1.0) for
// EVERY block -- i.e. the scale is already, trivially, both per-channel
// (constant across chunks for a given weight row) AND per-token (constant
// across chunks for a given activation row). That means `ceiling_mode=true`
// is mathematically EXACT on this exact data (no format change needed to
// prototype it) -- we just need a PRODUCTION-REPRESENTATIVE size (not the
// tiny M=100,N=90,K=256 correctness-test size) and hipEvent timing to
// measure the kernel-side ceiling: how much of the gap closes if the
// per-chunk rescale is removed ENTIRELY (not just one of the two scales,
// as T163-E attempted on the real per-block-scale production path).
static bool run_ceiling_bench(int64_t M, int64_t N, int64_t K, double & ms_normal, double & ms_ceiling, double & ms_deferdw, long & max_abs_err_out) {
    std::mt19937 rng(163163); // T163
    const int64_t n_blocks_k = K / QK_IU4;
    std::uniform_int_distribution<int> actdist(-8, 7);
    std::uniform_int_distribution<int> wdist(-8, 7);

    std::vector<std::vector<int>> act_logical(M, std::vector<int>(K));
    std::vector<std::vector<int>> w_logical(N, std::vector<int>(K));
    for (int64_t m = 0; m < M; ++m) { for (int64_t k = 0; k < K; ++k) { act_logical[m][k] = actdist(rng); } }
    for (int64_t n = 0; n < N; ++n) { for (int64_t k = 0; k < K; ++k) { w_logical[n][k] = wdist(rng); } }

    std::vector<block_iu4> act_blocks((size_t) (M * n_blocks_k));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t c = 0; c < n_blocks_k; ++c) {
            pack_iu4_block(&act_logical[m][c * QK_IU4], act_blocks[m * n_blocks_k + c]);
        }
    }
    std::vector<block_iu4> w_blocks((size_t) (N * n_blocks_k));
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < n_blocks_k; ++c) {
            pack_iu4_block(&w_logical[n][c * QK_IU4], w_blocks[n * n_blocks_k + c]);
        }
    }

    void * d_w_raw = nullptr;
    block_iu4 * d_act = nullptr;
    float * d_dst_normal = nullptr;
    float * d_dst_ceiling = nullptr;
    if (hipMalloc(&d_w_raw, w_blocks.size() * sizeof(block_iu4)) != hipSuccess ||
        hipMalloc(&d_act, act_blocks.size() * sizeof(block_iu4)) != hipSuccess ||
        hipMalloc(&d_dst_normal, (size_t) (M * N) * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_dst_ceiling, (size_t) (M * N) * sizeof(float)) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    CUDA_CHECK(hipMemcpy(d_w_raw, w_blocks.data(), w_blocks.size() * sizeof(block_iu4), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_act, act_blocks.data(), act_blocks.size() * sizeof(block_iu4), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(d_dst_normal, 0, (size_t) (M * N) * sizeof(float)));
    CUDA_CHECK(hipMemset(d_dst_ceiling, 0, (size_t) (M * N) * sizeof(float)));

    constexpr int BM = MMQ_IU4_BM;
    constexpr int BN = MMQ_IU4_BN;
    constexpr int NWARPS = MMQ_IU4_NWARPS;
    const int64_t nb01 = n_blocks_k * (int64_t) sizeof(block_iu4);
    const bool need_check = (M % BM != 0) || (N % BN != 0);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
    const dim3 block(32, NWARPS, 1);

    hipEvent_t ev0, ev1, ev2, ev3;
    hipEventCreate(&ev0); hipEventCreate(&ev1); hipEventCreate(&ev2); hipEventCreate(&ev3);

    const int n_reps = 5;
    // Warmup + timed normal (ceiling_mode=false, current production math).
    for (int r = 0; r < 2; ++r) {
        if (need_check) {
            k_mul_mat_iu4_mmq<IU4Loader, BM, BN, NWARPS, true, false><<<grid, block>>>(
                    (const char *) d_w_raw, d_act, d_dst_normal, M, N, nb01, n_blocks_k, N);
        } else {
            k_mul_mat_iu4_mmq<IU4Loader, BM, BN, NWARPS, false, false><<<grid, block>>>(
                    (const char *) d_w_raw, d_act, d_dst_normal, M, N, nb01, n_blocks_k, N);
        }
    }
    CUDA_CHECK(hipDeviceSynchronize());
    hipEventRecord(ev0);
    for (int r = 0; r < n_reps; ++r) {
        if (need_check) {
            k_mul_mat_iu4_mmq<IU4Loader, BM, BN, NWARPS, true, false><<<grid, block>>>(
                    (const char *) d_w_raw, d_act, d_dst_normal, M, N, nb01, n_blocks_k, N);
        } else {
            k_mul_mat_iu4_mmq<IU4Loader, BM, BN, NWARPS, false, false><<<grid, block>>>(
                    (const char *) d_w_raw, d_act, d_dst_normal, M, N, nb01, n_blocks_k, N);
        }
    }
    hipEventRecord(ev1);
    CUDA_CHECK(hipDeviceSynchronize());

    // Warmup + timed ceiling (ceiling_mode=true, zero per-chunk rescale).
    for (int r = 0; r < 2; ++r) {
        if (need_check) {
            k_mul_mat_iu4_mmq<IU4Loader, BM, BN, NWARPS, true, true><<<grid, block>>>(
                    (const char *) d_w_raw, d_act, d_dst_ceiling, M, N, nb01, n_blocks_k, N);
        } else {
            k_mul_mat_iu4_mmq<IU4Loader, BM, BN, NWARPS, false, true><<<grid, block>>>(
                    (const char *) d_w_raw, d_act, d_dst_ceiling, M, N, nb01, n_blocks_k, N);
        }
    }
    CUDA_CHECK(hipDeviceSynchronize());
    hipEventRecord(ev2);
    for (int r = 0; r < n_reps; ++r) {
        if (need_check) {
            k_mul_mat_iu4_mmq<IU4Loader, BM, BN, NWARPS, true, true><<<grid, block>>>(
                    (const char *) d_w_raw, d_act, d_dst_ceiling, M, N, nb01, n_blocks_k, N);
        } else {
            k_mul_mat_iu4_mmq<IU4Loader, BM, BN, NWARPS, false, true><<<grid, block>>>(
                    (const char *) d_w_raw, d_act, d_dst_ceiling, M, N, nb01, n_blocks_k, N);
        }
    }
    hipEventRecord(ev3);
    CUDA_CHECK(hipDeviceSynchronize());

    float t_normal_ms = 0.0f, t_ceiling_ms = 0.0f;
    hipEventElapsedTime(&t_normal_ms, ev0, ev1);
    hipEventElapsedTime(&t_ceiling_ms, ev2, ev3);
    ms_normal  = (double) t_normal_ms  / n_reps;
    ms_ceiling = (double) t_ceiling_ms / n_reps;
    hipEventDestroy(ev0); hipEventDestroy(ev1); hipEventDestroy(ev2); hipEventDestroy(ev3);

    // T164-1b(3): defer_dw=true (weight-scale-only hoist, activation stays
    // per-chunk) on the SAME synthetic data (also valid here since
    // IU4Loader's pack_iu4_block gives constant d=1.0/block, so the weight
    // scale genuinely is chunk-invariant for this data too) -- TIMED this
    // time, to decompose the T163-F ceiling into weight-side vs
    // activation-side fractions at the isolated-GEMM level (coordinator's
    // "reality-check the speed side" ask, before committing to Phase 2 on
    // any new accuracy scheme).
    float * d_dst_deferdw = nullptr;
    CUDA_CHECK(hipMalloc(&d_dst_deferdw, (size_t) (M * N) * sizeof(float)));
    CUDA_CHECK(hipMemset(d_dst_deferdw, 0, (size_t) (M * N) * sizeof(float)));
    hipEvent_t ev4, ev5;
    hipEventCreate(&ev4); hipEventCreate(&ev5);
    for (int r = 0; r < 2; ++r) {
        if (need_check) {
            k_mul_mat_iu4_mmq<IU4Loader, BM, BN, NWARPS, true, false, true><<<grid, block>>>(
                    (const char *) d_w_raw, d_act, d_dst_deferdw, M, N, nb01, n_blocks_k, N);
        } else {
            k_mul_mat_iu4_mmq<IU4Loader, BM, BN, NWARPS, false, false, true><<<grid, block>>>(
                    (const char *) d_w_raw, d_act, d_dst_deferdw, M, N, nb01, n_blocks_k, N);
        }
    }
    CUDA_CHECK(hipDeviceSynchronize());
    hipEventRecord(ev4);
    for (int r = 0; r < n_reps; ++r) {
        if (need_check) {
            k_mul_mat_iu4_mmq<IU4Loader, BM, BN, NWARPS, true, false, true><<<grid, block>>>(
                    (const char *) d_w_raw, d_act, d_dst_deferdw, M, N, nb01, n_blocks_k, N);
        } else {
            k_mul_mat_iu4_mmq<IU4Loader, BM, BN, NWARPS, false, false, true><<<grid, block>>>(
                    (const char *) d_w_raw, d_act, d_dst_deferdw, M, N, nb01, n_blocks_k, N);
        }
    }
    hipEventRecord(ev5);
    CUDA_CHECK(hipDeviceSynchronize());
    float t_deferdw_ms = 0.0f;
    hipEventElapsedTime(&t_deferdw_ms, ev4, ev5);
    ms_deferdw = (double) t_deferdw_ms / n_reps;
    hipEventDestroy(ev4); hipEventDestroy(ev5);

    std::vector<float> out_normal((size_t) (M * N));
    std::vector<float> out_ceiling((size_t) (M * N));
    std::vector<float> out_deferdw((size_t) (M * N));
    CUDA_CHECK(hipMemcpy(out_normal.data(), d_dst_normal, out_normal.size() * sizeof(float), hipMemcpyDeviceToHost));
    CUDA_CHECK(hipMemcpy(out_ceiling.data(), d_dst_ceiling, out_ceiling.size() * sizeof(float), hipMemcpyDeviceToHost));
    CUDA_CHECK(hipMemcpy(out_deferdw.data(), d_dst_deferdw, out_deferdw.size() * sizeof(float), hipMemcpyDeviceToHost));
    (void) hipFree(d_w_raw); (void) hipFree(d_act); (void) hipFree(d_dst_normal); (void) hipFree(d_dst_ceiling); (void) hipFree(d_dst_deferdw);

    long max_abs = 0;
    long max_abs_deferdw = 0;
    for (size_t idx = 0; idx < out_normal.size(); ++idx) {
        max_abs = std::max(max_abs, (long) std::labs((long) out_normal[idx] - (long) out_ceiling[idx]));
        max_abs_deferdw = std::max(max_abs_deferdw, (long) std::labs((long) out_normal[idx] - (long) out_deferdw[idx]));
    }
    GGML_LOG_INFO("%s: defer_dw-only check vs normal: max_abs_err=%ld\n", __func__, max_abs_deferdw);
    max_abs_err_out = max_abs;
    return max_abs == 0 && max_abs_deferdw == 0;
}

} // namespace ggml_cuda_mul_mat_iu4_mmq_selftest_detail

bool ggml_cuda_mul_mat_iu4_mmq_selftest() {
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        return true; // vacuously true off RDNA4, mma_iu4() is NO_DEVICE_CODE there
    }

    using namespace ggml_cuda_mul_mat_iu4_mmq_selftest_detail;
    std::mt19937 rng(162162); // card 162
    bool all_pass = true;
    long worst_iu4 = 0, worst_q2_0 = 0;
    const int n_trials = 10;
    // M, N deliberately not multiples of BM/BN (boundary clamp); K spans
    // several double-buffer iterations. Exercises need_check=true (the
    // template bool added in T162 iterate-1, see k_mul_mat_iu4_mmq comment).
    for (int t = 0; t < n_trials; ++t) {
        long e = 0;
        if (!run_trial<LoaderKind::IU4, true>(rng, /*M=*/100, /*N=*/90, /*K=*/256, e)) { all_pass = false; }
        worst_iu4 = std::max(worst_iu4, e);
    }
    for (int t = 0; t < n_trials; ++t) {
        long e = 0;
        if (!run_trial<LoaderKind::Q2_0, true>(rng, /*M=*/100, /*N=*/90, /*K=*/256, e)) { all_pass = false; }
        worst_q2_0 = std::max(worst_q2_0, e);
    }
    // M=128, N=128 ARE exact multiples of BM/BN=64 -- exercises the new
    // need_check=false (zero-boundary-overhead) instantiation, which is what
    // launch() actually selects for every real Bonsai-27B weight tensor.
    long worst_iu4_nc = 0, worst_q2_0_nc = 0;
    for (int t = 0; t < n_trials; ++t) {
        long e = 0;
        if (!run_trial<LoaderKind::IU4, false>(rng, /*M=*/128, /*N=*/128, /*K=*/256, e)) { all_pass = false; }
        worst_iu4_nc = std::max(worst_iu4_nc, e);
    }
    for (int t = 0; t < n_trials; ++t) {
        long e = 0;
        if (!run_trial<LoaderKind::Q2_0, false>(rng, /*M=*/128, /*N=*/128, /*K=*/256, e)) { all_pass = false; }
        worst_q2_0_nc = std::max(worst_q2_0_nc, e);
    }
    // T162 Q1_0 wiring: same need_check=true/false + ragged-tail coverage as
    // Q2_0 above (Q1_0 shares the same QK1_0==128 / GROUP-of-4 layout).
    long worst_q1_0 = 0, worst_q1_0_nc = 0, worst_q1_0_ragged = 0;
    for (int t = 0; t < n_trials; ++t) {
        long e = 0;
        if (!run_trial<LoaderKind::Q1_0, true>(rng, /*M=*/100, /*N=*/90, /*K=*/256, e)) { all_pass = false; }
        worst_q1_0 = std::max(worst_q1_0, e);
    }
    for (int t = 0; t < n_trials; ++t) {
        long e = 0;
        if (!run_trial<LoaderKind::Q1_0, false>(rng, /*M=*/128, /*N=*/128, /*K=*/256, e)) { all_pass = false; }
        worst_q1_0_nc = std::max(worst_q1_0_nc, e);
    }
    for (int t = 0; t < n_trials; ++t) {
        long e = 0;
        if (!run_trial<LoaderKind::Q1_0, true>(rng, /*M=*/100, /*N=*/90, /*K=*/224, e)) { all_pass = false; }
        worst_q1_0_ragged = std::max(worst_q1_0_ragged, e);
    }
    GGML_LOG_INFO("%s: k_mul_mat_iu4_mmq, %d trials/loader (M=100,N=90,K=256 need_check=true) -> %s (max_abs_err iu4=%ld, q2_0=%ld, q1_0=%ld); "
                   "(M=128,N=128,K=256 need_check=false) -> (max_abs_err iu4=%ld, q2_0=%ld, q1_0=%ld); "
                   "(M=100,N=90,K=224 ragged, q1_0) -> (max_abs_err=%ld)\n",
                   __func__, n_trials, all_pass ? "PASS" : "FAIL",
                   worst_iu4, worst_q2_0, worst_q1_0, worst_iu4_nc, worst_q2_0_nc, worst_q1_0_nc, worst_q1_0_ragged);

    // T163 iterate-F: RC2 speed-ceiling prototype (see run_ceiling_bench's
    // comment). Production-representative size: M=512 (matches pp512's
    // token count), N=5120/K=5120 (matches Bonsai-27B's attn_qkv/ffn hidden
    // dim -- see gguf tensor dump in the KB write-up).
    {
        double ms_normal = 0.0, ms_ceiling = 0.0, ms_deferdw = 0.0;
        long   ceiling_err = 0;
        const bool ceiling_ok = run_ceiling_bench(/*M=*/512, /*N=*/5120, /*K=*/5120, ms_normal, ms_ceiling, ms_deferdw, ceiling_err);
        if (!ceiling_ok) { all_pass = false; }
        const double speedup_full     = ms_normal > 0.0 ? ms_normal / ms_ceiling  : 0.0;
        const double speedup_deferdw  = ms_normal > 0.0 ? ms_normal / ms_deferdw  : 0.0;
        // T164-1b(3): decompose the full ceiling into weight-side (deferdw)
        // vs activation-side (the remainder) fractions, on a log scale (each
        // step's ms reduction, as a fraction of the total ms reduction from
        // normal->ceiling) so "weight_frac + act_frac == 1" exactly.
        const double total_reduction_ms  = ms_normal - ms_ceiling;
        const double weight_reduction_ms = ms_normal - ms_deferdw;
        const double weight_frac = total_reduction_ms > 0.0 ? weight_reduction_ms / total_reduction_ms : 0.0;
        GGML_LOG_INFO("%s: RC2 ceiling-bench (M=512,N=5120,K=5120, IU4Loader synthetic per-channel/per-token data) -> %s "
                       "(max_abs_err=%ld) normal=%.4fms deferdw(weight-only)=%.4fms ceiling(both)=%.4fms "
                       "speedup_deferdw=%.3fx speedup_full=%.3fx weight_side_frac_of_ceiling=%.1f%%\n",
                       __func__, ceiling_ok ? "PASS" : "FAIL", ceiling_err, ms_normal, ms_deferdw, ms_ceiling,
                       speedup_deferdw, speedup_full, 100.0 * weight_frac);
    }
    return all_pass;
}
