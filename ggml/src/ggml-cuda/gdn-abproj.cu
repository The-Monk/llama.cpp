#include "gdn-abproj.cuh"
#include "mma.cuh"

static __device__ __forceinline__ nv_bfloat162 abproj_bf2(const float a, const float b) {
    nv_bfloat162 r;
    r.x = __float2bfloat16(a);   // round-to-nearest-even, as convert_unary<float, bf16>
    r.y = __float2bfloat16(b);
    return r;
}

// One block = 16 tokens x (48 + 48) outputs. 8 waves split K; each wave accumulates 6 WMMA 16x16 tiles (3 per matrix) over its
// contiguous K slice, then the partial tiles are reduced through LDS. RDNA4 only (gfx12 bf16 WMMA).
#define ABPROJ_NWAVES 8
#define ABPROJ_NRT    3     // 48 features = 3 row tiles per matrix

static __global__ void __launch_bounds__(ABPROJ_NWAVES * 32) gdn_ab_proj_kernel(
        const float * __restrict__ x, const nv_bfloat162 * __restrict__ wa, const nv_bfloat162 * __restrict__ wb,
        float * __restrict__ da, float * __restrict__ db,
        const int K, const int n_tok, const int64_t sx, const int64_t sw, const int64_t sda, const int64_t sdb) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, nv_bfloat162> tile_ab;
    typedef tile<16, 16, float>       tile_c;

    __shared__ float red[ABPROJ_NWAVES][2 * ABPROJ_NRT][8][32];

    const int wave   = threadIdx.x / 32;
    const int lane   = threadIdx.x % 32;
    const int tok0   = blockIdx.x * 16;
    const int ksteps = K / 16;
    const int ks0    = (wave * ksteps) / ABPROJ_NWAVES;
    const int ks1    = ((wave + 1) * ksteps) / ABPROJ_NWAVES;

    tile_c acc[2 * ABPROJ_NRT];

    const int  tok_l  = tok0 + lane % 16;
    const bool tok_ok = tok_l < n_tok;
    const float * xrow = x + (int64_t) (tok_ok ? tok_l : 0) * sx + (lane / 16) * 8;

    // 4 k-steps per iteration: the 8 float4 activation loads of the unrolled steps are issued before their WMMAs (memory-level parallelism;
    // v1 without unroll read the 21 MB activation at ~330 GB/s, half of the card)
#pragma unroll 4
    for (int ks = ks0; ks < ks1; ++ks) {
        tile_ab B;
        {
            float4 v0 = make_float4(0.f, 0.f, 0.f, 0.f), v1 = v0;
            if (tok_ok) {
                v0 = *(const float4 *) (xrow + ks * 16);
                v1 = *(const float4 *) (xrow + ks * 16 + 4);
            }
            B.x[0] = abproj_bf2(v0.x, v0.y);
            B.x[1] = abproj_bf2(v0.z, v0.w);
            B.x[2] = abproj_bf2(v1.x, v1.y);
            B.x[3] = abproj_bf2(v1.z, v1.w);
        }
#pragma unroll
        for (int t = 0; t < 2 * ABPROJ_NRT; ++t) {
            const nv_bfloat162 * w = (t < ABPROJ_NRT ? wa : wb) + (int64_t) ((t % ABPROJ_NRT) * 16 + lane % 16) * sw + ks * 8 + (lane / 16) * 4;
            tile_ab A;
            const uint4 u = *(const uint4 *) w;
            *(uint4 *) &A.x[0] = u;
            mma(acc[t], A, B);
        }
    }

#pragma unroll
    for (int t = 0; t < 2 * ABPROJ_NRT; ++t) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            red[wave][t][l][lane] = acc[t].x[l];
        }
    }
    __syncthreads();

    // idx -> (t, token, f): f fastest so a wave stores contiguous features of one token
    for (int idx = threadIdx.x; idx < 2 * ABPROJ_NRT * 16 * 16; idx += ABPROJ_NWAVES * 32) {
        const int f   = idx % 16;
        const int tk  = (idx / 16) % 16;
        const int t   = idx / 256;
        const int ln  = (f / 8) * 16 + tk;   // C fragment (RDNA4): lane%16 = token, 8*(lane/16)+l = feature
        const int l   = f % 8;
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < ABPROJ_NWAVES; ++w) {
            s += red[w][t][l][ln];
        }
        if (tok0 + tk < n_tok) {
            if (t < ABPROJ_NRT) {
                da[(int64_t) (tok0 + tk) * sda + t * 16 + f] = s;
            } else {
                db[(int64_t) (tok0 + tk) * sdb + (t - ABPROJ_NRT) * 16 + f] = s;
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, wa, wb, da, db, K, n_tok, sx, sw, sda, sdb);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
}

bool ggml_cuda_gdn_ab_proj_supported(const ggml_tensor * a, const ggml_tensor * b, int cc) {
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        return false;
    }
    const ggml_tensor * wa  = a->src[0];
    const ggml_tensor * wb  = b->src[0];
    const ggml_tensor * act = a->src[1];
    return wa->type == GGML_TYPE_BF16 && wb->type == GGML_TYPE_BF16 && act->type == GGML_TYPE_F32 && b->src[1] == act &&
           a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
           wa->ne[1] == 16 * ABPROJ_NRT && wb->ne[1] == wa->ne[1] && wb->ne[0] == wa->ne[0] && wa->ne[0] % 16 == 0 &&
           wa->ne[2] == 1 && wa->ne[3] == 1 && wb->ne[2] == 1 && wb->ne[3] == 1 && act->ne[2] == 1 && act->ne[3] == 1 &&
           act->ne[0] == wa->ne[0] && act->ne[1] >= 64 && ggml_is_contiguous(act) &&
           ggml_are_same_stride(wa, wb) && wa->nb[0] == 2 && wa->nb[1] % 16 == 0 &&
           (uintptr_t) wa->data % 16 == 0 && (uintptr_t) wb->data % 16 == 0 && (uintptr_t) act->data % 16 == 0 &&
           a->ne[0] == wa->ne[1] && b->ne[0] == wa->ne[1] && a->ne[1] == act->ne[1] && b->ne[1] == act->ne[1] &&
           a->ne[2] == 1 && a->ne[3] == 1 && b->ne[2] == 1 && b->ne[3] == 1 &&
           a->nb[0] == 4 && b->nb[0] == 4 && a->nb[1] % 4 == 0 && b->nb[1] % 4 == 0;
}

void ggml_cuda_gdn_ab_proj(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b) {
    GGML_ASSERT(ggml_cuda_gdn_ab_proj_supported(a, b, ggml_cuda_info().devices[ctx.device].cc));
    const ggml_tensor * wa  = a->src[0];
    const ggml_tensor * wb  = b->src[0];
    const ggml_tensor * act = a->src[1];
    const int n_tok = (int) act->ne[1];
    const dim3 grid((n_tok + 15) / 16, 1, 1);
    const dim3 block(ABPROJ_NWAVES * 32, 1, 1);
    gdn_ab_proj_kernel<<<grid, block, 0, ctx.stream()>>>(
        (const float *) act->data, (const nv_bfloat162 *) wa->data, (const nv_bfloat162 *) wb->data,
        (float *) a->data, (float *) b->data, (int) wa->ne[0], n_tok,
        (int64_t) (act->nb[1] / 4), (int64_t) (wa->nb[1] / 4), (int64_t) (a->nb[1] / 4), (int64_t) (b->nb[1] / 4));
    CUDA_CHECK(cudaGetLastError());
}
