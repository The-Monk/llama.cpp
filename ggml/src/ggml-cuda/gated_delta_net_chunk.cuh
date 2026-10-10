// T405: chunked gated delta rule for prefill (default on; GGML_GDN_CHUNK=0 disables).
// Scalar gate, S_v == 128, RDNA4 WMMA. Writes the final state only; with K > 1 snapshots the caller runs
// it over the first n_tokens - K tokens and the sequential kernel over the tail. Included by gated_delta_net.cu.
//
// The sequential kernel computes, per (seq, head), with S[i][c] (key i, value c):
//   kv = S^T k_t,  delta = beta_t (v_t - g_t kv),  S = g_t S + k_t delta^T,  o_t = scale * S^T q_t
// i.e. S_t = g_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T, g_t = exp(gl_t).
// Chunk form (FLA chunk_gated_delta_rule / WY): within a chunk of C tokens, b_t = cumsum gl,
// pseudo-values u_t with S_t = exp(b_t) S_0 + sum_{j<=t} exp(b_t - b_j) k_j u_j^T:
//   (I + L) U = diag(beta) (V - diag(exp b) K S_0),  L[t][j] = beta_t exp(b_t - b_j) k_t.k_j  (j < t)
//   T = (I + L)^-1 diag(beta)  ->  U = T X,  X = V - diag(exp b) K S_0
//   O = scale * (diag(exp b) Q S_0 + Pm U),  Pm[t][j] = exp(b_t - b_j) q_t.k_j  (j <= t)
//   S_C = exp(b_C) S_0 + K^T diag(exp(b_C - b)) U
// Only differences b_t - b_j (<= 0) are exponentiated, so nothing overflows for decaying gates.
//
// gdn_chunk_prep: one workgroup per (chunk, k-head, seq), fully parallel. Builds K K^T and Q K^T
//   once per k-head (WMMA), then for each v-head sharing it: gates, cumsum, blocked inverse of
//   I + L (16x16 diagonal blocks by forward substitution, off-diagonal blocks by block
//   substitution), writes T, Pm and per-token meta.
// gdn_chunk_scan: one wave per (seq, v-head, 16-column tile of the value dim), sequential over
//   chunks. Columns of the state are independent, and the WMMA D layout equals the B layout, so
//   the state tile, X and U never leave registers; only K, Q, T, Pm are loaded. One barrier per
//   chunk (double-buffered meta).
// Matmuls: v_wmma_f32_16x16x16_f16, fp32 accumulate. PREC 3 splits both operands into f16 hi+lo
// (a*b ~ ah*bh + ah*bl + al*bh, ~2^-22 relative); PREC 1 is plain f16 (GGML_GDN_CHUNK_PREC=1).
// gfx12 wave32 layout (verified on gfx1201): A[m][k] lane l: m = l%16, k = 8*(l/16)+r;
// B[k][n] lane l: n = l%16, k = 8*(l/16)+r; D[m][n] lane l: n = l%16, m = 8*(l/16)+r.
#pragma once

#define GDN_CH_C     32   // chunk length (tokens), multiple of 16
#define GDN_CH_D     128  // S_v
#define GDN_CH_WAVES 8    // scan: 16-column tiles (waves) per workgroup (8 = one head per workgroup)

#if defined(GGML_USE_HIP) && defined(RDNA4)
#define GDN_CHUNK_AVAILABLE 1
typedef _Float16 gdn_h8 __attribute__((ext_vector_type(8)));
typedef float    gdn_f8 __attribute__((ext_vector_type(8)));

template <typename X>
static __device__ __forceinline__ void gdn_split8(const X & x, gdn_h8 & hi, gdn_h8 & lo) {
#pragma unroll
    for (int r = 0; r < 8; r++) {
        const _Float16 h = (_Float16) x[r];
        const float   hf = gdn_opaque((float) h);
        hi[r] = h;
        lo[r] = (_Float16) (x[r] - hf);
    }
}

template <int PREC>
static __device__ __forceinline__ gdn_f8 gdn_mma(gdn_f8 acc, const gdn_h8 & ah, const gdn_h8 & al,
                                                const gdn_h8 & bh, const gdn_h8 & bl) {
    if constexpr (PREC >= 3) {
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, bl, acc);
    }
    if constexpr (PREC >= 2) {
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(al, bh, acc);
    }
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, bh, acc);
    return acc;
}

static __device__ __forceinline__ int64_t gdn_min64(int64_t a, int64_t b) {
    return a < b ? a : b;
}

// 8 contiguous floats (16-byte aligned)
static __device__ __forceinline__ void gdn_load8(const float * p, float (&a)[8], const float s) {
    const float4 x0 = reinterpret_cast<const float4 *>(p)[0];
    const float4 x1 = reinterpret_cast<const float4 *>(p)[1];
    a[0] = s * x0.x; a[1] = s * x0.y; a[2] = s * x0.z; a[3] = s * x0.w;
    a[4] = s * x1.x; a[5] = s * x1.y; a[6] = s * x1.z; a[7] = s * x1.w;
}
#endif // defined(GGML_USE_HIP) && defined(RDNA4)

// meta per (seq, v-head, chunk, token): [0] k scale, [1] q scale, [2] b = cumsum(log g), [3] beta
template <bool FusedBA, bool L2Norm>
__global__ void __launch_bounds__(256, 1)
gdn_chunk_prep(const float * q, const float * k, const float * g, const float * beta,
               float * Tbuf, float * Pbuf, float * meta,
               int64_t H, int64_t HK, int64_t n_tokens, int n_chunks,
               int64_t sq1, int64_t sq2, int64_t sq3, int64_t sb1, int64_t sb2, int64_t sb3,
               const uint3 rq3_magic, const float * ssm_dt, const float * ssm_a, float l2norm_eps) {
#ifdef GDN_CHUNK_AVAILABLE
    constexpr int C = GDN_CH_C;
    constexpr int D = GDN_CH_D;
    const int ch   = blockIdx.x;
    const int kh   = blockIdx.y;
    const int s    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int w    = tid / 32;
    const int ln   = lane % 16;
    const int lh   = 8 * (lane / 16);
    const int64_t t0   = (int64_t) ch * C;
    const int64_t tmax = n_tokens - 1;

    const uint32_t iq3 = fastdiv((uint32_t) s, rq3_magic);
    const float * qb = q + iq3 * sq3 + kh * sq1;
    const float * kb = k + iq3 * sq3 + kh * sq1;

    __shared__ float sG[C][C + 1];   // K K^T (scaled)
    __shared__ float sP[C][C + 1];   // Q K^T (scaled)
    __shared__ float sT[C][C + 1];   // (I + L)^-1
    __shared__ float sM[C / 16][16][17];
    __shared__ float s_sk[C], s_sq[C], s_beta[C], s_gl[C], s_b[C];
    constexpr int NT = C / 16;                 // token tiles per chunk
    constexpr int NJOB = NT * (NT + 1);        // lower tiles of G and of P

    ggml_cuda_pdl_sync();

    // per-token q/k scales (shared by all v-heads of this k-head); wave w: tokens w, w+8, ...
    for (int tl = w; tl < C; tl += 8) {
        const int64_t t = t0 + tl;
        float sk = 0.0f, sq = 0.0f;
        if (t < n_tokens) {
            if constexpr (L2Norm) {
                float ks = 0.0f, qs = 0.0f;
#pragma unroll
                for (int r = 0; r < D / 32; r++) {
                    const float kv = kb[t * sq2 + r * 32 + lane];
                    const float qv = qb[t * sq2 + r * 32 + lane];
                    ks = gdn_opaque(ks + kv * kv);
                    qs = gdn_opaque(qs + qv * qv);
                }
                ks = warp_reduce_sum<32>(ks);
                qs = warp_reduce_sum<32>(qs);
                sk = rsqrtf(fmaxf(ks, l2norm_eps * l2norm_eps));
                sq = rsqrtf(fmaxf(qs, l2norm_eps * l2norm_eps));
            } else {
                sk = 1.0f;
                sq = 1.0f;
            }
        }
        if (lane == 0) {
            s_sk[tl] = sk;
            s_sq[tl] = sq;
        }
    }
    __syncthreads();

    // G = K K^T and P = Q K^T, lower 16x16 tiles only, spread over 8 waves
    for (int job = w; job < NJOB; job += 8) {
        const bool isP = job >= NJOB / 2;
        int jj = isP ? job - NJOB / 2 : job;
        int ti = 0;
        while (jj > ti) { jj -= ti + 1; ti++; }
        const int tj = jj;
        const float * Ab = isP ? qb : kb;
        const int64_t ta = gdn_min64(t0 + ti * 16 + ln, tmax);
        const int64_t tb = gdn_min64(t0 + tj * 16 + ln, tmax);
        gdn_f8 acc = {};
#pragma unroll 2
        for (int ks = 0; ks < D / 16; ks++) {
            float a[8], b[8];
            gdn_load8(Ab + ta * sq2 + ks * 16 + lh, a, 1.0f);
            gdn_load8(kb + tb * sq2 + ks * 16 + lh, b, 1.0f);
            gdn_h8 ah, al, bh, bl;
            gdn_split8(a, ah, al);
            gdn_split8(b, bh, bl);
            acc = gdn_mma<3>(acc, ah, al, bh, bl);
        }
        const int jl = tj * 16 + ln;
#pragma unroll
        for (int r = 0; r < 8; r++) {
            const int tl = ti * 16 + lh + r;
            if (isP) {
                sP[tl][jl] = acc[r] * (s_sq[tl] * s_sk[jl]);
            } else {
                sG[tl][jl] = acc[r] * (s_sk[tl] * s_sk[jl]);
            }
        }
    }
    __syncthreads();

#define GDN_L(R, J) (s_beta[R] * expf(s_b[R] - s_b[J]) * sG[R][J])

    const int64_t rep = H / HK;
    for (int64_t m = 0; m < rep; m++) {
        const int64_t h = kh + m * HK;
        if (tid < C) {
            const int64_t t = t0 + tid;
            float bt = 0.0f, gl = 0.0f;
            if (t < n_tokens) {
                const int64_t gbo = s * sb3 + t * sb2 + h * sb1;
                if constexpr (FusedBA) {
                    bt = 1.0f / (1.0f + expf(-beta[gbo]));
                    const float ab = g[gbo] + ssm_dt[h];
                    const float sp = (ab > 20.0f) ? ab : logf(1.0f + expf(ab));
                    gl = sp * ssm_a[h];
                } else {
                    bt = beta[gbo];
                    gl = g[gbo];
                }
            }
            s_beta[tid] = bt;
            s_gl[tid]   = gl;
        }
        __syncthreads();
        if (tid < C) {
            float acc = 0.0f;
            for (int j = 0; j <= tid; j++) {
                acc += s_gl[j];
            }
            s_b[tid] = acc;
        }
        __syncthreads();

        // diagonal blocks: X_bb = (I + L_bb)^-1, one column per thread
        if (tid < C) {
            const int bb = tid / 16;
            const int c  = tid % 16;
            float x[16];
#pragma unroll
            for (int r = 0; r < 16; r++) {
                float acc = r == c ? 1.0f : 0.0f;
#pragma unroll
                for (int j = 0; j < r; j++) {
                    acc -= GDN_L(bb * 16 + r, bb * 16 + j) * x[j];
                }
                x[r] = acc;
            }
#pragma unroll
            for (int r = 0; r < 16; r++) {
                sT[bb * 16 + r][bb * 16 + c] = x[r];
            }
        }
        __syncthreads();
        // off-diagonal blocks: X_ij = -X_ii sum_{kb=j}^{i-1} L_{i,kb} X_{kb,j}
        for (int i = 1; i < NT; i++) {
            for (int e = tid; e < i * 256; e += 256) {
                const int j = e / 256, r = (e / 16) % 16, c = e % 16;
                float acc = 0.0f;
                for (int kk = j; kk < i; kk++) {
#pragma unroll
                    for (int qq = 0; qq < 16; qq++) {
                        acc += GDN_L(i * 16 + r, kk * 16 + qq) * sT[kk * 16 + qq][j * 16 + c];
                    }
                }
                sM[j][r][c] = acc;
            }
            __syncthreads();
            for (int e = tid; e < i * 256; e += 256) {
                const int j = e / 256, r = (e / 16) % 16, c = e % 16;
                float acc = 0.0f;
#pragma unroll
                for (int qq = 0; qq < 16; qq++) {
                    acc += sT[i * 16 + r][i * 16 + qq] * sM[j][qq][c];
                }
                sT[i * 16 + r][j * 16 + c] = -acc;
            }
            __syncthreads();
        }

        const int64_t blk = ((int64_t) s * H + h) * n_chunks + ch;
        float * Tg = Tbuf + blk * C * C;
        float * Pg = Pbuf + blk * C * C;
        float * Mg = meta + blk * C * 4;
        for (int e = tid; e < C * C; e += 256) {
            const int tl = e / C, jl = e % C;
            if (jl / 16 <= tl / 16) {
                // T = (I + L)^-1 diag(beta); the diagonal blocks of X are lower triangular
                Tg[e] = jl <= tl ? sT[tl][jl] * s_beta[jl] : 0.0f;
                Pg[e] = jl <= tl ? expf(s_b[tl] - s_b[jl]) * sP[tl][jl] : 0.0f;
            }
        }
        if (tid < C) {
            Mg[tid * 4 + 0] = s_sk[tid];
            Mg[tid * 4 + 1] = s_sq[tid];
            Mg[tid * 4 + 2] = s_b[tid];
            Mg[tid * 4 + 3] = s_beta[tid];
        }
        __syncthreads();
    }
#undef GDN_L
#else
    GGML_UNUSED_VARS(q, k, g, beta, Tbuf, Pbuf, meta, H, HK, n_tokens, n_chunks, sq1, sq2, sq3, sb1, sb2, sb3,
                     rq3_magic, ssm_dt, ssm_a, l2norm_eps);
    NO_DEVICE_CODE;
#endif
}

// grid (H, n_seqs, D / (16 * GDN_CH_WAVES)), one wave per 16-column tile
template <int PREC>
__global__ void __launch_bounds__(GDN_CH_WAVES * 32, 1)
gdn_chunk_scan(const float * q, const float * k, const float * v, const float * curr_state,
               const float * Tbuf, const float * Pbuf, const float * meta,
               float * dst, float * state,
               int64_t H, int64_t n_tokens, int64_t dst_tokens, int n_chunks,
               int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
               const uint3 neqk1_magic, const uint3 rq3_magic, float scale) {
#ifdef GDN_CHUNK_AVAILABLE
    constexpr int C = GDN_CH_C;
    constexpr int D = GDN_CH_D;
    const int h    = blockIdx.x;
    const int s    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int w    = tid / 32;
    const int c0   = (blockIdx.z * GDN_CH_WAVES + w) * 16;
    const int ln   = lane % 16;
    const int lh   = 8 * (lane / 16);
    const int64_t tmax = n_tokens - 1;

    const uint32_t iq1 = fastmodulo((uint32_t) h, neqk1_magic);
    const uint32_t iq3 = fastdiv((uint32_t) s, rq3_magic);
    const float * qb = q + iq3 * sq3 + iq1 * sq1;
    const float * kb = k + iq3 * sq3 + iq1 * sq1;
    const float * vb = v + s * sv3 + h * sv1 + c0 + ln;

    // A operands of the chunk, shared by the workgroup's waves, pre-scaled and split into f16 hi/lo.
    // K/Q rows are unpadded with the 8-half groups XOR-swizzled by row (conflict-free b128 reads);
    // K^T and T/Pm use a 40-half pitch (80 B: 16 lanes hit 16 disjoint 4-bank groups).
    constexpr int PT = C + 8;
    constexpr int NTH = GDN_CH_WAVES * 32;
    __shared__ __align__(16) _Float16 sKh[C][D], sKl[C][D], sQh[C][D], sQl[C][D];
    __shared__ __align__(16) _Float16 sKTh[D][PT], sKTl[D][PT];
    __shared__ __align__(16) _Float16 sTh[C][PT], sTl[C][PT], sPh[C][PT], sPl[C][PT];
    __shared__ float m_eb[C], m_ebc[C];
    auto swz = [](int row, int col) { return (((col >> 3) ^ (row & 15)) << 3) | (col & 7); };

    const int64_t soff = ((int64_t) s * H + h) * D * D + (int64_t) (c0 + ln) * D;
    curr_state += soff;
    state      += soff;
    dst        += ((int64_t) s * dst_tokens * H + h) * D + c0 + ln;

    ggml_cuda_pdl_sync();

    // S[it]: rows i = it*16 + lh + r, column c0 + ln (D layout == B layout for k-step it)
    gdn_f8 S[D / 16];
#pragma unroll
    for (int it = 0; it < D / 16; it++) {
#pragma unroll
        for (int r = 0; r < 8; r++) {
            S[it][r] = curr_state[it * 16 + lh + r];
        }
    }

    for (int ch = 0; ch < n_chunks; ch++) {
        const int64_t t0  = (int64_t) ch * C;
        const int64_t blk = ((int64_t) s * H + h) * n_chunks + ch;
        const float * Tg = Tbuf + blk * C * C;
        const float * Pg = Pbuf + blk * C * C;
        const float * Mg = meta + blk * C * 4;

        __syncthreads();   // previous chunk is done with the staging buffers
        if (tid < C) {
            const float bC = Mg[(C - 1) * 4 + 2];
            const float bt = Mg[tid * 4 + 2];
            m_eb[tid]  = expf(bt);
            m_ebc[tid] = expf(bC - bt);
        }
        // K, Q rows (scaled by their per-token norm scale), coalesced float4
        for (int e = tid; e < C * D / 4; e += NTH) {
            const int     tl = e / (D / 4);
            const int     i4 = (e % (D / 4)) * 4;
            const int64_t t  = gdn_min64(t0 + tl, tmax);
            const float4 kx = *reinterpret_cast<const float4 *>(kb + t * sq2 + i4);
            const float4 qx = *reinterpret_cast<const float4 *>(qb + t * sq2 + i4);
            const float  skv = Mg[tl * 4 + 0];
            const float  sqv = Mg[tl * 4 + 1];
            const float ka[4] = { skv * kx.x, skv * kx.y, skv * kx.z, skv * kx.w };
            const float qa[4] = { sqv * qx.x, sqv * qx.y, sqv * qx.z, sqv * qx.w };
            const int ip = swz(tl, i4);
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const _Float16 khr = (_Float16) ka[r];
                const _Float16 qhr = (_Float16) qa[r];
                const _Float16 klr = (_Float16) (ka[r] - gdn_opaque((float) khr));
                sKh[tl][ip + r]     = khr;
                sKl[tl][ip + r]     = klr;
                sKTh[i4 + r][tl]    = khr;
                sKTl[i4 + r][tl]    = klr;
                sQh[tl][ip + r]     = qhr;
                sQl[tl][ip + r]     = (_Float16) (qa[r] - gdn_opaque((float) qhr));
            }
        }
        // T, Pm (lower 16x16 blocks only; the others are never read)
        for (int e = tid; e < C * C / 4; e += NTH) {
            const int tl = e / (C / 4);
            const int j4 = (e % (C / 4)) * 4;
            if (j4 / 16 > tl / 16) {
                continue;
            }
            const float4 tx = *reinterpret_cast<const float4 *>(Tg + tl * C + j4);
            const float4 px = *reinterpret_cast<const float4 *>(Pg + tl * C + j4);
            const float ta[4] = { tx.x, tx.y, tx.z, tx.w };
            const float pa[4] = { px.x, px.y, px.z, px.w };
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const _Float16 thr = (_Float16) ta[r];
                const _Float16 phr = (_Float16) pa[r];
                sTh[tl][j4 + r] = thr;
                sTl[tl][j4 + r] = (_Float16) (ta[r] - gdn_opaque((float) thr));
                sPh[tl][j4 + r] = phr;
                sPl[tl][j4 + r] = (_Float16) (pa[r] - gdn_opaque((float) phr));
            }
        }
        __syncthreads();
        const float * eb  = m_eb;
        const float * ebc = m_ebc;

        // step 1: KS = Kn S0, QS = Qn S0 (token tiles tt, this column tile)
        gdn_f8 KS[C / 16], QS[C / 16];
#pragma unroll
        for (int tt = 0; tt < C / 16; tt++) {
            KS[tt] = gdn_f8{};
            QS[tt] = gdn_f8{};
        }
#pragma unroll
        for (int ks = 0; ks < D / 16; ks++) {
            gdn_h8 bh, bl;
            gdn_split8(S[ks], bh, bl);
#pragma unroll
            for (int tt = 0; tt < C / 16; tt++) {
                const int tl = tt * 16 + ln;
                const int i0 = swz(tl, ks * 16 + lh);
                KS[tt] = gdn_mma<PREC>(KS[tt], *reinterpret_cast<const gdn_h8 *>(&sKh[tl][i0]),
                                       *reinterpret_cast<const gdn_h8 *>(&sKl[tl][i0]), bh, bl);
                QS[tt] = gdn_mma<PREC>(QS[tt], *reinterpret_cast<const gdn_h8 *>(&sQh[tl][i0]),
                                       *reinterpret_cast<const gdn_h8 *>(&sQl[tl][i0]), bh, bl);
            }
        }

        // step 2: X = V - diag(exp b) KS (in place)
        gdn_h8 Bh[C / 16], Bl[C / 16];
#pragma unroll
        for (int tt = 0; tt < C / 16; tt++) {
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const int     tl = tt * 16 + lh + r;
                const int64_t t  = t0 + tl;
                const float  vv  = t < n_tokens ? vb[t * sv2] : 0.0f;
                KS[tt][r] = vv - eb[tl] * KS[tt][r];
            }
            gdn_split8(KS[tt], Bh[tt], Bl[tt]);
        }

        // step 3: U = T X, T lower triangular by 16x16 blocks (KS now holds U)
#pragma unroll
        for (int tt = 0; tt < C / 16; tt++) {
            gdn_f8 acc = {};
#pragma unroll
            for (int jt = 0; jt <= tt; jt++) {
                acc = gdn_mma<PREC>(acc, *reinterpret_cast<const gdn_h8 *>(&sTh[tt * 16 + ln][jt * 16 + lh]),
                                    *reinterpret_cast<const gdn_h8 *>(&sTl[tt * 16 + ln][jt * 16 + lh]), Bh[jt], Bl[jt]);
            }
            KS[tt] = acc;
        }
#pragma unroll
        for (int tt = 0; tt < C / 16; tt++) {
            gdn_split8(KS[tt], Bh[tt], Bl[tt]);
        }

        // step 4: O = scale * (diag(exp b) QS + Pm U)
#pragma unroll
        for (int tt = 0; tt < C / 16; tt++) {
            gdn_f8 acc = {};
#pragma unroll
            for (int jt = 0; jt <= tt; jt++) {
                acc = gdn_mma<PREC>(acc, *reinterpret_cast<const gdn_h8 *>(&sPh[tt * 16 + ln][jt * 16 + lh]),
                                    *reinterpret_cast<const gdn_h8 *>(&sPl[tt * 16 + ln][jt * 16 + lh]), Bh[jt], Bl[jt]);
            }
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const int     tl = tt * 16 + lh + r;
                const int64_t t  = t0 + tl;
                if (t < n_tokens) {
                    dst[t * D * H] = (eb[tl] * QS[tt][r] + acc[r]) * scale;
                }
            }
        }

        // step 5: S = exp(b_C) S + Kn^T diag(exp(b_C - b)) U
#pragma unroll
        for (int tt = 0; tt < C / 16; tt++) {
#pragma unroll
            for (int r = 0; r < 8; r++) {
                KS[tt][r] *= ebc[tt * 16 + lh + r];
            }
            gdn_split8(KS[tt], Bh[tt], Bl[tt]);
        }
        const float ebC = eb[C - 1];
#pragma unroll
        for (int it = 0; it < D / 16; it++) {
            S[it] *= ebC;
            const int i = it * 16 + ln;
#pragma unroll
            for (int kt = 0; kt < C / 16; kt++) {
                S[it] = gdn_mma<PREC>(S[it], *reinterpret_cast<const gdn_h8 *>(&sKTh[i][kt * 16 + lh]),
                                      *reinterpret_cast<const gdn_h8 *>(&sKTl[i][kt * 16 + lh]), Bh[kt], Bl[kt]);
            }
        }
    }

#pragma unroll
    for (int it = 0; it < D / 16; it++) {
#pragma unroll
        for (int r = 0; r < 8; r++) {
            state[it * 16 + lh + r] = S[it][r];
        }
    }
#else
    GGML_UNUSED_VARS(q, k, v, curr_state, Tbuf, Pbuf, meta, dst, state, H, n_tokens, dst_tokens, n_chunks,
                     sq1, sq2, sq3, sv1, sv2, sv3, neqk1_magic, rq3_magic, scale);
    NO_DEVICE_CODE;
#endif
}

// The chunked path is on by default (T405 phase 4) for n_tokens >= GGML_GDN_CHUNK_MIN (default 64);
// GGML_GDN_CHUNK=0 restores the sequential kernel. Read per call (a few dozen calls per ubatch) so a
// process can A/B both paths.
static int gdn_chunk_min_tokens() {
    const char * e = getenv("GGML_GDN_CHUNK");
    if (e != nullptr && atoi(e) == 0) {
        return -1;
    }
    const char * m = getenv("GGML_GDN_CHUNK_MIN");
    return m ? atoi(m) : 64;
}

static int gdn_chunk_prec() {
    const char * e = getenv("GGML_GDN_CHUNK_PREC");
    return e && atoi(e) == 1 ? 1 : 3;
}

// returns false if the shape/layout is not supported (caller falls back to the sequential kernel)
static bool gdn_chunk_launch(ggml_backend_cuda_context & ctx, cudaStream_t stream,
        const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d,
        const float * s_d, float * dst_d, float * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t dst_tokens, int64_t n_seqs, int64_t neqk1, int64_t rq3,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, float scale,
        bool fused_ba, const float * ssm_dt_d, const float * ssm_a_d, bool l2norm_qk, float l2norm_eps, int K) {
    const int chunk_min = gdn_chunk_min_tokens();
    if (chunk_min < 0 || n_tokens < chunk_min || S_v != GDN_CH_D || H % neqk1 != 0) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ctx.device].cc)) {
        return false;
    }
    // float4 loads of q/k rows
    if (((uintptr_t) q_d % 16) || ((uintptr_t) k_d % 16) || (sq1 % 4) || (sq2 % 4) || (sq3 % 4)) {
        return false;
    }
    static const bool trace = getenv("GGML_GDN_CHUNK_TRACE") != nullptr;
    const int prec = gdn_chunk_prec();
    if (trace) {
        GGML_LOG_INFO("gdn_chunk: n_tokens=%lld of %lld K=%d H=%lld n_seqs=%lld fused_ba=%d l2norm=%d prec=%d\n",
            (long long) n_tokens, (long long) dst_tokens, K, (long long) H, (long long) n_seqs, (int) fused_ba,
            (int) l2norm_qk, prec);
    }
    const int    n_chunks = (int) ((n_tokens + GDN_CH_C - 1) / GDN_CH_C);
    const size_t nblk     = (size_t) n_seqs * H * n_chunks;
    ggml_cuda_pool_alloc<float> Tbuf(ctx.pool(), nblk * GDN_CH_C * GDN_CH_C);
    ggml_cuda_pool_alloc<float> Pbuf(ctx.pool(), nblk * GDN_CH_C * GDN_CH_C);
    ggml_cuda_pool_alloc<float> meta(ctx.pool(), nblk * GDN_CH_C * 4);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const dim3 pgrid(n_chunks, neqk1, n_seqs);
#define GDN_PREP(FB, L2) gdn_chunk_prep<FB, L2><<<pgrid, 256, 0, stream>>>(q_d, k_d, g_d, b_d, \
        Tbuf.get(), Pbuf.get(), meta.get(), H, neqk1, n_tokens, n_chunks, sq1, sq2, sq3, sb1, sb2, sb3, \
        rq3_magic, ssm_dt_d, ssm_a_d, l2norm_eps)
    if (fused_ba) {
        if (l2norm_qk) { GDN_PREP(true, true); } else { GDN_PREP(true, false); }
    } else {
        if (l2norm_qk) { GDN_PREP(false, true); } else { GDN_PREP(false, false); }
    }
#undef GDN_PREP
    CUDA_CHECK(cudaGetLastError());

    const dim3 sgrid(H, n_seqs, GDN_CH_D / (16 * GDN_CH_WAVES));
#define GDN_SCAN(P) gdn_chunk_scan<P><<<sgrid, GDN_CH_WAVES * 32, 0, stream>>>(q_d, k_d, v_d, s_d, \
        Tbuf.get(), Pbuf.get(), meta.get(), dst_d, state_d, H, n_tokens, dst_tokens, n_chunks, \
        sq1, sq2, sq3, sv1, sv2, sv3, neqk1_magic, rq3_magic, scale)
    if (prec == 1) { GDN_SCAN(1); } else { GDN_SCAN(3); }
#undef GDN_SCAN
    CUDA_CHECK(cudaGetLastError());
    return true;
}
