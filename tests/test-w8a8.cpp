// T513: tests that REACH the enterprise W8A8 path (GGML_N1_W8, T505), which test-backend-ops never does (its Q8_0 MUL_MATs have N < 64 / no
// GGML_N1_W8 and are compared against the CPU reference at a tolerance the different math cannot meet).
//
//   test-w8a8 kernels                     the W8 kernels (decode GEMV, M3-class GEMM, ROW GEMM) on caller-supplied int8 data at the real
//                                         shapes and small / odd N, against an EXACT integer reference: Y[t][f] = ((float) sum_k X*W * sx[t]) * rs[f]
//                                         (bit-exact; a never-written output element is NaN-filled and fails)
//   test-w8a8 op <out.bin> [csv-of-N]     routed MUL_MATs through ggml (Q8_0 weights, F32 activations with a few outlier channels) on the HIP
//                                         backend; writes every output; prints the NMSE vs a double-precision reference and the W8 counters.
//                                         Run it under different GGML_N1_W8* environments (tests/test-w8a8.sh does).
//   test-w8a8 cmp <a.bin> <b.bin> equal | close <max-nmse> | differs <max-nmse>
//                                         equal: bit-identical; differs: not identical (the path really engaged) and NMSE(a, b) <= max
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern int  ggml_cuda_n1_w8_selftest(const int8_t * W, const float * rs, const int8_t * X, const float * sx, float * Y, int N, int F, int K, int which);
extern void ggml_cuda_n1_w8_counters(unsigned long long out[4]);

struct shape { int K, F; const char * why; };

// Qwen3.8-27B (K 5120/6144/17408: the specialised kernels), Qwen3-8B / Llama-3.x-8B (4096, 12288, 14336) and Qwen2.5-14B (5120, 13824) generic-K shapes
static const shape SHAPES[] = {
    {5120, 5120, "q8 qwen38 attn"}, {5120, 1024, "qwen38 kv"}, {5120, 17408, "qwen38 ffn up/gate"}, {17408, 5120, "qwen38 ffn down"},
    {6144, 6144, "qwen38 6144"}, {4096, 4096, "8B attn"}, {4096, 1024, "8B kv"}, {4096, 14336, "llama8b ffn"}, {14336, 4096, "llama8b down"},
    {12288, 4096, "qwen3-8b down"}, {13824, 5120, "qwen2.5-14b down (512-block rotation)"}, {3584, 3584, "qwen2.5-7b hidden (512-block)"}, {18944, 3584, "qwen2.5-7b ffn down (256-block, 74 blocks)"}, {2688, 1024, "128-block (K = 21 x 128)"}, {128, 128, "minimum"}, {256, 384, "tiny odd tiles"},
};
static const int NS[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 384};

static int run_kernels() {
    std::mt19937 rng(1234);
    int fails = 0, checks = 0;
    for (const shape & sh : SHAPES) {
        const int K = sh.K, F = sh.F;
        const int Nmax = 384;
        std::vector<int8_t> W((size_t) F * K), X((size_t) Nmax * K);
        std::vector<float> rs(F), sx(Nmax);
        auto fill = [&](std::vector<int8_t> & v) {
            for (auto & x : v) {
                const uint32_t r = rng();
                x = (r & 63) == 0 ? (int8_t) ((r & 64) ? 127 : -127) : (int8_t) ((int) ((r >> 8) % 255) - 127);
            }
        };
        fill(W); fill(X);
        for (auto & v : rs) v = 1e-4f * (1.f + (rng() % 1000) * 1e-3f);
        for (auto & v : sx) v = 1e-3f * (1.f + (rng() % 1000) * 1e-3f);
        // reference rows: all of them when small, else 96 spread over F (always incl. the first and last)
        std::vector<int> rows;
        const int nr = F * (size_t) Nmax * K <= (size_t) 4e8 ? F : 96;
        for (int i = 0; i < nr; ++i) rows.push_back(nr == F ? i : (int) ((size_t) i * (F - 1) / (nr - 1)));
        std::vector<float> REF((size_t) Nmax * rows.size());   // exact integer reference, token t row index ri
        for (int t = 0; t < Nmax; ++t) {
            for (size_t ri = 0; ri < rows.size(); ++ri) {
                int acc = 0;
                const int8_t * w = &W[(size_t) rows[ri] * K], * x = &X[(size_t) t * K];
                for (int k = 0; k < K; ++k) acc += (int) w[k] * (int) x[k];
                REF[(size_t) t * rows.size() + ri] = ((float) acc * sx[t]) * rs[rows[ri]];
            }
        }
        for (int N : NS) {
            for (int which : {0, 1, 2, 3}) {
                if (which == 1 && N > 16) continue;
                std::vector<float> Y((size_t) N * F);
                if (ggml_cuda_n1_w8_selftest(W.data(), rs.data(), X.data(), sx.data(), Y.data(), N, F, K, which) != 0) {
                    printf("FAIL kernels K=%d F=%d N=%d which=%d: selftest refused\n", K, F, N, which); ++fails; continue;
                }
                // the decode GEMV is bit-exact against ((float) acc * sx) * rs; the WMMA GEMMs (compiled with -funsafe-math: the epilogue products are
                // re-associated) agree with the exact integer sum and differ from that float expression by at most 2 ulp
                const bool exact = which == 1 || (which == 0 && N <= 8);
                size_t bad = 0, nan = 0;
                for (size_t i = 0; i < Y.size(); ++i) nan += std::isnan(Y[i]);   // a never-written element keeps the 0xFF NaN fill
                for (int t = 0; t < N; ++t) {
                    for (size_t ri = 0; ri < rows.size(); ++ri) {
                        const float ref = REF[(size_t) t * rows.size() + ri], y = Y[(size_t) t * F + rows[ri]];
                        if (exact) { bad += memcmp(&ref, &y, sizeof(float)) != 0; continue; }
                        const float ulp = std::nextafter(std::fabs(ref), INFINITY) - std::fabs(ref);
                        bad += !(std::fabs(y - ref) <= 2.f * ulp);
                    }
                }
                ++checks;
                if (bad || nan) {
                    printf("FAIL kernels K=%d F=%d (%s) N=%d which=%d: %zu inexact, %zu unwritten\n", K, F, sh.why, N, which, bad, nan); ++fails;
                }
            }
        }
        printf("ok   kernels K=%d F=%d (%s): %zu rows x %zu N values x 4 kernels\n", K, F, sh.why, rows.size(), sizeof(NS) / sizeof(NS[0]));
        fflush(stdout);
    }
    printf("KERNELS %d checks, %d failed\n", checks, fails);
    return fails ? 1 : 0;
}

// ---- routed op test ----
static int run_op(const char * outf, const std::vector<int> & Ns) {
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!be) { printf("FAIL no GPU backend\n"); return 1; }
    FILE * fo = fopen(outf, "wb");
    if (!fo) { printf("FAIL cannot write %s\n", outf); return 1; }
    std::mt19937 rng(777);
    std::normal_distribution<float> nd(0.f, 1.f);
    double worst = 0, worst_all = 0;
    for (const shape & sh : SHAPES) {
        const int K = sh.K, F = sh.F;
        if (K < 1024 || F < 1024) continue;   // the tiny shapes are covered by the kernel test
        ggml_init_params ip = { ggml_tensor_overhead() * 4 + 4096, nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_2d(c, GGML_TYPE_Q8_0, K, F);
        ggml_set_name(w, "blk.0.ffn_up.weight");
        ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors(c, be);
        ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);   // what the model loader does; the in-place route requires it
        std::vector<float> wf((size_t) F * K);
        for (auto & v : wf) v = 0.02f * nd(rng);
        std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_Q8_0, K) * F);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), wq.data(), 0, F, K, nullptr);
        ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
        std::vector<float> wd((size_t) F * K);   // what the Q8_0 weights actually hold
        {
            const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q8_0);
            for (int f = 0; f < F; ++f) tt->to_float(wq.data() + (size_t) f * ggml_row_size(GGML_TYPE_Q8_0, K), &wd[(size_t) f * K], K);
        }
        for (int N : Ns) {
            std::vector<float> x((size_t) N * K);
            for (int t = 0; t < N; ++t) {
                for (int k = 0; k < K; ++k) x[(size_t) t * K + k] = nd(rng);
                x[(size_t) t * K + 7] *= 40.f;              // massive-activation channels: what per-token int8 suffers from
                x[(size_t) t * K + K / 2 + 3] *= 25.f;
            }
            ggml_init_params gp = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 4096, nullptr, true };
            ggml_context * g = ggml_init(gp);
            ggml_tensor * xin = ggml_new_tensor_2d(g, GGML_TYPE_F32, K, N);
            ggml_set_name(xin, "inp");
            ggml_tensor * y = ggml_mul_mat(g, w, xin);
            ggml_set_name(y, "y");
            ggml_cgraph * gf = ggml_new_graph(g);
            ggml_build_forward_expand(gf, y);
            ggml_backend_buffer_t gb = ggml_backend_alloc_ctx_tensors(g, be);
            ggml_backend_tensor_set(xin, x.data(), 0, x.size() * sizeof(float));
            if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { printf("FAIL graph compute K=%d F=%d N=%d\n", K, F, N); return 1; }
            std::vector<float> yo((size_t) N * F);
            ggml_backend_tensor_get(y, yo.data(), 0, yo.size() * sizeof(float));
            // double reference on a sample of tokens / rows
            double num = 0, den = 0;
            for (int t = 0; t < N; t += std::max(1, N / 8)) {
                for (int f = 0; f < F; f += std::max(1, F / 64)) {
                    double acc = 0;
                    for (int k = 0; k < K; ++k) acc += (double) wd[(size_t) f * K + k] * x[(size_t) t * K + k];
                    const double d = yo[(size_t) t * F + f] - acc;
                    num += d * d; den += acc * acc;
                }
            }
            const double nmse = num / den;
            worst_all = std::max(worst_all, nmse);
            worst = std::max(worst, nmse);
            fwrite(yo.data(), sizeof(float), yo.size(), fo);
            printf("op K=%-5d F=%-5d N=%-4d nmse_vs_double=%.3e\n", K, F, N, nmse);
            ggml_backend_buffer_free(gb);
            ggml_free(g);
        }
        ggml_backend_buffer_free(wb);
        ggml_free(c);
        fflush(stdout);
    }
    fclose(fo);
    unsigned long long cnt[4];
    ggml_cuda_n1_w8_counters(cnt);
    printf("W8_COUNTERS mm=%llu converted=%llu gemv=%llu refused=%llu worst_nmse=%.3e worst_nmse_all=%.3e\n", cnt[0], cnt[1], cnt[2], cnt[3], worst, worst_all);
    ggml_backend_free(be);
    return 0;
}

static std::vector<float> slurp(const char * f) {
    std::vector<float> v;
    FILE * fp = fopen(f, "rb");
    if (!fp) return v;
    fseek(fp, 0, SEEK_END);
    const long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    v.resize(n / sizeof(float));
    if (fread(v.data(), sizeof(float), v.size(), fp) != v.size()) v.clear();
    fclose(fp);
    return v;
}

static int run_cmp(const char * fa, const char * fb, const std::string & mode, double maxnmse) {
    const auto a = slurp(fa), b = slurp(fb);
    if (a.empty() || a.size() != b.size()) { printf("FAIL cmp: empty or different sizes (%zu vs %zu)\n", a.size(), b.size()); return 1; }
    size_t diff = 0;
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        diff += memcmp(&a[i], &b[i], sizeof(float)) != 0;
        const double d = (double) a[i] - b[i];
        num += d * d; den += (double) b[i] * b[i];
    }
    const double nmse = num / den;
    printf("CMP %s vs %s: %zu / %zu elements differ, nmse %.3e\n", fa, fb, diff, a.size(), nmse);
    if (mode == "equal") return diff == 0 ? 0 : (printf("FAIL expected bit-identical\n"), 1);
    if (mode == "close") return nmse <= maxnmse ? 0 : (printf("FAIL nmse %.3e > %.3e\n", nmse, maxnmse), 1);
    if (diff == 0) { printf("FAIL outputs are identical: the W8 path did not engage\n"); return 1; }
    if (!(nmse <= maxnmse)) { printf("FAIL nmse %.3e > %.3e\n", nmse, maxnmse); return 1; }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s kernels | op <out.bin> [N,N,...] | cmp a b equal|differs <max-nmse>\n", argv[0]); return 2; }
    const std::string cmd = argv[1];
    if (cmd == "kernels") return run_kernels();
    if (cmd == "op" && argc >= 3) {
        std::vector<int> Ns = {1, 3, 8, 9, 17, 64, 100, 129, 512};
        if (argc >= 4) { Ns.clear(); for (char * p = strtok(argv[3], ","); p; p = strtok(nullptr, ",")) Ns.push_back(atoi(p)); }
        return run_op(argv[2], Ns);
    }
    if (cmd == "cmp" && argc >= 5) return run_cmp(argv[2], argv[3], argv[4], argc >= 6 ? atof(argv[5]) : 1.0);
    fprintf(stderr, "bad arguments\n");
    return 2;
}
