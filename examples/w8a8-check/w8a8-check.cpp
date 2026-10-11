// T513 engine-level check for the enterprise W8A8 path (GGML_N1_W8 + _INPLACE [+ _ROT]).
//   llama-w8a8-check run <model> <out.bin> [ngl=999] [reloads=1] [load_mode=-1]
//       loads the model, evaluates a fixed prompt (a 256-token prefill, 16 forced single-token decode steps, two 4-token verify batches) and
//       writes the logits of 40 positions to out.bin. reloads > 1 loads / evaluates / frees the model again in the SAME process and requires the
//       logits of every later load to be bit-identical to the first (a stale in-place table would make them garbage). Prints the W8 counters.
//   llama-w8a8-check cmp <a.bin> <b.bin> equal | close <max-kld> | differs <max-mean-kld>
//       equal: bit-identical (the off-vs-off control); differs: not identical (the path really engaged) and mean KLD(b || a) <= max
#include "llama.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern void ggml_cuda_n1_w8_counters(unsigned long long out[4]);

static std::vector<llama_token> tok(const llama_vocab * v, const std::string & s) {
    std::vector<llama_token> t(s.size() + 16);
    int n = llama_tokenize(v, s.c_str(), (int) s.size(), t.data(), (int) t.size(), false, true);
    if (n < 0) { fprintf(stderr, "FAIL: tokenize\n"); exit(1); }
    t.resize(n);
    return t;
}

static void dec(llama_context * ctx, const llama_token * t, int n, int pos0, int nlogits_tail, std::vector<float> & rows, int nv) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = t[i]; b.pos[i] = pos0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = i >= n - nlogits_tail;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "FAIL: decode n=%d\n", n); exit(1); }
    for (int i = n - nlogits_tail; i < n; ++i) {
        const float * l = llama_get_logits_ith(ctx, i);
        rows.insert(rows.end(), l, l + nv);
    }
    llama_batch_free(b);
}

static bool eval_once(const char * path, int ngl, int load_mode, std::vector<float> & rows, int & nv_out) {
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl; mp.load_mode = (enum llama_load_mode) load_mode;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "FAIL: load\n"); return false; }
    const llama_vocab * v = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(v);
    nv_out = nv;
    std::string text;
    for (int i = 0; i < 40; ++i)
        text += "In the old harbour town the lighthouse keeper counted ships, wrote letters to his sister, and "
                "repaired the lamp every evening while storms rolled in from the grey northern sea. ";
    auto all = tok(v, text);
    if (all.size() < 256 + 16 + 8) { fprintf(stderr, "FAIL: text too short\n"); return false; }
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 512; cp.n_ubatch = 512; cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "FAIL: context\n"); return false; }
    rows.clear();
    int pos = 0;
    dec(ctx, all.data(), 256, pos, 16, rows, nv); pos += 256;
    for (int i = 0; i < 16; ++i) { dec(ctx, all.data() + pos, 1, pos, 1, rows, nv); ++pos; }
    for (int i = 0; i < 2; ++i) { dec(ctx, all.data() + pos, 4, pos, 4, rows, nv); pos += 4; }
    llama_free(ctx);
    llama_model_free(model);
    return true;
}

static bool load(const char * f, int & nv, std::vector<float> & rows) {
    FILE * fp = fopen(f, "rb");
    if (!fp) return false;
    int32_t h[2];
    bool ok = fread(h, 4, 2, fp) == 2;
    if (ok) { nv = h[0]; rows.resize((size_t) h[0] * h[1]); ok = fread(rows.data(), 4, rows.size(), fp) == rows.size(); }
    fclose(fp);
    return ok;
}

// mean KLD(P_a || P_b) over rows (a = base) and top-1 agreement
static void kld(const std::vector<float> & a, const std::vector<float> & b, int nv, double & mean, double & top1) {
    const size_t R = a.size() / nv;
    mean = 0; top1 = 0;
    for (size_t r = 0; r < R; ++r) {
        const float * la = &a[r * nv], * lb = &b[r * nv];
        double ma = -1e300, mb = -1e300;
        int ia = 0, ib = 0;
        for (int i = 0; i < nv; ++i) { if (la[i] > ma) { ma = la[i]; ia = i; } if (lb[i] > mb) { mb = lb[i]; ib = i; } }
        double za = 0, zb = 0;
        for (int i = 0; i < nv; ++i) { za += std::exp(la[i] - ma); zb += std::exp(lb[i] - mb); }
        const double lza = std::log(za) + ma, lzb = std::log(zb) + mb;
        double k = 0;
        for (int i = 0; i < nv; ++i) { const double pa = std::exp(la[i] - lza); if (pa > 0) k += pa * ((la[i] - lza) - (lb[i] - lzb)); }
        mean += k; top1 += ia == ib;
    }
    mean /= R; top1 /= R;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s run <model> <out.bin> [ngl] [reloads] [load_mode] | cmp a b equal|differs <max-kld>\n", argv[0]); return 2; }
    const std::string cmd = argv[1];
    if (cmd == "cmp") {
        int nva = 0, nvb = 0;
        std::vector<float> a, b;
        if (argc < 5 || !load(argv[2], nva, a) || !load(argv[3], nvb, b) || nva != nvb || a.size() != b.size() || a.empty()) { printf("FAIL cmp: cannot read the inputs\n"); return 1; }
        const bool same = memcmp(a.data(), b.data(), a.size() * 4) == 0;
        double mk, t1;
        kld(a, b, nva, mk, t1);
        printf("CMP mean_kld=%.6f top1=%.4f bit_identical=%d (%zu positions)\n", mk, t1, (int) same, a.size() / nva);
        const std::string mode = argv[4];
        if (mode == "equal") return same ? 0 : (printf("FAIL expected bit-identical\n"), 1);
        if (mode == "close") { const double mx = argc >= 6 ? atof(argv[5]) : 1.0; return mk <= mx ? 0 : (printf("FAIL mean KLD %.6f > %.6f\n", mk, mx), 1); }
        if (same) { printf("FAIL identical to the base: the W8 path did not engage\n"); return 1; }
        const double mx = argc >= 6 ? atof(argv[5]) : 1.0;
        if (!(mk <= mx)) { printf("FAIL mean KLD %.6f > %.6f\n", mk, mx); return 1; }
        return 0;
    }
    if (cmd != "run" || argc < 4) { fprintf(stderr, "bad arguments\n"); return 2; }
    const int ngl = argc > 4 ? atoi(argv[4]) : 999, reloads = argc > 5 ? atoi(argv[5]) : 1;
    const int load_mode = argc > 6 ? atoi(argv[6]) : -1;   // llama_load_mode: -1 auto, 0 none (read), 1 mmap, 2 mlock, 3 mmap+mlock, 4 direct I/O
    llama_backend_init();
    std::vector<float> first;
    int nv = 0;
    int rc = 0;
    for (int r = 0; r < reloads; ++r) {
        std::vector<float> rows;
        if (!eval_once(argv[2], ngl, load_mode, rows, nv)) return 1;
        unsigned long long c[4];
        ggml_cuda_n1_w8_counters(c);
        printf("LOAD %d: W8_COUNTERS mm=%llu converted=%llu gemv=%llu refused=%llu\n", r, c[0], c[1], c[2], c[3]);
        if (r == 0) {
            first = rows;
            FILE * fo = fopen(argv[3], "wb");
            if (!fo) { printf("FAIL cannot write %s\n", argv[3]); return 1; }
            const int32_t h[2] = { nv, (int32_t) (rows.size() / nv) };
            fwrite(h, 4, 2, fo); fwrite(rows.data(), 4, rows.size(), fo); fclose(fo);
        } else {
            const bool same = rows.size() == first.size() && memcmp(rows.data(), first.data(), rows.size() * 4) == 0;
            double mk = 0, t1 = 0;
            if (!same && rows.size() == first.size()) kld(first, rows, nv, mk, t1);
            printf("RELOAD %d: %s (mean_kld vs load 0 = %.6f)\n", r, same ? "bit-identical" : "DIFFERENT", mk);
            if (!same) rc = 1;
        }
    }
    printf("W8A8_CHECK_RUN %s\n", rc ? "FAIL" : "ok");
    return rc;
}
