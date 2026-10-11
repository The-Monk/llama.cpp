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
    // varied, non-repeating prose (an exactly repeating prompt makes the model near-deterministic and the KLD of a perturbed run unrepresentative)
    static const char * S[] = {
        "The committee met on Tuesday to review the proposed changes to the municipal water supply contract. ",
        "Chlorophyll absorbs light most strongly in the blue and red parts of the spectrum, which is why leaves look green. ",
        "After three weeks of rain the river burst its banks and flooded the lower fields near the mill. ",
        "A binary search tree keeps its keys ordered, so lookups take time proportional to the height of the tree. ",
        "She opened the letter slowly, afraid of what her brother might have written from the front. ",
        "The central bank raised interest rates by a quarter of a percentage point, citing persistent inflation in services. ",
        "Mount Kilimanjaro, the highest peak in Africa, has lost most of its glaciers over the past century. ",
        "To make the sauce, melt the butter, whisk in the flour, and add the milk a little at a time. ",
        "Early telescopes suffered from chromatic aberration, which smeared the image of bright stars with coloured halos. ",
        "The defence argued that the evidence had been collected without a proper warrant and should be excluded. ",
        "Honeybees communicate the location of flowers to their hive mates through a dance performed on the comb. ",
        "When the train finally pulled into the station, the platform was empty except for an old man with a violin case. ",
        "Quantum error correction encodes one logical qubit into many physical qubits so that local noise can be detected. ",
        "The museum's new wing, designed by a Danish architect, opens to the public in the spring of next year. ",
        "In 1859 a solar storm set telegraph offices on fire and produced auroras visible as far south as the Caribbean. ",
        "He had never liked the city, with its crowds and its noise, but the job paid well and the flat was cheap. ",
        "Sourdough starter is a living culture of wild yeast and lactic acid bacteria that must be fed regularly. ",
        "The expedition ran out of fresh water on the ninth day and was forced to turn back toward the coast. ",
        "A well-written unit test fails for exactly one reason and tells you which behaviour has stopped working. ",
        "Volcanic islands rise from the sea floor over millions of years, then slowly erode and sink beneath the waves. ",
        "The orchestra tuned to the oboe, and a hush fell over the hall as the conductor raised his baton. ",
        "Researchers found that sleep deprivation impairs the consolidation of memories formed earlier in the day. ",
        "The village council voted to repair the bridge rather than build a new one, to the relief of the farmers. ",
        "Prime numbers become sparser as they grow, yet there are infinitely many of them, as Euclid proved. ",
        "On the last night of the festival the streets filled with lanterns, drums, and the smell of roasting chestnuts. ",
        "The compiler inlines small functions, unrolls short loops, and removes code that can never be reached. ",
        "Her grandmother kept bees, grew beans along the fence, and could name every bird by its song alone. ",
        "Tides are caused mainly by the gravitational pull of the moon, with a smaller contribution from the sun. ",
        "The negotiators worked through the night, but by morning the two delegations were still far apart on fishing quotas. ",
        "Copper conducts electricity well, resists corrosion, and has been worked by humans for roughly ten thousand years. ",
        "A heavy fog settled over the moor, and the shepherd followed the stone wall home by touch alone. ",
        "The library reduced its opening hours in winter, which drew complaints from students preparing for examinations. ",
        "Neural networks trained on large corpora can complete sentences, translate languages, and summarise long documents. ",
        "He folded the map along its worn creases and tucked it into his coat before climbing the last ridge. ",
        "Glaciers carve broad U-shaped valleys, leaving behind moraines, lakes, and long ridges of gravel when they retreat. ",
        "The bakery on the corner sells out of rye bread by nine o'clock, so regulars arrive before the doors open. ",
        "A sudden gust scattered the papers across the courtyard, and the clerks chased them between the fountains. ",
        "Economists disagree about whether the minimum wage reduces employment, and the evidence depends on the region studied. ",
        "The astronaut described the silence of space as unlike any silence on Earth, full of the hum of the machines. ",
        "Rust forms when iron reacts with oxygen and water, and salt in the air speeds the reaction considerably. ",
    };
    std::string text;
    for (const char * s : S) text += s;
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
static void kld(const std::vector<float> & a, const std::vector<float> & b, int nv, double & mean, double & top1, std::vector<double> * per_row = nullptr) {
    const size_t R = a.size() / nv;
    mean = 0; top1 = 0;
    if (per_row) per_row->assign(R, 0.0);
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
        if (per_row) (*per_row)[r] = k;
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
        std::vector<double> pr;
        kld(a, b, nva, mk, t1, &pr);
        printf("CMP mean_kld=%.6f top1=%.4f bit_identical=%d (%zu positions)\n", mk, t1, (int) same, a.size() / nva);
        auto seg = [&](const char * n, size_t lo, size_t hi) { double m = 0; for (size_t i = lo; i < hi && i < pr.size(); ++i) m += pr[i]; printf("  %s kld=%.6f", n, m / (hi - lo)); };
        seg("prefill-tail[0..16)", 0, 16); seg("decode N=1[16..32)", 16, 32); seg("verify N=4[32..40)", 32, 40);
        printf("\n");
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
