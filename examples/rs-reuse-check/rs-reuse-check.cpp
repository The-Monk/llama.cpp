// T409 safety test: can a REUSED graph read the wrong recurrent cache row after a sequence copy?
// Scenario (all single-sequence ubatches of the same size N, so llama.cpp may reuse the graph):
//   X1  decode seq0 A        (fresh cell 0)
//   X2  decode seq1 B        (fresh cell 1: head=1, rs_z=1, s_copy(0)=1  -> builds a graph)
//   X3  seq_rm(1); seq_cp(0 -> 1)
//   X4  decode seq1 C @ pos N..2N-1 (seq1 shares cell 0 -> moved to fresh cell 1 with src=0:
//       head=1, rs_z=1, s_copy(0)=0 -> SAME reuse key as X2, but the state must be gathered from cell 0)
//   X5  G greedy single-token steps on seq1 (decode path: T360/T368 in-place rungs)
// Reference R (fresh context): decode seq0 A, then seq0 C @ N..2N-1, then G greedy steps.
// Output: per-step graph reuse counters, max|dlogit| and argmax X4 vs R, generated tokens.
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<llama_token> tok(const llama_vocab * v, const std::string & s) {
    std::vector<llama_token> t(s.size() + 16);
    int n = llama_tokenize(v, s.c_str(), (int) s.size(), t.data(), (int) t.size(), false, true);
    if (n < 0) { fprintf(stderr, "FAIL: tokenize\n"); exit(1); }
    t.resize(n);
    return t;
}

static int reused(llama_context * ctx) { return llama_perf_context(ctx).n_reused; }

static void dec(llama_context * ctx, const std::vector<llama_token> & t, int seq, int pos0, const char * tag) {
    llama_batch b = llama_batch_init((int) t.size(), 0, 1);
    for (size_t i = 0; i < t.size(); ++i) {
        b.token[i] = t[i]; b.pos[i] = pos0 + (int) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = seq;
        b.logits[i] = i + 1 == t.size();
    }
    b.n_tokens = (int) t.size();
    int r0 = reused(ctx);
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "FAIL: decode %s\n", tag); exit(1); }
    printf("STEP %-14s seq=%d n=%zu pos0=%d graph_reused=%d\n", tag, seq, t.size(), pos0, reused(ctx) - r0);
    llama_batch_free(b);
}

static int argmax(const float * l, int n) { int a = 0; for (int i = 1; i < n; ++i) if (l[i] > l[a]) a = i; return a; }

static std::vector<llama_token> greedy(llama_context * ctx, int seq, int pos0, int G, int nv, int & reuse_cnt) {
    std::vector<llama_token> out;
    int r0 = reused(ctx);
    llama_token t = argmax(llama_get_logits_ith(ctx, -1), nv);
    for (int g = 0; g < G; ++g) {
        out.push_back(t);
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = t; b.pos[0] = pos0 + g; b.n_seq_id[0] = 1; b.seq_id[0][0] = seq; b.logits[0] = 1; b.n_tokens = 1;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "FAIL: decode gen\n"); exit(1); }
        llama_batch_free(b);
        t = argmax(llama_get_logits_ith(ctx, -1), nv);
    }
    reuse_cnt = reused(ctx) - r0;
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model [N=64] [G=32]\n", argv[0]); return 2; }
    const int N = argc > 2 ? atoi(argv[2]) : 64;
    const int G = argc > 3 ? atoi(argv[3]) : 32;
    // T483: ref_env (KEY=VAL) is set only for the reference context; "nocp" runs the X side as the plain
    // reference scenario (no seq_cp), so X-vs-R then isolates the effect of ref_env alone (e.g. the fused-vs-unfused
    // conv path, or GGML_GDN_PERTURB_ULP=1 for the 1-ulp chaos floor).
    const std::string ref_env = argc > 4 ? argv[4] : "-";
    const bool nocp = argc > 5 && std::string(argv[5]) == "nocp";
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "FAIL: load\n"); return 1; }
    const llama_vocab * v = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(v);

    std::string text;
    for (int i = 0; i < 40; ++i)
        text += "In the old harbour town the lighthouse keeper counted ships, wrote letters to his sister, and "
                "repaired the lamp every evening while storms rolled in from the grey northern sea. ";
    auto all = tok(v, text);
    if ((int) all.size() < 3 * N) { fprintf(stderr, "FAIL: text too short\n"); return 1; }
    std::vector<llama_token> A(all.begin(), all.begin() + N), B(all.begin() + N + 7, all.begin() + 2 * N + 7),
                             C(all.begin() + 2 * N + 13, all.begin() + 3 * N + 13);
    // make B clearly different from A/C so a zero/wrong state shows up
    for (int i = 0; i < N; ++i) B[i] = (B[i] * 7 + 13) % 5000 + 100;

    auto cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 1024; cp.n_ubatch = 1024; cp.n_seq_max = 2; cp.kv_unified = true;
    cp.no_perf = false;

    std::vector<float> LX, LR; std::vector<llama_token> TX, TR; int rX = 0, rR = 0;
    {
        llama_context * ctx = llama_init_from_model(model, cp);
        llama_memory_t mem = llama_get_memory(ctx);
        const int xs = nocp ? 0 : 1;
        dec(ctx, A, 0, 0, "X1 seq0 A");
        if (!nocp) {
            dec(ctx, B, 1, 0, "X2 seq1 B");
            llama_memory_seq_rm(mem, 1, -1, -1);
            llama_memory_seq_cp(mem, 0, 1, -1, -1);
        }
        dec(ctx, C, xs, N, "X4 seq C");
        const float * l = llama_get_logits_ith(ctx, -1); LX.assign(l, l + nv);
        TX = greedy(ctx, xs, 2 * N, G, nv, rX);
        llama_free(ctx);
    }
    {
        const size_t eq = ref_env.find('=');
        if (eq != std::string::npos) setenv(ref_env.substr(0, eq).c_str(), ref_env.c_str() + eq + 1, 1);
        llama_context * ctx = llama_init_from_model(model, cp);
        dec(ctx, A, 0, 0, "R1 seq0 A");
        dec(ctx, C, 0, N, "R2 seq0 C");
        const float * l = llama_get_logits_ith(ctx, -1); LR.assign(l, l + nv);
        TR = greedy(ctx, 0, 2 * N, G, nv, rR);
        llama_free(ctx);
    }
    double md = 0; for (int i = 0; i < nv; ++i) md = std::max(md, (double) std::fabs(LX[i] - LR[i]));
    printf("PREFILL_CMP max_abs_dlogit=%.4f argmax_X=%d argmax_R=%d\n", md, argmax(LX.data(), nv), argmax(LR.data(), nv));
    int same = 0; while (same < G && TX[same] == TR[same]) same++;
    printf("GEN_CMP common_prefix=%d/%d gen_reused_X=%d gen_reused_R=%d\n", same, G, rX, rR);
    auto pr = [&](const char * n, const std::vector<llama_token> & T) {
        std::string s; char buf[256];
        for (auto t : T) { int k = llama_token_to_piece(v, t, buf, sizeof buf, 0, true); if (k > 0) s.append(buf, k); }
        for (auto & c : s) if (c == '\n') c = ' ';
        printf("TEXT_%s %s\n", n, s.c_str());
    };
    pr("X", TX); pr("R", TR);
    printf("VERDICT %s\n", (md < 0.5 && same == G) ? "MATCH" : "MISMATCH");
    llama_model_free(model);
    return 0;
}
