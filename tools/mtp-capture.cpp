// mtp-capture: teacher-force a corpus through a model and dump per-position
// pre-output-norm hidden states (the MTP/nextn head input) + token ids.
// Output: <outdir>/hidden.bf16 (n_tok x n_embd, bf16), <outdir>/tokens.i32,
//         <outdir>/chunks.i32 (length of each independent chunk), meta.txt.
// Usage: mtp-capture <model.gguf> <text.raw> <outdir> <max_tokens> [n_ctx]
#include "llama.h"
#include "../src/llama-ext.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>

static uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1;          // round-to-nearest-even
    return (uint16_t)((u + 0x7fff + lsb) >> 16);
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <model> <text> <outdir> <max_tokens> [n_ctx]\n", argv[0]); return 1; }
    const char * model_path = argv[1];
    const char * text_path  = argv[2];
    const std::string outdir = argv[3];
    const long max_tokens = atol(argv[4]);
    const int n_ctx = argc > 5 ? atoi(argv[5]) : 2048;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = n_ctx;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }
    llama_set_embeddings_nextn(ctx, true, /*masked*/ false);

    // read + tokenize whole corpus
    std::ifstream f(text_path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    fprintf(stderr, "corpus: %zu bytes\n", text.size());
    std::vector<llama_token> toks(text.size() / 2 + 64);
    int n_tok = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                               toks.data(), (int32_t) toks.size(), /*add_bos*/ true, /*special*/ false);
    if (n_tok < 0) { fprintf(stderr, "tokenize failed (%d)\n", n_tok); return 1; }
    toks.resize(n_tok);
    fprintf(stderr, "corpus tokens: %d (capturing up to %ld)\n", n_tok, max_tokens);

    std::ofstream fh(outdir + "/hidden.bf16", std::ios::binary);
    std::ofstream ft(outdir + "/tokens.i32", std::ios::binary);
    std::ofstream fc(outdir + "/chunks.i32", std::ios::binary);
    std::ofstream f8i(outdir + "/top8_ids.i32", std::ios::binary);
    std::ofstream f8v(outdir + "/top8_logits.bf16", std::ios::binary);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    std::vector<uint16_t> hrow(n_embd);
    long written = 0;
    for (long off = 0; off + 2 < n_tok && written < max_tokens; off += n_ctx) {
        const int len = (int) std::min((long) n_ctx, (long) (n_tok - off));
        if (len < 16) break;
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = len;
        for (int i = 0; i < len; ++i) {
            batch.token[i] = toks[off + i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;   // output at every position
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at off %ld\n", off); return 1; }
        for (int i = 0; i < len; ++i) {
            const float * h = llama_get_embeddings_nextn_ith(ctx, i);
            if (!h) { fprintf(stderr, "null nextn embd at %d\n", i); return 1; }
            for (int j = 0; j < n_embd; ++j) hrow[j] = f32_to_bf16(h[j]);
            fh.write((const char *) hrow.data(), (std::streamsize) n_embd * 2);
            // top-8 target logits at this position (distillation signal)
            const float * lg = llama_get_logits_ith(ctx, i);
            if (!lg) { fprintf(stderr, "null logits at %d\n", i); return 1; }
            int32_t ids[8]; uint16_t vals[8];
            {
                int32_t best[8]; float bv[8];
                for (int k = 0; k < 8; ++k) { best[k] = -1; bv[k] = -1e30f; }
                for (int v = 0; v < n_vocab; ++v) {
                    const float x = lg[v];
                    if (x <= bv[7]) continue;
                    int k = 7;
                    while (k > 0 && x > bv[k-1]) { bv[k] = bv[k-1]; best[k] = best[k-1]; --k; }
                    bv[k] = x; best[k] = v;
                }
                for (int k = 0; k < 8; ++k) { ids[k] = best[k]; vals[k] = f32_to_bf16(bv[k]); }
            }
            f8i.write((const char *) ids, 32);
            f8v.write((const char *) vals, 16);
        }
        ft.write((const char *) (toks.data() + off), (std::streamsize) len * 4);
        int32_t clen = len;
        fc.write((const char *) &clen, 4);
        written += len;
        if ((written / n_ctx) % 32 == 0) fprintf(stderr, "  %ld tokens captured\n", written);
    }
    std::ofstream fm(outdir + "/meta.txt");
    fm << "n_embd " << n_embd << "\nn_tokens " << written << "\nn_ctx " << n_ctx << "\nmodel " << model_path << "\n";
    fprintf(stderr, "DONE: %ld tokens, %.1f GiB hidden\n", written, written * (double) n_embd * 2 / (1 << 30));
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
