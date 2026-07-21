// mtp-extract: dump the qwen35 backbone's post-output-norm hidden state
// ("h_nextn", the exact tensor the MTP/nextn head consumes at inference --
// see src/models/qwen35.cpp: `cb(cur, "h_nextn", -1); res->t_h_nextn = cur;`
// right after `build_norm(cur, model.output_norm, ...)`) plus the token ids,
// one row per token, for offline MTP-head fine-tuning (card 147/151/152).
//
// Output layout per corpus "chunk" (an independent KV-cache window, no
// carryover between chunks -- keeps this simple and correct; the small
// cold-start context loss at each chunk's first few tokens is an accepted
// approximation for head-only fine-tuning):
//   <out>/features.f32   -- flat float32[total_tokens, n_embd], row-major
//   <out>/tokens.i32      -- flat int32[total_tokens], the token id at each row
//   <out>/chunk_lens.i32  -- int32[n_chunks], length of each chunk in tokens
//     (training script reconstructs chunk boundaries via cumsum; valid
//      (h[p], tok[p+1]) -> tok[p+2] triples only within one chunk, never
//      spanning two -- see mtp-finetune training script for the exact
//      offset this must match: MTP predicts tok[p+2] from (h_nextn[p],
//      embed(tok[p+1])), matching common/speculative.cpp's
//      common_speculative_impl_draft_mtp::process() h-shift-by-one +
//      token-at-k pairing exactly.)
//
// Usage:
//   llama-mtp-extract -m <gguf> --corpus <file> -o <outdir> [--chunk 2048] [--device ROCm0]

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn (see common/speculative.cpp)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct extract_ctx {
    std::ofstream * f_feat  = nullptr;
    std::ofstream * f_tok   = nullptr;
    int64_t         n_embd  = 0;
    int64_t         n_rows_written = 0;
    bool            saw_h_nextn_this_ubatch = false;
};

static bool mtp_extract_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * ex = (extract_ctx *) user_data;

    if (ask) {
        // Only request data copy-back for the tensor we care about -- avoids
        // paying the sync cost for every node in the graph.
        return strcmp(t->name, "h_nextn") == 0;
    }

    if (strcmp(t->name, "h_nextn") != 0) {
        return true;
    }

    GGML_ASSERT(t->type == GGML_TYPE_F32 && "h_nextn expected f32 (post output_norm)");
    GGML_ASSERT(t->ne[2] == 1 && t->ne[3] == 1);

    const int64_t n_embd   = t->ne[0];
    const int64_t n_tokens = t->ne[1];

    if (ex->n_embd == 0) {
        ex->n_embd = n_embd;
    }
    GGML_ASSERT(n_embd == ex->n_embd && "n_embd changed mid-run?!");

    // h_nextn may not be contiguous row-major if nb[1] != n_embd*sizeof(float)
    // (shouldn't happen for this tensor, but check rather than silently
    // misreading transposed data).
    GGML_ASSERT((size_t) t->nb[0] == sizeof(float));

    std::vector<float> buf((size_t) n_embd * n_tokens);
    if (t->nb[1] == (size_t) n_embd * sizeof(float)) {
        // fully contiguous -- one shot copy
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
    } else {
        // row-by-row (defensive; not expected on this path)
        for (int64_t i1 = 0; i1 < n_tokens; ++i1) {
            ggml_backend_tensor_get(t, buf.data() + i1 * n_embd, (size_t) i1 * t->nb[1], (size_t) n_embd * sizeof(float));
        }
    }

    // sanity: finite values only (card 151/152 discipline -- catch a broken
    // extraction immediately rather than silently training on garbage/NaNs)
    for (float v : buf) {
        if (!std::isfinite(v)) {
            LOG_ERR("%s: non-finite value in h_nextn -- aborting extraction\n", __func__);
            GGML_ABORT("non-finite h_nextn");
        }
    }

    ex->f_feat->write(reinterpret_cast<const char *>(buf.data()), (std::streamsize) (buf.size() * sizeof(float)));
    ex->n_rows_written += n_tokens;
    ex->saw_h_nextn_this_ubatch = true;

    return true;
}

int main(int argc, char ** argv) {
    std::string corpus_path;
    std::string out_dir = "./mtp-extract-out";
    int64_t     chunk_tokens = 2048;

    // Pull our custom flags out of argv BEFORE handing off to
    // common_params_parse, which rejects unrecognized flags -- build a
    // filtered argv containing only the flags common_params_parse knows about.
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--corpus" && i + 1 < argc) {
            corpus_path = argv[++i];
        } else if ((a == "-o" || a == "--out") && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (a == "--chunk" && i + 1 < argc) {
            chunk_tokens = std::atoll(argv[++i]);
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    int filtered_argc = (int) filtered_argv.size();

    common_params params;

    common_init();

    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (corpus_path.empty()) {
        LOG_ERR("%s: --corpus <file> is required\n", __func__);
        return 1;
    }

    std::ifstream corpus_f(corpus_path, std::ios::binary);
    if (!corpus_f) {
        LOG_ERR("%s: failed to open corpus file '%s'\n", __func__, corpus_path.c_str());
        return 1;
    }
    std::stringstream ss;
    ss << corpus_f.rdbuf();
    std::string corpus_text = ss.str();

    system(("mkdir -p " + out_dir).c_str());

    llama_backend_init();
    llama_numa_init(params.numa);

    params.warmup = false;
    if (params.n_ctx == 0) {
        params.n_ctx = (uint32_t) chunk_tokens + 8;
    }

    extract_ctx ex;
    std::ofstream f_feat(out_dir + "/features.f32", std::ios::binary);
    std::ofstream f_tok (out_dir + "/tokens.i32",   std::ios::binary);
    std::ofstream f_len (out_dir + "/chunk_lens.i32", std::ios::binary);
    ex.f_feat = &f_feat;
    ex.f_tok  = &f_tok;

    // cb_eval must be wired into llama_context_params BEFORE context creation
    // (there is no runtime setter in this fork) -- mirrors examples/eval-callback.
    params.cb_eval           = mtp_extract_cb_eval;
    params.cb_eval_user_data = &ex;

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to load model/context\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    // Match common_speculative_impl_draft_mtp's ctor exactly for ctx_tgt:
    // masked=false so h_nextn is retained for every position, not just the
    // last one used for standard next-token sampling.
    llama_set_embeddings_nextn(ctx, true, /*masked=*/ false);

    // Tokenize the whole corpus once, then walk it in non-overlapping chunks.
    std::vector<llama_token> all_tokens = common_tokenize(ctx, corpus_text, add_bos, true);
    LOG_INF("%s: corpus tokenized: %zu tokens, chunk size %lld\n", __func__, all_tokens.size(), (long long) chunk_tokens);

    llama_batch batch = llama_batch_init((int32_t) chunk_tokens, 0, 1);

    int64_t n_chunks = 0;
    for (size_t off = 0; off < all_tokens.size(); off += (size_t) chunk_tokens) {
        const size_t len = std::min((size_t) chunk_tokens, all_tokens.size() - off);
        if (len < 4) break; // too short to be useful (need p, p+1, p+2)

        llama_memory_clear(llama_get_memory(ctx), true);

        common_batch_clear(batch);
        for (size_t i = 0; i < len; ++i) {
            // logits=true on every position: we don't need the actual logits,
            // but this forces the graph to keep every position "live" through
            // to result_norm/t_embd, which is what makes h_nextn (feeding
            // into t_embd's dependency chain -- see qwen35.cpp) retained for
            // ALL positions rather than pruned for interior ones.
            common_batch_add(batch, all_tokens[off + i], (llama_pos) i, {0}, true);
        }

        ex.saw_h_nextn_this_ubatch = false;
        const int32_t rc = llama_decode(ctx, batch);
        if (rc != 0) {
            LOG_ERR("%s: llama_decode failed rc=%d at chunk offset %zu\n", __func__, rc, off);
            return 1;
        }
        if (!ex.saw_h_nextn_this_ubatch) {
            LOG_ERR("%s: h_nextn never fired for this chunk -- extraction is broken\n", __func__);
            return 1;
        }

        // token ids for this chunk, in order
        for (size_t i = 0; i < len; ++i) {
            int32_t tok = (int32_t) all_tokens[off + i];
            f_tok.write(reinterpret_cast<const char *>(&tok), sizeof(int32_t));
        }
        int32_t len32 = (int32_t) len;
        f_len.write(reinterpret_cast<const char *>(&len32), sizeof(int32_t));
        ++n_chunks;

        if (n_chunks % 20 == 0) {
            LOG_INF("%s: %lld chunks, %lld rows written\n", __func__, (long long) n_chunks, (long long) ex.n_rows_written);
        }
    }

    llama_batch_free(batch);
    f_feat.close();
    f_tok.close();
    f_len.close();

    LOG_INF("%s: DONE. %lld chunks, %lld rows, n_embd=%lld -> %s\n",
            __func__, (long long) n_chunks, (long long) ex.n_rows_written, (long long) ex.n_embd, out_dir.c_str());

    llama_backend_free();
    return 0;
}
