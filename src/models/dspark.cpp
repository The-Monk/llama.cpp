#include "models.h"

#include "llama-kv-cache.h"

// DSpark: a DFlash-family block-diffusion drafter (see src/models/dflash.cpp for
// the shared design: fc + hidden_norm fuse the target's extracted layer features
// into the draft's hidden size, then a small non-causal transformer denoises a
// block of <mask> tokens in one shot).
//
// Differences from DFlash, and what this port does and does not implement:
//   - tensor names carry a "dspark." prefix (dspark.fc, dspark.hidden_norm, ...)
//     instead of DFlash's bare names, and DSpark's custom hparams live under the
//     nested "dspark.dspark.*" gguf key namespace instead of DFlash's flat
//     "dflash.*" -- see LLM_KV_DSPARK_* in llama-arch.{h,cpp}.
//   - DSpark carries its own token_embd/output tensors (DFlash instead borrows
//     the target's via ctx_other), so no ctx_other fallback is needed here.
//   - DSpark additionally conditions the block on a log-SNR (noise level) value
//     via a small sinusoidal-embedding MLP (dspark.log_snr_fc1/fc2). This port
//     implements that conditioning for a SINGLE-SHOT draft: the block starts
//     fully masked, i.e. at maximum noise, so it is conditioned on
//     hparams-provided dspark.min_log_snr rather than a multi-step schedule.
//   - DSpark also carries a low-rank Markov consistency head
//     (dspark.markov_head_a/b) and a confidence head (dspark.confidence_head)
//     used upstream for iterative re-masking / early-exit refinement across
//     multiple diffusion steps. Those tensors are loaded (so nothing in the
//     gguf is orphaned) but NOT wired into the compute graph: this is a
//     single-shot port, not a faithful multi-step diffusion sampler. This is
//     the known simplification versus the full DSpark algorithm.

void llama_model_dspark::load_arch_hparams(llama_model_loader & ml) {

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (!ml.get_arr(LLM_KV_DSPARK_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DSpark model requires 'dspark.target_layers' in GGUF metadata");
    }

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    ml.get_key(LLM_KV_DSPARK_BLOCK_SIZE,  dspark_block_size,  false);
    ml.get_key(LLM_KV_DSPARK_MIN_LOG_SNR, dspark_min_log_snr, false);
    ml.get_key(LLM_KV_DSPARK_MAX_LOG_SNR, dspark_max_log_snr, false);

    LLAMA_LOG_INFO("%s: DSpark extract_layers = [", __func__);
    for (size_t i = 0; i < target_layer_ids.size(); ++i) {
        LLAMA_LOG_INFO("%d%s", target_layer_ids[i], i + 1 < target_layer_ids.size() ? ", " : "");
    }
    LLAMA_LOG_INFO("]\n");
    LLAMA_LOG_INFO("%s: DSpark block_size=%u, min_log_snr=%.3f, max_log_snr=%.3f "
                    "(markov head / confidence head loaded but not used by this port)\n",
                    __func__, dspark_block_size, dspark_min_log_snr, dspark_max_log_snr);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dspark::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    // DSpark carries its own token_embd / output (unlike DFlash, which borrows
    // the target model's via ctx_other).
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab }, 0);

    fc              = create_tensor(tn(LLM_TENSOR_DSPARK_FC,          "weight"), { n_embd_inp, n_embd }, 0);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_DSPARK_HIDDEN_NORM, "weight"), { n_embd },             0); // hidden_norm (after fc)
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,        "weight"), { n_embd },             0); // decoder final norm

    // log-SNR conditioning MLP: 128-dim sinusoidal embedding -> n_embd -> n_embd
    dspark_log_snr_fc1   = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC1, "weight"), { 128,    n_embd }, 0);
    dspark_log_snr_fc1_b = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC1, "bias"),   { n_embd },         0);
    dspark_log_snr_fc2   = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC2, "weight"), { n_embd, n_embd }, 0);
    dspark_log_snr_fc2_b = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC2, "bias"),   { n_embd },         0);

    // Markov consistency head + confidence head: loaded but not used by this
    // single-shot port (see file header comment).
    const int64_t markov_rank = 256; // dspark.dspark.markov_rank; fixed by the released checkpoint
    dspark_markov_head_a   = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_A,   "weight"), { markov_rank, n_vocab },         0);
    dspark_markov_head_b   = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_B,   "weight"), { markov_rank, n_vocab },         0);
    dspark_confidence_head   = create_tensor(tn(LLM_TENSOR_DSPARK_CONFIDENCE, "weight"), { n_embd + markov_rank, 1 },  0);
    dspark_confidence_head_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONFIDENCE, "bias"),   { 1 },                       0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dspark::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

template <>
ggml_tensor * llama_model_dspark::graph<true>::build_inp_embd_enc() const {
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// DSpark encoder: fuse target features (fc + hidden_norm), same shape as DFlash.
template <>
llama_model_dspark::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// DSpark decoder, dual-mode by batch type:
//   * embd batch  -> fused target features: project + inject K/V into the cache.
//   * token batch -> noise-block diffusion: attend over [committed, MASK...] to
//                    generate draft tokens, additively conditioned on a fixed
//                    log-SNR embedding (single-shot: see file header comment).
template <>
llama_model_dspark::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_kv * inp_attn = build_attn_inp_kv();

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    // KV cache injection
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_g = inp->embd;
        cb(inp_g, "inp_g_embeddings", -1);

        res->add_input(std::move(inp));

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * Kcur = build_lora_mm(layer.wk, inp_g);
            ggml_tensor * Vcur = build_lora_mm(layer.wv, inp_g);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur_injected", il);
            cb(Vcur, "Vcur_injected", il);

            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
        }

        res->t_embd = inp_g;

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    // tok_embd is DSpark's own (unlike DFlash, which borrows the target's)
    auto * tok_embd = model.tok_embd;
    GGML_ASSERT(tok_embd != nullptr && "DSpark decoder requires its own token embeddings");

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    // kept as a raw pointer for the Markov-head correction below: `inp` is moved
    // into res->add_input() but the underlying ggml_tensor is context-owned and
    // outlives that move.
    ggml_tensor * tokens_t = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    // log-SNR conditioning: OFF BY DEFAULT (opt-in via DSPARK_LOG_SNR_ENABLE=1).
    // v1 added a single shift ONCE to the pre-norm input and measured ZERO
    // effect on drafting (byte-identical accept counts swept across the whole
    // -9..+9 trained range) -- renormalized away by the very next RMSNorm. v2
    // tried re-injecting the SAME vector as a post-norm shift at every layer
    // (AdaLN-lite) to survive normalization -- measured RESULT: it makes accept
    // worse at every setting tested (5.06% no-cond vs 1.63% with-cond, isolated
    // from the Markov correction; 13.4% markov-only vs 3.7% markov+cond). The
    // DSpark paper (arXiv:2607.05147) confirms the reference architecture does
    // NOT condition on log-SNR at all (it's a KV-injection backbone like
    // DFlash), so these tensors are a PrismML-checkpoint-specific addition this
    // port cannot calibrate correctly from the gguf alone -- left load-only,
    // graph-computable via the env var for further research, off by default.
    ggml_tensor * snr_cond = nullptr;
    if (getenv("DSPARK_LOG_SNR_ENABLE")) {
        const int64_t n_snr_dim  = 128;
        const int64_t n_snr_half = n_snr_dim / 2;

        ggml_tensor * snr_idx  = ggml_arange(ctx0, 0.0f, (float) n_snr_half, 1.0f);
        ggml_tensor * snr_freq = ggml_exp(ctx0, ggml_scale(ctx0, snr_idx, -logf(10000.0f) / (float) n_snr_half));
        // DSPARK_LOG_SNR_OVERRIDE: dev-only knob for the noise-level ablation.
        float log_snr = model.dspark_min_log_snr;
        if (const char * e = getenv("DSPARK_LOG_SNR_OVERRIDE")) {
            log_snr = strtof(e, nullptr);
        }
        {
            ggml_tensor * snr_arg = ggml_scale(ctx0, snr_freq, log_snr);

            ggml_tensor * snr_emb = ggml_concat(ctx0, ggml_sin(ctx0, snr_arg), ggml_cos(ctx0, snr_arg), 0);
            cb(snr_emb, "dspark_log_snr_emb", -1);

            ggml_tensor * snr_h = build_lora_mm(model.dspark_log_snr_fc1, snr_emb);
            snr_h = ggml_add(ctx0, snr_h, model.dspark_log_snr_fc1_b);
            snr_h = ggml_silu(ctx0, snr_h);
            snr_h = build_lora_mm(model.dspark_log_snr_fc2, snr_h);
            snr_h = ggml_add(ctx0, snr_h, model.dspark_log_snr_fc2_b);
            cb(snr_h, "dspark_log_snr_cond", -1);

            snr_cond = snr_h;
        }
    }

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        if (snr_cond) {
            noise_norm = ggml_add(ctx0, noise_norm, snr_cond);
        }
        cb(noise_norm, "noise_norm", il);

        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, noise_norm);
        ggml_tensor * Vcur = build_lora_mm(layer.wv, noise_norm);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);

        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // cache-aware, non-causal attention
        ggml_tensor * cur = build_attn(inp_attn, layer.wo, NULL, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        if (snr_cond) {
            cur = ggml_add(ctx0, cur, snr_cond);
        }
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    // output is DSpark's own (unlike DFlash, which borrows the target's)
    auto * output = model.output;
    GGML_ASSERT(output != nullptr && "DSpark decoder requires its own output projection");

    // Markov-head sequential correction (arXiv:2607.05147 sec. on the Markov
    // head): the parallel backbone above produces per-position hidden states
    // h_1..h_{n-1} (h_0 is the anchor, already known -- not a draft candidate).
    // A lightweight low-rank bigram correction is then applied SEQUENTIALLY,
    // left-to-right, using the token actually sampled at k-1:
    //   B_k(x_{k-1}, .) = W1[x_{k-1}] @ W2                (dspark_markov_head_a/b)
    //   p_k(.)          = softmax(U_k + B_k)
    // Greedy (temp=0) sampling lets this run entirely on-device: x_{k-1} for
    // k==1 is the known anchor token (a view into the already-populated
    // `tokens_t` input); for k>1 it is ggml_argmax() of the PREVIOUS corrected
    // logits, fed back into ggml_get_rows() on markov_head_a -- a dynamic
    // (graph-computed) index, not a host readback, so the whole chain stays
    // in one graph. This DOES require DSPARK_MARKOV=1 sampling to be greedy
    // (temp=0); it is disabled automatically if the tensors are absent.
    // Skipped: the confidence head (dspark.confidence_head) -- computable from
    // the same [h_k; W1[x_{k-1}]] concat, but the paper uses it for THROUGHPUT
    // scheduling (variable-length verification under batched load), which this
    // single-stream greedy harness has no decision point for; left dormant.
    // the sequential per-position loop below is only meaningful (and only
    // cheap) at the real block size (dspark.dspark.block_size, e.g. 4); the
    // n_tokens==512-ish worst-case graphs used for one-time memory RESERVATION
    // must fall back to the plain single mul_mat, or unrolling ~500 argmax/
    // get_rows/mul_mat nodes blows the ggml scratch-context memory pool.
    const bool use_markov = model.dspark_markov_head_a != nullptr &&
                             model.dspark_markov_head_b != nullptr &&
                             !getenv("DSPARK_NO_MARKOV") &&
                             n_tokens <= 32;

    if (!use_markov || n_tokens < 2) {
        cur = build_lora_mm(output, cur);
        cb(cur, "result_output", -1);
        res->t_logits = cur;
    } else {
        std::vector<ggml_tensor *> logits_per_pos(n_tokens);

        // position 0 (the anchor) is not a draft candidate but t_logits must
        // cover the whole block for the generic logits-getters downstream.
        ggml_tensor * h0 = ggml_view_2d(ctx0, cur, n_embd, 1, cur->nb[1], 0);
        logits_per_pos[0] = build_lora_mm(output, h0);

        ggml_tensor * prev_idx = ggml_view_1d(ctx0, tokens_t, 1, 0); // x_0 = anchor (known)

        for (int64_t k = 1; k < n_tokens; ++k) {
            ggml_tensor * hk = ggml_view_2d(ctx0, cur, n_embd, 1, cur->nb[1], k * cur->nb[1]);
            ggml_tensor * Uk = build_lora_mm(output, hk);

            // DSPARK_MARKOV_SWAP=1: dev-only ablation swapping which of the two
            // identically-shaped (256,vocab) tensors plays embedding-table vs
            // projection role (the gguf gives no way to tell apart from shape
            // alone; a/b naming + dtype (a=bf16, b=q4_1) is our best guess).
            ggml_tensor * markov_embed_table = getenv("DSPARK_MARKOV_SWAP") ? model.dspark_markov_head_b : model.dspark_markov_head_a;
            ggml_tensor * markov_proj        = getenv("DSPARK_MARKOV_SWAP") ? model.dspark_markov_head_a : model.dspark_markov_head_b;

            ggml_tensor * embed_prev = ggml_get_rows(ctx0, markov_embed_table, prev_idx);
            ggml_tensor * Bk         = build_lora_mm(markov_proj, embed_prev);

            ggml_tensor * Uk_corr = ggml_add(ctx0, Uk, Bk);
            cb(Uk_corr, "dspark_markov_logits", (int) k);

            logits_per_pos[k] = Uk_corr;
            prev_idx = ggml_argmax(ctx0, Uk_corr); // greedy x_k, fed into the next iteration
        }

        cur = logits_per_pos[0];
        for (int64_t k = 1; k < n_tokens; ++k) {
            cur = ggml_concat(ctx0, cur, logits_per_pos[k], 1);
        }
        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}
