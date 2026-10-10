#include "models.h"
#include "llama-memory-recurrent.h"

#include <cstring>

// GGML_GDN_STATE_INPLACE promoted to default-on (T360, 2026-10-07): unset = on, "0" = off. Exact on R9700
// Bonsai-27B (greedy text and top-5 logprobs identical to the gather/write-back graph), Q1_0 +4.0%, Q2_0 +3.5%.
// GGML_GDN_FUSED_BA and GGML_GDN_FUSED_L2NORM default-on too (T361, 2026-10-08), same unset = on, "0" = off.
// Bit-exact vs the unfused graph since the T361 kernel fixes (server text + top-5 logprobs identical, decode
// KL equal to the off/off control); on top of INPLACE: Q1_0 +6.3%, Q2_0 +3.9%.
// GGML_GDN_CONV_INPLACE default-on (T368, 2026-10-08), same convention: exact (12/12 logprob-identical, decode KL =
// control), Q1_0 +5.0%, Q2_0 +4.2% tg128 (144 fewer dependent launches/token).
static bool gdn_rung_default_on(const char * name) {
    const char * e = getenv(name);
    return e == nullptr || strcmp(e, "0") != 0;
}

// The fused BA and L2-norm paths exist only in the ROCm/CUDA gated_delta_net kernel; the CPU op asserts on
// them, so a CPU-placed layer (partial offload) must take the unfused path.
static bool gdn_layer_on_cuda_like(const llama_model & model, int il) {
    ggml_backend_dev_t dev = model.dev_layer(il);
    if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
    return reg != nullptr && (strcmp(reg, "ROCm") == 0 || strcmp(reg, "CUDA") == 0);
}

// [TAG_MMVQ_MTAB] T395: by default (GGML_GEMV_FUSE=0 disables) the four GDN input projections (qkv, z,
// beta, alpha: all read attn_norm; GGML_GEMV_FUSE_GROUPS bit 1) and q/k/v (bit 2) are
// pinned next to each other in single-token graphs, so the CUDA backend can run each
// set as ONE matrix-table GEMV. Execution order only; every op and value is unchanged.
// GGML_GEMV_FUSE=0 = graph untouched. GROUPS defaults to 3 (both).
static int qwen35_hfuse_mask() {
    static const int m = [] {
        const char * e = getenv("GGML_GEMV_FUSE");
        if (e != nullptr && atoi(e) == 0) {
            return 0;
        }
        const char * g = getenv("GGML_GEMV_FUSE_GROUPS");
        return g ? atoi(g) : 3;
    }();
    return m;
}

void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // NextN/MTP (Qwen3.5/3.6): extra decoder block appended beyond the main stack
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < n_layer_impl");

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, mtp_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, mtp_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen35::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    const bool hfuse_pin = (qwen35_hfuse_mask() & 1) && ubatch.n_tokens == 1;   // [TAG_MMVQ_MTAB]
    // T439 GGML_CUDA_B2_ADDNORM_FUSE: emit the Hadamard-folded qkv/z matmul chains right behind the attn_norm that the layer loop
    // expanded. Without it a cold recurrent state (first ubatch of a prompt) gets build_rs's zero-clear and gather nodes between the
    // norm and its sign flip, which breaks the ADD+RMS_NORM+MUL+MUL+RESHAPE+MUL_MAT adjacency the CUDA producer fusion matches.
    static const bool b2_order = [] { const char * e = getenv("GGML_CUDA_B2_ADDNORM_FUSE"); return e && atoi(e) != 0; }();
    const bool pin_order = hfuse_pin || (b2_order && ubatch.n_tokens > 1);
    if (pin_order) {
        ggml_build_forward_expand(gf, qkv_mixed);
    }
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);
    if (pin_order) {
        ggml_build_forward_expand(gf, z);
    }

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);
    const bool hfuse_pin = (qwen35_hfuse_mask() & 2) && ubatch.n_tokens == 1;   // [TAG_MMVQ_MTAB]
    if (hfuse_pin) {
        ggml_build_forward_expand(gf, Qcur_full);
    }

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);
    if (hfuse_pin) {
        ggml_build_forward_expand(gf, Kcur);
    }

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);
    if (hfuse_pin) {
        ggml_build_forward_expand(gf, Vcur);
    }

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    // GGML_GDN_GATED_NORM (T447): expand the z projection here so its nodes precede the recurrence. Graph order is a
    // depth-first walk from the output, which otherwise puts z's quantize+GEMM BETWEEN the gated norm's RMS_NORM/MUL and its
    // SILU(z)*normed; hoisting makes RMS_NORM, MUL, SILU, MUL adjacent so the CUDA backend can run them as one kernel. Same
    // values, different (still valid) topological order; prefill-sized ubatches only.
    if (n_seq_tokens > 32) {   // prefill-sized only: spec-verify batches keep the existing graph and fusions
        const char * gn = getenv("GGML_GDN_GATED_NORM");   // default ON, =0 disables
        if (gn == nullptr || atoi(gn) != 0) {
            ggml_build_forward_expand(gf, z);
        }
    }

    // EXPERIMENT (env-gated GGML_GDN_FUSED_BA, ported from PrismML
    // megakernel/rmsnorm-qmv-fuse commit c92cf5ebc): fold sigmoid(beta) and
    // softplus(alpha+ssm_dt)*ssm_a into the GDN kernel, skipping 4 separate
    // elementwise dispatches/layer. Fused K=1 decode path only.
    //
    // Only the CUDA/HIP kernel implements FusedBA (gated_delta_net.cu); the
    // CPU op asserts on it, so gdn_layer_on_cuda_like keeps CPU-placed layers
    // on the unfused path.
    //
    // Gate requires cparams.n_rs_seq == 0: build_recurrent_attn's keep==true
    // branch (rollback active) passes gate/beta straight into the base
    // ggml_gated_delta_net call with explicit nullptr,nullptr for ssm_dt/
    // ssm_a, so raw unactivated values must never reach that path.
    const bool gdn_fused_ba = gdn_rung_default_on("GGML_GDN_FUSED_BA") && gdn_layer_on_cuda_like(model, il) &&
                              n_seq_tokens == 1 && cparams.n_rs_seq == 0;

    // GGML_GDN_FUSED_L2NORM (T180 follow-on): fold the two upstream
    // GGML_OP_L2_NORM dispatches (q_conv, k_conv) into the GDN kernel,
    // matching GGML_GDN_FUSED_BA's exact gating discipline above (cparams.
    // fused_gdn_ar/fused_gdn_ch are hardcoded true in this fork, so
    // n_seq_tokens==1 && n_rs_seq==0 is sufficient to guarantee the fused
    // K=1 kernel path is actually taken; build_delta_net/build_delta_net_fused
    // carry a GGML_ASSERT that would catch it if that assumption ever broke).
    // GGML_GDN_FUSED_L2NORM_PF (T447, default off): the same fold for prefill-sized ubatches. The chunked GDN kernel's prep stage already
    // takes raw q/k and derives the per-token L2 scale itself (same lane order and rsqrt as l2_norm_f32<32>), so the two L2_NORM
    // launches (and their 8 MB-per-tensor round trip) disappear. n_seq_tokens >= 64 is the chunk kernel's own threshold
    // (GGML_GDN_CHUNK_MIN): below it the sequential kernel runs, whose multi-token L2 mode was never gated, and spec-verify batches stay unfused.
    const bool gdn_fused_l2norm_pf = []() { const char * e = getenv("GGML_GDN_FUSED_L2NORM_PF"); return e == nullptr || atoi(e) != 0; }() &&   // default ON, =0 disables
                                     gdn_layer_on_cuda_like(model, il) && n_seq_tokens >= 64 && cparams.n_rs_seq == 0 &&
                                     hparams.ssm_d_state == 128;
    const bool gdn_fused_l2norm = (gdn_rung_default_on("GGML_GDN_FUSED_L2NORM") && gdn_layer_on_cuda_like(model, il) &&
                                   n_seq_tokens == 1 && cparams.n_rs_seq == 0) || gdn_fused_l2norm_pf;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    const bool hfuse_pin = (qwen35_hfuse_mask() & 1) && ubatch.n_tokens == 1;   // [TAG_MMVQ_MTAB]
    if (hfuse_pin) {
        ggml_build_forward_expand(gf, beta);
    }
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    if (hfuse_pin) {
        ggml_build_forward_expand(gf, alpha);
    }
    alpha = ggml_reshape_4d(ctx0, alpha, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * gate;
    if (gdn_fused_ba) {
        // raw beta/alpha ride straight through to ggml_gated_delta_net; the
        // kernel computes sigmoid(beta) / softplus(alpha+ssm_dt)*ssm_a itself.
        gate = alpha;
    } else {
        beta = ggml_sigmoid(ctx0, beta);
        cb(beta, "beta_sigmoid", il);

        ggml_tensor * alpha3 = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
        ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha3, model.layers[il].ssm_dt);
        ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
        cb(alpha_softplus, "a_softplus", il);

        gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
        cb(gate, "gate", il);

        gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    }

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    // EXPERIMENT (GGML_GDN_CONV_INPLACE, default-on: unset = on, "0" = off; card T368): single-token decode reads the
    // conv state straight from its cache row and updates it in place (vLLM causal_conv1d_update pattern) in
    // ONE kernel, replacing build_rs's get_rows gather + concat + ssm_conv/silu + the write-back cpy (3
    // dependent launches per GDN layer removed). Same arithmetic as the ssm_conv path. Gated exactly like
    // gdn_state_inplace below (same direct-view validity argument and the same cold-start rule: a pending
    // zero-clear, rs_z >= 0, falls back to build_rs), plus n_seq_tokens == 1 since the op is single-token.
    const bool gdn_conv_inplace =
        gdn_rung_default_on("GGML_GDN_CONV_INPLACE") &&
        n_seq_tokens == 1 &&
        cparams.n_rs_seq == 0 &&
        n_seqs == 1 &&
        inp->mctx->get_n_rs() == 1 &&
        inp->mctx->get_rs_z() < 0;

    // GGML_GDN_GLUE_FUSE (default ON, =0 disables; card T409, gated 2026-10-09: KLD = 1-ulp floor, seq_cp reuse matrix 14/14, +3% prefill): prefill counterpart of GGML_GDN_CONV_INPLACE.
    // One multi-token ssm_conv_update reads the conv state row directly (or treats it as zeros on a fresh cell)
    // and the new tokens in qkv_mixed's native layout, applies silu, and writes the new state back in place:
    // removes build_rs's gather, the transposed concat (21 MB at ub 1024), the separate silu fusion and the
    // write-back cpy. n_seq_tokens > 32 keeps every small batch (spec verify M=2..9) on the default path.
    // Same direct-view validity rule as gdn_conv_inplace (single cell, no rollback slots), except a pending
    // zero-clear of THAT cell (rs_z == head, the fresh first ubatch of a prompt) is folded into the kernel as
    // state_is_zero instead of falling back. s_copy(0) == head checks that the gather is the identity.
    // GGML_GDN_GLUE_L2=1 additionally folds the two q/k L2_NORM ops into the same kernel (producer side, not
    // into the recurrence loop: T296).
    const auto gdn_env_on = [](const char * name) {
        const char * e = getenv(name);
        return e != nullptr && strcmp(e, "1") == 0;
    };
    const int32_t rs_head_cell = (int32_t) inp->mctx->get_head();
    const bool gdn_glue_fuse =
        !gdn_conv_inplace &&
        gdn_rung_default_on("GGML_GDN_GLUE_FUSE") &&
        gdn_layer_on_cuda_like(model, il) &&
        n_seq_tokens > 32 &&
        cparams.n_rs_seq == 0 &&
        n_seqs == 1 &&
        inp->mctx->get_n_rs() == 1 &&
        inp->mctx->s_copy(0) == rs_head_cell &&
        (inp->mctx->get_rs_z() < 0 || inp->mctx->get_rs_z() == rs_head_cell) &&
        conv_channels % 128 == 0;
    const bool gdn_glue_l2 = gdn_glue_fuse && gdn_env_on("GGML_GDN_GLUE_L2") && head_k_dim == 128 &&
                             !gdn_fused_l2norm;

    ggml_tensor * conv_input       = nullptr;
    ggml_tensor * conv_output_silu = nullptr;
    if (gdn_glue_fuse) {
        // the kernel never reads s_copy: a reused graph must recheck s_copy(0) == head (seq_cp'd cell)
        inp->s_copy_identity = true;
        ggml_tensor * conv_state = build_rs_state_view(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
        conv_state = ggml_reshape_3d(ctx0, conv_state, conv_kernel_size - 1, conv_channels, n_seqs);
        cb(conv_state, "conv_state_inplace_view", il);

        const int32_t l2_n = gdn_glue_l2 ? (int32_t) (2 * head_k_dim * num_k_heads) : 0;
        conv_output_silu = ggml_ssm_conv_update_ext(ctx0, conv_state, qkv_mixed, conv_kernel, true,
                                                    inp->mctx->get_rs_z() >= 0, l2_n, hparams.f_norm_rms_eps);
        cb(conv_output_silu, "conv_output_silu", il);
        static bool logged[2] = { false, false }; // once per state_is_zero value
        if (!logged[inp->mctx->get_rs_z() >= 0]) {
            logged[inp->mctx->get_rs_z() >= 0] = true;
            LLAMA_LOG_INFO("%s: T409 GDN glue fusion active (n_tokens=%d, state_is_zero=%d, l2=%d)\n", __func__,
                           (int) n_seq_tokens, (int) (inp->mctx->get_rs_z() >= 0), (int) gdn_glue_l2);
        }
    } else if (gdn_conv_inplace) {
        ggml_tensor * conv_state = build_rs_state_view(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
        conv_state = ggml_reshape_3d(ctx0, conv_state, conv_kernel_size - 1, conv_channels, n_seqs);
        cb(conv_state, "conv_state_inplace_view", il);

        conv_output_silu = ggml_ssm_conv_update(ctx0, conv_state, qkv_mixed, conv_kernel, true);
        cb(conv_output_silu, "conv_output_silu", il);
    } else {
        conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);
    }

    // EXPERIMENT (env-gated GGML_GDN_STATE_INPLACE, ported from PrismML
    // megakernel/rmsnorm-qmv-fuse commit 6d8333b3d): at plain batch=1 decode
    // the per-layer state gather is an identity permutation -- read a direct
    // view of the cache row instead. Requires n_rs_seq == 0 (rollback is a
    // structurally separate code path, see the `keep` bool in
    // build_recurrent_attn) and get_n_rs() == 1 (single occupied cache row --
    // false the instant a second sequence's row exists, e.g. multi-slot
    // serving).
    //
    // FOUND DURING PORTING (real bug, not shipped broken): build_rs does two
    // things, not one -- gather AND a cache-hygiene zero-clear
    // (ggml_scale_inplace(state_zero, 0)) that fires whenever get_rs_z() >= 0
    // (a fresh/reset sequence slot -- true on the very first ubatch of a cold
    // context, since GPU buffers are not zero-initialized on alloc). The
    // direct-view path skips build_rs entirely, so it would also skip that
    // clear and read uninitialized memory on a cold start. Measured: with
    // only this rung enabled, greedy output diverged from baseline starting
    // partway through generation (root-caused to this, not FP reordering).
    // Extra gate: only take the fast path when there is no pending clear
    // this call; every case that needs the clear falls back to build_rs.
    const bool gdn_state_inplace =
        gdn_rung_default_on("GGML_GDN_STATE_INPLACE") &&
        cparams.n_rs_seq == 0 &&
        n_seqs == 1 &&
        inp->mctx->get_n_rs() == 1 &&
        inp->mctx->get_rs_z() < 0;

    ggml_tensor * state;
    if (gdn_state_inplace) {
        state = build_rs_state_view(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
        state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
        cb(state, "state_inplace_view", il);
    } else {
        state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
        state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
        cb(state, "state_predelta", il);
    }

    if (!gdn_conv_inplace && !gdn_glue_fuse) {
        ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
        cb(conv_output_proper, "conv_output_raw", il);

        conv_output_silu = ggml_silu(ctx0, conv_output_proper);
        cb(conv_output_silu, "conv_output_silu", il);
    }

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    // GGML_GDN_FUSED_L2NORM: skip both L2_NORM dispatches here, pass RAW
    // q_conv/k_conv through -- the GDN kernel normalizes internally (see
    // ggml_gated_delta_net's l2norm_qk doc comment / gated_delta_net_cuda's
    // L2Norm template branch). When the flag is off, unchanged behavior.
    if (!gdn_fused_l2norm && !gdn_glue_l2) {
        q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
        k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);
    }

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il,
            gdn_fused_ba ? model.layers[il].ssm_dt : nullptr,
            gdn_fused_ba ? model.layers[il].ssm_a  : nullptr,
            gdn_fused_l2norm,
            eps_norm);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
llama_model_qwen35::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // hparams.n_layer includes both main model layers and MTP layers. The MTP
    // layers are stored immediately after the main layers in model.layers[],
    // one per trained head. Multi-block MTP (n_layer_nextn > 1, DeepSeek-V3 /
    // Step3.5 style): the DECODER_MTP graph runs the head selected by
    // cparams.nextn_layer_offset (0 = first trained head, chained head->head+1
    // by the speculative driver for depth>1 drafts). offset 0 keeps
    // single-block behavior identical to before. See step35.cpp graph_mtp.
    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range [0, n_layer_nextn)");
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
