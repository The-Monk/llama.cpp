#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>
#include <thread>
#include <cstdlib>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model
    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;

    // mirror tools/server/server-context.cpp's has_draft/spec_mtp split so MTP
    // self-speculation works from this example too (previously it unconditionally
    // tried to load a second full model from an empty --spec-draft-model path).
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp  = std::find(params.speculative.types.begin(),
                                      params.speculative.types.end(),
                                      COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    // [FIX card ASYNC-ACCEPT-1] Bug3: MTP is not the only drafter type whose draft()
    // depends on ctx_tgt's *just-computed* forward output. EAGLE3 and DFlash/DSpark
    // (feature-injection block-diffusion drafters, see src/models/dflash.cpp /
    // dspark.cpp) read the target's intermediate-layer activations
    // (llama_get_embeddings_layer_inp(ctx_tgt, ...) inside
    // common_speculative_impl_draft_dflash::process()) for the *current* verify
    // batch and inject them into ctx_dft before draft() can meaningfully sample the
    // next block. need_n_rs_seq() > 0 is exactly the set of types with this
    // property (it's why they need bounded recurrent-state rollback in the first
    // place: they're deeply coupled to the target's own positions/features, unlike
    // draft-simple which is a fully independent model). Any code path that skips
    // process() for these types, or runs it concurrently with the ctx_tgt decode it
    // reads from, feeds the draft stale-or-wrong features -- this collapsed real
    // measured dspark acceptance from ~28.6% (sequential) to ~1% (async modes 1/2)
    // while target-verify stayed exact (coherent output, just a starved draft).
    const bool spec_needs_tgt_process = params.speculative.need_n_rs_seq() > 0;

    // TODO: simplify this logic
    if (has_draft) {
        const auto & params_spec = params.speculative.draft;

        auto params_dft = params;

        params_dft.devices      = params_spec.devices;
        params_dft.model        = params_spec.mparams;
        params_dft.n_gpu_layers = params_spec.n_gpu_layers;
        params_dft.cache_type_k = params_spec.cache_type_k;
        params_dft.cache_type_v = params_spec.cache_type_v;

        if (params_spec.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads       = params.speculative.draft.cpuparams.n_threads;
            params_dft.cpuparams_batch.n_threads = params.speculative.draft.cpuparams_batch.n_threads;
        }

        params_dft.tensor_buft_overrides = params.speculative.draft.tensor_buft_overrides;

        auto mparams_dft = common_model_params_to_llama(params_dft);

        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (model_dft == nullptr) {
            LOG_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
            return 1;
        }

        auto cparams = common_context_params_to_llama(params_dft);
        if (spec_mtp) {
            // a real second model was loaded (e.g. -md pointing at the same gguf so it
            // can sit on a different device); restrict the graph to the nextn subgraph
            // instead of running the full stack as the "draft". n_rs_seq=0: this draft
            // context does not itself need the recurrent-state lookahead slots that
            // params_dft may have inherited (SSM/hybrid archs) for the target side.
            cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            cparams.n_rs_seq = 0;
        }
        cparams.ctx_other = ctx_tgt;
        ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return 1;
        }

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft.get();
    } else if (spec_mtp) {
        // MTP self-speculation, no separate draft model given: the nextn head lives
        // inside the target gguf, so reuse model_tgt's already-loaded weights and just
        // add a lightweight MTP context (own KV-cache + compute buffers, no second copy
        // of the model weights).
        auto cparams_mtp = common_context_params_to_llama(params);
        cparams_mtp.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
        cparams_mtp.type_k    = params.speculative.draft.cache_type_k;
        cparams_mtp.type_v    = params.speculative.draft.cache_type_v;
        cparams_mtp.n_rs_seq  = 0;
        cparams_mtp.ctx_other = ctx_tgt;

        ctx_dft.reset(llama_init_from_model(model_tgt, cparams_mtp));
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create MTP context\n");
            return 1;
        }

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft.get();
    } else {
        LOG_ERR("%s", "no draft model given (--spec-draft-model) and --spec-type does not include draft-mtp\n");
        return 1;
    }

    // check if the context supports partial sequence removal
    const bool use_ckpt_tgt = (common_context_can_seq_rm(ctx_tgt)       == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);
    const bool use_ckpt_dft = (common_context_can_seq_rm(ctx_dft.get()) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    llama_seq_id seq_id = 0;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // init the speculator before the prompt eval: draft-mtp's process() hook needs to run
    // on the same prompt batch that's decoded into ctx_tgt (it pulls h_tgt via
    // llama_get_embeddings_nextn(ctx_tgt) and feeds ctx_dft with embeddings, not raw token
    // ids -- unlike draft-simple's process(), which does the equivalent of a plain
    // llama_decode(ctx_dft, batch) with real tokens). Previously this example only ever did
    // a manual llama_decode(ctx_dft, ...) with tokens here, which is correct for
    // draft-simple but leaves an MTP draft context's prefill state never populated
    // (0% useful predictions, ~1-2% accept by chance).
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);

    // eval the prompt. common_speculative_process (needed by draft-mtp) reads
    // batch.n_seq_id[k]/seq_id[k], which llama_batch_get_one() leaves null (it's meant
    // for the simplest single-call decode path, not for feeding into the speculator) --
    // build a real batch with seq_id populated instead.
    llama_batch batch_prompt = llama_batch_init((int32_t) inp.size() - 1, 0, 1);
    for (size_t i = 0; i + 1 < inp.size(); ++i) {
        common_batch_add(batch_prompt, inp[i], (llama_pos) i, { seq_id }, false);
    }
    llama_decode(ctx_tgt, batch_prompt);
    if (!common_speculative_process(spec, batch_prompt)) {
        LOG_ERR("%s", "failed to process speculative batch during prompt eval\n");
        return 1;
    }
    llama_batch_free(batch_prompt);

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    common_speculative_begin(spec, seq_id, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    size_t n_draft = 0;

    llama_tokens draft;
    common_prompt_checkpoint ckpt;

    const auto t_enc_end = ggml_time_us();

    // T112: async overlap of target-verify and draft-reeval on separate streams.
    // These two llama_decode calls consume the same batch_tgt but are mutually
    // independent (distinct contexts, neither reads the other's output; the accept
    // step reads only ctx_tgt). Running them concurrently on their per-context
    // non-blocking CUDA streams hides the smaller draft-reeval behind the target
    // verify. Gated by env so a single binary can A/B sequential vs async.
    // 0 = sequential baseline, 1 = overlap verify||reeval, 2 = pipeline (draft-gen[N+1] || verify[N])
    const int spec_mode = [] {
        const char * e = getenv("LLAMA_SPEC_ASYNC");
        return e ? atoi(e) : 0;
    }();
    LOG_INF("T112 spec-async mode: %d (%s)\n", spec_mode,
            spec_mode == 2 ? "PIPELINE" : spec_mode == 1 ? "verify||reeval" : "sequential");

    const auto t_dec_start = ggml_time_us();

    // [FIX card ASYNC-ACCEPT-1] Bug3: this whole pipeline block launches gen() (==
    // common_speculative_draft(), i.e. draft() only) for block N+1 concurrently
    // with -- or, worse, entirely INSTEAD of ever calling -- common_speculative_
    // process() on block N's just-verified batch. For draft-simple that's fine
    // (draft() there IS the whole story). For DFlash/DSpark/EAGLE3/MTP, process()
    // is the step that reads ctx_tgt's just-computed intermediate-layer features
    // for block N and injects them into ctx_dft; draft() for block N+1 then cross-
    // attends to that injected KV. Skipping process() (mode 2 never calls it at
    // all) or racing it against the ctx_tgt decode it reads from (what mode 1 used
    // to do) feeds these drafters STALE conditioning -- measured: dspark accept
    // 28.6% (sequential, correct ordering) -> 0.78% (this pipeline, pre-fix).
    //
    // The dependency chain is hard and sequential no matter how it's scheduled:
    //   decode(ctx_tgt, block N) -> process(block N)  [reads ctx_tgt's output]
    //     -> draft(block N+1)    [reads process()'s injected KV]
    // draft(block N+1) cannot legitimately start until process(block N) has run,
    // and process(block N) cannot start until decode(ctx_tgt, block N) has
    // returned -- so for these types there is NO valid placement of "draft(N+1) ||
    // verify(N)" overlap at the call-ordering level; it collapses to exactly what
    // sequential mode already does. (A genuine partial overlap would require
    // exposing a mid-forward-pass hook so dspark's *shallow* target_layer_ids
    // features -- computed early in ctx_tgt's stack -- could be read and injected
    // while ctx_tgt's later layers are still computing; that needs a per-layer
    // callback this driver doesn't have today, not a call-ordering fix. Left as
    // future work, noted below.) So: for spec_needs_tgt_process types, skip this
    // specialized (broken-for-them) pipeline entirely and fall through to the
    // plain sequential loop below, which already calls process() correctly.
    if (spec_mode == 2 && !spec_needs_tgt_process) {
        // ===== T112 PIPELINE: overlap draft-gen[N+1] (ctx_dft) with verify[N] (ctx_tgt) =====
        // The draft over-generates by one: the extra token is a *guess* for the target's bonus
        // token. Block N is kept in ctx_dft KV, so while the target verifies block N the draft
        // speculatively generates block N+1 seeded from that bonus-guess at its confirmed
        // position. On a hit (full accept && bonus == guess) ctx_dft KV already equals the
        // confirmed state -> reuse it, no reeval. On a miss, roll ctx_dft/ctx_tgt back to the
        // confirmed prefix and regenerate. Output is identical to sequential in every case.
        // Only needs suffix seq_rm (always available), not FULL-checkpoint support.

        auto trim = [&](llama_context * c, int pos) {
            llama_memory_seq_rm(llama_get_memory(c), seq_id, pos, -1);
        };

        // [FIX card ASYNC-PIPELINE-POS-1] `draft()` (common_speculative_impl_draft_dflash)
        // always requests exactly `params.n_max` (the configured/clamped block width) new
        // mask tokens every call -- it ignores dp.n_max (-1 here) and just uses its own
        // default. On a "hit", the *next* seed (bonus_guess) is the LAST of those
        // already-decoded block tokens, so it already occupies a position at the tail of
        // ctx_dft's KV cache. Re-submitting it as position `past` via a plain decode
        // collides with what's already there (llama_decode's consecutive-position check
        // fails with X == Y instead of Y == X + 1, observed as "llama_decode returned -1"
        // repeating every iteration once the pipeline goes async). The block-diffusion
        // draft only needs the *identity* of the seed token, not its stale hidden state
        // from the speculative pass that guessed it -- trim ctx_dft back to `past` first so
        // the position is free to be recomputed as the head of the new block.
        auto gen = [&](llama_token seed, int past, llama_tokens & out) {
            out.clear();
            trim(ctx_dft.get(), past);
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting = */ true,
                /* .n_max    = */ -1,
                /* .n_past   = */ past,
                /* .id_last  = */ seed,
                /* .prompt   = */ &prompt_tgt,
                /* .result   = */ &out,
            };
            common_speculative_draft(spec);
        };
        auto split_guess = [](llama_tokens & blk) -> llama_token {
            if ((int) blk.size() >= 2) { llama_token g = blk.back(); blk.pop_back(); return g; }
            return LLAMA_TOKEN_NULL;
        };

        // block 0 (sequential; nothing to overlap yet). Keep its ctx_dft KV.
        gen(id_last, n_past, draft);
        llama_token bonus_guess = split_guess(draft);

        while (true) {
            const int p0 = n_past;

            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, id_last, p0, { seq_id }, true);
            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], p0 + 1 + (int) i, { seq_id }, true);
            }

            const int  spec_past = p0 + (int) draft.size() + 1;   // the bonus position
            const bool can_spec  = (bonus_guess != LLAMA_TOKEN_NULL) && !draft.empty();

            llama_tokens spec_next;
            std::thread th_dft;
            if (can_spec) {
                th_dft = std::thread([&] { gen(bonus_guess, spec_past, spec_next); });
            }
            llama_decode(ctx_tgt, batch_tgt);
            if (can_spec) th_dft.join();

            auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
            GGML_ASSERT(ids.size() > 0);

            const bool         full_accept = (ids.size() - 1 == draft.size());
            const llama_token  bonus       = ids.back();
            const bool         hit         = full_accept && can_spec && (bonus == bonus_guess);

            common_speculative_accept(spec, seq_id, ids.size() - 1);

            bool stop = false;
            for (size_t i = 0; i < ids.size(); ++i) {
                prompt_tgt.push_back(id_last);
                id_last = ids[i];
                if (llama_vocab_is_eog(vocab, id_last)) { has_eos = true; stop = true; break; }
                LOG("%s", common_token_to_piece(ctx_tgt, id_last).c_str());
            }
            n_past    += (int) ids.size();
            n_drafted += (int) draft.size();
            n_accept  += (int) ids.size() - 1;
            n_predict += (int) ids.size();

            trim(ctx_tgt, n_past);   // drop any rejected verify positions

            if (stop || (params.n_predict >= 0 && n_predict > params.n_predict)) break;

            if (hit) {
                // ctx_dft already holds block N+1 (+ its bonus guess) at confirmed positions.
                draft = std::move(spec_next);
                bonus_guess = split_guess(draft);
            } else {
                // miss/partial: discard the speculative draft, roll ctx_dft back to the
                // confirmed prefix, and regenerate the next block from the true id_last.
                trim(ctx_dft.get(), n_past);
                gen(id_last, n_past, draft);
                bonus_guess = split_guess(draft);
            }
        }
    } else
    while (true) {
        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // generate a new draft
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ -1,
                /* .n_past     = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft, // output
            };
            common_speculative_draft(spec);

            // save the original draft size
            n_draft = draft.size();

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            if (!draft.empty()) {
                if (use_ckpt_tgt) {
                    ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }

            {
                ckpt.load_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, ckpt.pos_max + 1, -1);
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            if (use_ckpt_tgt) {
                GGML_ASSERT(!ckpt.empty());
            }
        }

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { seq_id }, true);

        // build the verify batch: [id_last, draft0, draft1, ..., draftN-1]
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
        }

        //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

        if (spec_mode == 1 && !spec_needs_tgt_process) {
            // T112: run draft-reeval concurrently with target-verify on a separate
            // thread → separate per-context stream → GPU overlaps the two graphs.
            // batch_tgt is read-only to both decodes, so shared access is safe.
            // Only valid for draft-simple (and other types with NO dependency on
            // ctx_tgt's own just-computed output): draft-mtp/eagle3/dflash/dspark's
            // process() reads ctx_tgt's just-computed embeddings_nextn / intermediate
            // layer activations, so it must run strictly after ctx_tgt's decode, not
            // concurrently with it -- falls through to the sequential path below.
            // [card ASYNC-ACCEPT-1] this used to be gated on `!spec_mtp` only, which
            // let dflash/dspark through this raw-llama_decode(ctx_dft, batch_tgt)
            // path -- WRONG on two counts: (1) it decodes ctx_tgt's real token ids
            // straight into ctx_dft, bypassing dspark's mask-diffusion input format
            // entirely; (2) even calling the correct process() hook here would race
            // ctx_tgt's decode for the features it reads. Measured effect: dspark
            // accept collapsed 28.6% (sequential) -> 1.59% (this path). Broadened the
            // guard to `!spec_needs_tgt_process` (== need_n_rs_seq() > 0) so DFlash/
            // DSpark/EAGLE3 fall through to the correct sequential process() call below.
            std::thread th_dft([&] {
                llama_decode(ctx_dft.get(), batch_tgt);
            });
            llama_decode(ctx_tgt, batch_tgt);
            th_dft.join();
        } else {
            // evaluate the target model, then process the draft/MTP side. Using
            // common_speculative_process() here (not a raw llama_decode(ctx_dft, ...))
            // is required for draft-mtp -- it pulls ctx_tgt's embeddings_nextn output
            // and feeds the nextn head; for draft-simple its process() is exactly the
            // equivalent llama_decode(ctx_dft, batch), so this is a safe generalization.
            llama_decode(ctx_tgt, batch_tgt);
            if (!common_speculative_process(spec, batch_tgt)) {
                LOG_ERR("%s", "failed to process speculative batch during verify\n");
                break;
            }
        }

        // only save the sampler sampler state if we use checkpoints
        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_ckpt_tgt && ids.size() - 1 < draft.size()) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, draft.size());

            draft = std::move(ids);

            {
                ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            }

            {
                ckpt.load_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, ckpt.pos_max + 1, -1);
            }

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        common_speculative_accept(spec, seq_id, ids.size() - 1);

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt),       seq_id, n_past, -1);
            llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, n_past, -1);
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());

    llama_batch_free(batch_tgt);

    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
