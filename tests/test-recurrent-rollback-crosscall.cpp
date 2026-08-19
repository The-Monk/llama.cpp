// Repro/diagnostic harness for cross-call recurrent-state rollback on hybrid
// attn+recurrent (delta-net) models. Unlike test-recurrent-state-rollback.cpp
// (which decodes the whole prompt window in ONE llama_decode() call, so the
// rollback stays inside the snapshot planes that same call produced), this
// harness decodes the prefix ONE TOKEN AT A TIME across SEPARATE llama_decode()
// calls (the real shape of interactive/chat autoregressive generation), then
// rolls back R > 1 tokens and checks the resulting state against a from-scratch
// ground truth that used the identical per-token call pattern up to the same
// point. If the recurrent-state rollback snapshots don't carry history across
// call boundaries, this diverges (large max|logit_src - logit_ref|); a correct
// implementation matches to fp noise (~1e-5 in fp32, looser in quantized paths).
//
// Also runs the same-call (single-batch) case as a positive control so a
// harness bug can't be confused with the cross-call bug.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <vector>

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t n_rs_seq) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = n_rs_seq;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (n_rs_seq + 4));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (n_rs_seq + 4));
    return llama_init_from_model(model, cparams);
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// decode tokens[0 .. count) one token per llama_decode() call (positions 0..count-1)
static bool decode_sequential(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    for (uint32_t pos = 0; pos < count; ++pos) {
        if (!decode_one(ctx, tokens[pos], (llama_pos) pos)) {
            return false;
        }
    }
    return true;
}

static double max_abs_logit_diff(const float * a, const float * b, int n_vocab) {
    double m = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        m = std::max(m, (double) std::fabs(a[i] - b[i]));
    }
    return m;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    // probe context to find the (possibly clamped) n_rs_seq for this arch
    const uint32_t n_rs_seq_req = 6;
    llama_context * ctx_probe = make_ctx(params, model, n_rs_seq_req);
    if (ctx_probe == nullptr) {
        fprintf(stderr, "%s : failed to init probe context\n", __func__);
        return 1;
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_probe);
    llama_free(ctx_probe);

    if (n_rs_seq == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled/unsupported for this arch\n", __func__);
        return 0;
    }

    const llama_vocab * vocab_top = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(vocab_top, "The quick brown fox jumps over the lazy dog and then runs", true);
    const uint32_t n_decode = std::min<uint32_t>((uint32_t) tokens.size() - 1, n_rs_seq + 2);
    if (n_decode < 3) {
        fprintf(stderr, "%s : not enough prompt tokens (need >= 3, have %u)\n", __func__, n_decode);
        return 1;
    }

    // Case 6: mirrors the new tools/server guard added at the chat-turn
    // prompt-prefix-reuse call site (server-context.cpp update_slots(), ~line
    // 3464): before calling seq_rm, check n_rollback = pos_cur - p0 + 1 against
    // n_rs_seq; deep rollback (> n_rs_seq) falls back to a full clear (rm_all,
    // which always succeeds) instead of an unguarded bounded seq_rm that would
    // GGML_ABORT(). Shallow rollback (<= n_rs_seq) must still take the fast
    // bounded-rollback path (not silently fall back every time). Own process
    // invocation, same rationale as case 3/4.
    if (getenv("XCALL_CASE6_ONLY") != nullptr) {
        // --- deep rollback: must NOT abort, must fall back to a full clear ---
        {
            llama_context * ctx = make_ctx(params, model, n_rs_seq);
            if (ctx == nullptr || !decode_sequential(ctx, tokens, n_decode)) {
                fprintf(stderr, "%s : case 6 (deep): setup failed\n", __func__);
                return 2;
            }

            const llama_pos p0_deep       = 1; // rollback = n_decode - 1, guaranteed > n_rs_seq
            const llama_pos pos_cur       = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
            const int64_t   n_rollback    = (int64_t) pos_cur - (int64_t) p0_deep + 1;
            const bool      deep          = n_rollback > (int64_t) n_rs_seq;

            fprintf(stderr, "%s : case 6 (deep): pos_cur=%d p0=%d n_rollback=%" PRId64 " n_rs_seq=%u deep=%d\n",
                    __func__, pos_cur, p0_deep, n_rollback, n_rs_seq, (int) deep);

            if (!deep) {
                fprintf(stderr, "%s : case 6 (deep): FAILED -- test construction bug, not actually deep\n", __func__);
                return 1;
            }

            // guard logic: fall back to full clear instead of the unguarded bounded seq_rm
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1)) {
                fprintf(stderr, "%s : case 6 (deep): FAILED -- even the full-clear fallback was rejected\n", __func__);
                return 1;
            }

            // context must remain usable after the fallback (fresh sequence at pos 0)
            if (!decode_one(ctx, tokens[0], 0)) {
                fprintf(stderr, "%s : case 6 (deep): FAILED -- context unusable after full-clear fallback\n", __func__);
                return 1;
            }
            fprintf(stderr, "%s : case 6 (deep): PASSED -- no abort, full-clear fallback taken, context usable\n", __func__);
        }

        // --- shallow rollback: must take the fast bounded path, not fall back ---
        {
            llama_context * ctx = make_ctx(params, model, n_rs_seq);
            if (ctx == nullptr || !decode_sequential(ctx, tokens, n_decode)) {
                fprintf(stderr, "%s : case 6 (shallow): setup failed\n", __func__);
                return 2;
            }

            const llama_pos p0_shallow = (llama_pos) (n_decode - 2); // rollback depth 2, well within n_rs_seq=6
            const llama_pos pos_cur    = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
            const int64_t   n_rollback = (int64_t) pos_cur - (int64_t) p0_shallow + 1;
            const bool      deep       = n_rollback > (int64_t) n_rs_seq;

            fprintf(stderr, "%s : case 6 (shallow): pos_cur=%d p0=%d n_rollback=%" PRId64 " n_rs_seq=%u deep=%d\n",
                    __func__, pos_cur, p0_shallow, n_rollback, n_rs_seq, (int) deep);

            if (deep) {
                fprintf(stderr, "%s : case 6 (shallow): FAILED -- test construction bug, should be shallow\n", __func__);
                return 1;
            }

            // guard logic: this depth must take the fast bounded seq_rm, not the full-clear fallback
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, p0_shallow, -1)) {
                fprintf(stderr, "%s : case 6 (shallow): FAILED -- fast bounded rollback path was rejected "
                                 "(should have succeeded within n_rs_seq)\n", __func__);
                return 1;
            }
            fprintf(stderr, "%s : case 6 (shallow): PASSED -- fast bounded rollback path taken, no fallback needed\n", __func__);
        }

        return 0;
    }

    // Case 5: decode-throughput overhead of the rotation fix's extra cont+concat
    // ops, n_rs_seq=0 (untouched path) vs n_rs_seq=6 (fix's code path active).
    // Own process invocation for a clean, uncontended timing run.
    if (const char * bench_n_str = getenv("XCALL_BENCH_N")) {
        const int n_bench = atoi(bench_n_str);
        const uint32_t bench_n_rs_seq = (uint32_t) atoi(getenv("XCALL_BENCH_RS_SEQ") ? getenv("XCALL_BENCH_RS_SEQ") : "0");
        llama_context * ctx = make_ctx(params, model, bench_n_rs_seq);
        if (ctx == nullptr) {
            fprintf(stderr, "%s : case 5: failed to init context\n", __func__);
            return 2;
        }
        // warmup (JIT/cache/HIP-graph capture settle)
        for (int i = 0; i < 8; ++i) {
            if (!decode_one(ctx, tokens[i % tokens.size()], i)) {
                fprintf(stderr, "%s : case 5: warmup decode failed\n", __func__);
                return 2;
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_bench; ++i) {
            if (!decode_one(ctx, tokens[i % tokens.size()], 8 + i)) {
                fprintf(stderr, "%s : case 5: bench decode failed at i=%d\n", __func__, i);
                return 2;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        fprintf(stderr, "%s : case 5: n_rs_seq=%u n=%d wall=%.4fs tps=%.2f\n",
                __func__, bench_n_rs_seq, n_bench, secs, n_bench / secs);
        return 0;
    }

    // Case 3 is run as its own fresh process invocation (env-selected) rather
    // than via fork(): HIP/ROCm contexts do not survive fork() reliably, so a
    // forked child here would confound "did it SIGABRT" with "did fork+HIP break".
    if (getenv("XCALL_CASE3_ONLY") != nullptr) {
        llama_context * ctx = make_ctx(params, model, n_rs_seq);
        if (ctx == nullptr) {
            fprintf(stderr, "%s : case 3: failed to init context\n", __func__);
            return 2;
        }
        if (!decode_sequential(ctx, tokens, n_decode)) {
            fprintf(stderr, "%s : case 3: prefix decode failed\n", __func__);
            return 2;
        }
        // p0=0 would take the trivial rm_all shortcut (always succeeds); use p0=1 so the
        // rollback-bound check actually runs: rollback = cell.pos - (p0-1) = n_decode-1 > n_rs_seq.
        fprintf(stderr, "%s : case 3: requesting rollback depth %u via common_context_seq_rm() (n_rs_seq=%u) "
                         "-- this is the exact helper tools/server calls on every prompt-prefix-reuse\n",
                __func__, n_decode - 1, n_rs_seq);
        common_context_seq_rm(ctx, 0, 1, -1); // expected to GGML_ABORT()
        fprintf(stderr, "%s : case 3: did NOT abort (unexpected)\n", __func__);
        return 0;
    }

    // Case 4: session save/restore with a MISMATCHED n_rs_seq between the
    // saving and loading contexts (e.g. server restarted with a different
    // draft width, or a session file replayed against a different config).
    // Own process invocation, same rationale as case 3.
    if (getenv("XCALL_CASE4_ONLY") != nullptr) {
        llama_context * ctx_a = make_ctx(params, model, n_rs_seq);
        llama_context * ctx_b = make_ctx(params, model, n_rs_seq > 2 ? n_rs_seq - 2 : n_rs_seq + 2);
        if (ctx_a == nullptr || ctx_b == nullptr) {
            fprintf(stderr, "%s : case 4: failed to init contexts\n", __func__);
            return 2;
        }
        if (!decode_sequential(ctx_a, tokens, n_decode)) {
            fprintf(stderr, "%s : case 4: ctx_a prefix decode failed\n", __func__);
            return 2;
        }
        const llama_pos p0_case4 = (llama_pos) (n_decode - std::max<uint32_t>(2, n_rs_seq / 2));
        if (!llama_memory_seq_rm(llama_get_memory(ctx_a), 0, p0_case4, -1)) {
            fprintf(stderr, "%s : case 4: ctx_a rollback failed unexpectedly\n", __func__);
            return 2;
        }

        const size_t n_bytes = llama_state_seq_get_size(ctx_a, 0);
        std::vector<uint8_t> buf(n_bytes);
        const size_t written = llama_state_seq_get_data(ctx_a, buf.data(), buf.size(), 0);
        fprintf(stderr, "%s : case 4: saved %zu bytes (n_rs_seq_a=%u) from ctx_a mid-rollback state, "
                         "loading into ctx_b (n_rs_seq_b=%u)\n",
                __func__, written, n_rs_seq, llama_n_rs_seq(ctx_b));

        const size_t read = llama_state_seq_set_data(ctx_b, buf.data(), written, 0);
        fprintf(stderr, "%s : case 4: llama_state_seq_set_data() returned %zu (0 == rejected cleanly, "
                         "nonzero == accepted)\n", __func__, read);

        // try to actually use ctx_b afterward -- continue the sequence from the position
        // the loaded state claims to be at (p0_case4), same as any real session-restore
        // caller would. A clean reject should leave ctx_b usable at pos 0 instead (not
        // exercised here); an accepted-but-mismatched load should either work correctly
        // or fail loudly, not silently corrupt.
        if (!decode_one(ctx_b, tokens[p0_case4], p0_case4)) {
            fprintf(stderr, "%s : case 4: ctx_b unusable after set_data, continuing from pos %d "
                             "(the position the loaded state itself reports)\n", __func__, p0_case4);
            return 2;
        }
        fprintf(stderr, "%s : case 4: ctx_b remained usable after set_data\n", __func__);
        return 0;
    }

    uint32_t R = std::max<uint32_t>(2, n_rs_seq / 2); // rollback depth, deliberately > 1 call
    if (const char * r_env = getenv("XCALL_R")) {
        R = (uint32_t) atoi(r_env);
    }
    if (R > n_rs_seq || R >= n_decode) {
        fprintf(stderr, "%s : rollback depth %u out of range (n_rs_seq=%u, n_decode=%u)\n", __func__, R, n_rs_seq, n_decode);
        return 1;
    }

    const llama_pos p0        = (llama_pos) (n_decode - R); // first position to roll back
    const llama_token replay_tok = tokens[p0];

    int n_fail = 0;

    // ----------------------------------------------------------------
    // Case 1: CROSS-CALL rollback. ctx_src does n_decode separate
    // single-token decode() calls, then rolls back R tokens (R spans
    // multiple prior calls), then replays tokens[p0] at pos p0.
    // ctx_ref performs the identical per-token call pattern but only up
    // to p0, then replays tokens[p0] at pos p0 -- this is ground truth.
    // ----------------------------------------------------------------
    {
        llama_context * ctx_src = make_ctx(params, model, n_rs_seq);
        llama_context * ctx_ref = make_ctx(params, model, n_rs_seq);
        if (ctx_src == nullptr || ctx_ref == nullptr) {
            fprintf(stderr, "%s : failed to init case-1 contexts\n", __func__);
            return 1;
        }

        if (!decode_sequential(ctx_src, tokens, n_decode)) {
            fprintf(stderr, "%s : case 1: ctx_src prefix decode failed\n", __func__);
            return 1;
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, p0, -1)) {
            fprintf(stderr, "%s : case 1: rollback of %u tokens (<=n_rs_seq=%u) was REJECTED by seq_rm "
                             "-- this itself is a finding (server would GGML_ABORT on this call site)\n",
                    __func__, R, n_rs_seq);
            n_fail++;
        } else {
            if (!decode_one(ctx_src, replay_tok, p0)) {
                fprintf(stderr, "%s : case 1: ctx_src replay failed\n", __func__);
                return 1;
            }

            if (!decode_sequential(ctx_ref, tokens, (uint32_t) p0)) {
                fprintf(stderr, "%s : case 1: ctx_ref prefix decode failed\n", __func__);
                return 1;
            }
            if (!decode_one(ctx_ref, replay_tok, p0)) {
                fprintf(stderr, "%s : case 1: ctx_ref replay failed\n", __func__);
                return 1;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src, 0);
            const float * logits_ref = llama_get_logits_ith(ctx_ref, 0);
            const llama_vocab * vocab = llama_model_get_vocab(model);
            const int n_vocab = llama_vocab_n_tokens(vocab);
            const double d = max_abs_logit_diff(logits_src, logits_ref, n_vocab);

            fprintf(stderr, "%s : CASE 1 (cross-call rollback) n_decode=%u R=%u p0=%d max|dlogit|=%.6f\n",
                    __func__, n_decode, R, p0, d);

            constexpr double eps = 1.0; // codebase precedent: chunked-vs-AR fp noise ~0.1-0.2, corruption ~1.7-8 (see [FIX card SSM-ROLLBACK-1])
            if (d > eps) {
                fprintf(stderr, "%s : CASE 1 FAILED (state desync) -- max|dlogit|=%.6f > eps=%.6f\n", __func__, d, eps);
                n_fail++;
            } else {
                fprintf(stderr, "%s : CASE 1 PASSED\n", __func__);
            }
        }

        llama_free(ctx_src);
        llama_free(ctx_ref);
    }

    // ----------------------------------------------------------------
    // Case 2 (positive control): SAME-CALL rollback, mirroring
    // test-recurrent-state-rollback.cpp -- decode n_decode tokens in ONE
    // batch call, roll back R tokens (within that call's own snapshot
    // window), replay. This is the documented-correct spec-decode path.
    // ----------------------------------------------------------------
    {
        llama_context * ctx_src = make_ctx(params, model, n_rs_seq);
        llama_context * ctx_ref = make_ctx(params, model, n_rs_seq);
        if (ctx_src == nullptr || ctx_ref == nullptr) {
            fprintf(stderr, "%s : failed to init case-2 contexts\n", __func__);
            return 1;
        }

        llama_batch batch = llama_batch_init(n_decode, 0, 1);
        for (uint32_t pos = 0; pos < n_decode; ++pos) {
            common_batch_add(batch, tokens[pos], (llama_pos) pos, { 0 }, true);
        }
        const bool ok = llama_decode(ctx_src, batch) == 0;
        llama_batch_free(batch);
        if (!ok) {
            fprintf(stderr, "%s : case 2: ctx_src batch decode failed\n", __func__);
            return 1;
        }

        if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, p0, -1)) {
            fprintf(stderr, "%s : case 2: rollback rejected unexpectedly\n", __func__);
            n_fail++;
        } else {
            if (!decode_one(ctx_src, replay_tok, p0)) {
                fprintf(stderr, "%s : case 2: ctx_src replay failed\n", __func__);
                return 1;
            }

            if (!decode_sequential(ctx_ref, tokens, (uint32_t) p0)) {
                fprintf(stderr, "%s : case 2: ctx_ref prefix decode failed\n", __func__);
                return 1;
            }
            if (!decode_one(ctx_ref, replay_tok, p0)) {
                fprintf(stderr, "%s : case 2: ctx_ref replay failed\n", __func__);
                return 1;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src, 0);
            const float * logits_ref = llama_get_logits_ith(ctx_ref, 0);
            const llama_vocab * vocab = llama_model_get_vocab(model);
            const int n_vocab = llama_vocab_n_tokens(vocab);
            const double d = max_abs_logit_diff(logits_src, logits_ref, n_vocab);

            fprintf(stderr, "%s : CASE 2 (same-call rollback, positive control) max|dlogit|=%.6f\n", __func__, d);

            constexpr double eps = 1.0; // same-call rollback vs an *independently AR-computed* reference: expect chunked-vs-AR fp noise, not exactness
            if (d > eps) {
                fprintf(stderr, "%s : CASE 2 FAILED -- harness or same-call rollback itself is broken (max|dlogit|=%.6f)\n", __func__, d);
                n_fail++;
            } else {
                fprintf(stderr, "%s : CASE 2 PASSED\n", __func__);
            }
        }

        llama_free(ctx_src);
        llama_free(ctx_ref);
    }

    return n_fail > 0 ? 1 : 0;
}
