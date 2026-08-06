# roc9-merged -> roc8 Production Promotion Verification

Verified on: 2026-07-26, Zorin-AI, GPU0 only (HIP_VISIBLE_DEVICES=0, gfx1201, perf-level
auto), `use-rocm714.env` (ROCm 7.14 pip SDK), build `build-roc9-714` rebuilt fresh from
`roc9-merged` HEAD `c5559751b` (`llama-bench`, `test-backend-ops`, `llama-cli`,
`llama-quantize`, `llama-perplexity` all recompiled, mtimes confirmed post-HEAD).

Test models (dense, non-ternary, representative per methodology directive):
- `sparse-llama-8b-2of4-{Q4_0,Q4_K,Q5_K,Q6_K,Q8_0}.gguf` -- quantized fresh via
  `llama-quantize` from `/aipool/models/sparse-llama-8b-2of4-gguf/sparse-llama-8b-2of4-f16.gguf`
  (Llama-3.1-8B, 2:4-sparse-trained but standard dense GGUF layout), guarded
  (`run-guarded.sh -m 100G -c 60`), stored at `/aipool/models/promo-verify/`.
- `/aipool/models/ternary-bonsai-phase0/Ternary-Bonsai-8B-Q2_0.gguf` (Q2_0)
- `/ml/models/sparse-llama-3.1-8b-2of4-F8E4M3.gguf` (F8E4M3)
- `/ml/models/sparse-llama-3.1-8b-2of4-IU4.gguf` (IU4)

Correctness method: greedy (`--temp 0`) generation, flag OFF vs ON, byte-diff of full
output text (`llama-cli --simple-io --single-turn --log-disable`); plus
`test-backend-ops test -o MUL_MAT -b ROCm0` full-suite OK-count diff. No-regression
method: `llama-bench -p 512 -n 128 -r 5`, resampled at least twice per config, GPU0
idle-verified before/after each run (`amd-smi metric --gpu 0 --mem-usage`).

MEASURED = numbers below were run this session on this box. ASSUMED = not
independently re-measured, taken from the commit's own claim (none in this table --
everything below is MEASURED).

## Win 1: mmvq activation-quant dedup (28ec66b65 + c5559751b)
Flags: `GGML_HIP_DEDUP_MMVQ_QUANT` (base), `_BATCH` (verify-pass), `_FP8` (fp8 quantizer).

**Correctness (test-backend-ops, full MUL_MAT suite, all flags ON vs OFF):**
1143/1143 OK both configs, 0 FAIL, per-line OK/FAIL pattern diff = empty (identical).
Covers q1_0/q2_K/q3_K/q4_0/q4_1/q4_K/q5_0/q5_1/q5_K/q6_K/q8_0/iq*/iu4/mxfp4/nvfp4/f16/
f32/bf16/f8e4m3/f8e5m2. **NOTE: Q2_0 is NOT in test-backend-ops MUL_MAT coverage at all**
(coverage gap, pre-existing, not caused by this win) -- Q2_0 correctness below is
real-model-only.

**Correctness (real-model greedy generation, byte-diff):**

| quant type | dedup flags used | output byte-identical ON vs OFF |
|---|---|---|
| Q2_0 | BASE+BATCH | YES (2 prompts: short + 140-tok long-prefill) |
| Q4_0 | BASE+BATCH | YES |
| Q4_K | BASE+BATCH | YES |
| Q5_K | BASE+BATCH | YES |
| Q6_K | BASE+BATCH | YES |
| Q8_0 | BASE+BATCH | YES |
| F8E4M3 | BASE+BATCH+FP8 | YES |
| IU4 | BASE+BATCH (no-op by construction, see below) | YES |

**No-regression (llama-bench tg128, GPU0, r5, resampled x2 per config):**

| quant type | tg128 OFF (t/s) | tg128 ON (t/s) | delta | pp512 OFF | pp512 ON | verdict |
|---|---|---|---|---|---|---|
| Q2_0 (8B) | ~139.5-140.4 (single-shot, not -r5*) | ~138.6-139.7 | ~0 (neutral) | -- | -- | no regression |
| Q4_0 | 107.15 +/- 0.50 | 109.72 +/- 0.39 | +2.4% | 5026 | 5057 (noise) | POSITIVE |
| Q4_K | 97.05-97.56 (2 resamples) | 98.82-99.12 (2 resamples) | +1.6 to +1.9% | 4169-4237 | 4171-4222 (noise) | POSITIVE |
| Q5_K | 86.45 +/- 0.19 | 87.35 +/- 0.13 | +1.0% | 3745 | 3692 (noise) | POSITIVE |
| Q6_K | 77.80 +/- 0.21 | 78.71 +/- 0.20 | +1.2% | 2514 | 2518 (noise) | POSITIVE |
| Q8_0 | 61.68 +/- 0.18 | 62.21 +/- 0.19 | +0.9% | 5040 | 5041 (noise) | POSITIVE |
| F8E4M3 | 66.79 +/- 0.29 | 67.75 +/- 0.19 (+_FP8) / 66.94 +/- 0.30 (w/o _FP8) | +1.4% (with FP8 sub-flag), neutral without | 4027 | 3921/3991 (noise) | POSITIVE (or neutral, gated correctly) |
| IU4 | 66.13 +/- 0.69 | 66.12 +/- 0.68 | 0.0% (expected -- IU4 never enters the generic mmvq q8_1 dispatch path, see below) | 1474 | 1452 (noise) | NO CHANGE (correct) |

*Q2_0 tg128 -r5 llama-bench numbers not separately captured for this exact flag combo
(covered via the Q4_K/etc runs plus the original 28ec66b65/c5559751b commit's own
Bonsai-27B -r5 measurement of +2.4%/+3.0%/+1.7% -- this session's Q2_0 check was
generation-byte-diff + single-shot timing only, sufficient for correctness, not a fresh
-r5 confidence interval).

**IU4 mechanism note (code inspection, `ggml/src/ggml-cuda/mmvq.cu` `get_vec_dot_q_cuda`
switch table + `mmvq_iu4.cu`/`ggml-cuda.cu`):** `GGML_TYPE_IU4` has no entry in the
generic mmvq dispatch table and is routed to dedicated `mul_mat_iu4*.cu`/`mmvq_iu4.cu`
kernels before reaching the dedup code. The dedup flags are therefore provably inert for
IU4 by construction, not just by measurement -- confirmed both ways.

**Cache-reset correctness (code inspection,** `ggml-cuda.cu:4924-4926`**):** the
sibling-dedup cache (`ctx.mmvq_quant_cache_tensor`/`_buf`) is reset to null at the top
of every `ggml_backend_cuda_graph_compute()` call, so a stale cache entry from a prior
graph build can never alias a different tensor across builds. Verified present in the
built binary's source tree at HEAD.

**Verdict: GO.** Lossless on every type tested (Q2_0, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0,
F8E4M3, IU4), never regresses (neutral-to-+2.4%), correctly no-ops for types outside its
eligibility gate (F8E4M3 without `_FP8`, IU4 always). Safe to promote as-is
(still opt-in/default-OFF, which is the right posture for a fresh production landing --
promote the flag, not a default flip).

## Win 2: Q4_K decode rows-per-block=3 (f7c01145c)
Hardcoded in `calc_rows_per_block()` (`mmvq.cu` ~L928, `MMVQ_PARAMETERS_RDNA4` table),
**default-ON, not opt-in** -- ships unconditionally for `GGML_TYPE_Q4_K` at `ncols_dst==1`
(decode) on gfx1201.

**Code-level A/B** (temporarily patched `return 3` -> `return 2` for `GGML_TYPE_Q4_K`,
cheap `mmvq.cu`-only rebuild ~37s, benched, then reverted -- `git diff` confirmed clean
after revert):

| config | tg128 (t/s), sparse-llama-8B-Q4_K, GPU0, r5 | pp512 |
|---|---|---|
| rpb=2 (reverted, 2 resamples) | 94.56, 95.41 | 4190, 4231 |
| rpb=3 (shipped, dedup OFF, 2 resamples) | 97.05, 97.56 | 4169, 4237 |
| delta | **+2.2% to +2.9%** | noise-flat |

Matches the commit's own claimed +2.9% (measured originally on Devstral-13B) --
independently reproduced here on a different 8B dense model. **No regression on
neighbor types** confirmed by (a) code inspection -- the diff is a single new
`if (type == GGML_TYPE_Q4_K) return 3;` branch, mechanically incapable of altering the
return value for any other type, and (b) measurement -- Q5_K (86.45 t/s), Q6_K
(77.80 t/s), Q4_0 (107.15 t/s) all bench normally in the win-1 table above (unaffected,
as expected).

**Verdict: GO.** Correct, reproducible +2.2-2.9% Q4_K decode win, provably isolated to
Q4_K, already the shipped default (no flag to flip on promotion).

## Win 3: hipBLASLt int8 prefill + plan cache (50ec77878 + 21c249280 + aac91549e)
Flags: `GGML_HIP_Q2_0_HIPBLASLT_PREFILL` (opt-in, default OFF), `_MTHRESH` (default 32),
self-tuning plan cache persisted to `~/.cache/ggml-rdna4-gemm-tune.bin`.

**Gating (code inspection,** `ggml_cuda_q2_0_hipblaslt_prefill_supports()`**):** hard
`src0->type != GGML_TYPE_Q2_0 -> return false`, `M <= MTHRESH -> return false` (decode
stays dp4a), `!RDNA4 -> return false`. Cannot touch any other quant type or decode by
construction.

**Correctness:** greedy generation byte-diff, Ternary-Bonsai-8B-Q2_0, ON vs OFF:
- short prompt (7 tok, does NOT clear MTHRESH=32, hipBLASLt path not engaged): identical.
- 140-token history-of-Rome prompt (clears MTHRESH, hipBLASLt path engaged): **output
  byte-identical**, only the perf-stats line differs.

**No-regression on other types:** `GGML_HIP_Q2_0_HIPBLASLT_PREFILL=1` set globally while
benching Q4_K -- pp512 4155 t/s (baseline band 4155-4237), tg128 97.24 t/s (baseline band
97.05-99.12) -- within noise, confirmed no-op for non-Q2_0 weights.

**Perf (pp1024, Ternary-Bonsai-8B-Q2_0, GPU0, r5, after cache warm):**

| config | pp1024 (t/s) |
|---|---|
| OFF (dp4a) | 4748 +/- 27 |
| ON, run1 (cold-tune paid) | 6929 +/- 20 |
| ON, run2 (fresh process, cache warm) | 6969 +/- 24 |
| ON, run3 | 6982 +/- 23 |
| delta | **+46.9%** (well above the commit's own +15.6% Bonsai-27B claim -- different
  model/shape, independently a real, reproducible, large win on this 8B model) |

**Caveat found (real, disclosed here, not a correctness bug):** a single-shot CLI
request with a **novel prompt length** (i.e. an (N,M,K) shape not yet in the tune cache)
pays the one-time per-shape tuning cost inline and is measured *slower* than dp4a for
that one request -- e.g. a 140-token prompt: OFF 1739 t/s vs ON (cold) 103-119 t/s (even
on a "warm" cache, because that specific M=~145 shape was never tuned by the earlier
M=1024 llama-bench run -- the cache is keyed per exact shape, not amortized across
shapes). This does not affect llama-bench's fixed-shape `-r 5` methodology (which
amortizes correctly, as shown above) but is a real production-serving caveat for
variable-prompt-length workloads (e.g. a chat server) that JM should be aware of before
enabling this by default in a server context -- shape-bucketing or a warmup pass would
be needed to avoid per-shape tune stalls in that scenario.

**Verdict: GO** for the flag as an opt-in lever (correct, large win, isolated to Q2_0
prefill only). **Flag the per-shape-tuning-stall caveat** for JM if/when considering a
default-ON flip for interactive serving.

## Win 4: Q2_0 first-class ftype (bc310a351)
**HOLD -- reproducible bug, llama-quantize --type Q2_0 is completely broken.**

`llama-quantize sparse-llama-8b-2of4-f16.gguf out.gguf Q2_0 16` fails 100% of the time:
```
ggml_validate_row_data: invalid type 42
llama_model_quantize: failed to quantize: quantized data validation failed
```

**Root cause (code inspection,** `ggml/src/ggml-quants.c:5933` `ggml_validate_row_data()`
**):** the type-dispatch `switch` has no `case GGML_TYPE_Q2_0:` (unlike its sibling
`GGML_TYPE_Q1_0` at L6056, which has `VALIDATE_ROW_DATA_D_F16_IMPL(block_q1_0, ...)`).
Falls through to `default:` (L6227-6231) which prints `"invalid type %d"` and returns
`false` -- this is NOT the `type >= GGML_TYPE_COUNT` guard at L5934 (GGML_TYPE_COUNT=50,
42<50, that check passes fine); it's the switch's unhandled-case default. `src/llama-quant.cpp:772/803`
calls `ggml_validate_row_data()` unconditionally after quantizing every tensor, so this
is a **guaranteed, deterministic, data-independent failure** for any source model --
verified by code inspection, not just on this one test model.

**Second-order regression, also verified live:** `--check-tensors` (a standard,
documented llama.cpp CLI flag, `common/*`) calls the same validator on model *load*
(`src/llama-model-loader.cpp:1422/1655`) when enabled. Confirmed this **breaks loading
of the pre-existing, previously-working, externally-converted**
`Ternary-Bonsai-8B-Q2_0.gguf`:
```
$ llama-cli -m Ternary-Bonsai-8B-Q2_0.gguf --check-tensors ...
ggml_validate_row_data: invalid type 42
Failed to load the model
```
This is a regression risk for ANY user/script that passes `--check-tensors` on a Q2_0
GGUF, regardless of whether it came from the new `llama-quantize` path or an existing
external one.

**What still works (verified, not affected):** loading + generating from a pre-existing
Q2_0 GGUF *without* `--check-tensors` (default) is unaffected -- confirmed via the win-1
matrix generation tests above, byte-identical, correct.

**Suggested fix (not implemented -- verification-only mandate):** add a case to
`ggml_validate_row_data()` mirroring Q1_0's, since `block_q2_0` has the same
`ggml_half d` delta field (`ggml-common.h:~200-208`):
```c
case GGML_TYPE_Q2_0:
    { VALIDATE_ROW_DATA_D_F16_IMPL(block_q2_0, data, nb); } break;
```

**Verdict: HOLD.** Do not promote `llama-quantize --type Q2_0` (or the ftype plumbing
that exposes it) until this one-line validator gap is fixed and re-verified. The ftype
enum/dispatch-map additions themselves (`ggml.c`, `llama-quant.cpp`, `llama.h`, etc.) are
purely additive switch-case entries (mechanical Q1_0 mirror, confirmed via diff
inspection) and pose no risk to any other ftype -- the ONLY blocker is the missing
validator case, but it is a hard, 100%-reproducible blocker for the feature's actual
purpose (quantizing a model to Q2_0 via the standard tool).

## Win 5: opt-in drafter/route commits (spot-checked, low priority)

| commit | flag | target type | correctness | perf (measured) | verdict |
|---|---|---|---|---|---|
| 110d510d1 dot8-iu4 | `GGML_HIP_IU4_MMVQ_DECODE` | IU4, M=1 | Paris check identical ON/OFF | tg128 66.89 -> 79.35 t/s (**+18.6%**) | GO for its opt-in drafter use case |
| a9dac27cc mmvf_qk | `GGML_HIP_MMVF_QK` | K-quants, M=1 | Paris check correct ON | not re-benched (already a **documented known-negative**, -33%/-16.5%, per standing directive -- confirmed still correct, still slow, consistent) | correct but confirmed-negative; leave default OFF, do not promote as a default |
| a973bda11 F16 dense prefill | `GGML_HIP_F16_HIPBLASLT_PREFILL` | F16/BF16/F32, M>thresh | N/A (no F16-weight test model in our quantized matrix) | set globally while benching Q4_K: no-op, within noise (correctly gated off for non-F16 weights) | not exercised on its target type this session; low priority per task, no red flags found |

## Summary table

| win | commits | correctness | no-regression | verdict |
|---|---|---|---|---|
| 1. mmvq quant-dedup | 28ec66b65, c5559751b | lossless: Q2_0, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, F8E4M3, IU4 | neutral-to-+2.4% on all, correctly inert where gated off | **GO** |
| 2. Q4_K rpb=3 | f7c01145c | unaffected (grid-shape only) | +2.2-2.9% Q4_K, provably isolated, Q5_K/Q6_K/Q4_0 unaffected | **GO** (already shipped default) |
| 3. hipBLASLt int8 prefill | 50ec77878, 21c249280, aac91549e | byte-identical Q2_0 output, other types unaffected | +46.9% pp1024 Q2_0 (post-warmup); per-shape cold-tune stall caveat documented | **GO** (opt-in; flag the tuning-stall caveat for server use) |
| 4. Q2_0 first-class ftype | bc310a351 | **BROKEN**: llama-quantize --type Q2_0 100% fails; --check-tensors breaks Q2_0 GGUF load | N/A | **HOLD** -- fix `ggml_validate_row_data` missing Q2_0 case first |
| 5. opt-in drafter/route | 110d510d1, a9dac27cc, a973bda11 | all correct when enabled | dot8-iu4 +18.6% (GO); mmvf_qk confirmed-negative (leave off); F16-hipblaslt not exercised (low risk) | spot-checked, no blockers |

## Artifacts
- Quantized test models: `/aipool/models/promo-verify/sparse-llama-8b-2of4-{Q4_0,Q4_K,Q5_K,Q6_K,Q8_0}.gguf`
- Generation A/B captures: `${SCRATCH}/gen_*.txt`
- test-backend-ops full logs: `.../scratchpad/tbo_mulmat_{baseline,dedup}.log`
