# T405 phase 4: chunked GDN prefill on rock8-release (Bonsai-27B)

Port: gdn-chunk 324f9e4c32 + f20645b3c9 onto rock8-release 8eb1723169 (commit b45c981c57), identical to
`git merge-tree rock8-release gdn-chunk` (clean, 4 files). T409 glue fusion (gdn-glue ad486f6f2a) is not in the
release; a test-only tree (release + gdn-glue + chunk files) built and ran to check coexistence.
Ship bar fixed before measuring: pp2048 >= +3% on Q2_0 and Q1_0, all exactness gates pass, KLD <= ~1.1x 1-ulp floor.
Scripts and logs: ~/scratch/t405r8/{scripts,out,out-wrx}; zorin jobs 545-550, wrx90 jobs 86-87.

## (a) test-backend-ops (zorin job 545)
Port, default vs GGML_GDN_CHUNK=1: GATED_DELTA_NET 50/50 both (chunk arm 14 chunked calls, 7 split/snapshot),
SSM_CONV 45/45, SSM_CONV_UPDATE 24/24, MUL_MAT 1260/1260. chunk+glue tree: GATED_DELTA_NET 36/36, SSM_CONV_UPDATE 87/87.

## (b) state / snapshot slots at the Bonsai shape (Hk 16, Hv 48, S 128; job 546)
20 checks (K=8/16 snapshot, K=1 final state, 1-2 seqs, fused BA+L2, slow decay, zero/random init): worst output/slot
NMSE 2.8e-12 vs sequential, sequential-vs-sequential control exactly 0.

## (c) KLD, wikitext -c 2048 -b 2048 -ub 1024, 12 chunks (wrx90 jobs 86-87)
| model | n_rs_seq | 1-ulp floor | chunk | ratio | same top (ulp / chunk) |
|---|---|---|---|---|---|
| Q2_0 | 0 | 0.000710 | 0.000716 | 1.01x | 98.71 / 98.84% |
| Q2_0 | 7 | 0.000710 | 0.000696 | 0.98x | 98.71 / 98.63% |
| Q1_0 | 0 | 0.000505 | 0.000513 | 1.02x | 98.74 / 98.82% |
| Q1_0 | 7 | 0.000505 | 0.000515 | 1.02x | 98.74 / 98.79% |
| Q2_0 nksc | 0 / 7 | same as Q2_0 to all printed digits (bit-identical logits INFERRED from T399) | | | |
Release HEAD binary vs port default: KLD 0, same-top 100% (all models). Control max KLD 6e-5 (run-to-run, not GDN).

## (d) llama-bench -ub 1024 -fa 1 q8_0 KV, 4 interleaved rounds (zorin job 547), t/s and % of 7,200
| model | pp512 | pp2048 | pp4096 |
|---|---|---|---|
| Q2_0 | 1870 -> 2094 (+12.0%, 29.1%) | 2251 -> 2546 (+13.1%, 35.4%) | 2194 -> 2485 (+13.3%, 34.5%) |
| Q1_0 | 1858 -> 2047 (+10.2%, 28.4%) | 2175 -> 2451 (+12.7%, 34.0%) | 2118 -> 2389 (+12.8%, 33.2%) |
| nksc | 1902 -> 2094 (+10.1%, 29.1%) | 2252 -> 2554 (+13.4%, 35.5%) | 2198 -> 2482 (+13.0%, 34.5%) |
Release binary = port default within -1.1..+0.4%. chunk+glue tree, 2 rounds, pp2048: glue +3.3..3.4%, chunk +12.5..13.2%,
both +16.9..17.8% (Q2_0 2653 t/s = 36.8%): the two compose.

## (e) llama-server, Bonsai Q2_0 (zorin job 548)
DFlash qwen3.6-27b drafter, k=6, n_rs_seq=6 (912 split calls/run): prefill 3.6K-token prompts default ~1366 -> chunk
~1462 t/s (+7%). Release binary = port default: 4/4 greedy identical, identical draft counts (1511/123).
Default and chunk each reproduce themselves 4/4 over 3 rounds. Chunk vs default: p3 identical; p0 diverges at token 5
(the 1-ulp arm diverges at the same token, same pair), p1 token 27 (re-query margin 0.0004), p2 token 0 (3-way near-tie
198/695/1084 within 0.005 nats; the ulp arm flips p2 the same way under -np 2 concurrency). Acceptance 123/1511 vs
110/1584: per-prompt shift is p2 (56 -> 41), whose text differs from token 0. On p3 (text identical to default) chunk's
draft counts (457,18) equal the 1-ulp control's exactly, vs default (451,19): ulp-class drift in the drafter's injected
features plus text divergence (INFERRED), not drafter breakage.
-np 2 no drafter: release = default 4/4 (sequential and 2 concurrent clients); default concurrent pair reproduces 4/4;
chunk diverges 2/4 sequential (tokens 4, 0; margins 0.009, 0.005), 1-ulp arm 2/4 (tokens 9, 48; 0.001, 0.017).

## (f) graph reuse / seq_cp (zorin job 549)
llama-rs-reuse-check (chunk+glue tree, glue flag off unless noted), N 64/128/256, arms def / def_nocg / def_noreuse /
chunk / chunk_nocg / chunk_noreuse / chunk+glue: all VERDICT MATCH, max |dlogit| 0.0000, chunk path fired (240 calls),
graph reused at X4 (after seq_cp) with chunk on; 31/31 decode reuse. llama-parallel (shared system prompt, 12 wikitext
prompts, np 1/4): chunk == chunk_nocg on every common input; np4 default is not self-reproducible (as T409 found);
the trace was not printed at the default log level. Follow-up (job 551, flipped build, -lv 4, np 1): chunk path
fires 576 times (prompts 416-492 tokens, n_seqs=1), GGML_GDN_CHUNK=0 gives 0.
