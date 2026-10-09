# T407 activation-path fusion set (prefill, MMQ consumers)

Env `GGML_ACT_FUSE=1` (default off). `GGML_ACT_FUSE_MASK`: 1 quantize dedup, 2 swiglu -> q8_1,
4 [ADD ->] RMS_NORM -> MUL -> q8_1. `GGML_ACT_FUSE_VERIFY=1` byte-compares every fused producer
against the unfused quantize; `GGML_ACT_FUSE_STATS=1` prints per-graph counts. Code: `ggml-cuda/act-fuse.{cu,cuh}`,
cache in `mmq.cu` (`ggml_cuda_mul_mat_q`), graph hooks in `ggml-cuda.cu` (`[TAG_ACT_FUSE]`).

## Gates (Bonsai-27B Q2_0 and Q1_0, GPU0 R9700, jobs 350/360)

- Fire counts per 512-token ubatch (all 20 KLD graphs, incl. the CUDA-graph-captured one): MMQ activation
  hit 432 / miss 64 (64 = wo + ssm_out, no fused producer), glu 64, add+norm 127, norm 1.
- VERIFY: 0 differing bytes of 1,019,215,872 fused q8_1 bytes per ubatch, both models.
- test-backend-ops MUL_MAT 1246/1246, SWIGLU 24/24, RMS_NORM 51/51, ADD 99/99, default and fused
  (single-op graphs: these do not exercise the producers; VERIFY + KLD do).
- KLD vs default base, wikitext-2 c512 20 chunks: control, mask 1, 2, 4, 7 all mean 0.000000,
  max 0.000057 (Q2_0) / 0.000060 (Q1_0) = control floor, same top-1 100%.
- CUDA-graph replay (job 370, -b 512 -ub 512, 20 chunks): 2 graphs evaluated, 18 replayed with the
  producers + cache captured; KLD vs default base = control floor (max 5.9e-5), top-1 100%, both models.
- Default path unchanged: default pp512/pp2048 1305/1356 vs T403 census 1307/1353.
- Decode tg128 r3 x2 rounds: Q2_0 def 63.52/63.76 vs fused 63.65/63.66; Q1_0 def 91.65/91.30 vs fused 91.18/91.67 (noise).

## Speed (llama-bench -ngl 99 -n 0 -p 512,2048 -r 3, 4 interleaved rounds, mean [min-max] t/s)

| model | arm | pp512 | pp2048 |
|---|---|---|---|
| Q2_0 | default | 1305.3 [1302.8-1307.9] | 1355.6 [1354.4-1356.4] |
| Q2_0 | mask 1 dedup | 1331.6 | 1372.0 |
| Q2_0 | mask 2 glu | 1327.7 | 1376.9 |
| Q2_0 | mask 4 norm/add | 1353.1 | 1374.3 |
| Q2_0 | mask 7 all | 1371.4 [1365.7-1377.7] (+5.1%) | 1398.4 [1396.7-1399.8] (+3.2%) |
| Q1_0 | default | 1317.6 | 1371.0 |
| Q1_0 | mask 7 all | 1399.1 (+6.2%) | 1415.3 (+3.2%) |

Of 7,200 t/s: Q2_0 19.0% / 19.4%, Q1_0 19.4% / 19.7% (default 18.1-19.0%).

## Kernel trace, Q2_0 pp2048 (rocprofv3, measured pass, us/token)

| | default | mask 7 |
|---|---|---|
| launches per 512-token ubatch | 2121 | 1562 (-559) |
| non-GEMM kernel time | 203.3 | 177.4 (-25.9) |
| traced wall | 741.4 | 713.1 (-28.3) |
| quantize_mmq_q8_1 | 20.0 (496/ubatch) | 1.9 (64/ubatch) |
| unary_gated (swiglu + gated norm) | 24.7 (128) | 4.0 (64, gated-norm only) |
| rms_norm<1024> | 7.3 (129) | 0.1 (1) |
| act_glu_quant (new) | - | 15.3 (64, 122 us/call ~ 650 GB/s) |
| act_norm_quant (new) | - | 15.9 (128, 64 us/call) |

Launch saving is 559 not the predicted ~750 because the 192 producer launches replace the removed ones.

## N1 plug-in

Producers = load functor x store functor. N1 adds a store struct `operator()(row, col0, float4)` for its
layout under the same warp-collective contract (lane L owns columns [4L, 4L+4) of a 128-aligned span),
a layout id from `ggml_cuda_mmq_act_layout`'s N1 twin, and a case in the layout switches of
`ggml_cuda_act_glu_quant` / `ggml_cuda_act_norm_quant`; the cache key already includes the layout.
