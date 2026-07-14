# The Rock8 — Native fp8 LLM Inference on AMD RDNA4: A Performance Deep-Dive

*Lemonade Developer Challenge submission — performance evaluation track.*
*Hardware: 2× AMD Radeon AI PRO R9700 (gfx1201 / Navi 48, RDNA4). Runtime: The Rock8
(a llama.cpp/ggml fork with native RDNA4 fp8 kernels) on AMD TheRock ROCm 7.13, served
through Lemonade. All numbers measured on real gfx1201 hardware, drift-controlled.*

- **Source:** github.com/The-Monk/The-Rock8 (kernels, patch series, container appliance)
- **Models:** the *Quacken / RDNA4-fp8* collection on Hugging Face (Apache-2.0, benched)
- **Container:** `ghcr.io/the-monk/the-rock8:rdna4-tr713` (one-command Lemonade appliance)

---

## TL;DR — what we measured

1. **Native fp8 (E4M3) matmul on RDNA4 works and wins on prefill** — Qwen3.6-27B fp8
   prefill **~1310 t/s = +6.7% over int8 and +42% over Vulkan**, at +0.4–0.6% PPL vs BF16.
   Decode is bandwidth-bound and ties int8.
2. **Beating Vulkan on both axes** via fp8 tensor cores + MTP self-speculation:
   27B decode **95 t/s > Vulkan 91**; prefill **+42%**.
3. **Multi-user throughput (the vLLM-replacement story):** the 35B-A3B fp8 **MoE** scales
   **65 → 537 t/s aggregate decode (7.4×)** with continuous batching, peaking at ~114
   concurrent sequences — natively, with fp8 (vLLM silently dequantizes fp8 on gfx1201).
4. **Single-stream latency via speculative decode:** MTP takes the dense 27B **18 → 45 t/s
   (2.43×)**; drafter-free n-gram lookup takes it **18 → 134 t/s** on repetitive/code output.
5. **Mixed FP4/FP8 (NVFP4 ingest):** a mixed-precision Qwen3.6-27B is **23 GB (single-card),
   PPL 6.88 (beats our fp8 7.14), +30% decode** — smaller, more accurate, faster than fp8.
6. **A hardware finding worth its own report:** RDNA4's performance-monitor counters are
   ~96% dead on gfx1201 (only ~7 of 207 read). Documented, root-caused, and filed upstream
   (ROCm/rocprofiler-sdk #155).

Everything below includes the caveats — where a lever *doesn't* help is as important as where it does.

---

## 1. Why native fp8 on RDNA4

RDNA4 (gfx1201) exposes **native fp8 WMMA tensor-core** instructions
(`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`) and an fp8 dot-product for decode
(`v_dot4_f32_fp8_fp8`). Stock llama.cpp doesn't use them — fp8 models fall back to a
dequantize-to-f16 path. The Rock8 adds the real kernels: fp8 E4M3/E5M2 weights (WMMA
prefill + dp4a decode), fp8 KV-cache, fp8 MoE (`MUL_MAT_ID`), and MXFP8, validated against
the ISA via `llvm-objdump --mcpu=gfx1201` (not header greps).

**Accuracy (wikitext PPL, n_ctx=512):** fp8 E4M3 costs **+0.42–0.57% vs BF16** (int8 is
−0.06%); E5M2/bf8 costs +2.5% (2 vs 3 mantissa bits). fp8 is the frozen-model sweet spot.

**Prefill (Qwen3.6-27B, 1-GPU):** fp8 **1310 t/s = +6.7% over int8, +42% over Vulkan.**
**Decode:** bandwidth-bound, so fp8 ties int8 (both ~8-bit weight traffic per token).
The prefill win is real compute (fp8 tensor cores); decode is physics (memory bandwidth).

---

## 2. Single-stream latency — speculative decoding

Single-token decode is **memory-bound**: one token reads the full active weight set. The
only way to speed a *single* stream is to manufacture a batch dimension — draft N tokens,
verify them in one batched pass. Results (temp 0, greedy):

| Model | Method | Baseline | With spec | Gain |
|---|---|---:|---:|---:|
| 27B dense | **MTP self-spec** (`--spec-type draft-mtp`, n=8) | 18 t/s | **45 t/s** | **2.43×** |
| 27B dense | **n-gram lookup** (repetitive/code) | 18 t/s | **134 t/s** | **7.6×** |
| 35B-A3B MoE | **n-gram lookup** (repetitive/code) | 66 t/s | **298 t/s** | **4.5×** |

**MTP** (the model's own multi-token head) makes the 27B decode **95 t/s — beating Vulkan's
91** on the same model, entirely via fp8 tensor cores Vulkan structurally lacks.

**Honest caveats (measured):**
- **n-gram lookup is content-dependent.** It only wins when the output *echoes the input*
  (code reproduction, RAG-quoting). On general/novel chat, acceptance collapses
  (100%→25%→8% as draft length grows) and you get baseline. The 298 is a
  *repetitive-reproduction* number, not a chat number.
- **MTP *hurts* the A3B MoE** (66.5 vs 76 t/s) — a sparse MoE's single-stream decode is
  already fast (~3B active params), so the MTP head's overhead outweighs the gain. MTP is a
  lever for slow *dense* decode, not fast sparse MoE.
- **Async draft‖verify pipeline** (draft on GPU1 ‖ verify on GPU0) hits **+66%** on a dense
  target with an ultra-cheap ternary draft (Bonsai-8B: 63→105 t/s). The lever is
  **target-weight × draft-cheapness**: with the identical Q4 draft, a *heavy* BF16 target
  gives +14.5% while a *light* fp8 target gives −9%. It does **not** compose with hybrid
  SSM/GatedDeltaNet targets (a rollback/`seq_rm` gap — a real llama.cpp core limitation).

---

## 3. Multi-user throughput — the vLLM replacement on RDNA4

For *concurrent* users, continuous batching amortizes weight reads across the batch. On a
sparse MoE this compounds via **expert-union amortization** (more tokens per step share the
same routed-expert reads). Measured on the **35B-A3B fp8 MoE** (`llama-batched-bench`,
drift-controlled):

| Concurrent seqs | Aggregate decode | Scaling |
|---:|---:|---:|
| 1 | 65.8 t/s | 1.00× |
| 8 | 249.9 t/s | 3.80× |
| 32 | 355.3 t/s | 5.40× |
| 64 | 432.7 t/s | 6.58× |
| **114** | **537.4 t/s** | **7.39×** ← peak |
| 128 | 464.5 t/s | 7.06× (knee) |

**One dual-R9700 box serves ~110 concurrent users at ~537 tok/s aggregate.** It knees at
~114 (expert-union saturation — all 256 experts pulled per step), **not** a VRAM wall
(npl=128 fits). This is the same amortization that was vLLM's "server win" — but *native*
and *with fp8* (vLLM silently dequantizes fp8 on gfx1201, so its edge evaporates here).

**Two findings that only show up at scale:**
- **Throughput oscillates ±11% with `npl mod 4`** — a batch/ubatch tiling-alignment effect,
  reproducible across runs. `npl ≡ 2 mod 4` is favorable (110/114/118 = 531–537), `≡ 0 mod 4`
  is worst (112/116 = 451–490). Coarse power-of-two sweeps *systematically sample the bad
  alignment* and undersell the ceiling by ~11%. **Practical: pick a ≡2-mod-4 `--parallel`.**
- **fp8 KV-cache is incompatible with batched decode on this hybrid-SSM MoE** (breaks B>1) —
  so multi-user serving is f16-KV only here; fp8-KV remains a single-stream lever.

**Prefill vs decode under load:** prefill plateaus (~4000 t/s, compute-bound) while decode
keeps climbing (memory-bound, keeps amortizing) — a clean illustration of the two regimes.

---

## 4. Mixed-precision FP4/FP8 (NVFP4 ingest)

RDNA4 has **no native FP4 matmul** (that's CDNA4/gfx950) — FP4 runs an int8/dp4a fallback.
So the interesting result is *mixed* precision. Ingesting Unsloth's NVFP4 Qwen3.6-27B
(FP4 MLPs + FP8 attention) through The Rock8's mixed-precision compressed-tensors converter:

| Metric | Mixed FP4/FP8 | Our fp8 27B |
|---|---:|---:|
| Size | **23.2 GB (fits 1 card)** | ~29 GB (2 cards) |
| PPL | **6.88** | 7.14 |
| Decode | **24.0 t/s (+30%)** | 18.5 t/s |
| Prefill | 1069 t/s (−15%) | 1251 t/s |

**Smaller, more accurate, and faster to decode than pure fp8** — because the FP4 MLPs move
far less data in the memory-bound decode phase, while FP8 attention stays on native tensor
cores. Honest counter-result: **full-W4A4** (79.5% NVFP4) *loses* — the fallback tax
dominates and it doesn't batch-amortize like fp8 WMMA. **The RDNA4 sweet spot is mixed
FP4-MLP / FP8-attention, not full 4-bit.**

---

## 5. The RDNA4 profiling problem (a bonus finding)

Optimizing this was hard because **RDNA4's hardware performance counters are almost entirely
dead on gfx1201.** A full census (356 counters via ROCm 7.13 rocprofiler-sdk) found **only
15 live (4.2%), and ~8 of those are static device constants — so ~7 real counters work**
(`GRBM` busy-bits, `SQ_WAVES`, `SQ_BUSY_CYCLES`, `SQC_ICACHE`). Every armed per-block counter
— GL2C/L2 memory traffic, TCP, SPI, CPC, and the `SQ_INSTS_*` VALU/LDS instruction counts —
reads **0**, while the *same blocks work on gfx1151 (RDNA3.5) in the same ROCm*. It's a
below-userspace (kernel/firmware perfmon-readback) gap, confirmed unfixed through ROCm 7.13
and independent of firmware (our RLC microcode is byte-identical to upstream `linux-firmware`
HEAD). Filed upstream: **ROCm/rocprofiler-sdk #155**, with the full per-counter map.

**Workaround we used throughout:** an analytical bytes/time roofline + `amd-smi UMC_ACTIVITY`
(SMU telemetry, which *does* work) for the memory axis, plus per-kernel time attribution
(we ported a `soc_gfx1201.py` into rocprofiler-compute so it runs on gfx1201 at all).

---

## 6. Lemonade integration

The Rock8 ships as a **one-command Lemonade appliance**: a rootless-Podman image
(`ubuntu:24.04` + TheRock ROCm 7.13, one extra dep `libatomic1`) that wires Lemonade's rocm
backend to the fp8 kernels and registers a mounted model. It runs **pure-7.13 with zero
`/opt/rocm` on the host**, requires `crun` for GPU passthrough, and bakes in the validated
serving configs (continuous batching, MTP, and an n-gram `lookup` subcommand).

```bash
podman run -d --rm --runtime crun --name lemonade \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  -v /path/to/model:/models:ro -p 13305:13305 \
  ghcr.io/the-monk/the-rock8:rdna4-tr713 serve
```

---

## 7. Methodology & reproducibility

- **Drift control:** every headline number is a fresh, cold, isolated re-measure — we caught
  and corrected a session-wide llama-bench read drift and several measurement artifacts (e.g.
  a spec-decode t/s that looked like 2000+ but was a dual-context timing-parse error; the real
  numbers are the modest, defensible ones above).
- **Honesty over headlines:** where a lever loses (MTP on MoE, full-W4A4, async on hybrid-SSM,
  lookup on novel chat) we report it — a performance evaluation is only useful if the negative
  results are in it.
- **Reproduce it:** kernels + patch series on GitHub, benched models on Hugging Face, the
  container on ghcr. Each artifact links to the others.

*The Rock8 is a fork-experiment on AMD's open Lemonade/ROCm stack — built to show that
RDNA4 desktop/workstation GPUs can run modern LLMs natively at fp8, and to map exactly what
the silicon can and can't do today.*
