# The Rock8 — Got any weights? 💪🦆
### RDNA4 (gfx1201) native-fp8 llama.cpp + Lemonade appliance

The Rock8 is a fork of llama.cpp/ggml that adds **native low-precision matrix
kernels for AMD RDNA4** (gfx1201 — Radeon AI PRO R9700, RX 9070 / 9070 XT,
W-series), packaged as a one-command rootless-Podman appliance on the
AMD TheRock ROCm 7.13 toolchain.

Everything below is **validated on real gfx1201 hardware** (dual R9700), against
the RDNA4 ISA, not inferred from benchmarks alone. Where a capability is *not*
yet usable, we say so plainly and explain what would unblock it.

---

## 1. Shipped features — what they do and when to use them

### Precision / matrix kernels
| Feature | What it is | Use case |
|---|---|---|
| **fp8 E4M3 weights** | Native RDNA4 WMMA fp8 for prefill + `v_dot4_f32_fp8_fp8` for decode | The default 8-bit format. No-compromise quality-per-byte; prefill ~+42% vs Vulkan. Use for any dense model where you want fp8. |
| **fp8 E5M2 (bf8) weights** | Native bf8 WMMA + decode | Wider dynamic range than E4M3 (5 exp bits) at the cost of ~+2.5% PPL (2 mantissa bits). Use when activations/outliers need the range. |
| **fp8 KV-cache** (`-ctk/-ctv f8e4m3`) | fp8 K and V cache in flash-attn | Halves KV memory → longer context or more concurrent sequences on a 32 GB card. |
| **fp8 MoE** (`MUL_MAT_ID`) | fp8 for mixture-of-experts expert matmuls | fp8 for MoE models (e.g. Qwen3.6-35B-A3B). Quantizes the *experts*, not just attention. |
| **MXFP8** (`block_mxfp8`, e8m0 group-32 scale) | OCP Microscaling fp8; T77 hardware-dot2 decode | Ingests MLX `mx.quantize(mode="mxfp8")` models (e.g. OsaurusAI). T77 makes it the fastest fp8 *decode* path. |
| **2:4 structured-sparse fp8** (`2OF4_FP8`, SWMMAC) | RDNA4 2:4 sparse-tensor-core fp8 | Trained-2:4-sparse models (e.g. Sparse-Llama) at fp8 — 5.5 bpw, sparse-tensor-core throughput. |
| **2:4 structured-sparse fp16** (`2OF4_F16`, SWMMAC) | RDNA4 2:4 sparse f16 A/B | Trained-2:4-sparse models with **no quantization loss** — full-precision values on the sparse tensor cores. |

### Decode / serving levers
| Feature | What it is | Use case |
|---|---|---|
| **MTP self-speculative decode** (`--spec-type draft-mtp`) | Uses the model's own next-token (MTP) head as the draft | Single-GPU interactive latency — decode **95 t/s > Vulkan 91** on Qwen3.6-27B. The single-GPU champion. |
| **DFlash spec-decode** (`--draft-dflash`) | Q8/fp8 DFlash drafter + fp8 target | 1-GPU 52 t/s while leaving the 2nd GPU free for an agent fleet. |
| **Async spec-decode pipeline** (`LLAMA_SPEC_ASYNC=2`) | Draft-gen ‖ verify on **separate GPUs** (disjoint compute); needs a draft model + 2 cards | **A 2-GPU RDNA4 box serving one latency-critical stream** — a local coding assistant, an interactive chat, or an agent loop where you want the highest tokens/sec and have both cards. The draft model sits on GPU1 generating candidates *while* GPU0 verifies the previous batch, so neither card idles waiting on the other → **+75% decode vs running them serially**. Single-GPU is a wash (the two saturating kernels time-share one card), so this is specifically a *dual-card* lever; auto-enable when 2 GPUs + a draft are detected is on the roadmap (§3). |
| **Register-spill fixes** | fattn-vec fp8-KV (hd128/256, ncols=2) → 0 spill; mmq fp8/bf8/mxfp8 48-tile → 0 spill | Removes VGPR spills in fp8-KV + spec-decode/batched-verify, and in specific prefill batch widths (+17.8% on the 48-tile). Automatic — no flag. |
| **Continuous batching — multi-user serving** (`--cont-batching`, default) | Merges concurrent decode steps: one weight read advances **N** sequences (the amortization that makes serving fast) | **Serve *dozens* of concurrent users/agents off one 2-GPU box — the vLLM replacement on RDNA4.** On the 35B-A3B fp8 MoE, aggregate decode scales **65 → 537 t/s (8.2×), peaking at npl≈114**, then knees as the MoE's routed-expert union saturates (all 256 experts pulled per step → no more weight-read amortization; regression past ~114, *not* a VRAM wall — npl=128 fits). **Throughput oscillates with npl mod 4** (a batch/ubatch-tiling alignment effect, reproducible ~±11%): npl ≡ 2 mod 4 is the favorable alignment (110/114/118 ≈ 531–537 t/s) while ≡ 0 mod 4 is worst (112/116 ≈ 451–490) — **set `--parallel` to a ≡2-mod-4 value** for peak serving. Ladder at favorable alignment: npl 1/8/32/64/**114** → 66 / 250 / 355 / 432 / **537** t/s. This is the same batching amortization that was vLLM's "server win" — but *natively* and *with fp8* (vLLM silently dequants fp8 on gfx1201, so its win evaporates here). **Caveat: fp8-KV (`-ctk/-ctv f8e4m3`) is INCOMPATIBLE with batched decode on this hybrid-SSM MoE** (breaks B>1) — batching runs f16-KV only here; fp8-KV remains a single-stream lever. The remaining vLLM-only piece is PagedAttention's fragmentation-free *paging* (landing in llama.cpp #21961). |
| **Prompt-lookup / n-gram decode — drafter-free single-stream** (`llama-lookup`, `--spec-type ngram-cache`/`ngram-map-k`) | Drafts the next N tokens by **scanning recent context for matching n-grams** (zero draft model, zero extra VRAM), verifies them in one batched pass — turns a *single* stream into a batched-verify workload | **The single-stream latency win, free.** Captures the batching amortization for ONE user: on repetitive/code/RAG/agentic output where the model echoes its context, single-stream decode jumps **A3B 66→298 t/s (+348%, 4.5×; peak 349 = 92% of the npl=8 batch ceiling)** and **27B dense 18→134 t/s (+658%, 7.6×)** — *beating MTP on every model×workload*, most dramatically on the **A3B MoE where MTP regressed below baseline** (near-zero draft cost vs MTP's extra neural-head pass). Prose is lower but non-null (LLM self-repetition still gives material): A3B 66→177 (2.7×), 27B 18→87 (4.9×). **Best default for coding assistants / RAG / agent loops.** ⚠️ Caveats: (1) **lookahead/Jacobi (`llama-lookahead`) is INCOMPATIBLE** with Qwen3.6 M-RoPE (crashes `position X<Y`) — n-gram lookup only. (2) Output is exact *modulo fp8 rounding*: byte-identical short-horizon, but diverges ~40–120 tok at near-tied greedy logits (batch-size-dependent fp8-verify rounding + MoE routing noise — a numerical effect, **not** the ngram accept/reject logic). Fine for sampling; a minor caveat for strict greedy determinism. |

### Operations
| Feature | What it is | Use case |
|---|---|---|
| **Auto-ctx / OOM-guard** (`roc8-autoctx`) | Sizes context to free VRAM (model + KV, fp8-KV-aware); CPU-spill + desktop/GDM headroom aware; caps `-ngl` ≤ cores−1; clamps API-forced ctx; `--bypass-oom` escape | Prevents the classic "load OOMs the card" failure. The Lemonade default so users never hand-tune ctx. Rejects an oversized API `ctx_size` and forces it back to the computed max (or errors, your choice). |
| **The Lemonade appliance** | Rootless-Podman image (`ubuntu:24.04` + TheRock 7.13, one extra dep: `libatomic1`) | One-command portable RDNA4 AI stack. Proven to run pure-7.13 with **zero `/opt/rocm`** on the host. Requires `crun` for GPU passthrough (not `runc`). |

---

## 2. Dormant / model-blocked capabilities — honest status

These are **hardware-validated and correct**, but not usable end-to-end today
because no producer *model* exists (or an accuracy gate wasn't met). We ship
them dormant-complete — correct-when-a-model-exists — rather than pretend
they're ready. This is deliberate: the driver is complete for the *hardware*,
not just for today's models.

| Capability | State | Why it's blocked | What would unblock it |
|---|---|---|---|
| **iu4 W4A4** (`block_iu4`, native int4×int4 WMMA) | Kernel correct — the transposed-accumulator readback bug is now fixed in **both** the selftest *and* the real-model matmul (`mul_mat_iu4`, `DATA_LAYOUT_J_MAJOR`); selftest passes exact | **Model-blocked** — no packed uniform-int4 **W4A4** model exists that avoids an *online* activation rotation | A W4A4 model whose rotation/smoothing folds **offline** into weights (co-trained BitNet-a4.8, or a QuaRot/SpinQuant export with rotation pre-applied) |
| **Weight-only int4 (W4A16)** — Quark `int4_wo` → ggml Q4 | **Path verified:** Quark 0.12 produces runnable weight-only int4 (`int4_wo_128/64/32`, **no online rotation**), and llama.cpp Q4_K already runs it | **Converter-blocked, *not* model-blocked** — our converter ingests Quark **fp8** but not int4 yet; Quark's own GGUF export is Q4_1/toy-arch only (won't do hybrid archs) | Extend the converter to ingest Quark `int4_wo` → a ggml Q4 block. Unlike W4A4 above this needs **no special model** — it's plumbing. First use: the all-Quark spec-decode **draft** for the Omni bundle |
| **Native bf8 `V_DOT4_F32_BF8_BF8`** | Built, ISA-validated (emits the real opcode), VGPR-efficient | **Accuracy-gated OFF** — bf8 activations cost +2.5% PPL vs the int8-activation dot2 path | A lower-error bf8 activation scheme (or a model tolerant of the loss); the kernel is ready to flip on |
| **2:4-sparse int8** (`swmmac_iu8`) | Kernel validated | **Model-blocked** — no W8A8 2:4-sparse model on hand | A trained/exported W8A8 2:4-sparse model + its converter |
| **2:4-sparse iu4** (`swmmac_iu4`) | Kernel validated | **Model-blocked** — needs the iu4 activation path *and* a 2:4-int4 model | A co-trained Sparse-BitNet-class 2:4-int4 model |
| **Mixed fp8×bf8 WMMA** (E4M3 weights × E5M2 activations, `GGML_HIP_FP8_ACT=bf8`) | Runtime toggle, correct | **Accuracy-gated OFF** by default (+0.29% PPL) | Kept as an opt-in runtime lever; flip `GGML_HIP_FP8_ACT=bf8` if the trade suits your model |

**Why keep dead-looking kernels?** RDNA4 silicon supports these paths
(verified against the ISA via `llvm-objdump --mcpu=gfx1201`, not header greps).
Shipping them correct-but-dormant means the day a suitable model appears, it
runs with zero kernel work. It also documents *exactly* what RDNA4 can and
cannot do for low-precision inference.

---

## 3. Roadmap / optional levers (not yet built)
- **Auto-enable the async pipeline** when 2 GPUs + a draft model are detected (today it's the `LLAMA_SPEC_ASYNC=2` opt-in).
- **mmq_x=48 spill** on F8E4M3's headroom — spot-check if future changes add register pressure.
- **MXFP4→FP8 upcast** converter (lossless upcast onto the fp8 tensor-core path).
- **Hardware-aware quantizer** — detect GPU capability → pick the optimal quant → validate on-box → emit.
- **Multi-arch port** — capability-gated RDNA1/2/3/3.5 + CDNA (hardware-owner validated).

---

## 4. Where to get it
- **Models:** the *The Rock8 — RDNA4 fp8* collection on Hugging Face (`Quack-*-FP8`) — each Quark-quantized from full-precision BF16, with PPL + throughput benches on gfx1201.
- **Container:** `ghcr.io` / Docker Hub / Quay.io — `the-rock8:rdna4-tr713` (podman *and* docker pull the same image; use `--runtime crun` for GPU).
- **Source:** the `The-Rock8` GitHub repo — kernels, patch series, appliance recipe, and these docs.

*Every artifact links to the others — land on any one, reach them all.*
