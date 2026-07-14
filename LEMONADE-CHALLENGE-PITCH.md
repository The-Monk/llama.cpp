# The Rock8 — Lemonade Challenge Submission (one-page)

**A native-fp8 LLM runtime for AMD RDNA4, integrated into Lemonade, plus the deepest public
performance map of what gfx1201 can and can't do today.** Open-source (Apache/MIT):
github.com/The-Monk/The-Rock8 · models on Hugging Face · container on ghcr.

### What it is
Stock llama.cpp ignores RDNA4's fp8 tensor cores and dequantizes fp8 to f16. **The Rock8**
adds the real kernels — native fp8 E4M3/E5M2 WMMA prefill, `v_dot4` fp8 decode, fp8 KV-cache,
fp8 MoE, mixed FP4/FP8 — and ships as a **one-command Lemonade appliance** on TheRock ROCm 7.13.

### The numbers (real gfx1201, dual R9700, drift-controlled)
| Axis | Result |
|---|---|
| fp8 prefill (27B) | **1310 t/s — +42% vs Vulkan**, +6.7% vs int8, +0.5% PPL |
| Beating Vulkan | decode **95 > 91** (MTP), prefill +42% — both via fp8 tensor cores |
| Multi-user (A3B MoE) | **65 → 537 t/s aggregate (7.4×)** @ ~114 concurrent — the vLLM replacement |
| Single-stream latency | MTP 27B **18 → 45 (2.43×)**; n-gram lookup **18 → 134** on code/RAG |
| Mixed FP4/FP8 (27B) | **23 GB single-card, PPL 6.88 (beats fp8), +30% decode** |

### Original findings (the "deep-dive" the challenge asks for)
- **Batched throughput oscillates ±11% with `npl mod 4`** — a tiling-alignment effect that
  coarse power-of-two sweeps systematically miss; the true peak is ~11% above what everyone
  else reports. *Pick a ≡2-mod-4 `--parallel`.*
- **RDNA4 perfmon is ~96% dead on gfx1201** — only ~7 of 207 hardware counters read (full
  census included). Root-caused, firmware-ruled-out, **filed upstream: ROCm/rocprofiler-sdk
  #155**, with the exhaustive per-counter map.
- **Honest negative results** — MTP hurts sparse MoE; full-W4A4 loses to mixed FP4/FP8; async
  needs a heavy target + cheap draft; lookup only wins on repetitive output. A performance
  evaluation is only useful if the losses are in it.

### The bigger idea (bonus)
The Rock8 is the inference layer of **The Lemonade Stand** — a local AI-OS front-end:
hardware-detect → backend-select → model-tune → RAG/wiki → memory/kanban (MCP) → agent
harnesses, delivered as **downloadable, auto-wiring storefront configs** that ship
pre-optimized (a self-improving Tuning Agent writes validated wins back to a shared KB).

**Reproduce it:** `podman pull ghcr.io/the-monk/the-rock8:rdna4-tr713`. Full deep-dive:
`PERFORMANCE-DEEPDIVE.md`. Vision: the Lemonade Stand writeup.
