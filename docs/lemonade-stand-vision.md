# The Lemonade Stand — a local AI OS front-end

*Vision / architecture writeup. Lemonade runs the models; the Lemonade Stand is the whole
storefront around it — hardware-aware inference + knowledge + memory + tools + agents,
assembled into one coherent, local-first, AMD-native AI operating environment.*

---

## The pitch

Today, running local AI is a pile of disconnected parts: a runtime here, a model download
there, a vector DB, a note-taking wiki, an MCP server, and an agent CLI — each configured by
hand, none aware of the others or of the machine they run on. **The Lemonade Stand collapses
that pile into an OS-like layer.** You point it at a box; it figures out the silicon, picks
the best backend and quantization for *that* hardware, mounts your knowledge and memory,
exposes a tool fabric, and lets any agent harness plug in on top. Lemonade is the inference
kernel; the Stand is the OS around it.

Think of it as the layer between **raw hardware** and **user intent** — exactly what an OS
does: abstract the hardware, manage the resources, provide the services, run the apps.

---

## The stack (each layer maps to a real component we've built)

### Layer 0 — Hardware detection
Enumerate GPUs, arch (gfx1201/gfx1151/gfx942…), VRAM, NUMA, NPU, CPU. This is the
`lemonade-stand` tool's **detect** stage — it fingerprints the box (BDF-keyed VRAM,
discrete-vs-iGPU, per-SKU bandwidth) so nothing downstream has to guess.

### Layer 1 — Backend selection
Given the hardware, pick the *optimal runtime*, not a lowest-common-denominator one:
- **RDNA4 (gfx1201)** → **The Rock8** native-fp8 llama.cpp (fp8 WMMA prefill +42% vs Vulkan)
- **CDNA (gfx942/950)** → vLLM (datacenter concurrency, native FP4)
- **RDNA3/3.5, Strix Halo** → tuned llama.cpp HIP (arch-specialist optimizer agents)
- **No GPU** → CPU/Vulkan fallback

This is the `decide` stage + the **arch-specialist optimizer agent family**
(`llm-optimizer-<gfx>`) that carries validated per-arch tuning (a compounding KB: the next
box of an arch auto-configures with every win recorded).

### Layer 2 — Model management
Inventory local GGUFs + Hugging Face, and choose the **quantization per hardware**:
fp8 / mixed FP4-MLP+FP8-attn / int8 / MXFP4, sized to fit VRAM (fp8-KV-aware, per-slot ctx).
**Validate on-box** (measured tokens/s + PPL), then **emit** a Lemonade config. The
Quacken/RDNA4-fp8 model collection is the reference library; the "hardware-aware quantizer"
(HAQ-inspired: detect capability → pick per-layer format → validate → emit) is the
north-star of this layer.

### Layer 3 — Knowledge (RAG + wiki)
A persistent **knowledge tree** (the wiki: markdown, nightly-projected board state, tech
reports) + **RAG** over it and over authoritative corpora (policy/STIG/security docs). Agents
ground answers in cited sources instead of hallucinating. The wiki is both the human's notes
*and* the machine's retrieval index — one brain, human- and agent-readable.

### Layer 4 — Memory & state (DB + MCP)
A **single source of truth** for work state: a **kanban** (Vikunja) as the task board, a
hash-chained **memory-events DB**, and the nightly projection into the wiki. Agents read the
board before acting and write back on completion, so no session re-discovers a closed thread.
This is the *persistence* layer — the thing that makes a fleet of stateless agents behave
like one continuous operator.

### Layer 5 — Tool fabric (MCP)
The **MCP server** as the plug-in bus: GPU/system/infra tools, a security DB (CVE/KEV/STIG/
ATT&CK), policy/RAG tools, GitHub, the kanban tools — dozens of capabilities any agent can
call over one protocol. New capabilities are MCP servers, not bespoke integrations.

### Layer 6 — Agent harnesses (pluggable front-ends)
The apps that run on the OS. Because everything below speaks two standards — an **OpenAI-
compatible API** (Lemonade) and **MCP** (tools/memory) — the harness is *swappable*:
- **Claude Code** — the coding/ops agent
- **Hermes** — the voice/home-assistant agent
- **OpenClaude / open harnesses** — community front-ends
- **…whatever comes next** — it just needs to speak OpenAI + MCP

The Stand doesn't care which brain you bolt on; it provides the body, the senses, and the
memory.

---

## The cross-cutting piece — the Tuning Agent (the Stand tunes itself)

Everything above is *static* configuration. The **Tuning Agent** is what makes the Stand
**self-improving** — an autonomous, arch-specialist optimizer that continuously mines the
silicon for measurable speed and capability gains, then bakes the wins back in so the whole
stack gets faster over time without a human hand-tuning it.

**What it does:**
- **Runs a standing optimization playbook, unprompted** — physics-first questions asked of
  every model/kernel: *memory- or compute-bound? is single-stream leaving batch efficiency
  on the table (spec-decode / n-gram lookup)? what's the model-class-appropriate lever (dense
  → MTP, MoE → concurrency)? which compute units are idle (async overlap)? native tensor-core
  path or a fallback?* It generates its own investigations instead of waiting to be told.
- **Profiles on real hardware** — via the ported `rocprof-compute` on gfx1201 (per-kernel
  time attribution) plus the counter-free analytical roofline + `amd-smi UMC_ACTIVITY`, since
  RDNA4's hardware counters are ~96% dead (a limitation the agent knows and routes around).
- **Benchmarks before/after, drift-controlled** — cold, isolated re-measures; it catches its
  own inflated numbers (the discipline that turned a bogus "2000 t/s" into the honest one).
- **Writes validated wins to a shared KB + leaderboard** — a compounding memory keyed by
  arch, so **the next box of that arch auto-configures with every win ever recorded**. Never
  transfers a win across arches as fact — it enters the other arch's section as a *hypothesis*
  until re-swept there.
- **Proposes changes as kanban cards, applies approved ones** — it never silently ships risky
  changes; it lands proposals on the board (Layer 4), benchmarks, and writes the result back,
  so no future session re-litigates a closed thread.

The Tuning Agent is a **member of the agent-harness layer that acts on Layers 1–2**: it's an
agent that optimizes the OS it runs on. Every win it validates (the `calc_rows_per_block=2`
decode win, the `nwarps` dispatch entry, the fp8 kernels, the async target-weight lever) is a
permanent upgrade to how the Stand serves *that* hardware. It's the difference between a
config tool that ships once and an OS that gets better the longer it runs.

### Evidence — the Tuning Agent's validated wins (real, on-box, drift-controlled)

| Win | Change | Result |
|---|---|---|
| Native fp8 kernels | fp8 WMMA prefill + dp4a decode on RDNA4 | 27B prefill **+42% vs Vulkan**, +6.7% vs int8 |
| Decode dispatch | `calc_rows_per_block = 2` for RDNA4 decode | Q2_0 tg128 **148 → 164 (+11%)**, beats Vulkan 155 |
| Dispatch table | add `MXFP4→nwarps=3` RDNA4 entry | 35B-A3B MXFP4 tg128 **74.2 → 76.5 (+3.2%)** |
| Register-spill fix | mmq fp8/mxfp8 48-tile → 0 spill | **+17.8%** on the 48-tile prefill width |
| Async spec-decode | draft ‖ verify on 2 GPUs (target-weight lever) | Bonsai **63 → 105 t/s (+66%)**, byte-identical |
| MTP self-spec | model's own nextn head, n=8 | 27B **18 → 45 t/s (2.43×)**, decode 95 > Vulkan 91 |

Each was found by the playbook, benched cold/isolated, and written to the shared KB
leaderboard — so they're **permanent, portable upgrades**, not one-off tweaks. (And the
discipline runs both ways: the agent *rejected* its own inflated numbers — a "2000+ t/s"
single-stream reading was caught as a dual-context timing-parse artifact and corrected to
the honest 134/298.)

---

## The Storefront — configurations as downloadable, auto-wiring products

Here's where the name pays off: **the Stand is a storefront.** You don't hand-assemble the
seven layers — you browse a catalog, pick a **configuration**, and it arrives **wired**.
Every configuration is a complete, ready-to-serve bundle:

> **hardware target + backend + model(s) + RAG corpus + MCP tool set + agent harness +
> the validated tuning profile** — packaged as one downloadable unit.

**Automatic wiring** is the magic. On install, the bundle:
1. **detects the box** (Layer 0) and confirms the config fits (VRAM, arch) — or picks the
   right variant (fp8 on RDNA4, int8 on RDNA3, CPU fallback) from the same product;
2. **lays down the backend + model** with the arch's validated tuning profile already baked
   in (courtesy of the Tuning Agent — bundles ship pre-optimized, not stock);
3. **indexes the RAG corpus** and mounts the wiki;
4. **registers the MCP tool set** and connects the memory DB / kanban;
5. **launches the agent harness** already pointed at all of it.

No config files, no port wiring, no "which quant for my GPU" — you picked the product, and it
works. It's `apt install` for a *whole* local-AI setup, self-configuring to your silicon.

**Example storefront products:**
- **RDNA4 fp8 Coding Assistant** — Rock8 fp8 + Quacken-27B + code/repo RAG + git/filesystem
  MCP + Claude Code, with n-gram `lookup` on for code reproduction.
- **Home Voice Assistant** — A3B fp8 + Hermes + Home-Assistant MCP + TTS/STT, continuous-
  batching for the household.
- **Security Analyst** — fp8 + the security-DB RAG (CVE/KEV/STIG) + the CVE/ATT&CK/STIG MCP
  tools + a policy-guardrail agent.
- **Multi-user Serving Node** — A3B fp8 MoE + `--cont-batching --parallel 114` (the ≡2-mod-4
  peak) + an OpenAI endpoint, tuned for ~537 t/s aggregate.

**Who fills the store:** the Tuning Agent contributes validated per-arch **configs**; the
fp8/mixed-precision **model** library (Quacken) ships with on-box provenance; **MCP servers**
are pluggable capability packages; **agent harnesses** are swappable brains. Because the
Tuning Agent writes wins back to the shared KB, **the store's configs improve on their own** —
a bundle you download next month is faster than today's, on the same hardware, no action from
you.

---

## The flow (how a request moves through the Stand)

```
User / agent intent
      │
  [Agent harness]  ── Claude Code / Hermes / OpenClaude
      │  needs inference          needs a tool / memory        needs knowledge
      ▼                                   ▼                            ▼
 [Lemonade + backend]              [MCP tool fabric]             [RAG over wiki]
  (Rock8 fp8 on RDNA4,              (GPU/infra/security,          (cited grounding)
   auto-selected model)             kanban read/write)
      │                                   │                            │
      └───────────────── all pre-configured by ─────────────────────┘
                         Layers 0–2 (detect → decide → validate → emit)
      │
  work happens, results written back to Layer 4 (kanban + memory DB)
```

A user opens their agent. It doesn't know or care that the box has two R9700s — the Stand has
already detected them, chosen The Rock8 fp8 backend, loaded the right-sized model, and exposed
the wiki + tools + board. The agent just *works*, grounded and stateful, on hardware it never
had to think about.

---

## Why "AI OS front-end" is the right frame

| OS concept | Lemonade Stand |
|---|---|
| Drivers / HAL | Backend selection (Rock8 / vLLM / Vulkan per arch) |
| Resource manager | VRAM budgeting, model load/evict, per-slot ctx |
| System services | RAG, memory DB, kanban, tool fabric |
| Syscall ABI | OpenAI API + MCP (what every agent speaks) |
| Applications | Agent harnesses (Claude Code, Hermes, …) |
| Package manager | Model + MCP-server registry |

It's **local-first** (private, offline-capable), **AMD-native** (extracts the silicon's real
capability instead of a portable-lowest-common-denominator), **hardware-aware** (auto-optimal,
not hand-tuned), **memory-persistent** (a board + a brain, not a fresh context each time), and
**agent-agnostic** (bring your own harness).

---

## What already exists vs. what the Stand assembles

- ✅ **Rock8** (Layer 1) — native-fp8 llama.cpp for RDNA4, shipped as a Lemonade appliance.
- ✅ **lemonade-stand tool** (Layers 0–2) — detect → decide → validate → emit, with the
  arch-specialist optimizer agents.
- ✅ **MCP server** (Layers 4–5) — dozens of tools + a security DB, over MCP.
- ✅ **Wiki + RAG** (Layer 3) — the knowledge tree, nightly-projected, RAG-queryable.
- ✅ **Kanban / memory DB** (Layer 4) — Vikunja SSOT + hash-chained events.
- ✅ **Agent harnesses** (Layer 6) — Claude Code + Hermes already run on this stack.
- ✅ **Tuning Agent** (cross-cutting) — the arch-specialist optimizer family
  (`llm-optimizer-gfx1201/1151/1100/942`) + self-optimizer, with a shared KB/leaderboard,
  the standing playbook, and the rocprof-compute-on-gfx1201 profiling port. Proven wins:
  the fp8 kernels, `rows_per_block=2` decode (+11%), the `nwarps` dispatch entry, the async
  target-weight lever.

The pieces are built and proven in isolation. **The Lemonade Stand is the product that binds
them into one thing** — a coherent, installable, local AI OS whose front door is Lemonade and
whose apps are whatever agent you like.

---

## The one-liner

**Lemonade runs the model. The Lemonade Stand runs the machine, the memory, the knowledge,
the tools — and hands the whole thing to your agent, on your hardware, offline.**
