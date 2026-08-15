# TheRock ROCm 7.13 Pin — Ship Sign-Off (Section B, blocker #2)

> **SUPERSEDED 2026-08-15 — this sign-off is historical.**
>
> The pin it certifies (TheRock ROCm 7.13) is **retired**; the "keep inference on
> 7.13" directive of 2026-07-16 is rescinded and **ROCm 7.14 is the going-forward
> toolchain** (`use-rocm714.env`). There is no 7.15 — TheRock's gfx120X nightly
> index has published nothing newer than `7.14.0a20260612`.
>
> This document remains valid for exactly one purpose: it describes the toolchain
> bundled in the **frozen submission image** `ghcr.io/the-monk/the-rock8:rdna4-tr713`
> (built from `roc8` `70629174b1`). That image is frozen and is not being rebuilt
> or retagged, so its 7.13 contents stay accurate and reproducible. Nothing here
> should be read as a recommendation for new builds.
>
> **The successor recipe is `container/Containerfile.dev714`** (7.14 devel SDK).
> It is **not yet validated**: as of 2026-08-15 it has never been built — every
> image on the build host is a `tr713` tag — and it builds branch `roc8`, which
> carries the `MUL_MAT_ID` out-of-bounds defect. Before it can replace this
> sign-off it must be built from a fixed branch and pass `validate_image.sh`,
> and a new sign-off written against the resulting digest.
>
> Also note: the "Image" line below cites digest tag `f36df0f3c`, which is a
> **4-week-old** build, not the submitted `70629174b1`. Treat the digest here as
> referring to the 13 July validation run only.

**Date:** 2026-07-13
**Image:** `localhost/roc8-lemonade:tr713` (= `ghcr.io/the-monk/the-rock8:rdna4-tr713`, digest tag `f36df0f3c`)
**Target:** RDNA4 / gfx1201 (Radeon AI PRO R9700, RX 9070 XT/9070)

## Pin

| Item | Value |
|------|-------|
| ROCm distribution | **TheRock `gfx1201-7.13.0`** (bundled at `/opt/therock`) |
| `rocm-core` soname | `librocm-core.so.1.0.71300` → **ROCm 7.13.00** |
| HIP runtime | `libamdhip64.so.7` (HIP 7.x) |
| rocBLAS | `librocblas.so.5.4` |
| Base image | `docker.io/library/ubuntu:24.04` |
| `/opt/rocm` present | **NO** — zero host ROCm dependency |

## Verification (reproducible)

Run `./validate_image.sh` (ldd closure) or the inline check below. All four shipped
binaries resolve their full closure inside the container with **no `/opt/rocm` and no
missing libraries**:

```
llama-server     : CLEAN (self-contained, no /opt/rocm)
llama-bench       : CLEAN
llama-perplexity  : CLEAN
llama-quantize    : CLEAN

rocm sonames resolve to /opt/therock:
  libhipblas.so.3   => /opt/therock/lib/libhipblas.so.3
  librocblas.so.5   => /opt/therock/lib/librocblas.so.5
  libamdhip64.so.7  => /opt/therock/lib/libamdhip64.so.7
  libhipblaslt.so.1 => /opt/therock/lib/libhipblaslt.so.1
```

This confirms card-139's finding — the appliance runs on **pure TheRock 7.13**, independent
of the host's ROCm 7.2.4. Portability was already validated on a clean therock-7.13-only host
(card 138/139).

## Lemonade wiring

The Containerfile pre-seeds Lemonade's `rocm` llama_server slot with the ROC8 `build-tr713`
binaries and writes matching `version.txt` (`b1066`) + `backend.txt` (`rocm`) so
`install_llamacpp()` short-circuits and **never downloads the stock
`lemonade-sdk/llamacpp-rocm` build**. The shipped fp8/RDNA4 kernels are the ones served.

## Verdict

✅ **PIN LOCKED — ROCm 7.13.00 (TheRock gfx1201-7.13.0), self-contained, sign-off complete.**

Do not float this pin. Any ROCm bump requires re-running `validate_image.sh` + the
sustained load test (`loadtest.sh`) and a new sign-off entry here.
