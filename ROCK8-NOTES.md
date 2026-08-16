# The Rock8 — all-AMD release line

Banked kernel improvements from the roc10 development line, compiled and
guarded for **all consumer RDNA generations** in one fatbin. The frozen
hackathon submission (`roc8` branch) is untouched; this branch is the
downloadable release line.

## Build (all-AMD fatbin)

    cmake -B build -DGGML_HIP=ON \
      -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ \
      -DCMAKE_HIP_ARCHITECTURES="gfx1030;gfx1100;gfx1151;gfx1201" \
      -DAMDGPU_TARGETS="gfx1030;gfx1100;gfx1151;gfx1201" \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j

Add/remove gfx targets freely; per-arch behavior is compile-guarded from
the ISA grid (`tools/bench-card/isa-grid.{md,json}` — 17 arches x 32
datapath instructions, `llvm-mc`-probed).

## What engages where

| capability | RDNA4 (gfx120x) | RDNA3/3.5 (gfx110x/115x) | RDNA2 (gfx103x) | CDNA |
|---|---|---|---|---|
| Q1_0/Q2_0 SWAR tile unpacks (+12.6%/+8.9% pp measured on RDNA4) | Y | Y | Y | Y (generic int SWAR) |
| Q1_0/Q2_0 fold-identity dp4a decode | Y | Y | Y (i8 dot4) | Y (i8 dot4) |
| gated-delta-net DPP wave reduction (+2% pp) | Y | Y | Y | butterfly fallback |
| MXFP6 native MMQ tile (+40% kernel-local) | Y | compiled out | compiled out | compiled out |
| F8E4M3 T77 dot2 decode | Y | compiled out | compiled out | compiled out |
| 2:4 sparse SWMMAC prefill | Y | compiled out | compiled out | compiled out |
| RDNA4 launch-geometry tables (rpb/nwarps/fusion) | Y | upstream defaults | upstream defaults | upstream defaults |
| experimental routes (iu4 W4A4, fp8 SR, dot2f16, per-channel) | env-gated, RDNA4 runtime-checked | compiled out | compiled out | compiled out |

## RDNA3 testers — expected results

Correctness gates (bit-exact or noted; wikitext-2, `--chunks 4`):
- Bonsai-27B Q1_0 prefill-path PPL: **11.6466** (bit-exact vs perm/select
  reference paths — any other value is a bug, please report)
- Q2_0 g128 prefill-path PPL: **10.1465**
- gdn DPP path: PPL parity within ±0.02 of the above (reassociation only)

Performance: expect the SWAR prefill gains (+6–30% depending on baseline)
and the dp4a decode paths to transfer; RDNA4-only rows above will report
`compiled out`/fallback and that is correct behavior. Please capture
`llama-bench` tg128/pp2048 plus junction temps — bench-card tooling in
`tools/bench-card/` automates full provenance capture if you want it.

## Strix Halo (Ryzen AI Max, gfx1151) testers

Covered by the default build line as-is (`gfx1151` is in the target list;
RDNA 3.5 resolves to the RDNA3-tier guards). APU-specific notes:

- **Memory**: unified LPDDR5X (~256 GB/s class). Ternary decode is
  memory-bound, so expect proportionally lower tg than dGPUs — the
  interesting Halo result is that a 27B 1-bit model (3.5 GB) fits and
  decodes usefully at all, and the SWAR VALU savings are worth *more*
  per byte here than on discrete cards. Rough ceiling math: Q1_0 27B
  ≈ 256/3.4 → ~75 t/s theoretical; report whatever fraction you achieve
  together with the model of your unit.
- **Stack**: needs a ROCm build with gfx1151 enabled (ROCm 7.x /
  TheRock wheels both carry it). Use `-ngl 99`; unified memory means no
  VRAM-size gymnastics, but check the BIOS carveout if allocation fails.
- **Gates are identical**: the bit-exact PPL values above are
  architecture-independent — same numbers or it is a bug.
- If you capture results, junction/skin temps and the power profile
  (balanced vs performance) matter on APUs — note them alongside
  tg128/pp2048.
