"""
W4A4 ACTIVATION-RESCUE schemes for Bonsai (gfx1201 optimizer, task: clear the
int4-activation accuracy gate). Extends w4a4_gate_bonsai.py -- same model, same
corpus, same "weight is NEVER requantized" rule (Bonsai's HF safetensors are a
lossless fp16 unpack of an already-ternary model; the only thing that varies
here is the ACTIVATION quantization path).

w4a4_gate_bonsai.py already measured (results_bonsai.json, reproduced fresh by
this script's first three rows):
    int8_act           (W4A8 floor, the shipped path proxy)
    int4_act           (naive W4A4, round-to-nearest per-token group=32)
    int4_act_hadamard  (rotate-quantize-derotate per group=32, weight untouched
                        -- the "fused into adjacent norms, free at inference"
                        style QuaRot/SpinQuant probe)

This script ADDS the schemes the task asked for that were not yet covered:
    smoothquant_a32    SmoothQuant-style per-input-channel scale migration
                        (s_k = actmax_k^alpha / wmax_k^(1-alpha), alpha=0.5,
                        calibrated on a held-out slice of the SAME corpus text)
                        folded EXACTLY into weight (weight stays real-valued,
                        never rounded) + int4 act quant at group=32. At
                        inference this s_k would fuse into the preceding
                        RMSNorm weight -- free, no extra op (matches the task's
                        "fused into adjacent norms" requirement).
    int4_g16 / int4_g8  naive int4, finer activation granularity (16/8 instead
                        of the shipped 32) -- more scale-storage/rescale, less
                        outlier dilution per group.
    int4_calib_p999     per-group int4, but the clip scale is a STATIC,
                        offline-calibrated 99.9th-percentile magnitude (from
                        the calibration slice) instead of the per-token dynamic
                        round-to-nearest amax -- tests calibration vs naive.
    smoothquant_g16     combines the best scale-migration + the best finer
                        granularity found above.

Calibration = the FIRST ~1024 tokens of corpus.txt (same real text the ppl eval
uses -- no synthetic/fabricated data). Verdict bands match the existing gate:
PASS<=15% / MARGINAL 15-30% / FAIL>30%, measured against the int8_act floor.

Env: mtp713. GPU1 only (HIP_VISIBLE_DEVICES pinned by caller).
"""
import os, sys, time, math, json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/aipool/models/ternary-bonsai-phase0/unpacked"
CORPUS    = "/home/jmonk/w4a4-gate/corpus.txt"
OUT_JSON  = "/home/jmonk/w4a4-gate/results_bonsai_rescue.json"
CALIB_TOKENS = 1024

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device, "torch:", torch.__version__)


# ---------------------------------------------------------------------------
# shared fake-quant helpers (mirrors w4a4_gate_bonsai.py exactly for the
# reproduced rows, so numbers are directly comparable)
# ---------------------------------------------------------------------------
def groupwise_quant_dequant(x, bits, group, dim=-1, rotate=False, had=None):
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    orig_shape = x.shape
    n = orig_shape[dim]
    assert dim == -1
    pad = (-n) % group
    if pad:
        pad_shape = list(orig_shape); pad_shape[dim] = pad
        x = torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=dim)
    ng = x.shape[-1] // group
    xr = x.reshape(*x.shape[:-1], ng, group)
    if rotate:
        xr = xr @ had
    absmax = xr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = absmax / qmax
    q = torch.clamp(torch.round(xr / scale), qmin, qmax)
    deq = q * scale
    if rotate:
        deq = deq @ had.t()
    deq = deq.reshape(*x.shape[:-1], ng * group)
    if pad:
        deq = deq[..., :n]
    return deq.reshape(orig_shape)


_had_cache = {}
def get_hadamard(n, dev, dtype):
    key = (n, str(dev), dtype)
    if key in _had_cache:
        return _had_cache[key]
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    H = (H / math.sqrt(n)).to(device=dev, dtype=dtype)
    _had_cache[key] = H
    return H


def static_group_quant_dequant(x, bits, group, scale_per_group, dim=-1):
    """Like groupwise_quant_dequant but the per-group SCALE is a fixed,
    precomputed tensor (shape [..., ng, 1] or broadcastable) instead of the
    per-token dynamic amax -- the offline-calibration variant."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    orig_shape = x.shape
    n = orig_shape[dim]
    assert dim == -1
    pad = (-n) % group
    if pad:
        pad_shape = list(orig_shape); pad_shape[dim] = pad
        x = torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=dim)
    ng = x.shape[-1] // group
    xr = x.reshape(*x.shape[:-1], ng, group)
    scale = scale_per_group.clamp_min(1e-8).view(*([1] * (xr.dim() - 2)), ng, 1)
    q = torch.clamp(torch.round(xr / scale), qmin, qmax)
    deq = (q * scale).reshape(*x.shape[:-1], ng * group)
    if pad:
        deq = deq[..., :n]
    return deq.reshape(orig_shape)


_ternary_checked = False
def _ternary_sanity(w):
    global _ternary_checked
    if _ternary_checked:
        return
    _ternary_checked = True
    row = w[0].detach().float().flatten()
    uniq_abs = torch.unique(torch.round(row.abs() / (row.abs().amax().clamp_min(1e-9)) * 1e6) / 1e6)
    print(f"[ternary-check] row0: {uniq_abs.numel()} distinct |values|, "
          f"zero-fraction={(row.abs() < 1e-9).float().mean().item():.3f}")


# ---------------------------------------------------------------------------
# Linear wrappers
# ---------------------------------------------------------------------------
class NaiveIntActLinear(nn.Module):
    """int8 / int4 / int4+hadamard, group configurable -- weight untouched."""
    def __init__(self, lin, bits, group, rotate=False):
        super().__init__()
        self.bias = lin.bias
        w = lin.weight.data
        _ternary_sanity(w)
        self.register_buffer("w", w)
        self.bits, self.group, self.rotate = bits, group, rotate
        self.had = get_hadamard(group, w.device, torch.float32) if rotate else None

    def forward(self, x):
        xq = groupwise_quant_dequant(x.float(), bits=self.bits, group=self.group,
                                      rotate=self.rotate, had=self.had).to(x.dtype)
        return torch.nn.functional.linear(xq, self.w, self.bias)


class SmoothQuantLinear(nn.Module):
    """Exact (pre-quant) per-input-channel scale migration: x' = x / s,
    w' = w * s[None,:] (weight stays REAL-VALUED, never rounded -- s just
    redistributes dynamic range from activation to weight, same identity
    SmoothQuant uses, no accuracy cost until the int4 act quant step)."""
    def __init__(self, lin, s, bits, group):
        super().__init__()
        self.bias = lin.bias
        w = lin.weight.data.float()
        self.register_buffer("w", (w * s.unsqueeze(0)).to(lin.weight.dtype))
        self.register_buffer("inv_s", (1.0 / s).to(lin.weight.dtype))
        self.bits, self.group = bits, group

    def forward(self, x):
        xs = x.float() * self.inv_s.float()
        xq = groupwise_quant_dequant(xs, bits=self.bits, group=self.group).to(x.dtype)
        return torch.nn.functional.linear(xq, self.w, self.bias)


class CalibratedStaticLinear(nn.Module):
    """int4 activation quant with a fixed, offline-calibrated per-group scale
    (percentile-based clip, not per-token dynamic amax). weight untouched."""
    def __init__(self, lin, bits, group, scale_per_group):
        super().__init__()
        self.bias = lin.bias
        self.register_buffer("w", lin.weight.data)
        self.register_buffer("scale", scale_per_group.float())
        self.bits, self.group = bits, group

    def forward(self, x):
        xq = static_group_quant_dequant(x.float(), bits=self.bits, group=self.group,
                                         scale_per_group=self.scale).to(x.dtype)
        return torch.nn.functional.linear(xq, self.w, self.bias)


def target_linears(model):
    out = []
    for li, layer in enumerate(model.model.layers):
        for name in ["self_attn", "mlp"]:
            sub = getattr(layer, name)
            for pname, mod in list(sub.named_children()):
                if isinstance(mod, nn.Linear):
                    out.append((sub, pname, mod, f"L{li}.{name}.{pname}"))
    return out


def patch_with(model, factory):
    """factory(lin, key) -> replacement module."""
    count = 0
    for sub, pname, mod, key in target_linears(model):
        setattr(sub, pname, factory(mod, key))
        count += 1
    return count


# ---------------------------------------------------------------------------
# Calibration pass: per-layer input activation abs-max per channel, on the
# clean (unpatched) bf16 model, over the FIRST CALIB_TOKENS tokens of the SAME
# corpus the ppl eval uses.
# ---------------------------------------------------------------------------
@torch.no_grad()
def calibrate_actmax(model, tokenizer, text):
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[:, :CALIB_TOKENS].to(device)
    stats = {}
    handles = []

    def mk_hook(key):
        def hook(module, inp):
            x = inp[0].detach().float()
            am = x.reshape(-1, x.shape[-1]).abs().amax(dim=0)
            if key in stats:
                stats[key] = torch.maximum(stats[key], am)
            else:
                stats[key] = am
        return hook

    for sub, pname, mod, key in target_linears(model):
        handles.append(mod.register_forward_pre_hook(mk_hook(key)))

    model(ids)

    for h in handles:
        h.remove()
    return stats  # key -> [K] abs-max tensor (fp32, on device)


@torch.no_grad()
def calibrate_percentile(model, tokenizer, text, group, pct=99.9):
    """Per-group STATIC clip scale from the pct-th percentile of |activation|,
    computed over all calibration tokens, grouped along the same contiguous
    K-groups the online quantizer would use."""
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[:, :CALIB_TOKENS].to(device)
    buf = {}
    handles = []

    def mk_hook(key):
        def hook(module, inp):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])  # [T,K]
            buf.setdefault(key, []).append(x.cpu())
        return hook

    for sub, pname, mod, key in target_linears(model):
        handles.append(mod.register_forward_pre_hook(mk_hook(key)))

    model(ids)
    for h in handles:
        h.remove()

    qmax = 2 ** (4 - 1) - 1
    scales = {}
    for key, chunks in buf.items():
        x = torch.cat(chunks, dim=0)  # [T,K]
        K = x.shape[-1]
        pad = (-K) % group
        if pad:
            x = torch.cat([x, torch.zeros(x.shape[0], pad)], dim=-1)
        ng = x.shape[-1] // group
        xr = x.reshape(x.shape[0], ng, group).abs()
        clip = torch.quantile(xr.reshape(x.shape[0], ng, group).float(), pct / 100.0, dim=0)  # [ng,group]
        clip = clip.amax(dim=-1)  # [ng] -- one scale per group (percentile over tokens, max within group)
        scales[key] = (clip.clamp_min(1e-8) / qmax).to(device)
    return scales


# ---------------------------------------------------------------------------
# perplexity (identical methodology to w4a4_gate_bonsai.py)
# ---------------------------------------------------------------------------
@torch.no_grad()
def perplexity(model, tokenizer, text, seq_len=512, stride=512, max_tokens=4096):
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[:, :max_tokens].to(device)
    n = ids.shape[1]
    nlls, total_tok = [], 0
    for begin in range(0, n - 1, stride):
        end = min(begin + seq_len, n)
        input_ids = ids[:, begin:end]
        if input_ids.shape[1] < 2:
            continue
        out = model(input_ids, labels=input_ids.clone())
        ntok = input_ids.shape[1] - 1
        nlls.append(out.loss.float() * ntok)
        total_tok += ntok
        if end == n:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / total_tok).item()
    return ppl, total_tok


def load_fresh():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    model.to(device); model.eval()
    return model, tok


def run_variant(label, patch_fn, text):
    model, tok = load_fresh()
    t0 = time.time()
    n = patch_fn(model, tok, text)
    ppl, ntok = perplexity(model, tok, text)
    dt = time.time() - t0
    print(f"[{label:22s}] ppl={ppl:.4f} patched={n} ({dt:.1f}s)")
    del model; torch.cuda.empty_cache()
    return ppl


def main():
    text = open(CORPUS, encoding="utf-8").read()
    results = {}

    # --- bf16 reference + reproduce the existing gate's 3 rows fresh ---
    def _bf16(model, tok, text):
        return 0
    ppl_bf16 = run_variant("bf16_reference", _bf16, text)
    results["bf16_reference"] = ppl_bf16

    ppl_int8 = run_variant("int8_act(floor)", lambda m, t, tx: patch_with(
        m, lambda lin, k: NaiveIntActLinear(lin, bits=8, group=32)), text)
    results["int8_act"] = ppl_int8

    ppl_int4 = run_variant("int4_act(naive)", lambda m, t, tx: patch_with(
        m, lambda lin, k: NaiveIntActLinear(lin, bits=4, group=32)), text)
    results["int4_act"] = ppl_int4

    ppl_int4had = run_variant("int4_act_hadamard", lambda m, t, tx: patch_with(
        m, lambda lin, k: NaiveIntActLinear(lin, bits=4, group=32, rotate=True)), text)
    results["int4_act_hadamard"] = ppl_int4had

    # --- finer granularity, no rotation, no smoothing ---
    ppl_g16 = run_variant("int4_g16", lambda m, t, tx: patch_with(
        m, lambda lin, k: NaiveIntActLinear(lin, bits=4, group=16)), text)
    results["int4_g16"] = ppl_g16

    ppl_g8 = run_variant("int4_g8", lambda m, t, tx: patch_with(
        m, lambda lin, k: NaiveIntActLinear(lin, bits=4, group=8)), text)
    results["int4_g8"] = ppl_g8

    # --- SmoothQuant-style scale migration (calibrated on the same corpus) ---
    def _smoothquant(model, tok, tx, group, alpha=0.5):
        stats = calibrate_actmax(model, tok, tx)
        def factory(lin, key):
            w = lin.weight.data.float()
            wmax = w.abs().amax(dim=0).clamp_min(1e-5)
            actmax = stats[key].clamp_min(1e-5)
            s = (actmax ** alpha) / (wmax ** (1 - alpha))
            s = s.clamp(1e-4, 1e4)
            return SmoothQuantLinear(lin, s, bits=4, group=group)
        return patch_with(model, factory)

    ppl_sq32 = run_variant("smoothquant_a32", lambda m, t, tx: _smoothquant(m, t, tx, group=32), text)
    results["smoothquant_a32"] = ppl_sq32

    ppl_sq16 = run_variant("smoothquant_g16", lambda m, t, tx: _smoothquant(m, t, tx, group=16), text)
    results["smoothquant_g16"] = ppl_sq16

    # --- calibrated static percentile clip, group=32 ---
    def _calib_static(model, tok, tx, group=32, pct=99.9):
        scales = calibrate_percentile(model, tok, tx, group=group, pct=pct)
        def factory(lin, key):
            return CalibratedStaticLinear(lin, bits=4, group=group, scale_per_group=scales[key])
        return patch_with(model, factory)

    ppl_calib = run_variant("int4_calib_p999", lambda m, t, tx: _calib_static(m, t, tx), text)
    results["int4_calib_p999"] = ppl_calib

    # ---------------------------------------------------------------------
    print("\n=== SUMMARY (delta vs int8_act FLOOR = the shipped W4A8 path) ===")
    floor = results["int8_act"]
    order = ["bf16_reference", "int8_act", "int4_act", "int4_act_hadamard",
             "int4_g16", "int4_g8", "smoothquant_a32", "smoothquant_g16",
             "int4_calib_p999"]
    deltas = {}
    for k in order:
        v = results[k]
        d = (v / floor - 1) * 100
        deltas[k] = d
        verdict = "PASS" if d <= 15 else ("MARGINAL" if d <= 30 else "FAIL")
        print(f"  {k:22s} ppl={v:9.4f}  Δvs_int8={d:+7.3f}%  {verdict}")

    best_key = min([k for k in order if k not in ("bf16_reference", "int8_act")],
                   key=lambda k: deltas[k])
    print(f"\n=== BEST SCHEME: {best_key}  ({deltas[best_key]:+.3f}% vs int8 floor) ===")
    print(f"=== NAIVE int4_act (no rescue at all): {deltas['int4_act']:+.3f}% vs int8 floor ===")

    results["_deltas_vs_int8_floor_pct"] = deltas
    results["_best_scheme"] = best_key
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
