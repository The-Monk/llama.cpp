"""
W4A4 accuracy gate for the TERNARY BONSAI model (card 156, forked from w4a4_gate.py).

Difference from the T89 dense-proxy gate: Bonsai's HF safetensors are a LOSSLESS fp16
unpack of already-ternary weights, so we DO NOT quantize the weight side (doing so would
double-quantize and produce a meaningless number). Only the ACTIVATION path is varied:

  - none      : bf16 activations (upper bound)
  - int8      : per-token groupwise(32) symmetric int8  -> proxy for the SHIPPED dp4a path
                (Q2_0 x q8_1). THIS is the baseline the verdict is measured against.
  - int4      : per-token groupwise(32) symmetric int4  -> the native iu4xiu4 W4A4 path
                (mul_mat_q2_0_wmma.cu's k_quantize_act_iu4_q2)
  - int4+had  : int4 activations with a Sylvester-Hadamard rotation per group of 32
                applied before quant, inverted after (QuaRot/SpinQuant-style rescue probe)

Weights are the real ternary values in EVERY config -> the only ppl delta comes from the
activation precision, which is exactly the W4A4-vs-W4A8 question.

VERDICT (delta of int4-act ppl vs the int8-act floor):
  PASS     <= +15%   W4A4 viable as-is
  MARGINAL 15..30%   try the hadamard branch as a rescue
  FAIL     >  +30%   int4 activations too lossy on Bonsai; the -54% WMMA path isn't worth
                     pursuing even if fused

Env: run in mtp713 (transformers 5.14 / torch 2.13 / safetensors). HIP_VISIBLE_DEVICES pinned
by caller. The gate is a NUMERICAL fake-quant simulation -> ROCm-version-independent.
"""
import os, sys, time, math, json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/aipool/models/ternary-bonsai-phase0/unpacked"
CORPUS    = "/home/jmonk/w4a4-gate/corpus.txt"
OUT_JSON  = "/home/jmonk/w4a4-gate/results_bonsai.json"
GROUP = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device, "torch:", torch.__version__)


def groupwise_quant_dequant(x, bits, group=GROUP, dim=-1, rotate=False):
    """Fake-quant (quantize+dequant, straight-through) along `dim` in groups of `group`, symmetric."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    orig_shape = x.shape
    n = orig_shape[dim]
    pad = (-n) % group
    if pad:
        pad_shape = list(orig_shape)
        pad_shape[dim] = pad
        x = torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=dim)
    assert dim == -1
    ng = x.shape[-1] // group
    xr = x.reshape(*x.shape[:-1], ng, group)
    if rotate:
        H = _get_hadamard(group, x.device, x.dtype)
        xr = xr @ H
    absmax = xr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = absmax / qmax
    q = torch.clamp(torch.round(xr / scale), qmin, qmax)
    deq = q * scale
    if rotate:
        deq = deq @ H.t()
    deq = deq.reshape(*x.shape[:-1], ng * group)
    if pad:
        deq = deq[..., :n]
    return deq.reshape(orig_shape)


_had_cache = {}
def _get_hadamard(n, device, dtype):
    key = (n, str(device), dtype)
    if key in _had_cache:
        return _had_cache[key]
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    H = (H / math.sqrt(n)).to(device=device, dtype=dtype)
    _had_cache[key] = H
    return H


_ternary_checked = False
def _ternary_sanity(w):
    """Warn (do not crash) if the weight tensor does not look ternary {-s,0,+s} per row.
    Bonsai weights are a lossless fp16 unpack of ternary values with a per-row/group scale,
    so a sampled row should have very few distinct magnitudes. Catches accidental double-quant."""
    global _ternary_checked
    if _ternary_checked:
        return
    _ternary_checked = True
    row = w[0].detach().float().flatten()
    uniq_abs = torch.unique(torch.round(row.abs() / (row.abs().amax().clamp_min(1e-9)) * 1e6) / 1e6)
    n_uniq = uniq_abs.numel()
    frac_zero = (row.abs() < 1e-9).float().mean().item()
    print(f"[ternary-check] row0: {n_uniq} distinct |values| (expect ~1-3 if ternary), "
          f"zero-fraction={frac_zero:.3f}")
    if n_uniq > 8:
        print(f"[ternary-check] WARNING: {n_uniq} distinct magnitudes -> weights may NOT be "
              f"cleanly ternary. The gate assumes NO weight quant; verify the model.")


def groupwise_fp8_e4m3_quant_dequant(x, group=GROUP, dim=-1):
    """T165: per-group E4M3 fake-quant using torch's NATIVE float8_e4m3fn dtype
    (authentic hardware round-trip, not an approximation) -- the activation
    side of the W(ternary)*A8-fp8 design. Absmax-scaled per group into E4M3's
    representable range (max ~448) before the real fp8 cast/back-cast."""
    qmax = 448.0
    orig_shape = x.shape
    n = orig_shape[dim]
    assert dim == -1
    pad = (-n) % group
    if pad:
        pad_shape = list(orig_shape); pad_shape[dim] = pad
        x = torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=dim)
    ng = x.shape[-1] // group
    xr = x.reshape(*x.shape[:-1], ng, group)
    absmax = xr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = absmax / qmax
    xq = (xr / scale).to(torch.float8_e4m3fn).to(torch.float32)
    deq = (xq * scale).reshape(*x.shape[:-1], ng * group)
    if pad:
        deq = deq[..., :n]
    return deq.reshape(orig_shape)


class FakeQuantActLinear(nn.Module):
    """Wraps an nn.Linear. Weights kept AS-IS (already ternary -> no weight quant).
    Only the activation is fake-quantized per the configured mode."""
    def __init__(self, lin: nn.Linear, act_mode: str):
        super().__init__()
        self.bias = lin.bias
        w = lin.weight.data
        _ternary_sanity(w)
        # NO weight quantization: Bonsai weights are already ternary (lossless fp16 unpack).
        self.register_buffer("w", w)
        self.act_mode = act_mode

    def forward(self, x):
        if self.act_mode == "none":
            xq = x
        elif self.act_mode == "int8":
            xq = groupwise_quant_dequant(x.float(), bits=8, group=GROUP, dim=-1).to(x.dtype)
        elif self.act_mode == "int4":
            xq = groupwise_quant_dequant(x.float(), bits=4, group=GROUP, dim=-1).to(x.dtype)
        elif self.act_mode == "int4+had":
            xq = groupwise_quant_dequant(x.float(), bits=4, group=GROUP, dim=-1, rotate=True).to(x.dtype)
        elif self.act_mode == "fp8_e4m3":
            # T165: W(ternary, exact in fp8 via trivial {-1,0,1} LUT, untouched
            # here since weight stays lossless) * A8-fp8(E4M3). This is the
            # accuracy-gate proxy for the FP8-datapath ternary route.
            xq = groupwise_fp8_e4m3_quant_dequant(x.float(), group=GROUP, dim=-1).to(x.dtype)
        else:
            raise ValueError(self.act_mode)
        return torch.nn.functional.linear(xq, self.w, self.bias)


def patch_model(model, act_mode):
    count = 0
    for layer in model.model.layers:
        for name in ["self_attn", "mlp"]:
            sub = getattr(layer, name)
            for pname, mod in list(sub.named_children()):
                if isinstance(mod, nn.Linear):
                    setattr(sub, pname, FakeQuantActLinear(mod, act_mode))
                    count += 1
    return count


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


def main():
    text = open(CORPUS, encoding="utf-8").read()
    results = {}

    model, tok = load_fresh()
    t0 = time.time()
    ppl, ntok = perplexity(model, tok, text)
    print(f"[bf16-reference]              ppl={ppl:.4f}  tokens={ntok}  ({time.time()-t0:.1f}s)")
    results["bf16_reference"] = ppl
    del model; torch.cuda.empty_cache()

    for label, mode in [
        ("int8_act",         "int8"),      # the SHIPPED floor (verdict baseline)
        ("int4_act",         "int4"),      # naive W4A4
        ("int4_act_hadamard","int4+had"),  # rotation rescue probe
        ("noact_bf16",       "none"),      # activations free (sanity bound)
        ("fp8e4m3_act",      "fp8_e4m3"),  # T165: W(ternary)*A8-fp8 route (RC2-free datapath)
    ]:
        model, tok = load_fresh()
        n = patch_model(model, mode)
        t0 = time.time()
        ppl, ntok = perplexity(model, tok, text)
        d_bf16 = (ppl / results["bf16_reference"] - 1) * 100
        print(f"[{label:20s}] ppl={ppl:.4f}  Δvs_bf16={d_bf16:+.2f}%  patched={n}  ({time.time()-t0:.1f}s)")
        results[label] = ppl
        del model; torch.cuda.empty_cache()

    print("\n=== SUMMARY (delta vs the int8-act FLOOR = the shipped path) ===")
    floor = results["int8_act"]
    for k in ["bf16_reference", "noact_bf16", "int8_act", "int4_act", "int4_act_hadamard", "fp8e4m3_act"]:
        v = results[k]
        print(f"  {k:20s} ppl={v:8.4f}  Δvs_int8={(v/floor-1)*100:+7.2f}%")

    d = (results["int4_act"] / floor - 1) * 100
    dh = (results["int4_act_hadamard"] / floor - 1) * 100
    d_fp8 = (results["fp8e4m3_act"] / floor - 1) * 100
    verdict = "PASS" if d <= 15 else ("MARGINAL" if d <= 30 else "FAIL")
    verdict_fp8 = "PASS" if d_fp8 <= 15 else ("MARGINAL" if d_fp8 <= 30 else "FAIL")
    print(f"\n=== VERDICT: int4-act = {d:+.2f}% vs int8 floor -> {verdict} "
          f"(hadamard rescue = {dh:+.2f}%) ===")
    print(f"=== T165 VERDICT: fp8e4m3-act (W-ternary*A8-fp8 route) = {d_fp8:+.2f}% vs int8 floor -> {verdict_fp8} ===")
    print("  PASS<=15% viable | MARGINAL 15-30% try hadamard | FAIL>30% int4-acts too lossy")

    results["_verdict"] = verdict
    results["_int4_vs_int8_pct"] = d
    results["_int4had_vs_int8_pct"] = dh
    results["_fp8e4m3_vs_int8_pct"] = d_fp8
    results["_fp8e4m3_verdict"] = verdict_fp8
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
