"""
T164-1b: find a per-channel-weight-scale-compatible scheme (still spill-free --
one epilogue multiply, no accumulator array) at a FRACTION of the +12.66% ppl
cost that pure per-channel deferral (w4a4_gate_bonsai_weight_grid.py) measured.
Coordinator directive, priority order:

  1) Per-channel weight + Hadamard rotation (QuaRot-style): rotate both the
     weight's K-dim (in BLOCK-diagonal chunks, since Bonsai's K=4096/12288
     aren't themselves powers of 2) and the activation's K-dim by the SAME
     orthogonal (symmetric Sylvester-Hadamard) matrix before quantizing.
     Preserves y=xW^T exactly pre-quantization (block-orthogonal rotation is
     norm-preserving and exactly invertible); the THESIS is that rotation
     redistributes the INTER-BLOCK magnitude variance (measured in
     w4a4_gate_bonsai_weight_grid.py's native-granularity check: adjacent
     native-128 blocks have visibly different scales, e.g. 0.0284 vs 0.0248)
     so that ONE per-channel (whole-row) scale fits the rotated values far
     better than it fits the raw ones.

  2) Outlier-channel (row) protection: identify the ROWS with the largest
     per-channel-quantization error, keep THEM at native (per-128) scale,
     defer the REST to per-channel. Sweep the protected fraction K.

Both schemes remain SPILL-FREE in the kernel sense (the weight scale used at
write-out is either the row's native-128 scale (if protected -- same as
today's production path) or the row's single per-channel scale (if
deferred) -- no accumulator array needed either way, just a per-row branch
at read/write time, which is the "cheap epilogue mask" the coordinator noted).

Env: mtp713. Model/corpus paths shared with the other w4a4_gate_bonsai*.py scripts.
"""
import os, sys, time, math, json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/aipool/models/ternary-bonsai-phase0/unpacked"
CORPUS    = "/home/jmonk/w4a4-gate/corpus.txt"
OUT_JSON  = "/home/jmonk/w4a4-gate/results_bonsai_phase1b.json"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Hadamard helpers
# ---------------------------------------------------------------------------
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


def hadamard_rotate_blocks(x, block_size):
    """Right-multiply each `block_size`-wide chunk of the last dim by a
    symmetric, orthogonal Hadamard matrix H (H@H==I). Applied identically to
    weight (K-dim) and activation (K-dim), this is an EXACT (pre-quant)
    identity: (x@H) @ (W@H)^T == x @ H @ H^T @ W^T == x @ W^T."""
    n = x.shape[-1]
    assert n % block_size == 0, f"{n} not divisible by block_size={block_size}"
    H = get_hadamard(block_size, x.device, x.dtype)
    ng = n // block_size
    xr = x.reshape(*x.shape[:-1], ng, block_size)
    xr = xr @ H
    return xr.reshape(x.shape)


def ternary_weight_requant(w, group):
    orig_shape = w.shape
    n = orig_shape[-1]
    pad = (-n) % group
    if pad:
        pad_shape = list(orig_shape); pad_shape[-1] = pad
        w = torch.cat([w, torch.zeros(pad_shape, dtype=w.dtype, device=w.device)], dim=-1)
    ng = w.shape[-1] // group
    wr = w.reshape(*w.shape[:-1], ng, group)
    absmax = wr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    code = torch.round(wr / absmax).clamp(-1, 1)
    deq = (code * absmax).reshape(*w.shape[:-1], ng * group)
    if pad:
        deq = deq[..., :n]
    return deq.reshape(orig_shape)


def groupwise_act_quant_dequant(x, bits, group, dim=-1):
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
    absmax = xr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = absmax / qmax
    q = torch.clamp(torch.round(xr / scale), qmin, qmax)
    deq = (q * scale).reshape(*x.shape[:-1], ng * group)
    if pad:
        deq = deq[..., :n]
    return deq.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Scheme 1: per-channel weight + Hadamard rotation
# ---------------------------------------------------------------------------
class HadamardPerChannelLinear(nn.Module):
    def __init__(self, lin: nn.Linear, had_block: int, act_group):
        super().__init__()
        self.bias = lin.bias
        w = lin.weight.data.float()
        K = w.shape[-1]
        self.had_block = had_block if K % had_block == 0 else None
        if self.had_block is not None:
            w_rot = hadamard_rotate_blocks(w, self.had_block)
        else:
            w_rot = w
        w_req = ternary_weight_requant(w_rot, K)  # per-channel (whole row) on the ROTATED weight
        self.register_buffer("w", w_req.to(lin.weight.dtype))
        self.act_group = act_group
        self.K = K

    def forward(self, x):
        if self.had_block is not None:
            xr = hadamard_rotate_blocks(x.float(), self.had_block)
        else:
            xr = x.float()
        ag = self.K if self.act_group == "token" else self.act_group
        xq = groupwise_act_quant_dequant(xr, bits=4, group=ag, dim=-1).to(x.dtype)
        return torch.nn.functional.linear(xq, self.w, self.bias)


# ---------------------------------------------------------------------------
# Scheme 2: outlier-channel (row) protection
# ---------------------------------------------------------------------------
class OutlierProtectedLinear(nn.Module):
    def __init__(self, lin: nn.Linear, protect_frac: float, act_group):
        super().__init__()
        self.bias = lin.bias
        w = lin.weight.data.float()
        K = w.shape[-1]
        w_perchan = ternary_weight_requant(w, K)
        # per-row error: mean squared diff between native (real, already
        # ternary) and the per-channel-requantized version.
        err = (w - w_perchan).pow(2).mean(dim=-1)
        n_rows = w.shape[0]
        n_protect = int(round(protect_frac * n_rows))
        if n_protect > 0:
            protect_idx = torch.topk(err, n_protect).indices
            mask = torch.zeros(n_rows, dtype=torch.bool, device=w.device)
            mask[protect_idx] = True
            w_mixed = torch.where(mask.unsqueeze(-1), w, w_perchan)
        else:
            w_mixed = w_perchan
        self.register_buffer("w", w_mixed.to(lin.weight.dtype))
        self.act_group = act_group
        self.K = K

    def forward(self, x):
        ag = self.K if self.act_group == "token" else self.act_group
        xq = groupwise_act_quant_dequant(x.float(), bits=4, group=ag, dim=-1).to(x.dtype)
        return torch.nn.functional.linear(xq, self.w, self.bias)


def patch_model(model, cls, **kwargs):
    count = 0
    for layer in model.model.layers:
        for name in ["self_attn", "mlp"]:
            sub = getattr(layer, name)
            for pname, mod in list(sub.named_children()):
                if isinstance(mod, nn.Linear):
                    setattr(sub, pname, cls(mod, **kwargs))
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


def run_one(label, cls, **kwargs):
    model, tok = load_fresh()
    text = open(CORPUS, encoding="utf-8").read()
    n = patch_model(model, cls, **kwargs)
    t0 = time.time()
    ppl, ntok = perplexity(model, tok, text)
    print(f"[{label:28s}] ppl={ppl:.4f} patched={n} ({time.time()-t0:.1f}s)")
    del model; torch.cuda.empty_cache()
    return ppl


def main():
    results = {}

    # Floor (weight native-128, act int8-per32) and the plain per-channel
    # (w=channel, a=32, +12.66%) reference, both recomputed here so this
    # script is self-contained and directly comparable.
    model, tok = load_fresh()
    text = open(CORPUS, encoding="utf-8").read()
    count = 0
    for layer in model.model.layers:
        for name in ["self_attn", "mlp"]:
            sub = getattr(layer, name)
            for pname, mod in list(sub.named_children()):
                if isinstance(mod, nn.Linear):
                    class _FloorLinear(nn.Module):
                        def __init__(self, lin):
                            super().__init__()
                            self.bias = lin.bias
                            self.register_buffer("w", lin.weight.data)
                        def forward(self, x):
                            xq = groupwise_act_quant_dequant(x.float(), bits=8, group=32, dim=-1).to(x.dtype)
                            return torch.nn.functional.linear(xq, self.w, self.bias)
                    setattr(sub, pname, _FloorLinear(mod))
                    count += 1
    floor_ppl, ntok = perplexity(model, tok, text)
    print(f"[floor: weight=native,act=int8-g32] ppl={floor_ppl:.4f} tokens={ntok} patched={count}")
    results["floor_ppl"] = floor_ppl
    del model; torch.cuda.empty_cache()

    def delta(ppl):
        return (ppl / floor_ppl - 1) * 100

    # Plain per-channel reference (no rotation, no protection) -- reuses
    # OutlierProtectedLinear with protect_frac=0, which is exactly plain
    # per-channel requant.
    ppl_plain = run_one("plain_perchannel_a32", OutlierProtectedLinear, protect_frac=0.0, act_group=32)
    results["plain_perchannel_a32"] = {"ppl": ppl_plain, "delta_pct": delta(ppl_plain)}
    print(f"  -> delta vs floor = {delta(ppl_plain):+.2f}%  (sanity check vs weight_grid.py's +12.66%)")

    # --- Scheme 1: Hadamard rotation ---
    # CONCLUSIVE NEGATIVE (already run + root-caused, see
    # /home/jmonk/iu4-loop.status T164-1b(1)): rotated+per-channel-requant
    # RMSE on real layer0 weights = 0.0219 vs plain per-channel's 0.0038 (5.7x
    # WORSE); single-layer forward relative error 89% vs 16%; full-model ppl
    # blew up to +50,457,319% (garbage). Root cause: Bonsai's weights are
    # ternary-TRAINED (exactly 3-level per native-128 block via QAT) --
    # rotating them destroys that exploitable structure, turning an
    # exactly-representable signal into a dense near-Gaussian one that a
    # 3-level ternary code represents far worse. QuaRot-style rotation fights
    # outliers in generic higher-bit-width quantization; it does not help an
    # already-coarse ternary code. NOT re-run here (mechanism-level failure,
    # not a block-size tuning question).

    # --- Scheme 2: outlier-channel protection sweep ---
    for frac in [0.001, 0.01, 0.05]:
        ppl_out = run_one(f"outlier_protect_{frac*100:g}pct_a32", OutlierProtectedLinear,
                           protect_frac=frac, act_group=32)
        results[f"outlier_protect_{frac*100:g}pct_a32"] = {"ppl": ppl_out, "delta_pct": delta(ppl_out)}
        print(f"  -> delta vs floor = {delta(ppl_out):+.2f}%")

    print("\n=== T164-1b SUMMARY (delta vs floor=weight-native-128+act-int8-g32; PASS<=15%) ===")
    for k, v in results.items():
        if isinstance(v, dict):
            d = v["delta_pct"]
            verdict = "PASS" if d <= 15 else ("MARGINAL" if d <= 30 else "FAIL")
            print(f"  {k:32s} ppl={v['ppl']:8.4f}  delta={d:+7.2f}%  {verdict}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
