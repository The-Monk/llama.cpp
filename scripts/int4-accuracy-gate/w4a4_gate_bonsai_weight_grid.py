"""
T164 Phase 1: 2D accuracy grid for the APEX4-style per-channel WEIGHT-scale deferral
idea (coordinator directive, see /home/jmonk/iu4-loop.status T164 tags and
/home/jmonk/amd-submission/w4a4-external-intel.md RC2 attack #1).

Extends w4a4_gate_bonsai.py (which only varies the ACTIVATION precision/granularity,
weights untouched) by ALSO fake-requantizing the WEIGHT at a configurable group size,
sweeping:
    weight_group  in {128 (native, no-op sanity check), 512, CHANNEL (=whole row / per-output-feature)}
    act_group     in {32 (current shipped granularity), 64, 128, TOKEN (=whole row)}

CRITICAL PRE-CHECK (done first, see native_weight_granularity_check() below): Bonsai's
HF safetensors weights are a lossless fp16 unpack of an already-ternary model. Before
assuming per-channel weight deferral is "free," we must know the model's NATIVE
weight-scale granularity -- if it's already per-channel or per-tensor, deferral costs
nothing; if it's a finer group (e.g. 128, matching Q2_0's QK2_0), coarsening to
per-channel is a REAL re-quantization with a real, measurable accuracy cost.

Weight fake-requant: given the real-valued (already-ternary, value = code*native_scale)
fp16 weight, for the TARGET group size, compute a new scale (absmax over that group)
and round each element to the nearest of {-1,0,+1} times that new scale. This models
"what if this weight used the coarser group's scale instead of its native one" --
exactly the format change under test. group=native(128)/finer(32) should be a
near-exact no-op (every finer-or-equal group is a subset of one native-clean group,
so its absmax == the native scale already); group=CHANNEL is the real test.

VERDICT is reported as delta vs the SHIPPED FLOOR (weight untouched/native, act=int8
per-32 -- i.e. w4a4_gate_bonsai.py's own "int8_act" result), same PASS/MARGINAL/FAIL
bands (<=15/<=30/>30%) as the original gate.

Env: mtp713 (transformers 5.14 / torch 2.13 / safetensors). Corpus/model paths shared
with w4a4_gate_bonsai.py.
"""
import os, sys, time, math, json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

MODEL_DIR = "/aipool/models/ternary-bonsai-phase0/unpacked"
CORPUS    = "/home/jmonk/w4a4-gate/corpus.txt"
OUT_JSON  = "/home/jmonk/w4a4-gate/results_bonsai_weight_grid.json"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Native weight-scale granularity pre-check (answers the coordinator's
# "CRITICAL sub-question" before any perplexity run).
# ---------------------------------------------------------------------------
def native_weight_granularity_check():
    idx = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))
    wm = idx["weight_map"]

    def load_tensor(name):
        fname = wm[name]
        path = os.path.join(MODEL_DIR, fname)
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    names = ["model.layers.0.mlp.gate_proj.weight", "model.layers.0.self_attn.q_proj.weight",
             "model.layers.5.mlp.down_proj.weight"]
    print("=== T164 Phase 1 pre-check: Bonsai NATIVE weight-scale granularity ===")
    report = {}
    for name in names:
        if name not in wm:
            continue
        w = load_tensor(name).float()
        row = w[0]
        K = row.shape[0]
        line = {}
        for gs in [32, 64, 128, 256, K]:
            if K % gs != 0:
                continue
            ng = K // gs
            clean = 0
            for g in range(ng):
                seg = row[g*gs:(g+1)*gs]
                nz = seg[seg.abs() > 1e-9].abs()
                if nz.numel() == 0 or torch.unique(torch.round(nz*1e7)/1e7).numel() == 1:
                    clean += 1
            line[gs] = (clean, ng)
        report[name] = line
        print(f"  {name} K={K}: " + " ".join(f"g{gs}={c}/{n}({100*c/n:.0f}%)" for gs, (c, n) in line.items()))
    return report


# ---------------------------------------------------------------------------
# Fake-quant helpers
# ---------------------------------------------------------------------------
def groupwise_act_quant_dequant(x, bits, group, dim=-1):
    """Same as w4a4_gate_bonsai.py's groupwise_quant_dequant (activation side,
    symmetric int-N quant). `group` may equal the full last-dim size (per-token)."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    orig_shape = x.shape
    n = orig_shape[dim]
    assert dim == -1
    pad = (-n) % group
    if pad:
        pad_shape = list(orig_shape)
        pad_shape[dim] = pad
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


def ternary_weight_requant(w, group):
    """Re-quantize an already-ternary (value = code*native_scale) weight row-set to a
    NEW group size: absmax per group, round each element to nearest of {-1,0,+1}*scale.
    `group` may equal the full last dim (per-channel/per-output-feature)."""
    orig_shape = w.shape
    n = orig_shape[-1]
    pad = (-n) % group
    if pad:
        pad_shape = list(orig_shape)
        pad_shape[-1] = pad
        w = torch.cat([w, torch.zeros(pad_shape, dtype=w.dtype, device=w.device)], dim=-1)
    ng = w.shape[-1] // group
    wr = w.reshape(*w.shape[:-1], ng, group)
    absmax = wr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    code = torch.round(wr / absmax).clamp(-1, 1)
    deq = (code * absmax).reshape(*w.shape[:-1], ng * group)
    if pad:
        deq = deq[..., :n]
    return deq.reshape(orig_shape)


class FakeQuantWALinear(nn.Module):
    """Wraps an nn.Linear. BOTH weight (ternary re-quant at `weight_group`) and
    activation (int4 groupwise at `act_group`) are fake-quantized. weight_group/
    act_group may be the string "channel"/"token" meaning "the full last dim"."""
    def __init__(self, lin: nn.Linear, weight_group, act_group):
        super().__init__()
        self.bias = lin.bias
        w = lin.weight.data.float()
        K = w.shape[-1]
        wg = K if weight_group == "channel" else weight_group
        w_req = ternary_weight_requant(w, wg) if wg != None else w
        self.register_buffer("w", w_req.to(lin.weight.dtype))
        self.act_group = act_group
        self.K = K

    def forward(self, x):
        ag = self.K if self.act_group == "token" else self.act_group
        xq = groupwise_act_quant_dequant(x.float(), bits=4, group=ag, dim=-1).to(x.dtype)
        return torch.nn.functional.linear(xq, self.w, self.bias)


def patch_model(model, weight_group, act_group):
    count = 0
    for layer in model.model.layers:
        for name in ["self_attn", "mlp"]:
            sub = getattr(layer, name)
            for pname, mod in list(sub.named_children()):
                if isinstance(mod, nn.Linear):
                    setattr(sub, pname, FakeQuantWALinear(mod, weight_group, act_group))
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
    native_report = native_weight_granularity_check()

    text = open(CORPUS, encoding="utf-8").read()
    results = {"_native_weight_granularity": {k: {str(g): v for g, v in vv.items()} for k, vv in native_report.items()}}

    # Shipped floor: weight untouched (native), activation int8 per-32 (matches
    # w4a4_gate_bonsai.py's int8_act result -- recomputed here for a self-contained script).
    model, tok = load_fresh()
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
    t0 = time.time()
    floor_ppl, ntok = perplexity(model, tok, text)
    print(f"[floor: weight=native,act=int8-g32] ppl={floor_ppl:.4f} tokens={ntok} ({time.time()-t0:.1f}s)  patched={count}")
    results["floor_weight_native_act_int8_g32"] = floor_ppl
    del model; torch.cuda.empty_cache()

    weight_grid = [128, 512, "channel"]
    act_grid    = [32, 64, 128, "token"]

    grid_results = {}
    for wg in weight_grid:
        for ag in act_grid:
            model, tok = load_fresh()
            n = patch_model(model, weight_group=wg, act_group=ag)
            t0 = time.time()
            ppl, ntok = perplexity(model, tok, text)
            delta = (ppl / floor_ppl - 1) * 100
            key = f"w{wg}_a{ag}"
            grid_results[key] = {"ppl": ppl, "delta_vs_floor_pct": delta}
            verdict = "PASS" if delta <= 15 else ("MARGINAL" if delta <= 30 else "FAIL")
            print(f"[{key:16s}] ppl={ppl:.4f}  Δvs_floor={delta:+7.2f}%  -> {verdict}  ({time.time()-t0:.1f}s)")
            del model; torch.cuda.empty_cache()

    results["grid"] = grid_results
    results["floor_ppl"] = floor_ppl

    print("\n=== T164 Phase 1 GRID SUMMARY (delta vs shipped floor: weight=native-128, act=int8-per32) ===")
    print(f"{'':16s} " + " ".join(f"a={ag:>6}" for ag in act_grid))
    for wg in weight_grid:
        row = []
        for ag in act_grid:
            key = f"w{wg}_a{ag}"
            d = grid_results[key]["delta_vs_floor_pct"]
            row.append(f"{d:+7.2f}%")
        print(f"w={wg!s:>10}  " + " ".join(row))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
