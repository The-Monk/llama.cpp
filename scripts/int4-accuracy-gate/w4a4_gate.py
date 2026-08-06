"""
T89 STEP 1 accuracy gate: simulate native-iu4 W4A4 (int4 weight x int4 activation)
vs the CURRENT shipped path (int4 weight, dequant->int8 activation quant, i.e. what
mmq/dp4a actually rides today) vs BF16 reference, on a small real dense model
(Qwen3-0.6B), measuring perplexity delta. Fake-quant (quantize+immediately dequant,
straight-through) -- no real 4-bit GEMM needed to gauge the *numerical* effect.

Weight quant: groupwise symmetric int4, group=32 along the reduction (in_features)
dim, matching ggml Q4_0 blocking (scale = absmax/8 per block of 32).
Activation quant variants:
  - none   : bf16, no activation quant (upper bound / "if activations were free")
  - int8   : per-token groupwise (group=32) symmetric int8 -- proxy for TODAY's
             shipped Q4_0 path (weight nibbles upconverted to int8, activation
             quantized to q8_1 for the dp4a dot)
  - int4   : per-token groupwise (group=32) symmetric int4 -- the NAIVE W4A4
             simulation for the native iu4xiu4 WMMA path
  - int4+had : int4 activations, but with a random Hadamard-ish rotation (per
             group of 32, a fixed orthogonal butterfly mix) applied before quant
             and inverted after -- cheap smoothing/rotation feasibility probe
             (QuaRot/SpinQuant style), NOT a real Hadamard transform, just a
             representative orthogonal mixing to see if it helps at all.

GPU1 only (HIP_VISIBLE_DEVICES pinned by caller). CPU fallback if GPU unhappy.
"""
import os, sys, time, math, json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/aipool/models/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
CORPUS = os.environ.get("SCRATCH", "./scratch") + "/int4-gate/corpus.txt"
GROUP = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device, "torch:", torch.__version__)

def groupwise_quant_dequant(x, bits, group=GROUP, dim=-1, rotate=False):
    """x: float tensor, quantize along `dim` in groups of `group`, symmetric, return dequantized fake-quant tensor."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    orig_shape = x.shape
    n = orig_shape[dim]
    pad = (-n) % group
    if pad:
        pad_shape = list(orig_shape)
        pad_shape[dim] = pad
        x = torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=dim)
    # reshape last dim into (ng, group) assuming dim=-1 for simplicity (that's all we use)
    assert dim == -1
    ng = x.shape[-1] // group
    xr = x.reshape(*x.shape[:-1], ng, group)
    if rotate:
        # fixed pseudo-Hadamard butterfly mix within each group of 32 (orthogonal-ish,
        # normalized so energy is preserved) -- cheap smoothing/rotation probe.
        g = group
        H = _get_hadamard(g, x.device, x.dtype)
        xr = xr @ H  # rotate
    absmax = xr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = absmax / qmax
    q = torch.clamp(torch.round(xr / scale), qmin, qmax)
    deq = q * scale
    if rotate:
        deq = deq @ H.t()  # inverse rotation (H is orthogonal up to scale, use transpose)
    deq = deq.reshape(*x.shape[:-1], ng * group)
    if pad:
        deq = deq[..., :n]
    return deq.reshape(orig_shape)

_had_cache = {}
def _get_hadamard(n, device, dtype):
    key = (n, str(device), dtype)
    if key in _had_cache:
        return _had_cache[key]
    # real Hadamard matrix via Sylvester construction (n must be power of 2; 32 is)
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    H = (H / math.sqrt(n)).to(device=device, dtype=dtype)
    _had_cache[key] = H
    return H


class FakeQuantLinear(nn.Module):
    """Wraps an existing nn.Linear: weight fake-quantized to int4 groupwise ONCE (cached),
    activation fake-quantized per forward call per the configured mode."""
    def __init__(self, lin: nn.Linear, act_mode: str):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.bias = lin.bias
        w = lin.weight.data
        # weight: int4 groupwise along in_features (dim=-1), matches Q4_0 blocking
        self.register_buffer("w_q4", groupwise_quant_dequant(w.float(), bits=4, group=GROUP, dim=-1).to(w.dtype))
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
        else:
            raise ValueError(self.act_mode)
        return torch.nn.functional.linear(xq, self.w_q4, self.bias)


def patch_model(model, act_mode):
    """Replace every nn.Linear inside decoder layers (q/k/v/o/gate/up/down proj) with FakeQuantLinear.
    Leave embed_tokens and lm_head untouched (matches common practice of keeping i/o at higher precision;
    this is the SAME for every variant tested so it doesn't bias the relative comparison)."""
    count = 0
    for layer in model.model.layers:
        for name in ["self_attn", "mlp"]:
            sub = getattr(layer, name)
            for pname, mod in list(sub.named_children()):
                if isinstance(mod, nn.Linear):
                    setattr(sub, pname, FakeQuantLinear(mod, act_mode))
                    count += 1
    return count


@torch.no_grad()
def perplexity(model, tokenizer, text, seq_len=512, stride=512, max_tokens=4096):
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[:, :max_tokens].to(device)
    n = ids.shape[1]
    nlls = []
    total_tok = 0
    for begin in range(0, n - 1, stride):
        end = min(begin + seq_len, n)
        input_ids = ids[:, begin:end]
        if input_ids.shape[1] < 2:
            continue
        target_ids = input_ids.clone()
        out = model(input_ids, labels=target_ids)
        # out.loss is mean NLL per token over (seq_len-1) predictions (HF shifts internally)
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
    model.to(device)
    model.eval()
    return model, tok


def main():
    text = open(CORPUS, encoding="utf-8").read()
    results = {}

    # 0. true BF16 reference, no patching at all
    model, tok = load_fresh()
    t0 = time.time()
    ppl, ntok = perplexity(model, tok, text)
    print(f"[bf16-reference]      ppl={ppl:.4f}  tokens={ntok}  ({time.time()-t0:.1f}s)")
    results["bf16_reference"] = ppl
    del model
    torch.cuda.empty_cache()

    configs = [
        ("q4_weight_int8_act", "int8"),      # proxy for TODAY's shipped Q4_0 path
        ("q4_weight_int4_act", "int4"),      # naive W4A4 (native iu4xiu4 sim)
        ("q4_weight_int4_act_hadamard", "int4+had"),  # rotation/smoothing feasibility
        ("q4_weight_noact_quant", "none"),   # weight-only int4, bf16 activation (sanity bound)
    ]
    for label, mode in configs:
        model, tok = load_fresh()
        n = patch_model(model, mode)
        t0 = time.time()
        ppl, ntok = perplexity(model, tok, text)
        dt = time.time() - t0
        delta = (ppl / results["bf16_reference"] - 1) * 100
        print(f"[{label:28s}] ppl={ppl:.4f}  tokens={ntok}  Δvs_bf16={delta:+.2f}%  patched={n}  ({dt:.1f}s)")
        results[label] = ppl
        del model
        torch.cuda.empty_cache()

    print("\n=== SUMMARY ===")
    base = results["bf16_reference"]
    for k, v in results.items():
        print(f"{k:32s} ppl={v:8.4f}  delta_vs_bf16={(v/base-1)*100:+7.2f}%")

    with open(os.environ.get("SCRATCH", "./scratch") + "/int4-gate/results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
