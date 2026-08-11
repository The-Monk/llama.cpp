"""Diagnostic (not a new gate variant): why did int4_calib_p999 blow up to
+459%? Hypothesis: a position-dependent activation spike (attention-sink /
first-token phenomenon) occurs too rarely in the 1024-token calibration slice
to survive a 99.9th-percentile clip, so EVERY eval chunk's early tokens get
severely clipped by the resulting too-small static scale -- a real failure
mode of static calibration, not a script bug. Verify by comparing, for one
representative Linear (layer0 mlp.down_proj, the widest K=12288), the
calibrated static per-group scale against the TRUE per-token dynamic amax
seen during the ppl eval window, and by checking whether the largest gaps
concentrate at low token positions.
"""
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/aipool/models/ternary-bonsai-phase0/unpacked"
CORPUS = "/home/jmonk/w4a4-gate/corpus.txt"
device = torch.device("cuda:0")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
model.to(device); model.eval()

text = open(CORPUS, encoding="utf-8").read()
enc = tok(text, return_tensors="pt")
ids_calib = enc.input_ids[:, :1024].to(device)
ids_eval = enc.input_ids[:, :512].to(device)  # first eval chunk (seq_len=512, stride=512)

target = model.model.layers[0].mlp.down_proj
group = 32

captured = {}
def hook(module, inp):
    captured["x"] = inp[0].detach().float()
h = target.register_forward_pre_hook(hook)

with torch.no_grad():
    model(ids_calib)
x_calib = captured["x"].reshape(-1, captured["x"].shape[-1])  # [1024, 12288]

with torch.no_grad():
    model(ids_eval)
x_eval = captured["x"].reshape(-1, captured["x"].shape[-1])  # [512, 12288]
h.remove()

K = x_calib.shape[-1]
ng = K // group

# calibrated static scale exactly as int4_calib_p999 computed it
xr_c = x_calib.reshape(x_calib.shape[0], ng, group).abs()
clip = torch.quantile(xr_c.float(), 0.999, dim=0)          # [ng, group]
clip = clip.amax(dim=-1)                                    # [ng]
qmax = 7
static_scale = (clip.clamp_min(1e-8) / qmax)                 # [ng]

# true per-token dynamic amax/scale during the eval window
xr_e = x_eval.reshape(x_eval.shape[0], ng, group).abs()
dyn_amax = xr_e.amax(dim=-1)                                 # [T_eval, ng]
dyn_scale = dyn_amax.clamp_min(1e-8) / qmax                  # [T_eval, ng]

# how many (token,group) cells does the STATIC scale under-shoot vs the
# token's own true dynamic need? ratio>1 means static scale is too SMALL
# (would clip/saturate this token's group).
ratio = dyn_scale / static_scale.unsqueeze(0)                # [T_eval, ng]
bad_frac_per_token = (ratio > 1.0).float().mean(dim=-1)      # [T_eval]
worst_ratio_per_token = ratio.amax(dim=-1)                    # [T_eval]

print("K =", K, "ng =", ng, "group =", group)
print("static_scale stats: min", static_scale.min().item(), "median", static_scale.median().item(),
      "max", static_scale.max().item())
print("\nper-token (first 16 positions) fraction-of-groups-that-would-saturate, worst-ratio:")
for t in range(16):
    print(f"  tok {t:3d}: bad_frac={bad_frac_per_token[t].item():.4f}  worst_ratio={worst_ratio_per_token[t].item():8.2f}")
print("\noverall (all 512 eval tokens): mean bad_frac=%.4f  max worst_ratio=%.2f  "
      "count tokens with worst_ratio>10x=%d/%d"
      % (bad_frac_per_token.mean().item(), worst_ratio_per_token.max().item(),
         (worst_ratio_per_token > 10).sum().item(), worst_ratio_per_token.numel()))

# which token positions have the single worst saturation?
top = torch.topk(worst_ratio_per_token, 5)
print("\ntop-5 worst tokens by position index:", top.indices.tolist(), "ratios:", [f"{v:.1f}" for v in top.values.tolist()])
