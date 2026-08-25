#!/usr/bin/env python3
# kernel_roofline_v2 -- corrected decode roofline calculator (2026-08-19 audit).
#
# Replaces the retired kernel-RL formula. Corrections baked in, each one
# measured with per-dispatch GL2C_EA_RDREQ/WRREQ counters (bytes = reqs*256,
# calibrated 2026-08-18) on gfx1201 R9700:
#
#  1. NUMERATOR = *executed* GEMV weight bytes only:
#     - exclude token_embd (row lookup), 1-d tensors, and any nextn/MTP block
#       (qwen35 'nextn.*' + blk.[n_layer].* are present in fp8/MTP GGUFs but
#       never execute in plain decode; the old formula counted them: +1.7%).
#  2. TRAFFIC FACTOR per format (EA-true bytes / GGUF weight bytes), measured:
#       q2_0: 1.000-1.002   q1_0: 1.001 except ffn_down 1.101 (open finding)
#       f8e4m3: 1.03-1.05 (34B-block row-boundary line waste)  q6_k head: 1.000
#  3. NON-GEMV MANDATORY TRAFFIC (this arch, bs=1): ~0.5 GB/token
#     (GDN state ~6.3MB/layer/token read (state read twice: gather + kernel),
#      ssm_conv, flash-attn KV, embd row, quantize reads). Old formula: 0.
#  4. DENOMINATOR = measured same-clock-domain ceiling:
#       auto/production clocks: 640.3 GB/s (calib_read 4GiB warm, re-verified
#       2026-08-19); pinned 'high': 629.8; profile_standard: 170.3.
#     Durations MUST come from a trace-only pass at the same DPM state.
#  5. Trace timestamps under HIP graphs: DURATIONS are valid, START times are
#     NOT (piled/overlapped) -- never compute busy-union or per-kernel overlap
#     from a graphs-on trace. Serialize (counter pass) or graphs-off for that.
#  6. >100%% readings are formula errors by definition; EA-true bytes over
#     duration never exceeded 101.6%% of the calib ceiling (fp8, within
#     counter-noise + row-waste of the saturated interface).
import sys, json, collections
sys.path.insert(0,'/home/jmonk/roc10/gguf-py')
from gguf.gguf_reader import GGUFReader

ROOF = {'auto': 640.3e9, 'high': 629.8e9, 'profile_standard': 170.3e9}
TRAFFIC = {  # EA-true/GGUF-bytes, measured 2026-08-19
    41: 1.005,   # q1_0 blend (ffn_down 1.101 handled per-tensor below)
    42: 1.001,   # q2_0
    43: 1.045,   # f8e4m3
    14: 1.000,   # q6_k
}
NONGEMM_BYTES = 0.5e9  # qwen3.5-hybrid 27B, bs=1, measured 0.485-0.547 GB/token

def main(path, tps=None, clock='auto'):
    r = GGUFReader(path)
    n_layer = None
    for f in r.fields.values():
        if f.name.endswith('.block_count'): n_layer = int(f.parts[-1][0])
    nextn = 0
    for f in r.fields.values():
        if 'nextn_predict_layers' in f.name: nextn = int(f.parts[-1][0])
    exec_last = (n_layer or 10**9) - nextn
    tot = 0
    for t in r.tensors:
        if len(t.shape) < 2: continue
        if t.name == 'token_embd.weight': continue
        if t.name.startswith('nextn.'): continue
        if t.name.startswith('blk.'):
            if int(t.name.split('.')[1]) >= exec_last: continue
        f = TRAFFIC.get(int(t.tensor_type), 1.0)
        if 'ffn_down' in t.name and int(t.tensor_type) == 41: f = 1.101
        tot += int(t.n_bytes) * f
    tot += NONGEMM_BYTES
    print('executed decode traffic: %.3f GB/token (incl. %.2f GB non-GEMV)' % (tot/1e9, NONGEMM_BYTES/1e9))
    roof = ROOF[clock]
    print('wall-roofline token rate at %s ceiling (%.1f GB/s): %.2f t/s' % (clock, roof/1e9, roof/tot))
    if tps:
        print('measured %.2f t/s -> wall efficiency %.1f%%' % (float(tps), 100*float(tps)*tot/roof))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else None)
