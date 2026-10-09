#!/usr/bin/env python3
"""T404 trace analysis. analyze_trace.py <kernel_trace.csv> <n_runs>
Buckets kernel time per pp run (total / n_runs) and reports in-engine N1 TOPS per (F,K) shape (median dispatch)."""
import csv, re, statistics, sys
from collections import defaultdict

f, nruns = sys.argv[1], int(sys.argv[2])
# standalone job 236, N=512, mean of 2 rounds (results.md section 3), keyed (F, K)
SA = {
    'w4x2_rs1_w2': {(17408, 5120): 181.0, (5120, 17408): 182.5, (10240, 5120): 179.3, (6144, 5120): 176.9,
                    (5120, 6144): 176.9, (12288, 5120): 178.8, (5120, 5120): 174.9},
    'w4x2_rs0':    {(17408, 5120): 209.1, (5120, 17408): 216.2, (10240, 5120): 211.0, (6144, 5120): 208.8,
                    (5120, 6144): 208.4, (12288, 5120): 210.4, (5120, 5120): 206.4},
}
CATS = [
    ('N1 GEMM', r'gemm_iu8'),
    ('N1 act quantize', r'k_n1_quant_act'),
    ('N1 weight convert (one-time)', r'k_n1_pack|k_n1_row_fold'),
    ('q8_1/mmq act quantize', r'quantize'),
    ('MMQ GEMM (incl. stream-k fixup)', r'mul_mat_q|mmq'),
    ('MMVQ/GEMV', r'mul_mat_vec|mmvq'),
    ('other matmul (f16/cublas/hipblaslt)', r'gemm|Cijk|matmul|mul_mat'),
    ('attention (FA/softmax)', r'flash_attn|soft_max|fattn'),
    ('GDN / SSM / conv', r'gated_delta|delta_net|ssm|conv|gdn'),
    ('norms', r'norm'),
    ('rope', r'rope'),
    ('copy / get_rows / set_rows', r'cpy|copy|get_rows|set_rows|concat|cont'),
    ('elementwise / glu / other', r'.'),
]
tot = defaultdict(int); cnt = defaultdict(int)
n1 = defaultdict(list)
first_ts, last_ts = None, None
with open(f) as fh:
    for r in csv.DictReader(fh):
        if r.get('Kind', 'KERNEL_DISPATCH') != 'KERNEL_DISPATCH':
            continue
        name = r['Kernel_Name']; s, e = int(r['Start_Timestamp']), int(r['End_Timestamp'])
        d = e - s
        cat = next(c for c, rx in CATS if re.search(rx, name))
        tot[cat] += d; cnt[cat] += 1
        if cat == 'N1 GEMM':
            m = (re.search(r'gemm_iu8<128, 128, 4, 2, (\d+), 1, (\d+), (\d+)>', name) or
                 re.search(r'gemm_iu8ILi128ELi128ELi4ELi2ELi(\d)ELi1ELi(\d+)ELi(\d+)E', name))
            rs, fl, kt = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (-1, -1, 0)
            F = int(r['Grid_Size_Y']) * 128
            Npad = int(r['Grid_Size_X']) // 256 * 128
            n1[(rs, fl, kt, F, Npad)].append(d)
gpu = sum(tot.values()) - tot['N1 weight convert (one-time)']
print(f'per pp run (total/{nruns}), weight conversion excluded from total:')
for c, _ in CATS:
    if cnt[c]:
        extra = '' if c != 'N1 weight convert (one-time)' else '  [excluded]'
        print(f'  {c:38s} {tot[c]/nruns/1e6:9.3f} ms  {100*tot[c]/gpu if not extra else 0:5.1f}%  ({cnt[c]/nruns:.0f} dispatches){extra}')
print(f'  {"GPU busy total":38s} {gpu/nruns/1e6:9.3f} ms  -> kernel-time-bound {512/(gpu/nruns/1e9):.0f} t/s at pp512')
if n1:
    print('in-engine N1 GEMM per shape (median dispatch; TOPS = 2*F*Npad*K/t):')
    for (rs, fl, kt, F, Npad), ds in sorted(n1.items(), key=lambda kv: -sum(kv[1])):
        med = statistics.median(ds)
        K = kt if kt else 0
        cfg = 'w4x2_rs0' if rs == 0 else 'w4x2_rs1_w2'
        tops = 2 * F * Npad * K / med / 1e3 if K else float('nan')
        sa = SA[cfg].get((F, K)) if fl != 9 else SA['w4x2_rs1_w2'].get((F, K))
        cap = f'{100*tops/sa:5.1f}% of standalone {sa}' if sa and K else 'no standalone number'
        print(f'  rs{rs} FL{fl} F={F:6d} K={K:6d} Npad={Npad:5d} n={len(ds):4d} med {med/1e3:8.1f} us  {tops:6.1f} TOPS  {cap}  share {100*sum(ds)/tot["N1 GEMM"]:4.1f}%')
    fl_all = sum(2.0 * F * Npad * kt * len(ds) for (rs, fl, kt, F, Npad), ds in n1.items())
    t_all = sum(sum(ds) for ds in n1.values())
    print(f'  all N1 dispatches: {fl_all/nruns/1e12:.3f} TFLOP per run in {t_all/nruns/1e6:.3f} ms -> {fl_all/t_all/1e3:.1f} TOPS (mean, incl. outliers)')
    # time-weighted capture over shapes with a standalone number: sum(standalone time) / sum(in-engine time)
    t_sa = t_en = 0.0; t_med_all = 0.0
    for (rs, fl, kt, F, Npad), ds in n1.items():
        sa = SA['w4x2_rs0' if rs == 0 else 'w4x2_rs1_w2'].get((F, kt))
        if not sa: continue
        flops = 2.0 * F * Npad * kt * len(ds)
        t_sa += flops / (sa * 1e3)            # ns
        t_en += statistics.median(ds) * len(ds)
    if t_en:
        print(f'  capture (standalone time / in-engine median time, shapes with a standalone number): {100*t_sa/t_en:.1f}%  '
              f'covers {100*t_en/sum(statistics.median(ds)*len(ds) for ds in n1.values()):.1f}% of N1 GEMM time')
