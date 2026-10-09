#!/bin/bash
# T404 follow-ups: (1) q5_0 m=16 n=1 k=32 flake check, default vs N1 env, 5x each;
# (2) pp512 -r 10 json medians, default/g128/token, 2 interleaved rounds (Q2_0).
set -u
B=/home/jmonk/n1-engine/build/bin
M=/aipool/models/bonsai-27b/Ternary-Bonsai-27B-Q2_0.gguf
for f in $B/test-backend-ops $B/llama-bench; do [ -x $f ] || { echo "FAIL: no $f"; exit 2; }; done
[ -s $M ] || { echo "FAIL: no model"; exit 2; }
echo "# $(date -u +%FT%TZ) HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
P='type_a=q5_0,type_b=f32,m=16,n=1,k=32,bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1'
for arm in default n1; do
  for i in 1 2 3 4 5 6 7 8; do
    if [ $arm = n1 ]; then E=(GGML_N1_PREFILL=1); else E=(N1_ARM=default); fi
    out=$(env "${E[@]}" $B/test-backend-ops -o MUL_MAT -b ROCm0 -p "$P" 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
    line=$(echo "$out" | grep -E "tests passed" | tail -1)
    err=$(echo "$out" | grep -o "ERR = [0-9.e-]*" | head -1)
    echo "FLAKE $arm run=$i :: ${line:-NO PASSED LINE} ${err}"
  done
done
for round in 1 2; do
  for arm in default g128 token; do
    case $arm in
      default) E=(N1_ARM=default);;
      g128)    E=(GGML_N1_PREFILL=1 GGML_N1_ACT=g128);;
      token)   E=(GGML_N1_PREFILL=1 GGML_N1_ACT=token);;
    esac
    env "${E[@]}" $B/llama-bench -m $M -ngl 99 -n 0 -p 512 -r 10 -o json > pp512-$arm-$round.json 2> pp512-$arm-$round.err; rc=$?
    [ $rc -eq 0 ] || { echo "FAIL: $arm $round rc=$rc"; tail -5 pp512-$arm-$round.err; continue; }
    python3 -I - pp512-$arm-$round.json $arm $round <<'EOF'
import json, sys, statistics
d = json.load(open(sys.argv[1]))
for r in d:
    s = r.get('samples_ts') or []
    if not s: print('FAIL: no samples_ts'); continue
    print(f"PP512 round={sys.argv[3]} arm={sys.argv[2]} median {statistics.median(s):.1f} t/s  min {min(s):.1f} max {max(s):.1f}  samples {' '.join(f'{x:.0f}' for x in s)}")
EOF
  done
done
echo "# done $(date -u +%FT%TZ)"
