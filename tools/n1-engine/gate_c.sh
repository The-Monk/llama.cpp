#!/bin/bash
# T404 gate (c)+(e): llama-bench pp512/pp2048 (-ngl 99 -n 0 -r 3), 4 interleaved rounds, default vs N1 arms,
# with a peak-VRAM sampler (sysfs mem_info_vram_used of PCI 0000:04:00.0 = GPU0). gate_c.sh <model.gguf> <tag> [arms...]
set -u
M=$1; TAG=$2; shift 2
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(default g128 token rowffn)
B=/home/jmonk/n1-engine/build/bin
V=/sys/bus/pci/devices/0000:04:00.0/mem_info_vram_used
[ -x $B/llama-bench ] || { echo "FAIL: no llama-bench"; exit 2; }
[ -s "$M" ] || { echo "FAIL: no $M"; exit 2; }
[ -r $V ] || { echo "FAIL: no $V"; exit 2; }
echo "# $(date -u +%FT%TZ) $TAG HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset} idle_vram_MiB=$(( $(cat $V) / 1048576 ))"
fail=0
for round in 1 2 3 4; do
  for arm in "${ARMS[@]}"; do
    case $arm in
      default) E=(N1_ARM=default);;
      g128)    E=(GGML_N1_PREFILL=1 GGML_N1_ACT=g128);;
      token)   E=(GGML_N1_PREFILL=1 GGML_N1_ACT=token);;
      rowffn)  E=(GGML_N1_PREFILL=1 GGML_N1_ACT=g128 GGML_N1_ROW=ffn);;
    esac
    pk=$PWD/vram-$TAG-$arm-$round.txt; : > $pk
    ( while :; do cat $V >> $pk; sleep 0.25; done ) & SP=$!
    env "${E[@]}" $B/llama-bench -m "$M" -ngl 99 -n 0 -p 512,2048 -r 3 -o csv > bench-$TAG-$arm-$round.csv 2> bench-$TAG-$arm-$round.err; rc=$?
    kill $SP 2>/dev/null; wait $SP 2>/dev/null
    peak=$(sort -n $pk | tail -1)
    [ $rc -eq 0 ] || { echo "FAIL: $arm round $round rc=$rc"; tail -10 bench-$TAG-$arm-$round.err; fail=1; continue; }
    # columns: find n_prompt, avg_ts, stddev_ts by header name
    python3 -I - bench-$TAG-$arm-$round.csv "$TAG" "$arm" "$round" "${peak:-0}" <<'EOF'
import csv, sys
f, tag, arm, rnd, peak = sys.argv[1:6]
rows = list(csv.DictReader(open(f)))
if not rows: print(f"FAIL: empty csv {f}")
for r in rows:
    print(f"RES {tag} round={rnd} arm={arm} pp{r['n_prompt']} {float(r['avg_ts']):.1f} +- {float(r['stddev_ts']):.1f} t/s  peak_vram_MiB={int(peak)//1048576}")
EOF
    grep "\[N1\] stats" bench-$TAG-$arm-$round.err | sed -n '2p;4p'
  done
done
echo "# done $(date -u +%FT%TZ)"
exit $fail
