#!/bin/bash
# T404 gate (d): rocprofv3 kernel trace of llama-bench pp512 (-n 0 -r 3; warmup + 3 reps) per arm.
# gate_d.sh <model.gguf> <tag> [arms...]   -> kt-<tag>-<arm>/.../*kernel_trace.csv + analysis
set -u
M=$1; TAG=$2; shift 2
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(default g128 token rowffn)
B=/home/jmonk/n1-engine/build/bin
R=/home/jmonk/.cache/lemonade/bin/therock/gfx1201-7.13.0/bin/rocprofv3
AN=/home/jmonk/scratch/n1eng/analyze_trace.py
[ -x $B/llama-bench ] || { echo "FAIL: no llama-bench"; exit 2; }
[ -x $R ] || { echo "FAIL: no rocprofv3"; exit 2; }
[ -s $AN ] || { echo "FAIL: no $AN"; exit 2; }
echo "# $(date -u +%FT%TZ) $TAG HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
fail=0
for arm in "${ARMS[@]}"; do
  case $arm in
    default) E=(N1_ARM=default);;
    g128)    E=(GGML_N1_PREFILL=1 GGML_N1_ACT=g128);;
    token)   E=(GGML_N1_PREFILL=1 GGML_N1_ACT=token);;
    rowffn)  E=(GGML_N1_PREFILL=1 GGML_N1_ACT=g128 GGML_N1_ROW=ffn);;
  esac
  d=$PWD/kt-$TAG-$arm
  timeout 1200 env "${E[@]}" $R --kernel-trace --output-format csv -d $d -o kt -- \
    $B/llama-bench -m "$M" -ngl 99 -p 512 -n 0 -r 3 -o csv > $d.log 2>&1
  rc=$?; f=$(find $d -name "*kernel_trace.csv" | head -1)
  if [ -n "$f" ] && [ -s "$f" ]; then
    echo "=== TRACE $TAG $arm rc=$rc rows=$(wc -l < $f)"
    grep -E '^"?build_commit' -A3 $d.log | cut -d, -f33-41 | tail -1
    /home/jmonk/miniforge3/bin/python3 -I $AN "$f" 4 || fail=1
  else
    echo "=== TRACE $TAG $arm rc=$rc NO-CSV"; tail -8 $d.log; fail=1
  fi
done
echo "# done $(date -u +%FT%TZ)"
exit $fail
