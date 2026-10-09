#!/bin/bash
# T404 gate (b): llama-perplexity KLD, -c 512 --chunks 20, default path = base. gate_b.sh <model.gguf> <tag>
set -u
M=$1; TAG=$2
B=/home/jmonk/n1-engine/build/bin
PPL=$B/llama-perplexity
TXT=/home/jmonk/gfx1201-native-model/data/wikitext-2-raw/wiki.test.raw
[ -x "$PPL" ] || { echo "FAIL: no $PPL"; exit 2; }
for f in "$M" "$TXT"; do [ -s "$f" ] || { echo "FAIL: no $f"; exit 2; }; done
OUT=$PWD; BASE=$OUT/$TAG.kld
COMMON=(-c 512 --chunks 20 -ngl 99 -f "$TXT")
echo "# $(date -u +%FT%TZ) $TAG HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
echo "=== BASE default path ==="
"$PPL" -m "$M" "${COMMON[@]}" --kl-divergence-base "$BASE" > $OUT/$TAG-base.txt 2>&1; rc=$?
grep -E "Final estimate" $OUT/$TAG-base.txt | tail -1
[ $rc -eq 0 ] && [ -s "$BASE" ] || { echo "FAIL: base rc=$rc"; tail -20 $OUT/$TAG-base.txt; exit 4; }
fail=0
for arm in control g128 token rowffn; do
  case $arm in
    control) E=(N1_ARM=control);;
    g128)    E=(GGML_N1_PREFILL=1 GGML_N1_ACT=g128);;
    token)   E=(GGML_N1_PREFILL=1 GGML_N1_ACT=token);;
    rowffn)  E=(GGML_N1_PREFILL=1 GGML_N1_ACT=g128 GGML_N1_ROW=ffn);;
  esac
  echo "=== ARM $arm ${E[*]} ==="
  env "${E[@]}" "$PPL" -m "$M" "${COMMON[@]}" --kl-divergence-base "$BASE" --kl-divergence > $OUT/$TAG-$arm.txt 2>&1; rc=$?
  [ $rc -eq 0 ] || { echo "FAIL: $arm rc=$rc"; tail -20 $OUT/$TAG-$arm.txt; fail=1; continue; }
  grep -E "^Mean PPL\(Q\)|Mean ln\(PPL|Mean    KLD|99.9%   KLD|Maximum KLD|Same top p" $OUT/$TAG-$arm.txt
  grep "\[N1\]" $OUT/$TAG-$arm.txt | grep -E "stats" | head -4
  grep "\[N1\] fallback shape" $OUT/$TAG-$arm.txt | sort | uniq -c | sort -rn | head -5
  grep -q "Mean    KLD" $OUT/$TAG-$arm.txt || { echo "FAIL: no KLD lines in $arm"; fail=1; }
done
rm -f "$BASE"
echo "# done $(date -u +%FT%TZ)"
exit $fail
