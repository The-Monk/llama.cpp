#!/bin/bash
# T404 smoke: does N1 route, run, and produce sane PPL? (not a measurement)
set -u
B=/home/jmonk/n1-engine/build/bin
M=/aipool/models/bonsai-27b/Ternary-Bonsai-27B-Q2_0.gguf
TXT=/home/jmonk/gfx1201-native-model/data/wikitext-2-raw/wiki.test.raw
for f in $B/llama-bench $B/llama-perplexity; do [ -x "$f" ] || { echo "FAIL: no $f"; exit 2; }; done
for f in "$M" "$TXT"; do [ -s "$f" ] || { echo "FAIL: no $f"; exit 2; }; done
echo "# $(date -u +%FT%TZ) HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
for arm in default g128 token; do
  case $arm in
    default) E=(N1_ARM=default);;
    g128)    E=(GGML_N1_PREFILL=1 GGML_N1_ACT=g128);;
    token)   E=(GGML_N1_PREFILL=1 GGML_N1_ACT=token);;
  esac
  echo "=== $arm ${E[*]}"
  env "${E[@]}" $B/llama-perplexity -m "$M" -f "$TXT" -c 512 --chunks 3 -ngl 99 > ppl-$arm.txt 2>&1; rc=$?
  echo "ppl rc=$rc"; grep -E "Final estimate|\[N1\]" ppl-$arm.txt | head -8
  [ $rc -eq 0 ] || tail -15 ppl-$arm.txt
  env "${E[@]}" $B/llama-bench -m "$M" -ngl 99 -n 0 -p 512 -r 2 -o csv > bench-$arm.txt 2> bench-$arm.err; rc=$?
  echo "bench rc=$rc"; grep -E "pp512|avg_ts" bench-$arm.txt | head -3; grep "\[N1\]" bench-$arm.err | head -6
  [ $rc -eq 0 ] || tail -15 bench-$arm.err
done
echo "# done $(date -u +%FT%TZ)"
