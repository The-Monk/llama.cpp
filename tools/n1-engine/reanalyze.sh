#!/bin/bash
# T404: re-run analyze_trace.py on saved traces (CPU only). reanalyze.sh <job workdir> <tag>
set -u
W=$1; TAG=$2
AN=/home/jmonk/scratch/n1eng/analyze_trace.py
PY=/home/jmonk/miniforge3/bin/python3
[ -x $PY ] || { echo "FAIL: no python"; exit 2; }
for arm in default g128 token rowffn; do
  f=$(find $W/kt-$TAG-$arm -name "*kernel_trace.csv" 2>/dev/null | head -1)
  [ -n "$f" ] && [ -s "$f" ] || { echo "=== $arm NO-CSV"; continue; }
  echo "=== $TAG $arm"
  grep -o 'gemm_iu8<[^(]*' "$f" | sort | uniq -c | head -3
  $PY -I $AN "$f" 4 || echo "FAIL: analyze $arm"
done
echo "# done"
