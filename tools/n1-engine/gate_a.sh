#!/bin/bash
# T404 gate (a): test-backend-ops MUL_MAT on ROCm0, default vs N1 arms. Reports passed/total + [N1] routing stats.
set -u
B=/home/jmonk/n1-engine/build/bin
[ -x $B/test-backend-ops ] || { echo "FAIL: no test-backend-ops"; exit 2; }
echo "# $(date -u +%FT%TZ) HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
fail=0
for arm in default g128 token rowall; do
  case $arm in
    default) E=(N1_ARM=default);;
    g128)    E=(GGML_N1_PREFILL=1 GGML_N1_ACT=g128);;
    token)   E=(GGML_N1_PREFILL=1 GGML_N1_ACT=token);;
    rowall)  E=(GGML_N1_PREFILL=1 GGML_N1_ROW=all);;
  esac
  env "${E[@]}" $B/test-backend-ops -o MUL_MAT -b ROCm0 > tbo-$arm.raw 2>&1; rc=$?
  sed 's/\x1b\[[0-9;]*m//g' tbo-$arm.raw > tbo-$arm.txt
  line=$(grep -E "tests passed" tbo-$arm.txt | tail -1)
  nfail=$(grep -c "FAIL" tbo-$arm.txt)
  echo "=== $arm ${E[*]} rc=$rc fails=$nfail :: ${line:-NO PASSED LINE}"
  grep -o "\[N1\] stats.*" tbo-$arm.txt | sed -n '2,3p'
  # the new T404 cases (types q2_0/q1_0, n>=64): print their verdict lines
  grep -E "MUL_MAT\(type_a=q[12]_0,type_b=f32,m=(256|384|512|1024),n=(64|100|128|256|512|513)," tbo-$arm.txt | sed 's/^ *//' | cut -c1-150
  [ -n "$line" ] || fail=1
  [ "$nfail" = "0" ] || { grep "FAIL" tbo-$arm.txt | head -10; fail=1; }
done
echo "# done $(date -u +%FT%TZ)"
exit $fail
