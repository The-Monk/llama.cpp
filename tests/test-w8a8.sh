#!/bin/bash
# T513 gate for the enterprise W8A8 path (GGML_N1_W8). usage: test-w8a8.sh <bin-dir> [model.gguf ...]
# Every check fails if the path silently stops engaging: the "on" arms must differ from the off control, which must reproduce itself bit for bit.
#   1. kernels: decode GEMV, M3-class GEMM and the ROW GEMM bit-exact against the integer reference at the real shapes and small / odd N
#   2. routed ops (Q8_0 weight, F32 activation with outlier channels): off == off; in-place W8 != off with a bounded error; ROT lowers the error;
#      GEMV (N <= 16) ~ GEMM at ulp level; transient requant == in-place bit for bit (N >= 64); counters show the engagement
#   3. for each model argument: logits (256-token prefill, 16 decode steps, two 4-token verify batches) off == off, in-place+ROT != off with a
#      small KLD, reload in one process bit-identical, GEMV-vs-GEMM bit-identical, GGUF sha256 unchanged (with and without mmap)
set -u
B=${1:?bin dir}; shift
export LD_LIBRARY_PATH=$B:${LD_LIBRARY_PATH:-}
T=$B/test-w8a8
C=$B/llama-w8a8-check
O=${W8_TEST_OUT:-$(mktemp -d /tmp/test-w8a8.XXXXXX)}
mkdir -p "$O"
for f in $T $C; do [ -x "$f" ] || { echo "FATAL missing $f"; exit 2; }; done
fail=0
ok()   { echo "ok   $*"; }
bad()  { echo "FAIL $*"; fail=1; }
chk()  { # name -- command...   (the command's own output is shown only on failure)
  local n=$1; shift; shift
  local out; out=$("$@" 2>&1); local rc=$?
  if [ $rc -eq 0 ]; then ok "$n"; else bad "$n"; printf '%s\n' "$out" | tail -8; fi
}
ctr() { sed -n "s/.*W8_COUNTERS .*$2=\([0-9]*\).*/\1/p" "$1" | tail -1; }
worst() { sed -n 's/.*worst_nmse=\([0-9.e+-]*\).*/\1/p' "$1" | tail -1; }
IN="GGML_N1_W8=1 GGML_N1_W8_INPLACE=1"

# --- 1. kernels
if "$T" kernels > "$O/kernels.log" 2>&1; then ok "kernels: $(tail -1 "$O/kernels.log")"; else bad "kernels"; grep -E "^FAIL" "$O/kernels.log" | head -10; tail -2 "$O/kernels.log"; fi

# --- 2. routed ops
run_op() { local name=$1 envs=$2 ns=${3:-}; env $envs "$T" op "$O/$name.bin" $ns > "$O/$name.log" 2>&1 || { bad "op $name did not run"; tail -3 "$O/$name.log"; }; }
run_op off1 "GGML_N1_W8=0";  run_op off2 "GGML_N1_W8=0"
run_op on   "$IN";           run_op on_rot "$IN GGML_N1_W8_ROT=1"
run_op on_g0 "$IN GGML_N1_W8_GEMV_MAX=0"; run_op on_g16 "$IN GGML_N1_W8_GEMV_MAX=16"
run_op off_big "GGML_N1_W8=0" 64,129,512; run_op tr "GGML_N1_W8=1" 64,129,512; run_op tr_rot "GGML_N1_W8=1 GGML_N1_W8_ROT=1" 64,129,512; run_op on_rot_big "$IN GGML_N1_W8_ROT=1" 64,129,512; run_op on_big "$IN" 64,129,512
[ "$(ctr $O/off1.log mm)" = "0" ] && ok "off arm never reaches W8 (mm=0)" || bad "off arm reached W8: $(grep W8_COUNTERS $O/off1.log)"
[ "$(ctr $O/on.log mm)" -gt 0 ] 2>/dev/null && ok "on arm engaged: $(grep W8_COUNTERS $O/on.log)" || bad "on arm did not engage: $(grep W8_COUNTERS $O/on.log)"
[ "$(ctr $O/on.log gemv)" -gt 0 ] 2>/dev/null && ok "decode GEMV engaged" || bad "decode GEMV did not engage"
[ "$(ctr $O/on_g0.log gemv)" = "0" ] && ok "GEMV_MAX=0 routes everything to the GEMM" || bad "GEMV_MAX=0 still ran the GEMV"
chk "off == off (control reproduces itself)" -- "$T" cmp $O/off1.bin $O/off2.bin equal
chk "in-place W8 != off, error bounded (unrotated, outlier activations)" -- "$T" cmp $O/on.bin $O/off1.bin differs 0.5
chk "in-place W8+ROT != off, error bounded" -- "$T" cmp $O/on_rot.bin $O/off1.bin differs 5e-3
w_on=$(worst $O/on.log); w_rot=$(worst $O/on_rot.log)
python3 -I -c "import sys; a,b=float('${w_on:-1}'),float('${w_rot:-1}'); sys.exit(0 if b < a/5 else 1)" && ok "ROT lowers the error ($w_on -> $w_rot)" || bad "ROT does not lower the error ($w_on -> $w_rot)"
chk "GEMV ~ GEMM (N <= 16 via GEMV_MAX=16 vs 0, ulp-level)" -- "$T" cmp $O/on_g16.bin $O/on_g0.bin close 1e-12
chk "default GEMV_MAX=8 ~ GEMM (ulp-level)" -- "$T" cmp $O/on.bin $O/on_g0.bin close 1e-12
chk "transient requant == in-place, bit for bit (N >= 64)" -- "$T" cmp $O/tr.bin $O/on_big.bin equal
chk "transient+ROT requant == in-place+ROT (N >= 64), bit for bit" -- "$T" cmp $O/tr_rot.bin $O/on_rot_big.bin equal
chk "transient W8 != off (N >= 64)" -- "$T" cmp $O/tr.bin $O/off_big.bin differs 0.5

# --- 3. engine level
for M in "$@"; do
  [ -f "$M" ] || { bad "missing model $M"; continue; }
  n=$(basename "$M" .gguf); S0=$(sha256sum "$M" | cut -d' ' -f1)
  eng() { local name=$1 envs=$2; shift 2; env $envs "$C" run "$M" "$O/$n-$name.bin" "$@" > "$O/$n-$name.log" 2>&1 || { bad "$n $name run"; tail -4 "$O/$n-$name.log"; }; }
  eng off1 "GGML_N1_W8=0" 999 1; eng off2 "GGML_N1_W8=0" 999 1
  eng on "$IN GGML_N1_W8_ROT=1" 999 2; eng on_mmap "$IN GGML_N1_W8_ROT=1" 999 1 1; eng on_nomm "$IN GGML_N1_W8_ROT=1" 999 1 0; eng on_dio "$IN GGML_N1_W8_ROT=1" 999 1 4; eng on_g0 "$IN GGML_N1_W8_ROT=1 GGML_N1_W8_GEMV_MAX=0" 999 1
  eng on_part "$IN GGML_N1_W8_ROT=1" 20 1; eng off_part "GGML_N1_W8=0" 20 1; eng on_cpu "$IN GGML_N1_W8_ROT=1" 0 1
  chk "$n: off == off" -- "$C" cmp $O/$n-off1.bin $O/$n-off2.bin equal
  chk "$n: in-place+ROT != off, KLD bounded" -- "$C" cmp $O/$n-off1.bin $O/$n-on.bin differs 0.05
  [ "$(ctr $O/$n-on.log mm)" -gt 0 ] 2>/dev/null && ok "$n: engaged: $(grep -m1 W8_COUNTERS $O/$n-on.log)" || bad "$n: W8 did not engage"
  grep -q "RELOAD 1: bit-identical" $O/$n-on.log && ok "$n: second load in the same process bit-identical" || bad "$n: reload differs: $(grep RELOAD $O/$n-on.log)"
  chk "$n: no-mmap load == mmap load" -- "$C" cmp $O/$n-on_mmap.bin $O/$n-on_nomm.bin equal
  chk "$n: direct-io load == mmap load" -- "$C" cmp $O/$n-on_mmap.bin $O/$n-on_dio.bin equal
  chk "$n: GEMV ~ GEMM at engine level (KLD)" -- "$C" cmp $O/$n-on.bin $O/$n-on_g0.bin close 1e-4
  conv_full=$(ctr $O/$n-on.log converted); conv_part=$(ctr $O/$n-on_part.log converted)
  [ "${conv_part:-0}" -gt 0 ] && [ "${conv_part:-0}" -lt "${conv_full:-0}" ] && ok "$n: partial offload converts only the device tensors ($conv_part < $conv_full)" || bad "$n: partial offload conv=$conv_part full=$conv_full"
  chk "$n: partial offload: bounded vs the same offload off" -- "$C" cmp $O/$n-off_part.bin $O/$n-on_part.bin differs 0.05
  [ "$(ctr $O/$n-on_cpu.log converted)" = "0" ] && ok "$n: ngl=0 converts nothing" || bad "$n: ngl=0 converted tensors"
  S1=$(sha256sum "$M" | cut -d' ' -f1)
  [ "$S0" = "$S1" ] && ok "$n: GGUF sha256 unchanged ($S1)" || bad "$n: GGUF CHANGED ON DISK $S0 -> $S1"
done
[ $fail -eq 0 ] && echo "TEST_W8A8_RESULT: PASS" || echo "TEST_W8A8_RESULT: FAIL"
echo "outputs in $O"
exit $fail
