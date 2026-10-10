#!/bin/bash
# T483 gate for llama-rs-reuse-check. All arms must report max |dlogit| == 0.0000 (bit-exact), because:
#  - nocp:ref=GGML_GDN_GLUE_FUSE=0 compares the fused prefill conv (ssm_conv_prefill_f32) with the unfused one
#    (concat + ssm_conv_long_token_f32) with no seq_cp and no graph reuse involved. A `< 0.5` verdict hid a last-ulp
#    tap-order difference (amplified to dlogit up to 2.1 at N=128) for a whole day: never loosen this.
#  - cp: seq_cp then prefill (the fused path is not eligible after a copy: s_copy(0) != head) vs a fresh reference
#    that does use the fused path.
#  - cp_nofuse: the same with the fusion off, where the graph IS reused after the copy (reused=1).
# usage: gate.sh <bin-dir> [model ...]   (default models: Bonsai 2 PQ2_0 and Bonsai 1 Q2_0)
set -u
B=${1:?bin dir}; shift
MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=(/aipool/models/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf /aipool/models/bonsai-27b/Ternary-Bonsai-27B-Q2_0.gguf)
export LD_LIBRARY_PATH=$B:${LD_LIBRARY_PATH:-}
[ -x "$B/llama-rs-reuse-check" ] || { echo "FATAL missing $B/llama-rs-reuse-check"; exit 2; }
fail=0
chk() { # tag model N [env...] -- args
  local tag=$1 M=$2 N=$3; shift 3
  local envs=() ; while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local out rc d
  out=$(env "${envs[@]}" "$B/llama-rs-reuse-check" "$M" "$N" 16 "$@" 2>&1); rc=$?
  d=$(printf '%s\n' "$out" | sed -n 's/^PREFILL_CMP max_abs_dlogit=\([0-9.]*\).*/\1/p')
  if [ $rc -ne 0 ] || [ -z "$d" ]; then echo "FAIL $tag $(basename "$M") N=$N rc=$rc (no result)"; fail=1; return; fi
  local reused; reused=$(printf '%s\n' "$out" | sed -n 's/^STEP X4.*graph_reused=\([0-9]*\).*/\1/p')
  if [ "$d" = "0.0000" ]; then echo "ok   $tag $(basename "$M" | cut -c1-24) N=$N dlogit=$d reused=$reused"
  else echo "FAIL $tag $(basename "$M" | cut -c1-24) N=$N dlogit=$d reused=$reused"; fail=1; fi
}
for M in "${MODELS[@]}"; do
  for N in 64 96 128 160 256; do chk nocp_fuse0 "$M" $N X=0 -- GGML_GDN_GLUE_FUSE=0 nocp; done
  for N in 64 128 256;        do chk cp "$M" $N X=0 -- -; done
  for N in 64 128 256;        do chk cp_nofuse "$M" $N GGML_GDN_GLUE_FUSE=0 -- -; done
done
[ $fail -eq 0 ] && echo "GATE_RESULT: PASS" || echo "GATE_RESULT: FAIL"
exit $fail
