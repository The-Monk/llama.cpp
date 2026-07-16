#!/usr/bin/env bash
# Section B ship blocker #1: sustained/production load test on the ROC8 appliance.
# Concurrency + stability-over-time + thermal. Non-destructive (--rm, model ro).
set -uo pipefail

IMG=${IMG:-roc8-lemonade:tr713}
MODEL_DIR=${MODEL_DIR:-/aipool/models/qwen3-8b-fp8}
MODEL_FILE=${MODEL_FILE:-/models/Qwen3-8B-F8E4M3.gguf}   # path INSIDE container
MODEL_NAME=${MODEL_NAME:-Qwen3-8B-FP8}
MODEL_ID=${MODEL_ID:-user.Qwen3-8B-FP8}
LLAMA_ARGS=${LLAMA_ARGS:-"-ngl 999 --cont-batching --reasoning off"}
PORT=${PORT:-13405}
GPU=${GPU:-0}      # single "0" or "0,1" for tensor-split across both cards
CONC=${CONC:-8}            # concurrent workers
DURATION=${DURATION:-300} # seconds of sustained load
MAXTOK=${MAXTOK:-128}
NAME=roc8load
OUT=${OUT:-/home/jmonk/src/mainline-llama.cpp-mxfp8/container/loadtest-results}
mkdir -p "$OUT"
STAMP=$(cat /proc/sys/kernel/random/uuid | cut -c1-8)
RUNDIR="$OUT/run-$STAMP"; mkdir -p "$RUNDIR"

echo "=== ROC8 appliance sustained load test ==="
echo "image=$IMG  gpu=$GPU  conc=$CONC  duration=${DURATION}s  maxtok=$MAXTOK"
echo "results -> $RUNDIR"

podman rm -f $NAME >/dev/null 2>&1

echo "--- starting appliance (detached, GPU$GPU) ---"
podman run -d --rm --runtime crun --name $NAME \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined \
  -v "$MODEL_DIR":/models:ro \
  -e HIP_VISIBLE_DEVICES=$GPU \
  -e MODEL="$MODEL_FILE" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e LLAMA_ARGS="$LLAMA_ARGS" \
  -e LEMONADE_PORT=13305 \
  -p ${PORT}:13305 \
  $IMG serve >/dev/null

# Wait for health
ok=0
for i in $(seq 1 90); do
  h=$(curl -s -m 3 http://localhost:${PORT}/api/v1/health 2>/dev/null)
  if echo "$h" | grep -qiE 'ok|status|true|healthy|model'; then echo "health up in $((i*2))s"; ok=1; break; fi
  sleep 2
done
if [ $ok -eq 0 ]; then echo "FAIL: health never came up"; podman logs $NAME 2>&1 | tail -30; podman stop $NAME >/dev/null 2>&1; exit 1; fi

# GPU sampler — samples BOTH cards every 3s (temp/util/power per card)
SAMPLE="$RUNDIR/gpu-samples.csv"
echo "ts,temp0_c,util0,power0_w,temp1_c,util1,power1_w" > "$SAMPLE"
(
  while true; do
    read t0 u0 p0 t1 u1 p1 < <(rocm-smi --showtemp --showuse --showpower 2>/dev/null | python3 -c '
import sys,re
temps={};utils={};pows={}
for ln in sys.stdin:
    m=re.match(r"GPU\[(\d)\]",ln)
    if not m: continue
    g=m.group(1)
    if "edge" in ln.lower():
        v=re.findall(r"[0-9]+\.[0-9]+",ln);  temps[g]=v[-1] if v else "NA"
    elif "GPU use" in ln:
        v=re.findall(r": (\d+)",ln); utils[g]=v[-1] if v else "NA"
    elif "Socket" in ln or "Average Graphics Package" in ln:
        v=re.findall(r"[0-9]+\.[0-9]+",ln); pows[g]=v[-1] if v else "NA"
print(temps.get("0","NA"),utils.get("0","NA"),pows.get("0","NA"),temps.get("1","NA"),utils.get("1","NA"),pows.get("1","NA"))
')
    echo "$(date +%s),${t0},${u0},${p0},${t1},${u1},${p1}" >> "$SAMPLE"
    sleep 3
  done
) &
SAMPLER=$!

PROMPTS=("Explain how a hash table works." "Write a haiku about storage arrays." "What is RAID 10?" "Summarize TCP congestion control." "Name three ZFS features." "What does mdadm do?" "Describe NUMA in one paragraph." "How does TRIM work on SSDs?")

WORK="$RUNDIR/latencies.txt"; : > "$WORK"
ERRLOG="$RUNDIR/errors.txt"; : > "$ERRLOG"

worker() {
  local wid=$1 end=$2 n=0
  while [ "$(date +%s)" -lt "$end" ]; do
    local pr="${PROMPTS[$(( (wid + n) % ${#PROMPTS[@]} ))]}"
    local t0=$(date +%s.%N)
    local resp; resp=$(curl -s -m 120 http://localhost:${PORT}/api/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"$pr /no_think\"}],\"max_tokens\":$MAXTOK,\"temperature\":0.7}" 2>/dev/null)
    local t1=$(date +%s.%N)
    local dt=$(echo "$t1 - $t0" | bc)
    if echo "$resp" | grep -q '"content"'; then
      local ct=$(echo "$resp" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("usage",{}).get("completion_tokens",0))' 2>/dev/null || echo 0)
      echo "$dt $ct" >> "$WORK"
    else
      echo "w$wid n$n dt=$dt : $(echo "$resp" | head -c 200)" >> "$ERRLOG"
    fi
    n=$((n+1))
  done
}

END=$(( $(date +%s) + DURATION ))
echo "--- firing $CONC workers for ${DURATION}s ---"
for w in $(seq 1 $CONC); do worker $w $END & done
wait $(jobs -rp | grep -v $SAMPLER 2>/dev/null) 2>/dev/null
# ensure all workers done
for job in $(jobs -rp); do [ "$job" != "$SAMPLER" ] && wait "$job" 2>/dev/null; done

kill $SAMPLER 2>/dev/null

echo "--- container state after load ---"
podman ps --filter name=$NAME --format "{{.Status}}" | tee "$RUNDIR/container-status.txt"
podman logs $NAME 2>&1 | grep -iE "error|abort|assert|oom|crash|panic" | tail -20 | tee "$RUNDIR/container-errors.txt"

echo "--- stopping appliance ---"
podman stop $NAME >/dev/null 2>&1

# ---- Report ----
python3 - "$WORK" "$ERRLOG" "$SAMPLE" "$DURATION" "$CONC" <<'PY' | tee "$RUNDIR/report.txt"
import sys
lat_f, err_f, samp_f, dur, conc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
lats=[]; toks=[]
for ln in open(lat_f):
    a=ln.split()
    if len(a)>=2:
        lats.append(float(a[0])); toks.append(int(a[1]))
errs=sum(1 for _ in open(err_f) if _.strip())
n=len(lats)
def pct(v,p):
    if not v: return 0
    s=sorted(v); import math; k=min(len(s)-1,int(round(p/100*(len(s)-1)))); return s[k]
print("\n================= LOAD TEST REPORT =================")
print(f"requests OK     : {n}")
print(f"requests FAILED : {errs}   ({100*errs/max(1,n+errs):.2f}%)")
print(f"throughput      : {n/dur:.2f} req/s  ({conc} concurrent)")
if lats:
    print(f"latency p50     : {pct(lats,50):.2f}s")
    print(f"latency p90     : {pct(lats,90):.2f}s")
    print(f"latency p99     : {pct(lats,99):.2f}s")
    print(f"latency max     : {max(lats):.2f}s")
    tt=sum(toks)
    print(f"tokens total    : {tt}   agg tok/s: {tt/dur:.1f}")
# thermal — per card
def col(idx):
    out=[]
    for ln in open(samp_f):
        p=ln.strip().split(',')
        if p[0]=='ts' or len(p)<7: continue
        try:
            if p[idx]!='NA': out.append(float(p[idx]))
        except: pass
    return out
for g,(ti,ui,pi) in {0:(1,2,3),1:(4,5,6)}.items():
    temps=col(ti); utils=col(ui); powers=col(pi)
    if not temps: continue
    print(f"\n--- GPU{g} ---")
    print(f"  thermal start/peak/drift : {temps[0]:.0f}C / {max(temps):.0f}C / +{max(temps)-temps[0]:.0f}C")
    if powers: print(f"  power peak               : {max(powers):.0f}W")
    if utils:  print(f"  util avg/peak            : {sum(utils)/len(utils):.0f}% / {max(utils):.0f}%")
print("===================================================")
PY
echo "LOADTEST_DONE  ($RUNDIR)"