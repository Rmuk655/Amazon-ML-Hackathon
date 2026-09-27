#!/bin/bash
# Backup bi-encoder queue on Kaggle (max 2 GPU sessions). Every 5 min: download the output of any kernel that has
# finished, then push the next pending kernel while fewer than 2 are running. Lightweight (Kaggle API calls only).
#   setsid nohup kaggle/queue.sh > ~/kaggle_queue.log 2>&1 &     PID in ~/kaggle_queue.pid
# State: submissions/emb/<kernel>/ (downloaded output) and submissions/emb/<kernel>.done
cd "$(dirname "$0")"; echo $$ > ~/kaggle_queue.pid
K=../../../venv/bin/kaggle; OUT=../../../submissions/emb; mkdir -p $OUT
QUEUE="k_test_india k_test_us k_test_france k_train_india k_train_us"   # k_test_india was pushed before the laptop rebooted
ref() { [ "$1" = biencoder ] && echo rmuk16/ber-biencoder || python3 -c "import json;print(json.load(open('$1/kernel-metadata.json'))['id'])"; }
status() { $K kernels status "$(ref $1)" 2>&1 | tail -1 | grep -oE 'RUNNING|QUEUED|COMPLETE|ERROR|CANCEL[A-Z_]*|not found|404' | head -1; }
log() { echo "[$(date '+%a %H:%M')] $*"; }
started() { [ "$1" = biencoder ] || [ "$1" = k_test_india ] || [ -f $OUT/$1.pushed ]; }
while :; do
  running=0; pending=""
  for k in biencoder $QUEUE; do
    [ -f $OUT/$k.done ] && continue
    if ! started $k; then pending="$pending $k"; continue; fi
    s=$(status $k)
    case "$s" in
      COMPLETE) mkdir -p $OUT/$k; $K kernels output "$(ref $k)" -p $OUT/$k > $OUT/$k.download.log 2>&1 \
                  && touch $OUT/$k.done && log "$k COMPLETE -> $(ls $OUT/$k | tr '\n' ' ')" || log "$k download FAILED (retry next loop)";;
      ERROR|CANCEL*) mkdir -p $OUT/$k; $K kernels output "$(ref $k)" -p $OUT/$k > $OUT/$k.download.log 2>&1
                  touch $OUT/$k.done; log "$k ended with $s (log downloaded to $OUT/$k)";;
      *) running=$((running + 1));;
    esac
  done
  for k in $pending; do
    [ $running -ge 2 ] && break
    $K kernels push -p $k > $OUT/$k.push.log 2>&1 && { touch $OUT/$k.pushed; running=$((running + 1)); log "pushed $k"; } \
      || log "push $k FAILED: $(tail -1 $OUT/$k.push.log)"
  done
  [ -z "$pending" ] && [ $running -eq 0 ] && { log "QUEUE DONE"; exit 0; }
  sleep 300
done
