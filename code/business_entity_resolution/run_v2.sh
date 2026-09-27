#!/bin/bash
# v2 pipeline, end to end (data -> blocking -> pruning -> features -> matcher -> assignment -> output).
#   ./run_v2.sh                 all steps
#   FROM=block ./run_v2.sh      resume from a step (prep norm block prune feats train test predict validate)
# Env: BER_WORKERS (threads, default 8), BER_TRAIN_FRAC (matcher training subsample, default 0.35),
#      BER_K_NAME/ADDR/COMB/KEY (blocking top-K per channel), BER_FPW (threshold density setting, default 2.0).
set -u
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
L=../logs/run_v2.log
mkdir -p ../logs
export BER_WORKERS=${BER_WORKERS:-8} BER_TRAIN_FRAC=${BER_TRAIN_FRAC:-0.35}
STEPS="prep norm block prune feats train test predict validate"
FROM=${FROM:-prep}
started=0
step() { echo "[$(date +%H:%M:%S)] START $*" >> $L; /usr/bin/time -f "peak=%MKB wall=%e" "$@" >> $L 2>&1; rc=$?
         echo "[$(date +%H:%M:%S)] END rc=$rc $*" >> $L; [ $rc -eq 0 ] || { echo "RUN_V2 FAILED" >> $L; exit 1; }; }
for s in $STEPS; do
  [ "$s" = "$FROM" ] && started=1
  [ $started -eq 1 ] || continue
  case $s in
    prep)     step $P -m v2.prep raw --split train; step $P -m v2.prep raw --split test ;;
    norm)     step $P -m v2.run norm --split train; step $P -m v2.run norm --split test ;;
    block)    step $P -m v2.run block --split train; step $P -m v2.run block --split test ;;
    prune)    step $P -m v2.run prune --split train ;;
    feats)    step $P -m v2.run feats --split train ;;
    train)    step $P -m v2.run train ;;
    test)     step $P -m v2.run prune --split test; step $P -m v2.run feats --split test ;;
    predict)  step $P -m v2.run predict ;;
    validate) step $P ../../../utils/validate_submission.py --matching ../../../output/matching_results.tsv \
                   --candidate ../../../output/candidate_pairs.tsv --test-dir ../../../dataset/test ;;
  esac
done
echo "[$(date +%H:%M:%S)] RUN_V2 DONE" >> $L
