#!/bin/bash
# Overnight run 5 (one r5.2xlarge): new preprocessing/blocking/pruner -> submission5a (run-3 model, NO retrain)
# as an early safety net -> retrain with all new features + gated ensemble -> submission5b.
# Every zip is checked by the official validator (package_submission.py). Artifacts go to S3 after every stage
# through presigned URLs in ~/run5_urls.env (no AWS credentials on the instance). The instance stops at the end.
#   setsid nohup ./overnight_run5.sh > ~/run5.out 2>&1 &     (PID in ~/run5.pid)
set -u
echo $$ > ~/run5.pid
B=~/ber; C=$B/code/business_entity_resolution; L=$C/logs; mkdir -p $L; S=$L/run5.log
source ~/run5_urls.env                                   # URL_<name>=presigned PUT URL per artifact
log() { echo "[$(date +%T)] $*" | tee -a $S; }
up() { [ -f "$1" ] && log "upload $(basename $1): $(curl -s -o /dev/null -w '%{http_code}' -T "$1" "$2")"; }
backup() {  # small artifacts + logs after every stage
  cd $B && tar -czf ~/run5_logs.tgz code/business_entity_resolution/logs models/*.joblib \
     $(ls dataset/processed/eval_*.parquet dataset/processed/eval_loss_decomposition.csv dataset/processed/oof_train.parquet \
          dataset/processed/decision_trace_test.parquet 2>/dev/null) 2>/dev/null
  up ~/run5_logs.tgz "$URL_logs"
}
finish() { log "FINISHED: $1"; backup; sleep 300; sudo shutdown -h now; exit 0; }
export MEM_CAP=none PY=../../../venv/bin/python S1_FRAC=${S1_FRAC:-0.5} FOLDS=3 BER_K=30 BER_K_REV=3 BER_LR=0.1 \
       BER_EXTRA_FEATS=1 BER_DECOY_FEATS=1 BER_ENSEMBLE=1 BER_PRUNE_MAX_CANDS=5 BER_PRUNE_FIT_PAIRS=3000000 FORCE=1
cd $C/src
# ---------------------------------------------------------------- 1. smoke test
log "smoke test"
SM=$B/smoke5; rm -rf $SM; mkdir -p $SM/proc $SM/models; cp $B/dataset/processed/*.tsv $SM/proc/
( export BER_PROCESSED_DIR=$SM/proc BER_MODELS_DIR=$SM/models FOLDS=2
  set -e
  $PY preprocess.py --splits train test --backend indicxlit --cache-only --overwrite --limit 10000 --jobs 3
  $PY blocking.py --split train --eval
  $PY prune.py fit
  $PY matching.py train --s1-frac 0.5 --folds 2
  $PY matching.py evaluate --train-frac 0.5
  $PY blocking.py --split test
  $PY prune.py apply --split test --no-tsv
  $PY matching.py predict ) > $L/smoke5.log 2>&1 || finish "SMOKE TEST FAILED (logs/smoke5.log)"
log "smoke passed: $(grep -a 'ensemble gate' $L/smoke5.log | tail -1)"
rm -rf $B/output; mkdir -p $B/output
# ---------------------------------------------------------------- 2. preprocess + blocking (train || test)
cd $C
./run_pipeline.sh learn preprocess >> $S 2>&1 || finish "FAILED preprocess"
( ./run_pipeline.sh block_test > $L/r5_block_test.log 2>&1 ) & BT=$!
./run_pipeline.sh block_train >> $S 2>&1 || finish "FAILED block_train"
rm -rf $B/dataset/processed/candidates_train_preprune; cp -r $B/dataset/processed/candidates_train $B/dataset/processed/candidates_train_preprune
wait $BT || finish "FAILED block_test"; cat $L/r5_block_test.log >> $S
backup
# ---------------------------------------------------------------- 3. 5a: new candidates, run-3 pruner + model (no retrain)
P5A=$B/proc5a; rm -rf $P5A; mkdir -p $P5A
for f in $B/dataset/processed/*.parquet $B/dataset/processed/*.tsv; do ln -s "$f" $P5A/; done
cp -r $B/dataset/processed/candidates_test $P5A/candidates_test
( export BER_PROCESSED_DIR=$P5A BER_MODELS_DIR=$B/models_run3 BER_DECOY_FEATS=0 BER_ENSEMBLE=0
  cd $C/src && $PY -u prune.py apply --split test && $PY -u matching.py predict && $PY -u package_submission.py --out ~/submission5a.zip
) > $L/r5_5a.log 2>&1 & P5=$!
# ---------------------------------------------------------------- 4. new pruner + retrain (all features + ensemble), in parallel with 5a
./run_pipeline.sh prune_train >> $S 2>&1 || finish "FAILED prune_train"
./run_pipeline.sh match_train >> $S 2>&1 || finish "FAILED match_train"
backup
wait $P5 && { up ~/submission5a.zip "$URL_sub5a"; log "submission5a validated + uploaded"; } || log "5a FAILED (see logs/r5_5a.log)"
# ---------------------------------------------------------------- 5. 5b
./run_pipeline.sh prune_test predict >> $S 2>&1 || finish "FAILED prune_test/predict"
cd $C/src && $PY -u package_submission.py --out ~/submission5b.zip >> $S 2>&1 && up ~/submission5b.zip "$URL_sub5b" || log "5b package FAILED"
grep -aE "per S1" $L/prune_test.log | tail -1 | tee -a $S
backup
# ---------------------------------------------------------------- 6. held-out evaluation of 5b (after the deliverables)
log "evaluate 5b held-out"
$PY -u matching.py evaluate --train-frac 0.5 --blocked-dir ../../../dataset/processed/candidates_train_preprune > $L/r5_evaluate.log 2>&1
$PY -u analyze_missed.py > $L/r5_missed.log 2>&1
finish "OK"
