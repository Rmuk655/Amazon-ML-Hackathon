#!/bin/bash
# Run 5 preview (self-running, no laptop): all catalogue-independent changes - name cleaners + address key,
# reverse blocking, decoy features, pruner <= 5 candidates/S1, gated stacked ensemble, locked 10% holdout -
# retrain -> validated submission5_preview.zip + locked-holdout evaluation -> S3 -> stop the instance.
set -u
echo $$ > ~/run5p.pid
source ~/run5_urls.env
B=~/ber; C=$B/code/business_entity_resolution; L=$C/logs; mkdir -p $L; S=$L/run5p.log
log() { echo "[$(date +%T)] $*" | tee -a $S; }
up() { [ -f "$1" ] && log "upload $(basename $1): $(curl -s -o /dev/null -w '%{http_code}' -T "$1" "$2")"; }
backup() {
  (cd $B && tar -czf ~/run5p_logs.tgz code/business_entity_resolution/logs models/*.joblib \
    $(ls dataset/processed/eval_*.parquet dataset/processed/eval_*.csv dataset/processed/oof_train.parquet \
         dataset/processed/lost_pairs_classified.parquet dataset/processed/decision_trace_test.parquet 2>/dev/null) 2>/dev/null)
  up ~/run5p_logs.tgz "$URL_run5p_logs"
}
finish() { log "FINISHED: $1"; backup; sleep 300; sudo shutdown -h now; exit 0; }
export MEM_CAP=none PY=../../../venv/bin/python S1_FRAC=0.5 FOLDS=3 BER_K=30 BER_K_REV=3 BER_LR=0.1 \
       BER_EXTRA_FEATS=1 BER_DECOY_FEATS=1 BER_ENSEMBLE=1 BER_PRUNE_MAX_CANDS=5 BER_PRUNE_FIT_PAIRS=3000000 FORCE=1
cd $C/src
log "smoke test"
SM=$B/smoke5p; rm -rf $SM; mkdir -p $SM/proc $SM/models; cp $B/dataset/processed/*.tsv $SM/proc/
( export BER_PROCESSED_DIR=$SM/proc BER_MODELS_DIR=$SM/models
  set -e
  $PY preprocess.py --splits train test --backend indicxlit --cache-only --overwrite --limit 10000 --jobs 3
  $PY blocking.py --split train --eval
  $PY prune.py fit
  $PY matching.py train --s1-frac 0.5 --folds 2
  $PY matching.py evaluate --population holdout --train-frac 0.0
  $PY blocking.py --split test
  $PY prune.py apply --split test --no-tsv
  $PY matching.py predict ) > $L/smoke5p.log 2>&1 || finish "SMOKE TEST FAILED (logs/smoke5p.log)"
log "smoke passed: $(grep -a 'ensemble gate' $L/smoke5p.log | tail -1)"
rm -rf $B/output; mkdir -p $B/output
cd $C
./run_pipeline.sh learn preprocess >> $S 2>&1 || finish "FAILED preprocess"
( ./run_pipeline.sh block_test > $L/r5p_block_test.log 2>&1 ) & BT=$!
./run_pipeline.sh block_train >> $S 2>&1 || finish "FAILED block_train"
rm -rf $B/dataset/processed/candidates_train_preprune
cp -r $B/dataset/processed/candidates_train $B/dataset/processed/candidates_train_preprune
wait $BT || finish "FAILED block_test"; cat $L/r5p_block_test.log >> $S
./run_pipeline.sh prune_train >> $S 2>&1 || finish "FAILED prune_train"
( ./run_pipeline.sh prune_test > $L/r5p_prune_test.log 2>&1 ) & PT=$!
./run_pipeline.sh match_train >> $S 2>&1 || finish "FAILED match_train"
wait $PT || finish "FAILED prune_test"; cat $L/r5p_prune_test.log >> $S
backup
./run_pipeline.sh predict >> $S 2>&1 || finish "FAILED predict"
cd $C/src && $PY -u package_submission.py --out ~/submission5_preview.zip >> $S 2>&1 \
  && up ~/submission5_preview.zip "$URL_sub5p" || log "PACKAGE/VALIDATION FAILED"
grep -aE "per S1" $L/prune_test.log | tail -1 | tee -a $S
log "evaluate on the LOCKED HOLDOUT"
$PY -u matching.py evaluate --population holdout --train-frac 0.0 \
  --blocked-dir ../../../dataset/processed/candidates_train_preprune > $L/r5p_holdout_eval.log 2>&1
grep -aA3 "^slice " $L/r5p_holdout_eval.log | tee -a $S
$PY -u analyze_missed.py > $L/r5p_missed.log 2>&1
finish "OK"
