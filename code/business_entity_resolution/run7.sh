#!/bin/bash
# Run 7 (self-running, no laptop): noisy-channel token repair + run-5 preview + rule-based blocking (exact-core+address bypass of the
# top-K cut, sorted-token and acronym keys, full and 3-word glue joins) + legal-form added/dropped decoy features.: all catalogue-independent changes - name cleaners + address key,
# reverse blocking, decoy features, pruner <= 5 candidates/S1, gated stacked ensemble, locked 10% holdout -
# retrain -> validated submission7.zip + locked-holdout evaluation -> S3 -> stop the instance.
set -u
echo $$ > ~/run7.pid
source ~/run5_urls.env
B=~/ber7; C=$B/code/business_entity_resolution; L=$C/logs; mkdir -p $L; S=$L/run7.log
log() { echo "[$(date +%T)] $*" | tee -a $S; }
up() { [ -f "$1" ] && log "upload $(basename $1): $(curl -s -o /dev/null -w '%{http_code}' -T "$1" "$2")"; }
backup() {
  (cd $B && tar -czf ~/run7.logs.tgz code/business_entity_resolution/logs models/*.joblib \
    $(ls dataset/processed/eval_*.parquet dataset/processed/eval_*.csv dataset/processed/oof_train.parquet \
         dataset/processed/lost_pairs_classified.parquet dataset/processed/decision_trace_test.parquet 2>/dev/null) 2>/dev/null)
  up ~/run7.logs.tgz "$URL_run7_logs"
}
finish() { log "FINISHED: $1"; backup; sleep 300; sudo shutdown -h now; exit 0; }
export MEM_CAP=none PY=../../../venv/bin/python S1_FRAC=1.0 FOLDS=3 BER_K=30 BER_K_REV=3 BER_LR=0.1 \
       BER_EXTRA_FEATS=1 BER_DECOY_FEATS=1 BER_ENSEMBLE=0 BER_PRUNE_MAX_CANDS=5 BER_PRUNE_SPLIT_BUDGET=5 BER_PRUNE_FIT_PAIRS=3000000 BER_TOKEN_REPAIR=1 FORCE=1
cd $C/src
log "smoke test"
SM=$B/smoke7; rm -rf $SM; mkdir -p $SM/proc $SM/models; cp $B/dataset/processed/*.tsv $SM/proc/
( export BER_PROCESSED_DIR=$SM/proc BER_MODELS_DIR=$SM/models
  set -e
  $PY preprocess.py --splits train test --backend indicxlit --cache-only --overwrite --limit 10000 --jobs 3
  $PY blocking.py --split train --eval
  $PY prune.py fit
  $PY matching.py train --s1-frac 0.5 --folds 2
  $PY matching.py evaluate --population holdout --train-frac 0.0
  $PY blocking.py --split test
  $PY prune.py apply --split test --no-tsv
  $PY matching.py predict ) > $L/smoke7.log 2>&1
if [ $? -ne 0 ]; then       # guard: denoiser failed -> fall back to the run-6 configuration (denoiser off)
  log "smoke test FAILED with the denoiser -> retrying with BER_TOKEN_REPAIR=0 (run-6 configuration)"
  export BER_TOKEN_REPAIR=0
  ( export BER_PROCESSED_DIR=$SM/proc BER_MODELS_DIR=$SM/models
    set -e
    $PY preprocess.py --splits train test --backend indicxlit --cache-only --overwrite --limit 10000 --jobs 3
    $PY blocking.py --split train --eval
    $PY prune.py fit
    $PY matching.py train --s1-frac 0.5 --folds 2
    $PY blocking.py --split test
    $PY prune.py apply --split test --no-tsv
    $PY matching.py predict ) > $L/smoke7_fallback.log 2>&1 || finish "SMOKE TEST FAILED even without the denoiser"
fi
log "configuration: BER_TOKEN_REPAIR=$BER_TOKEN_REPAIR BER_NEW_KEYS=${BER_NEW_KEYS:-1} BER_ADDR_KEY=${BER_ADDR_KEY:-1} BER_BYPASS=${BER_BYPASS:-1}"
log "smoke passed: $(grep -a 'ensemble gate' $L/smoke7.log | tail -1)"
rm -rf $B/output; mkdir -p $B/output
log "denoiser report on the locked holdout (before the new preprocessing)"
(cd $C/src && BER_TOKEN_REPAIR=1 $PY -u analyze_denoise.py > $L/r7_denoise_report.log 2>&1); tail -3 $L/r7_denoise_report.log | tee -a $S
cd $C
./run_pipeline.sh learn preprocess >> $S 2>&1 || finish "FAILED preprocess"
( ./run_pipeline.sh block_test > $L/r7_block_test.log 2>&1 ) & BT=$!
./run_pipeline.sh block_train >> $S 2>&1 || finish "FAILED block_train"
rm -rf $B/dataset/processed/candidates_train_preprune
cp -r $B/dataset/processed/candidates_train $B/dataset/processed/candidates_train_preprune
wait $BT || finish "FAILED block_test"; cat $L/r7_block_test.log >> $S
./run_pipeline.sh prune_train >> $S 2>&1 || finish "FAILED prune_train"
( ./run_pipeline.sh prune_test > $L/r7_prune_test.log 2>&1 ) & PT=$!
./run_pipeline.sh match_train >> $S 2>&1 || finish "FAILED match_train"
wait $PT || finish "FAILED prune_test"; cat $L/r7_prune_test.log >> $S
backup
./run_pipeline.sh predict >> $S 2>&1 || finish "FAILED predict"
cd $C/src && $PY -u package_submission.py --out ~/submission7.zip >> $S 2>&1 \
  && up ~/submission7.zip "$URL_sub7" || log "PACKAGE/VALIDATION FAILED"
grep -aE "per S1" $L/prune_test.log | tail -1 | tee -a $S
log "evaluate on the LOCKED HOLDOUT"
$PY -u matching.py evaluate --population holdout --train-frac 0.0 \
  --blocked-dir ../../../dataset/processed/candidates_train_preprune > $L/r7_holdout_eval.log 2>&1
grep -aA3 "^slice " $L/r7_holdout_eval.log | tee -a $S
$PY -u analyze_missed.py > $L/r7_missed.log 2>&1
finish "OK"
