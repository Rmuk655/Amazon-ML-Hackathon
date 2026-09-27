#!/usr/bin/env bash
# Fast, labelled experiment loop on one country slice of TRAIN (laptop-sized, ~20 min).
#   ./dev_slice.sh                      India, 5 states, score the saved model on held-out S1s
#   TRAIN=1 ./dev_slice.sh              also retrain (3 folds, on the other half) before scoring
#   SLICE=us ./dev_slice.sh             US slice (NC, OH, IL)
#   NAME=myexp ./dev_slice.sh           separate working dir per experiment (dataset/dev/<NAME>)
# Held-out S1s = hash >= 0.5 (never used for training), same as `matching.py evaluate`.
# Everything goes to dataset/dev/<NAME> and models/dev/<NAME>; the real data and models are untouched.
# Steps whose inputs did not change are skipped (preprocess/blocking reuse; features via the feature store).
set -eu
cd "$(dirname "$0")/src"
PY=${PY:-../../../venv/bin/python}
SLICE=${SLICE:-india}
NAME=${NAME:-$SLICE}
TRAIN=${TRAIN:-0}
case $SLICE in
  india) COUNTRY=India; CNORM=india; TOKENS="maharashtra karnataka gujarat telangana kerala" ;;
  us)    COUNTRY=US;    CNORM=us;    TOKENS="nc oh il" ;;
  *) echo "SLICE must be india or us"; exit 1 ;;
esac
ROOT=$(cd ../../.. && pwd)
export BER_PROCESSED_DIR=$ROOT/dataset/dev/$NAME
export BER_MODELS_DIR=$ROOT/models/dev/$NAME
export BER_ONLY_COUNTRY=$COUNTRY
mkdir -p "$BER_PROCESSED_DIR" "$BER_MODELS_DIR" ../logs
for f in learned_tokens.tsv name_vocab.tsv translit_vocab_indicxlit.tsv; do
  [ -e "$BER_PROCESSED_DIR/$f" ] || cp "$ROOT/dataset/processed/$f" "$BER_PROCESSED_DIR/"
done
for f in pruner.joblib stage4_model.joblib; do       # start from the saved models
  [ -e "$BER_MODELS_DIR/$f" ] || cp "$ROOT/models/$f" "$BER_MODELS_DIR/"
done
L=../logs/dev_$NAME
run() {  # name, args...: skipped when stages.py says its inputs are unchanged
  local n=$1; shift
  if [ "${FORCE:-0}" != 1 ] && [ -n "${STAGE:-}" ] && $PY stages.py fresh "$STAGE" "$*" 2>/dev/null; then
    echo "[$(date +%T)] SKIP  $n (up to date)"; return 0; fi
  echo "[$(date +%T)] START $n"
  /usr/bin/time -v $PY -u "$@" > "$L.$n.log" 2>&1 || { echo "FAILED $n (see $L.$n.log)"; exit 1; }
  echo "[$(date +%T)] END   $n peak=$(( $(grep -oP 'Maximum resident set size \(kbytes\): \K\d+' "$L.$n.log") / 1024 ))MB"
  [ -n "${STAGE:-}" ] && $PY stages.py stamp "$STAGE" "$*"; return 0
}
STAGE=preprocess_train run preprocess preprocess.py --splits train --backend indicxlit --cache-only --overwrite --jobs 2
STAGE=block_dev run block blocking.py --split train --eval --k "${K:-30}" --countries $CNORM --s1-address-tokens $TOKENS
STAGE=prune_train run prune prune.py apply --split train --no-tsv
if [ "$TRAIN" = 1 ]; then
  STAGE=match_dev run train matching.py train --s1-frac 0.5 --folds "${FOLDS:-3}"
fi
STAGE= run evaluate matching.py evaluate --train-frac 0.5
grep -aA2 "^slice " "$L.evaluate.log"
echo "full report: $L.evaluate.log"
