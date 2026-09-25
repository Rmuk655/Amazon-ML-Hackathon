#!/usr/bin/env bash
# Run pipeline steps in order, each memory-capped (laptop-safe) and logged to logs/<step>.log.
# Stops at the first failing step. Usage: ./run_pipeline.sh step1 step2 ...   (no args = all)
# Launch detached so it survives the terminal:  setsid nohup ./run_pipeline.sh > logs/pipeline.log 2>&1 &
set -u
cd "$(dirname "$0")/src"
PY=${PY:-../../../venv/bin/python}
CAP=${MEM_CAP:-11G}
mkdir -p ../logs

declare -A CMD=(
  [learn]="learn_suffixes.py"
  [preprocess]="preprocess.py --splits train test --backend indicxlit --cache-only --overwrite"
  [sanity]="sanity_check.py"
  [block_dev]="blocking.py --split train --eval --limit-s1 50000"
  [block_train]="blocking.py --split train --eval"
  [prune_train]="prune.py fit"
  [prune_test]="prune.py apply --split test"
  [match_dev]="matching.py train --s1-frac 0.05"
  [match_train]="matching.py train --s1-frac ${S1_FRAC:-0.3}"
  [block_test]="blocking.py --split test"
  [predict]="matching.py predict"
  [rob_slices]="robutness.py slices"
  [rob_holdout]="robutness.py holdout"
  [rob_testshift]="robutness.py testshift"
  [package]="package_submission.py"
)
ORDER=(learn preprocess sanity block_dev block_train prune_train match_dev match_train block_test prune_test predict
       rob_slices rob_holdout rob_testshift package)
STEPS=("${@:-${ORDER[@]}}")

for s in "${STEPS[@]}"; do
  echo "[$(date +%T)] START $s: ${CMD[$s]}"
  WRAP=()   # memory cap on the laptop; SageMaker has no systemd -> MEM_CAP=none runs uncapped
  if [ "$CAP" != none ] && command -v systemd-run >/dev/null; then
    WRAP=(systemd-run --user --scope -q -p MemoryMax="$CAP" -p MemorySwapMax=0)
  fi
  TIME=(); [ -x /usr/bin/time ] && TIME=(/usr/bin/time -v)
  "${WRAP[@]}" "${TIME[@]}" $PY -u ${CMD[$s]} > "../logs/$s.log" 2>&1
  rc=$?
  peak=$(grep -oP "Maximum resident set size \(kbytes\): \K\d+" "../logs/$s.log")
  echo "[$(date +%T)] END   $s rc=$rc peak=$((${peak:-0} / 1024))MB"
  [ $rc -ne 0 ] && { echo "STOPPED at $s (see logs/$s.log)"; exit $rc; }
done
echo "[$(date +%T)] ALL DONE"
