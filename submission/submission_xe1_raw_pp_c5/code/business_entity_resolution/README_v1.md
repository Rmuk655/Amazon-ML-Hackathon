# Business Entity Resolution

Match every Source-1 (reference) business to its records in the noisy Sources 2/3.
Metric: macro F0.5 (precision-weighted). Output: `output/matching_results.tsv` + `output/candidate_pairs.tsv`.

## Pipeline
| Stage | What | Code |
|---|---|---|
| 1 | EDA -> design decisions (`eda/eda_findings.md`) | `src/eda.py` |
| 2 | Normalization: NFKC -> casefold -> script -> transliteration -> canonical name (`c4a`/`c4b`/`legal`); street-type abbreviations unified; legal forms/stopwords learned from the data | `src/text_utils.py`, `canonical.py`, `learn_suffixes.py`, `lang_id.py`, `transliterate.py`, `preprocess.py`, `sanity_check.py` |
| 3 | Blocking, within country: 8 key types (core name, tokens, consonant-skeleton bigrams, prefixes, postal code, name+address, address bigrams, glued tokens) + char n-gram TF-IDF channel; ambiguity context (forward/reverse rank, margins) | `src/blocking.py` |
| 3b | Supervised meta-blocking: small LightGBM on blocking signals + 3 cheap comparisons (core-name / address token-set, house number) prunes candidates, threshold chosen out-of-fold to keep 99.5% of found true pairs (dev slice: 57.6 -> 5.6 candidates per S1) | `src/prune.py` |
| 4 | ~40 pair features (name, address, blocking context, chain-name frequency) -> LightGBM, 5-fold grouped CV, isotonic calibration | `src/matching.py train` |
| 5 | Decision layer: thresholds per source x address-missing, one S1 per target, margin filter, optional top-1 fallback - all tuned on out-of-fold macro F0.5 | `src/matching.py predict` |
| 6 | Robustness (weak slices, leave-country-out, test shift), official validator, zip | `src/robutness.py`, `src/package_submission.py` |

## Run
`run_pipeline.sh` runs the steps in order, logs each to `logs/<step>.log` with peak memory, and stops at the first failure.

```bash
pip install -r requirements.txt
./run_pipeline.sh                                  # everything
./run_pipeline.sh block_test predict package       # or selected steps
```

| Step | Command | Notes |
|---|---|---|
| `learn` | `learn_suffixes.py` | legal forms / stopwords from all names (no labels) -> `dataset/processed/learned_tokens.tsv` |
| `preprocess` | `preprocess.py --splits train test --backend indicxlit --cache-only --overwrite` | uses the IndicXlit vocab cache; tokens missing from it fall back to unidecode |
| `sanity` | `sanity_check.py` | token overlap on true pairs before/after romanization |
| `block_dev` | `blocking.py --split train --eval --limit-s1 50000` | recall report on a slice |
| `block_train` | `blocking.py --split train --eval` | labelled candidates for training |
| `match_dev` / `match_train` | `matching.py train --s1-frac 0.05` / `0.3` | `S1_FRAC` env var overrides the full-run sample |
| `prune_train` | `prune.py fit` | fits the pruner on train candidates and prunes them |
| `block_test` / `prune_test` | `blocking.py --split test` / `prune.py apply --split test` | pruned test candidates -> `output/candidate_pairs.tsv` |
| `predict` | `matching.py predict` | writes `output/matching_results.tsv` (a row for every test S1) |
| `rob_*` | `robutness.py slices / holdout / testshift` | |
| `package` | `package_submission.py` | runs `utils/validate_submission.py`; refuses to zip on failure |

### Cheap iteration (no retraining)
`match_train` saves out-of-fold predictions and `predict` saves every test pair scored >= 0.05, so decision
experiments take minutes instead of a full run:
```bash
cd src
python matching.py retune --fp-weight 1.0 [--save]   # re-tune thresholds/rules on out-of-fold predictions
python matching.py decide [--thr-scale 0.95]        # rewrite output/matching_results.tsv from saved test scores
```

### Faster iteration
- **Stage skipping:** `run_pipeline.sh` skips a step whose code, settings, command and inputs are unchanged since its
  last success (`src/stages.py`, stamps in `dataset/processed/.stamps/`). `FORCE=1` reruns everything.
- **Feature store:** matching features are cached per candidate file in `dataset/processed/features_<split>/`, in
  groups (`base`, `amb`, `extra`). A group is recomputed only when its code version (`FEATURE_GROUPS` in
  `matching.py`) or its inputs change; a new group is computed alone and appended. `BER_FEATURE_CACHE=0` disables it.
- **Labelled dev loop on the laptop** (~20 min, real data and models untouched):
  ```bash
  ./dev_slice.sh                 # India, 5 states: score the saved model on held-out S1s
  TRAIN=1 ./dev_slice.sh         # retrain on the other half first (3 folds)
  SLICE=us NAME=exp1 ./dev_slice.sh
  ```
- **Laptop + AWS:** `./remote.sh start | push | run "<steps>" | status | pull <paths> | stop` runs memory-heavy steps
  on the EC2 instance and moves results through S3.

Environment variables: `MEM_CAP` (default `11G`, memory cap via systemd; `none` disables it, e.g. on SageMaker),
`PY` (python executable), `S1_FRAC`, `FORCE`, `BER_LR` (matcher learning rate, default 0.1), `BER_MODELS_DIR`
(where pruner/matcher models are read and written), `BER_PROCESSED_DIR`, `BER_EXTRA_FEATS` (0 disables the extra
similarity features), `BER_FEATURE_CACHE`, `BER_ONLY_COUNTRY` (preprocess one country; dev runs).

### Transliteration (IndicXlit)
Only ~1.5k distinct Indic tokens exist, so transliteration is done once at vocabulary level:
```bash
cd src
python preprocess.py --splits train test --backend indicxlit --dump-vocab   # -> dataset/processed/pending_vocab.tsv
python make_colab_notebook.py                                               # -> colab/translit_colab.ipynb (vocab embedded)
```
Run the notebook on Colab (Run all), put the downloaded `translit_vocab_indicxlit.tsv` in `dataset/processed/`,
then run the pipeline from `preprocess`.

### SageMaker
All steps are CPU (LightGBM, scipy); no GPU is needed. Measured peaks on a 15 GB laptop:
preprocess 10.2 GB; blocking of one country partition (India, 4.1M targets) 9.5 GB; 5-fold training on 350k pairs 2.1 GB.
The US partition has 1.5x the targets and full training at `S1_FRAC=0.3` is ~40M pairs, so use a
64-128 GB instance (e.g. ml.r5.2xlarge / ml.r5.4xlarge):
```bash
MEM_CAP=none PY=python ./run_pipeline.sh learn preprocess block_train prune_train match_train block_test prune_test predict package
```
Copy `dataset/` (raw TSVs + `processed/translit_vocab_indicxlit.tsv`) and `utils/` next to `code/` first.

## Key design facts (from EDA, 1M-row sample)
- 0% cross-country matches -> block within country. Test adds France (absent from train): legal forms are learned from test names too, and address abbreviations (`r.`/`rue`, `av`/`avenue`) are unified.
- No target belongs to two S1 entities -> one-S1-per-target rule.
- 7.1% of true pairs are cross-script (Latin S1 vs Indic target) -> vocabulary-level IndicXlit.
- ~40% of S1 rows share their core name with another entity -> name+address keys in blocking and ambiguity features (reverse rank/margin, core-name frequency) in matching.
