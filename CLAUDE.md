# Amazon ML Challenge 2026 – Business Entity Resolution

Match every Source-1 (clean) business to its records in the noisy Sources 2/3. Metric: macro F0.5 per S1
(a singleton with an empty prediction scores 1). Submission zip = `output/matching_results.tsv` +
`output/candidate_pairs.tsv` + code + `Documentation_template.md`; smaller candidate sets count in the ranking.
Deadline **Mon 28 Sep 2026 00:30 IST**; freeze (packaging / validation / docs only) from **Sun 20:30**.
Owner: Krishnan (GitHub `Rmuk655`, branch `baseline-pipeline`). The S2/S3 records and decoys come from a synthetic
generator: reverse-engineering its noise/decoy rules is the main lever.

## Scores (leaderboard ≈ locked-holdout / OOF macro F0.5 − ~0.015)
| Submission | What | Held-out | Leaderboard |
|---|---|---|---|
| submission1 | first AWS run (unidecode) | OOF 0.9313 | 0.916 |
| submission3_noqwen | IndicXlit + spelling snap, reverse blocking, 17 features | OOF 0.9541 | 0.939 |
| **submission4** | + name cleaners, address key (run-3 model, no retrain) | India slice 0.9423 | **0.943 (best)** |
| run 7 (submission7) | rules + subset keys + token repair + decoy feats, 90 % training | OOF 0.9742 | pending |
Files: `submissions/` (zips, `results*.tsv`, `models_run5p/`, `tracking/`, `eda/`, `emb/`).

## Pipeline (code/business_entity_resolution/src)
learn_suffixes → preprocess (normalize, IndicXlit cache, spelling snap, name cleaners, token repair) → blocking
(keys + TF-IDF + reverse top-K + bypass) → [merge_embed] → prune (LightGBM, ≤5 cand/S1) → matching train
(LightGBM, 3 folds, calibration, thresholds; optional gated ensemble) → predict → package (official validator).
`run_pipeline.sh <steps>` runs steps with stage skipping (`stages.py`, `FORCE=1` reruns). `matching.py evaluate
--population holdout --train-frac 0.0` = locked-holdout report (confusion by source × address, loss decomposition).
`matching.py retune / decide` re-tune / rewrite from saved scores (no retrain). Analysis only (excluded from the zip):
analyze_failures.py, analyze_missed.py, analyze_denoise.py.

## Env flags (all default ON unless noted)
| Flag | Effect |
|---|---|
| BER_NAME_CLEAN | strip aliases, www/com, titles; segment glued words |
| BER_TOKEN_REPAIR | noisy-channel repair of non-vocabulary tokens (name_clean.TokenRepair) |
| BER_NEW_KEYS | sorted-token, acronym, 3-word / whole-name glue keys |
| BER_ADDR_KEY | address-only key (house number + street word) |
| BER_SUBSET_KEYS | word / word-pair keys anchored by house number or street (dropped/added words) |
| BER_BYPASS | exact core name + address key survives the top-K cut |
| BER_EMBED_CANDS | (run8, default 1 there) merge Kaggle bi-encoder neighbours before pruning |
| BER_EXTRA_FEATS / BER_DECOY_FEATS | 17 similarity features / decoy-difference features |
| BER_ENSEMBLE (off) | LightGBM + importance-weighted Extra-Trees + approx-RBF, stacked, gated on the holdout (no gain so far) |
| S1_FRAC, FOLDS | training S1 fraction (1.0 = all non-holdout), CV folds |
| BER_K, BER_K_REV | candidates per S1 per source (30), reverse top-K per target (3) |
| BER_PRUNE_MAX_CANDS, BER_PRUNE_SPLIT_BUDGET | pruner candidate budget on train OOF / on each split (5) |
| BER_LR, BER_MODELS_DIR, BER_PROCESSED_DIR, BER_FEATURE_CACHE, BER_ONLY_COUNTRY | lr 0.1, model / data dirs, feature store, dev filter |

## Locked holdout (rule)
`config.is_holdout`: fixed 10 % of train S1 (hash key `lockedholdout001`), excluded from rule inference, pruner fit,
training and tuning. **Only upload a submission whose locked-holdout macro F0.5 beats the current best
(submission4 → needs ≥ ~0.960).** Held-out gains have carried over to the leaderboard 1:1 so far.

## Infrastructure
- AWS us-east-1, IAM user `claude-cli` (cannot request quotas; 8 vCPU on-demand = one r5.2xlarge at a time, GPU 0).
  Instance **i-0076c2fcc9a8df3ab** (`claude-ber-run3`, key `~/.ssh/claude-ber.pem`, IP changes on restart); stopped
  instance i-04c2898da3e86ea2b (old). Code dirs on the instance: `~/ber` (data + run ≤5), `~/ber7`, `~/ber8`
  (symlink `dataset`, `utils`, `venv` to `~/ber`). Runs are detached, write `~/runN.pid`, log
  `~/berN/code/business_entity_resolution/logs/runN.log`, upload via presigned URLs (`~/run5_urls.env`), stop the
  instance at the end (cost guard via `shutdown -h +N`). Never terminate; never put AWS keys on the instance.
- S3 `s3://sagemaker-us-east-1-134051031272/ber/`: run-2026-09-25, rescore-2026-09-26, run3-2026-09-26,
  run4-2026-09-26, run5p-2026-09-26 (eda_results.tgz, run5p_logs.tgz), run7-2026-09-26, run8-2026-09-27.
- Kaggle (`venv/bin/kaggle`, token `~/.kaggle/access_token`, user `rmuk16`, max 2 GPU sessions, ~30 GPU-h/week):
  dataset `rmuk16/ber-texts` (private); kernels `rmuk16/ber-biencoder` (all splits, running since Sat 12:40, 12 h cap)
  and backups `rmuk16/ber-emb-{test,train}-{india,us,france}` (code in `code/business_entity_resolution/kaggle/`).
  Queue driver `code/business_entity_resolution/kaggle/queue.sh` (detached): PID `~/kaggle_queue.pid`, log
  `~/kaggle_queue.log`, outputs `submissions/emb/<kernel>/nn_<split>_<country>_S<n>.parquet`.
- Run-8 launcher `code/business_entity_resolution/launch_run8.sh` (detached, laptop): PID `~/launch_run8.pid`, log
  `~/launch_run8.log`; waits for run 7 + both splits' embeddings, then starts run 8; gives up at Sun 11:00.
- **Status: `~/Amazon-ML-Hackathon/status.sh`** (instances, runs, Kaggle, queue, launcher, S3 and local submissions).

## State (Sun 00:20)
- Run 7 on the instance: predict running → submission7.zip ~01:40, locked-holdout eval ~02:10, then stop.
  Concern: test pruning needed threshold 0.130 (train 0.019) to hit 5 cand/S1 – may cost test recall.
- Kaggle: original job + backup test/india both slow (> 4.5 h); outputs not landed yet.
- Done: generator analysis (holdout: 65 % fully explained, 32 % up to dropped/added words, 3.2 % address-only,
  0.04 % unrecoverable; decoys = house number changed + legal form added), denoiser, subset keys, pruner holdout fix.
- Left: rule catalogue write-up, test self-training (needs rules confirmation), address-token repair, final
  100 % retrain (only if it fits before freeze), Documentation_template.md numbers.

## Safety rules
Nothing heavy on the laptop (15 GB RAM; tiny unit tests only). Held-out numbers for every claim; label estimates.
Skipped items in CAPS. Use PID files / exact `ps -p`, never `pkill -f` (it has killed our own shell). Confirm before
cost-incurring or outward-facing actions (new instances, uploads outside S3, Kaggle data). On "git push": just push.
