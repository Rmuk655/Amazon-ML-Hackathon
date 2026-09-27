# Business Entity Resolution: v2 pipeline

Matches every Source-1 business to its records in Sources 2/3 and writes `output/matching_results.tsv` and
`output/candidate_pairs.tsv`. All code is in `src/v2/`. It uses only the provided train/test files: no external data,
no pretrained models, and CPU only.

## Setup
```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt     # Python 3.12
```
The data is expected at `dataset/train/*.tsv` and `dataset/test/*.tsv`, in the `student_resource` root next to `code/`.

## Run end to end
```bash
cd code/business_entity_resolution
PYTHON=../../venv/bin/python BER_WORKERS=8 ./run_v2.sh          # about 4 h on 8 vCPU / 64 GB
```
Steps, which can be resumed with `FROM=<step> ./run_v2.sh`:

| Step | Command | Output (in `work_v2/`) |
|---|---|---|
| prep | `python -m v2.prep raw --split train\|test` | raw parquet, truth pairs |
| norm | `python -m v2.run norm --split train\|test` | `lexicon.pkl` (learned from train truth), normalised records |
| block | `python -m v2.run block --split train\|test` | candidate pairs from 5 channels (+ recall on train) |
| prune | `python -m v2.run prune --split train\|test` | at most 6 S1 candidates per record (cross-fitted LightGBM) |
| feats | `python -m v2.run feats --split train\|test` | about 55 pair features |
| train | `python -m v2.run train` | pass-1 + pass-2 matchers (3-fold OOF), thresholds, locked-holdout report |
| predict | `python -m v2.run predict` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |
| validate | `python utils/validate_submission.py ...` | PASS / issues |

`python -m v2.run <step> --dev` runs the same steps on a 10% slice of train (`python -m v2.prep slice` first).

## Key environment variables
`BER_WORKERS` (threads), `BER_TRAIN_FRAC` (matcher training subsample, default 0.35), `BER_K_NAME/ADDR/COMB/PAIR/KEY`
(blocking top-K per channel), `BER_CAP_SMALL/BIG` (document-frequency caps), `BER_FPW` (threshold tuned for 1x or
2x decoy density, default 2.0), `BER_T` (explicit threshold override).

## Files
- `src/v2/common.py`: paths, integer id encoding, TSV I/O, the locked 10% holdout of train S1s
- `src/v2/prep.py`: TSV to parquet, dev slice
- `src/v2/normalize.py`: normalisation and the learned lexicon (Indic→Latin tokens, state/abbreviation maps)
- `src/v2/block.py`: record-centric blocking (TF-IDF channels, composite name×address channel, same-name key channel)
- `src/v2/features.py`: pruner features, pair features, pass-2 cluster features
- `src/v2/model.py`: LightGBM helpers, pruning, exclusive assignment, macro F0.5
- `src/v2/run.py`: stage driver
- The older v1 pipeline is kept in `src/*.py` (see `README_v1.md`); it is not used by v2.
