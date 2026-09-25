# Business Entity Resolution

## Stage map
| Stage | What | Code |
|---|---|---|
| 1 | EDA -> design decisions + findings note | `src/eda.py` |
| 2 | Normalization: raw -> unicode -> lowercase -> script -> transliteration (c2, c1, c3, c4a/c4b done; c5 address parts and phonetic key pending) | `src/text_utils.py`, `canonical.py`, `lang_id.py`, `transliterate.py`, `preprocess.py`, `sanity_check.py` |
| 3 | Blocking -> `output/candidate_pairs.tsv`, recall@k | `src/blocking.py` |
| 4 | Pair features + LightGBM, calibrated probabilities | _todo_ |
| 5 | Decision layer: assignment, thresholds -> `output/matching_results.tsv` | _todo_ |
| 6 | Robustness (cross-country), validator, docs | _todo_ |

## Stage 1 - EDA
```bash
cd src
python eda.py --limit 200000     # dev
python eda.py                    # full; writes dataset/processed/eda/eda_findings.md
```

## Stage 2 — normalization + transliteration

```bash
pip install -r requirements.txt
cd src
# optional: unzip IndicLID FTN model (indiclid-ftn.zip from the AI4Bharat/IndicLID releases) into ../models/indiclid-ftn/

# 1) dev run (no heavy models)
python preprocess.py --splits train --backend unidecode --limit 200000

# 2) real run, Windows + Colab/WSL split
python preprocess.py --splits train test --backend indicxlit --dump-vocab     # local: writes dataset/processed/pending_vocab.tsv
python transliterate.py --fill pending_vocab.tsv --backend indicxlit           # Colab/WSL/Kaggle (fairseq): builds translit_vocab_indicxlit.tsv
#   copy translit_vocab_indicxlit.tsv back to dataset/processed/
python preprocess.py --splits train test --backend indicxlit                   # local: cache lookups only (add --cache-only to forbid engine use)

# 3) does romanization raise overlap on true matches? (name + address)
python sanity_check.py
```

Outputs (`dataset/processed/*.parquet`, atomic writes; existing files skipped unless `--overwrite`):
raw columns kept, plus `business_name_{norm,rom,script,lang}` and `business_address_{norm,rom,script,lang}`, `country_norm`.


## Stage 3 - blocking
```bash
cd src
python blocking.py --split train --eval --limit-s1 50000   # dev: recall report vs ground truth
python blocking.py --split train --eval                    # full; writes dataset/processed/candidates_train/*.parquet (with is_true labels)
python blocking.py --split test --write-tsv                # writes candidates_test/ + output/candidate_pairs.tsv
```
Memory control: `--countries india` (one partition at a time), `--k`. Tunables (key caps, K, chunk sizes, within-country) are in `config.py`.
Read the printed recall report and `dataset/processed/eda/missed_true_pairs.tsv` before tuning.