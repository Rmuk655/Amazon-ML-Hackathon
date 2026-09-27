# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** Krishnan
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
Every S2/S3 record belongs to at most one S1 entity (in train, 0 of 7.64M matched ids appear in two lists). We
therefore resolve entities **record by record**. Each S2/S3 record retrieves its most likely S1 businesses. A LightGBM
model scores these candidates, and the record is assigned to its single best S1 if the model is confident enough. The
per-S1 match lists are the union of these assignments. A second, **collective** pass rescores each candidate against
the other records tentatively assigned to the same S1. Finally, a **fine-tuned multilingual cross-encoder** reads the raw
name and address text of every candidate pair, and a stacked LightGBM combines its score with the pipeline's (Section 4.3).
The pipeline uses only the provided data (no external lookups). The one pretrained component is the cross-encoder's
starting checkpoint `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0), fine-tuned on our train pairs on
2x T4 GPUs (Kaggle). Everything else is LightGBM trained from scratch on CPU.

Locked-holdout macro F0.5 (10% of train S1 entities, 220,656 S1s, never used for any fitting or tuning): **0.9890**
(pair precision 0.9985, recall 0.970), up from 0.9851 for the pipeline without the cross-encoder. It is 0.9886 when
decoy false positives are doubled to match test density. Leaderboard: 0.975 without the cross-encoder, **0.983** with it.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Exclusivity:** no S2/S3 id is in two S1 lists. This turns matching into an assignment problem: pick one S1 or none per
  record.
- **Regular generator:** train has 3.46 matches per S1 and 5.6% singletons, identical in US and India.
- **Test is harder:** it has about 5.7 S2/S3 records per S1 against 4.67 in train. That means roughly twice as many
  unmatched records (decoys or orphans), so thresholds are tuned under simulated double decoy density (Section 4).
- **Noise catalogue** (measured on true pairs): words added or dropped (25%/23%), house-number format or value changes
  (21%), legal-form dropped, added or changed, Indic-script transliteration (18% of India pairs), website form
  (`name.com`), glued words, alias prefixes (`X DBA Y`, `aka`, `doing business as`), replaced names (3%), and missing
  addresses (4%).
- **Decoys:** unmatched records near an S1 usually have a changed house number plus an added legal form.
- **France** appears only in test. No step one-hot encodes or filters on country; France gets French street-type
  abbreviations (rue/avenue/boulevard/impasse…) and otherwise the same features.

### 2.2 Solution Strategy
**Approach type:** record-centric blocking → LightGBM pruner → LightGBM matcher (two passes) → exclusive assignment.
**Core ideas:** (1) exclusive per-record assignment; (2) composite name×address and same-name blocking keys that keep
common business names resolvable; (3) collective pass-2 features from the tentative clusters; (4) a cross-encoder that
reads the raw text of both records, stacked with the pipeline score; (5) thresholds tuned directly on macro F0.5 under
test-like decoy density.

---

## 3. Candidate Generation (Blocking)
**Normalisation** (`normalize.py`) with a lexicon **learned from train truth pairs** (holdout excluded):
- Indic→Latin token dictionary, built by position-aligning equal-length names. The Indic vocabulary is closed (~1.3k tokens).
- Address component map (state abbreviations and regional-script state names → canonical S1 form).
- Address token map (`st→street`, `ave→avenue`, frequent typos).
- Also: NFKC, mojibake/HTML repair, accent folding, placeholder removal (`null`, `N/A`), `&`→and, joining single-letter
  runs (`L.L.C`→`llc`), ordinal and leading-zero stripping, alias-prefix and website unwrapping.
- Names split into core / legal form / glued form. Addresses split into word tokens / house numbers / postcodes.

**Channels** (within country; every S2/S3 record queries the S1 index):

| Channel | Features (TF-IDF, binary tf) | top-K |
|---|---|---|
| name | core words + char 3-grams of the glued name | 10 |
| addr | address words, house numbers, first/last 3 digits of numbers, postcodes | 8 |
| comb | name words + address features | 10 |
| pair | composite `name word × address item` | 10 |
| key | all S1 sharing the glued or sorted-token core name, ranked by address cosine | 15 |

Features whose S1 document frequency exceeds a cap are dropped, which keeps the sparse products linear. The composite
and same-name channels exist because at full scale common names (shared by 30–100 S1s) tie on the name channel, and
their words exceed the frequency cap.

- **Blocking output (train):** 334M candidate pairs, about 32 per record. Blocking recall is at least the post-pruning recall below.
- **Pruner:** a small cross-fitted LightGBM on the channel cosines, ranks, margins, house-number and glued-name equality
  keeps at most 6 S1 per record (p ≥ 0.002). Recall after pruning: **0.9813** (holdout 0.9813), with 1.39 candidates per record on train and 1.44 on test (8.27 per S1).
**`candidate_pairs.tsv`** keeps, for each S1, its 5 highest-scoring candidates by the final stacked score, plus every
predicted match if there are more than 5. That is 4.97 per S1, and every submitted match is inside the candidate set.

---

## 4. Matching Model
**Pair features (~55):**
- Name: ratio, token-set, token-sort, partial ratio, Jaro-Winkler and Levenshtein on the glued name, containment,
  token Jaccard, IDF-weighted overlap and coverage, token counts.
- Legal form: equal / added / dropped / conflicting.
- Address: token-set/ratio/partial, Jaccard, IDF-weighted overlap.
- House numbers: first-number equality, record number contained in S1 numbers, Jaccard, absolute/relative difference.
- Postcode equality, missing-address flags, Indic-script flag.
- Name frequency among S1 (glued name, first word).
- Blocking and pruner scores, ranks and margins (competition among the record's candidates, and how many records
  rank the S1 first).

**Pass 2 (collective):** from pass-1 out-of-fold scores, each record's tentative S1 cluster is formed. For every candidate we
add its margin to the next-best S1, cluster size, same-source members, mean member score, and best/mean name and
address similarity to the members. We also add house-number and legal-form agreement with the members. This recovers
replaced names and missing addresses, and rejects decoys that disagree with the cluster.

**Model type:** LightGBM (binary), 3-fold out-of-fold grouped by the record's true S1. Holdout and test use the average of
the fold models. Training rows are subsampled to 35% of groups for speed.

**Decision:** each record goes to its argmax S1 if p ≥ t. `t` is tuned on out-of-fold macro F0.5 (singletons included),
with false positives from unmatched records weighted ×2 to mimic the test decoy density.

### 4.3 Cross-encoder reranker and stack (`code/business_entity_resolution/neural/`)
- **Why:** the LightGBM features describe a pair through hand-built similarities. A cross-encoder reads both records' raw
  "name | address" strings together, so it can pick up noise patterns the features miss. Its XLM-R vocabulary also covers
  Indic scripts and French.
- **Model:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0, 12 layers, 384 wide), one output logit.
  Fine-tuned with binary cross-entropy for 1 epoch on 1.6M candidate pairs (fp16, lr 3e-5, batch 128, max 96 tokens).
- **Leakage control:** non-holdout S1 groups are split into 2 folds (hash of the record's true S1). Fold k's model scores
  the other fold's pairs out-of-fold. Fold 0's model scores the locked holdout and test. The holdout is never trained on.
- **Stack:** LightGBM on the pipeline's pass-2 logit and the cross-encoder logit. For both, it adds the pair's rank and its
  margin to the best competitor within the record and within the S1, plus candidate counts. Trained 3-fold out-of-fold on
  non-holdout pairs; threshold 0.675 tuned on out-of-fold macro F0.5. Assignment is the same exclusive rule as before.
- **Alone vs together:** the cross-encoder alone is weaker (pair AUC 0.9991 vs 0.9996), but its errors differ, so the stack
  gains +0.0038 holdout macro F0.5. That is +0.002 precision, +0.003 recall, and +0.009 on singletons.
- **Decision checks** (all tuned out-of-fold, reported once on the holdout): per-(country, source) thresholds, an
  ambiguity margin, and cross-source rescue gave no gain or hurt. An empty-S1 rescue (t=0.5) plus the generator caps
  (≤5 S2, ≤6 S3 per S1) gave +0.0001. France has no labels; a stricter France threshold (0.85) left the leaderboard
  unchanged at 0.983.

---

## 5. Results & Error Analysis
| Stage | Metric | Value |
|---|---|---|
| Blocking + pruning | pair recall (train) | 0.9813 |
| Pass 1 | locked-holdout macro F0.5 (P / R) | 0.9810 (0.995 / 0.955) |
| Pass 2 | locked-holdout macro F0.5 (P / R) | 0.9851 (0.997 / 0.965) |
| Pass 2 + cross-encoder stack | locked-holdout macro F0.5 (P / R) | **0.9889** (0.9985 / 0.970) |
| Old v1 pipeline (run 7) | locked-holdout macro F0.5 | about 0.982 (India 0.978, US 0.985) |

Test sanity checks per country (matches per S1 / empty S1 share), against train truth 3.46 / 0.056:
France 3.45 / 0.054, India 3.38 / 0.059, US 3.41 / 0.058. France, which is unseen in training, behaves like the train countries.

Where the remaining holdout recall goes (as a share of true pairs): 1.87% never reach the matcher (blocking or pruning); 1.02% are the record's best S1 but fall below the threshold; 0.59% lose to another S1.

- **Remaining false negatives:** records with no address whose name is shared by many S1s (no signal can separate
  them); heavy name noise combined with a truncated address.
- **Remaining false positives:** decoys that keep the house number and differ only by an added word or legal form.

---

## 6. Conclusion
The decisive steps were recognising the exclusive assignment structure and measuring blocking at full scale. A 10%
slice showed 0.997 recall, but full data dropped it to 0.95 because common names tie and their words exceed the
frequency caps. Composite name×address and same-name keys restored it. Collective pass-2 features then added
precision and recall on top of a strong pairwise model.

---

## Appendix
### A. Code Artefacts
`code/business_entity_resolution/`: `src/v2/` (pipeline), `run_v2.sh` (end to end), `README.md`, `requirements.txt`.
`neural/`: `build_xe_pkg.py` (exports scored pairs and texts), `xe_kernel.py` (Kaggle fine-tuning and scoring, one run per fold),
`stack.py` (stack plus holdout report), `final.py` (decision rules), `trim_cands.py` (≤5 candidates per S1), `README.md`.
`./run_v2.sh` runs prep → norm → block → prune → feats → train → test → predict → validate and writes both output files.
The earlier v1 pipeline (`src/*.py`, `README_v1.md`) is kept for reference and is not used.

### B. Additional Results
- 10% train slice: blocking recall 0.9971; pass 1 holdout 0.9903; pass 2 holdout 0.9920 (0.9914 at ×2 decoy density).
- Top pass-1 features: pruner score, combined-channel cosine, house-number difference, glued Levenshtein, legal-form added.
