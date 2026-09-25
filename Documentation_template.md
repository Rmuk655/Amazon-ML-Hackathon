# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** Krishnan  
**Submission Date:** [Date]

---

## 1. Executive Summary
We resolve each Source-1 business against the noisy Sources 2/3 with a multi-key blocking stage (name, consonant-skeleton, name+address, address and glued-token keys plus a char n-gram TF-IDF channel), a LightGBM pair classifier with explicit chain-name ambiguity features, and a precision-oriented decision layer tuned directly on macro F0.5. Two ideas carry most of the gain: address-anchored blocking keys that keep chain names ("Global Trust", shared by hundreds of entities) reachable, and reverse-direction features that tell the model how many S1 entities compete for the same record.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on a 1M-row-per-source sample (`src/eda.py`):
- **Country:** 0.00% of true pairs cross country labels, so blocking runs within country. Test adds a country absent from train (France, ~15% of test rows), which drove data-learned legal forms and address-abbreviation unification.
- **One-to-one structure:** no S2/S3 record belongs to more than one S1 entity, which justifies a one-S1-per-target assignment rule. 5.6% of S1 entities have no match; 89% have several (mean 3.5, max 11).
- **Script:** 7.1% of true pairs pair a Latin S1 name with an Indic-script target (Devanagari about half of these). Only ~1.5k distinct Indic tokens exist, so transliteration is done once per token.
- **Chain names:** ~40% of S1 rows share their core name (legal suffixes removed) with another entity (top: meridian, summit, redwood). Name-only matching is unsafe; address and ambiguity signals are required.
- **Name noise:** Jaro-Winkler < 0.7 on 12.3% of true pairs: typos, digit-for-letter swaps (`va1ley`, `5ervices`), glued tokens (`antstechnology com`), token reordering, `M/s` and `d/b/a` prefixes, trailing ID numbers.
- **Address:** missing on ~3.3% of S2/S3 records, almost never on both sides of a true pair. French S2/S3 abbreviate street types (`r.`, `av`, `all`) where S1 writes them in full.

### 2.2 Solution Strategy
**Approach Type:** Blocking + gradient-boosted pair classifier + constrained assignment  
**Core Innovation:** Ambiguity-aware entity resolution: address-anchored blocking keys for chain names, and forward/reverse competition features (rank and score margin of a pair among all candidates of the S1 entity and among all S1 entities proposing the same record) used both in the model and in the decision rule.

---

## 3. Candidate Generation (Blocking)
- **Normalization first:** NFKC, casefolding, native digits to ASCII, accent folding, vocabulary-level IndicXlit transliteration, canonical name with abbreviation expansion, dotted-acronym rejoining (`p c` -> `pc`), ID-noise removal, and legal forms/stopwords learned from boundary-token frequencies per country (e.g. `sci`, `ets`, `pc`, `de`).
- **Blocking keys used (within country):** exact core name; core tokens; consonant-skeleton token bigrams (vowel/typo/transliteration variants, digit-for-letter repair); 4-char prefixes; postal code + name skeleton; **name skeleton + house number / street word**; **address token bigrams**; **glued tokens**. Keys shared by more targets than a per-type cap are ignored; pairs are scored by summed key IDF and the top 30 per S1 per source are kept.
- **Second channel:** char 3-4-gram TF-IDF on core names; top 10 per S1 per source by cosine are added, and the cosine is stored for every candidate.
- **Candidate pairs generated:** [total from the final test run] (~58 per S1 on the dev slice)
- **How you ensured true matches were not lost:** recall was measured against training ground truth after every change. On a 20k-entity India slice, pair recall went from 0.60 (name keys only) to 0.84 after adding the address-anchored and glued-token keys; S1 entities with all matches found went from 31% to 64%. The remaining misses are dominated by Indic names (addressed by IndicXlit) and identical chain names with different addresses.

---

## 4. Matching Model

**Features used:**
- Name features: ratio / token-set / token-sort / Jaro-Winkler on normalized and romanized names, consonant-skeleton similarity, exact-match flags, token Jaccard, first-token equality, length and token-count differences, partial ratio, TF-IDF cosine.
- Address features: token-set and partial ratios, token Jaccard, numeric-token Jaccard, postal-code match and conflict, missing flags.
- Other: blocking score, key-type bitmask and rank; forward rank/margin (among the S1's candidates), reverse rank/count/margin (among S1 entities proposing the same record); core-name frequency of the S1 and of the candidate within Source 1.

**Model type:** LightGBM, 5-fold cross-validation grouped by S1 entity (inner split for early stopping), fold ensemble at prediction time, isotonic calibration fitted on out-of-fold predictions.  
**Threshold selection method:** out-of-fold macro F0.5. Thresholds per source x address-missing; a grid over the one-S1-per-target rule, a competition-margin filter and a top-1 fallback picks the rule with the highest macro F0.5.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [final out-of-fold value, within candidates] x blocking recall [final]; dev slice: 0.95 within candidates.
- **Common false positives (wrong merges):** branches of chains with identical names and nearby addresses; generic names (`tech business`) where the address carries all the signal.
- **Common false negatives (missed matches):** Indic-script names poorly romanized; records whose name was replaced entirely (only the address links them); chain branches with different addresses that blocking cannot reach.

---

## 6. Conclusion
Recall at blocking was the binding constraint: name-only keys silently dropped chain-name entities. Address-anchored keys and competition features fixed most of it, and tuning every decision on out-of-fold macro F0.5 kept precision high. Main lesson: measure blocking recall first; no classifier can recover pairs the candidate stage never proposes.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `src/` (all code), `README.md` (pipeline, commands, compute notes), `requirements.txt` (pinned), `run_pipeline.sh` (entry point), `colab/translit_colab.ipynb` (IndicXlit vocabulary fill).
Reproduce both output files with `./run_pipeline.sh` (steps: `learn` -> `preprocess` -> `block_train` -> `match_train` -> `block_test` -> `predict` -> `package`). `predict` writes `output/matching_results.tsv`, `block_test` writes `output/candidate_pairs.tsv`, and `package` validates them with `utils/validate_submission.py` before zipping.

### B. Additional Results
- Blocking ablation (India slice, 20k S1 entities): name keys only 0.60 pair recall -> + address/glued keys 0.84; the TF-IDF channel added 111 true pairs over 138k extra candidates, so it is kept mainly for its cosine feature.
- Top model features (dev run): skeleton name ratio, address token Jaccard, forward score margin, partial name ratio, blocking score, reverse score margin.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
