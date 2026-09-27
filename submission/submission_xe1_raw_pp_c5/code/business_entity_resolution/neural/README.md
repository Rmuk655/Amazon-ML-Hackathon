# Cross-encoder reranker stacked on the v2 pipeline

A second opinion on v2's candidate pairs from a model that reads both records' raw text together.

1. `prep/build_xe_pkg.py`: exports v2's scored candidate pairs (train with out-of-fold `p2`, test) and the raw
   name/address texts as a Kaggle dataset. The locked holdout is flagged and never trained on.
2. `kernels/xe_template.py` (one Kaggle kernel per fold, 2x T4, ~1.5-3 h): fine-tunes the multilingual cross-encoder
   `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0; XLM-R vocabulary, so it covers Indic scripts) on
   1.6M pairs of "name | address" text (BCE, 1 epoch, fp16). Fold k trains on half of the non-holdout S1 groups and
   scores the other half out-of-fold. Fold 0 also scores the locked holdout and the test pairs.
3. `prep/stack.py`: LightGBM on v2's score plus the cross-encoder logit, with within-record and within-S1 rank and
   margin features for both. It is trained 3-fold out-of-fold on non-holdout pairs, and the threshold is tuned on those
   predictions. v2's exclusive assignment (`v2.model.assign`) then produces the submission. The locked holdout is scored
   once, and v2 alone goes through the identical procedure as the baseline.

Results (pair AUC, out-of-fold): cross-encoder 0.9991 vs v2 0.9996. It is weaker alone but makes different mistakes.
Early non-holdout estimate: stack 0.9890 vs v2 0.9874 OOF macro F0.5. See the documentation for the locked-holdout
numbers.
