"""Paths, constants, thresholds, model locations."""
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
CODE_DIR = SRC_DIR.parent                      # .../code/business_entity_resolution
try:
    STUDENT_RESOURCE = CODE_DIR.parents[1]     # .../student_resource
except IndexError:                             # e.g. only src/ copied to Colab
    STUDENT_RESOURCE = CODE_DIR
DATASET_DIR = STUDENT_RESOURCE / "dataset"
PROCESSED_DIR = Path(os.environ.get("BER_PROCESSED_DIR", DATASET_DIR / "processed"))
OUTPUT_DIR = STUDENT_RESOURCE / "output"       # final submission files
MODELS_DIR = CODE_DIR / "models"
# trained pipeline models (pruner, matcher); BER_MODELS_DIR lets dev experiments write elsewhere
RUN_MODELS_DIR = Path(os.environ.get("BER_MODELS_DIR", STUDENT_RESOURCE / "models"))
# matcher learning rate: 0.05 hit the 1500-tree cap in every fold; 0.1 early-stops ~2-3x sooner
LGB_LEARNING_RATE = float(os.environ.get("BER_LR", "0.1"))
INDICLID_FTN_DIR = MODELS_DIR / "indiclid-ftn"  # unzip IndicLID FTN model here (any *.bin)
PENDING_VOCAB = PROCESSED_DIR / "pending_vocab.tsv"   # (lang, token) pairs still to transliterate
NAME_VOCAB = PROCESSED_DIR / "name_vocab.tsv"          # clean S1 name tokens for OCR repair (learn_suffixes.py)
LEARNED_TOKENS = PROCESSED_DIR / "learned_tokens.tsv"  # legal forms / stopwords from learn_suffixes.py

TRAIN_GT = DATASET_DIR / "train" / "train_ground_truth.tsv"

# ---- language identification ----
# Scripts that map to one language in our data -> no LID call. Others go to IndicLID.
AMBIGUOUS_SCRIPTS = {"Deva", "Beng", "Arab"}   # hi/mr/ne/sa.. ; bn/as/mni.. ; ur/sd/ks
LID_CONFIDENCE_THRESHOLD = 0.5                 # below -> fallback language
LID_MIN_CHARS = 4                              # shorter Indic text -> LID unreliable -> fallback
# Configurable fallback / default IndicXlit language per script
SCRIPT_DEFAULT_LANG = {
    "Deva": "hi", "Beng": "bn", "Guru": "pa", "Gujr": "gu", "Orya": "or",
    "Taml": "ta", "Telu": "te", "Knda": "kn", "Mlym": "ml", "Sinh": "si", "Arab": "ur",
}

# ---- transliteration ----
BEAM_WIDTH = 4
CACHE_FLUSH_EVERY = 2000


# ---- locked holdout: a fixed 10% of TRAIN S1 entities (and their records) excluded from rule inference,
# pruner fitting, training and threshold tuning; used only for final reporting. Own hash key, so it is
# independent of the S1-sample and fold hashes (which use pandas' default key).
HOLDOUT_FRAC = 0.10
HOLDOUT_KEY = "lockedholdout001"                      # exactly 16 characters (pandas hash_key)


def is_holdout(s1_ids):
    import numpy as np
    import pandas as pd
    ids = pd.Series(s1_ids).astype(str).values
    return (pd.util.hash_array(ids, hash_key=HOLDOUT_KEY) % 10_000 < int(HOLDOUT_FRAC * 10_000)).astype(bool) \
        if len(ids) else np.zeros(0, bool)


def source_path(split: str, n: int) -> Path:
    return DATASET_DIR / split / f"{split}_source{n}.tsv"


def processed_path(split: str, n: int) -> Path:
    return PROCESSED_DIR / f"{split}_source{n}.parquet"


# ---- Stage 3: blocking ----
# new key types are appended AFTER tfidf so the existing mask bits keep their meaning
KTYPES = ("core", "tok", "skelbi", "pre", "addrpc", "nameaddr", "addrbi", "join", "tfidf", "sorted", "acro", "subset",
          "embed")                                           # embed: bi-encoder neighbour (mask bit only)
# A key shared by more targets than its cap is too common to be discriminative -> ignored.
BLOCK_CAPS = {"core": 300, "tok": 100, "skelbi": 100, "pre": 100, "addrpc": 100,
              "nameaddr": 100, "addrbi": 50, "join": 100, "tfidf": 0,   # tfidf: not a key (mask bit only)
              "sorted": 300, "acro": 100, "subset": 100, "embed": 0}
# pairs sharing the exact core name AND an address key skip the top-K cut (identical chain names otherwise
# crowd each other out of the K slots; the analysis found ~13k India true pairs lost that way)
BYPASS_KEYS = os.environ.get("BER_BYPASS", "1") != "0"
NEW_KEYS = os.environ.get("BER_NEW_KEYS", "1") != "0"      # sorted-token, acronym, 3-word / whole-name glue keys
ADDR_KEY = os.environ.get("BER_ADDR_KEY", "1") != "0"      # address-only key (house number + street word)
SUBSET_KEYS = os.environ.get("BER_SUBSET_KEYS", "1") != "0"  # dropped/added words: word pairs / words anchored by address
# char n-gram TF-IDF channel on core names (second, similarity-based blocking channel)
TFIDF_K = 5                # extra candidates per S1 per source by cosine (0 disables the channel)
TFIDF_NGRAM = (3, 4)       # char_wb n-gram range
TFIDF_MAX_DF = 5000        # drop n-grams in more targets than this (bounds sparse-product cost)
TFIDF_BLOCK = 2000         # S1 rows per sparse matmul
K_PER_SOURCE = int(os.environ.get("BER_K", "50"))   # candidates kept per S1 entity, separately for S2 and S3
# target-centric (reverse) candidates: every target also keeps its top K_REV S1 entities by key score, so a
# record that is outside its true S1's top-K is still compared with that S1 (not only with a lookalike)
K_REV = int(os.environ.get("BER_K_REV", "3"))
S1_CHUNK = 100_000         # S1 rows joined at a time (bounds memory)
KEYGEN_BLOCK = 500_000     # records turned into hashed keys at a time
WITHIN_COUNTRY = True      # block inside country_norm (set False if EDA shows cross-country matches)


# ---- Stage 3b: supervised meta-blocking (prune.py) ----
PRUNE_RECALL = 0.995       # keep this share of the true pairs blocking found (threshold chosen out-of-fold)
PRUNE_MAX_THR = 0.05       # never prune harder than this probability


def candidates_dir(split: str):
    return PROCESSED_DIR / f"candidates_{split}"