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
INDICLID_FTN_DIR = MODELS_DIR / "indiclid-ftn"  # unzip IndicLID FTN model here (any *.bin)
PENDING_VOCAB = PROCESSED_DIR / "pending_vocab.tsv"   # (lang, token) pairs still to transliterate

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


def source_path(split: str, n: int) -> Path:
    return DATASET_DIR / split / f"{split}_source{n}.tsv"


def processed_path(split: str, n: int) -> Path:
    return PROCESSED_DIR / f"{split}_source{n}.parquet"


# ---- Stage 3: blocking ----
KTYPES = ("core", "tok", "skelbi", "pre", "addrpc")
# A key shared by more targets than its cap is too common to be discriminative -> ignored.
BLOCK_CAPS = {"core": 300, "tok": 100, "skelbi": 100, "pre": 100, "addrpc": 100}
K_PER_SOURCE = 30          # candidates kept per S1 entity, separately for S2 and S3
S1_CHUNK = 100_000         # S1 rows joined at a time (bounds memory)
KEYGEN_BLOCK = 500_000     # records turned into hashed keys at a time
WITHIN_COUNTRY = True      # block inside country_norm (set False if EDA shows cross-country matches)


def candidates_dir(split: str):
    return PROCESSED_DIR / f"candidates_{split}"