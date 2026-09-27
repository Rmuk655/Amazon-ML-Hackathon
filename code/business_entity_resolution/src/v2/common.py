"""v2 pipeline: shared paths, id encoding, TSV loading and the locked holdout.

Records are kept as integer ids for speed:
  S1-<n>            -> n
  S2-<n> / S3-<n>   -> src * 10**10 + n     (src = 2 or 3; the numeric parts of S2 and S3 can collide)
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = SRC_DIR.parent
ROOT = CODE_DIR.parents[1]
DATASET_DIR = Path(os.environ.get("BER_DATASET_DIR", ROOT / "dataset"))
WORK = Path(os.environ.get("BER_V2_WORK", ROOT / "work_v2"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", ROOT / "output"))
SRC_MULT = 10 ** 10

# Locked holdout: same definition as the v1 pipeline (config.is_holdout) so scores stay comparable.
HOLDOUT_FRAC = 0.10
HOLDOUT_KEY = "lockedholdout001"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def is_holdout_str(s1_str_ids) -> np.ndarray:
    ids = pd.Series(s1_str_ids).astype(str).values
    if not len(ids):
        return np.zeros(0, bool)
    return pd.util.hash_array(ids, hash_key=HOLDOUT_KEY) % 10_000 < int(HOLDOUT_FRAC * 10_000)


def is_holdout(s1_int_ids) -> np.ndarray:
    return is_holdout_str(np.char.add("S1-", np.asarray(s1_int_ids).astype(str)))


def hash_frac(ids, salt: int, frac: float) -> np.ndarray:
    """Deterministic pseudo-random subset of integer ids."""
    x = np.asarray(ids, np.int64).astype(np.uint64) ^ np.uint64((salt * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
    h = pd.util.hash_array(x, hash_key="v2hashfraction01")
    return (h % np.uint64(1_000_000)) < np.uint64(int(frac * 1_000_000))


def encode_ids(str_ids) -> np.ndarray:
    s = pd.Series(str_ids).astype(str)
    src = s.str.slice(1, 2).astype(np.int64).values
    num = s.str.slice(3).astype(np.int64).values
    return np.where(src == 1, num, src * SRC_MULT + num)


def decode_ids(ids) -> np.ndarray:
    ids = np.asarray(ids, np.int64)
    src = np.where(ids >= SRC_MULT, ids // SRC_MULT, 1)
    num = np.where(ids >= SRC_MULT, ids % SRC_MULT, ids)
    return np.char.add(np.char.add("S", src.astype(str)), np.char.add("-", num.astype(str)))


def read_tsv(path) -> pd.DataFrame:
    """Read a challenge TSV exactly as written (tab separated, no quoting, empty stays empty)."""
    t = pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in ["entity_id", "business_name", "business_address", "country",
                                                   "source1_entity_id", "matched_entity_ids"]},
            strings_can_be_null=False),
        read_options=pacsv.ReadOptions(block_size=1 << 26))
    return t.to_pandas()


def raw_path(split, n):
    return DATASET_DIR / split / f"{split}_source{n}.tsv"


def load_raw(split) -> pd.DataFrame:
    """All three sources of a split: id (int), src, name, addr, country."""
    parts = []
    for n in (1, 2, 3):
        df = read_tsv(raw_path(split, n))
        parts.append(pd.DataFrame({
            "id": encode_ids(df["entity_id"]), "src": np.int8(n),
            "name": df["business_name"].values, "addr": df["business_address"].values,
            "country": df["country"].values}))
        del df
    return pd.concat(parts, ignore_index=True)


def load_gt() -> pd.DataFrame:
    """Train truth as (s1, rec) pairs; a singleton S1 appears once with rec = -1."""
    g = read_tsv(DATASET_DIR / "train" / "train_ground_truth.tsv")
    g = g.assign(m=g["matched_entity_ids"].str.split(",")).explode("m")
    has = (g["m"].notna() & (g["m"] != "")).values
    rec = np.full(len(g), -1, np.int64)
    rec[has] = encode_ids(g["m"].values[has])
    return pd.DataFrame({"s1": encode_ids(g["source1_entity_id"].values), "rec": rec})


def write_parquet(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def read_parquet(path, columns=None):
    return pq.read_table(path, columns=columns).to_pandas()
