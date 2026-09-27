"""Convert raw TSVs to parquet (one file per split) and build a small dev slice of train.

  python -m v2.prep raw  --split train|test      -> work_v2/raw/<split>.parquet (+ work_v2/raw/gt.parquet)
  python -m v2.prep slice --frac 0.1             -> work_v2/dev/raw/train.parquet + gt.parquet

The dev slice keeps a hashed fraction of train S1s, all of their true records, and the same fraction of
unclaimed S2/S3 records, so the density of decoys per S1 matches the full data.
"""
import argparse

import numpy as np

from .common import WORK, hash_frac, load_gt, load_raw, log, read_parquet, write_parquet


def cmd_raw(split):
    df = load_raw(split)
    write_parquet(df, WORK / "raw" / f"{split}.parquet")
    log(split, len(df), "records")
    if split == "train":
        gt = load_gt()
        write_parquet(gt, WORK / "raw" / "gt.parquet")
        log("gt", len(gt), "rows")


def cmd_slice(frac):
    df = read_parquet(WORK / "raw" / "train.parquet")
    gt = read_parquet(WORK / "raw" / "gt.parquet")
    keep_s1 = df.loc[(df.src == 1) & hash_frac(df.id.values, 11, frac), "id"].values
    gts = gt[np.isin(gt.s1.values, keep_s1)]
    claimed = set(gt.rec.values[gt.rec.values >= 0])
    rec = df[df.src > 1]
    unclaimed = rec.loc[~rec.id.isin(claimed) & hash_frac(rec.id.values, 12, frac), "id"].values
    keep = np.concatenate([keep_s1, gts.rec.values[gts.rec.values >= 0], unclaimed])
    out = df[df.id.isin(keep)]
    write_parquet(out, WORK / "dev" / "raw" / "train.parquet")
    write_parquet(gts, WORK / "dev" / "raw" / "gt.parquet")
    log("slice:", (out.src == 1).sum(), "S1,", (out.src > 1).sum(), "S2/S3 records,", len(unclaimed), "unclaimed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["raw", "slice"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--frac", type=float, default=0.1)
    a = ap.parse_args()
    cmd_raw(a.split) if a.cmd == "raw" else cmd_slice(a.frac)
