"""Merge bi-encoder nearest neighbours (Kaggle, multilingual-e5) into the blocking candidates, before pruning.
Enabled by BER_EMBED_CANDS=1 in the run scripts; gated on the locked holdout like everything else.

    python merge_embed.py --split train --dir ~/embed [--top 5]

Input: nn_<split>_<country>_S<n>.parquet (s1_id, cand_id, cos, rank) anywhere under --dir.
- existing candidate files get an `emb_cos` column (cosine where the pair is also a neighbour, else NaN)
- neighbours (rank < --top) that blocking did not propose are written to part-embed-<country>.parquet with the
  mask bit 'embed', neutral blocking values and is_true from the ground truth on train
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import config as C

KT = {k: i for i, k in enumerate(C.KTYPES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.dir), "**", f"nn_{a.split}_*.parquet"), recursive=True))
    if not files:
        print(f"[embed] no neighbour files for {a.split} under {a.dir}; nothing merged")
        return
    nn = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    nn = nn.sort_values("cos", ascending=False).drop_duplicates(["s1_id", "cand_id"])
    print(f"[embed] {a.split}: {len(nn):,} neighbour pairs from {len(files)} files")
    cdir = C.candidates_dir(a.split)
    parts = sorted(cdir.glob("part-*.parquet"))
    parts = [p for p in parts if not p.name.startswith("part-embed-")]
    seen, country_of = set(), {}
    cos_map = nn.set_index(nn["s1_id"].astype(str) + "|" + nn["cand_id"].astype(str))["cos"]
    for p in parts:
        d = pd.read_parquet(p)
        k = d["s1_id"].astype(str) + "|" + d["cand_id"].astype(str)
        d["emb_cos"] = k.map(cos_map).astype(np.float32).values
        d.to_parquet(p, index=False)
        seen.update(k)
        c = p.name.split("-")[1]
        country_of.update(dict.fromkeys(d["s1_id"].astype(str).unique(), c))
    new = nn[(nn["rank"] < a.top)].copy()
    new = new[~(new["s1_id"].astype(str) + "|" + new["cand_id"].astype(str)).isin(seen)]
    new["country"] = new["s1_id"].astype(str).map(country_of)
    new = new[new["country"].notna()]
    truth = None
    if a.split == "train":
        gt = pd.read_csv(C.TRAIN_GT, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        flat = pd.DataFrame({"s1": gt.iloc[:, 0].str.strip(), "t": gt.iloc[:, 1].str.split(",")}).explode("t")
        truth = set(flat["s1"] + "|" + flat["t"].fillna("").str.strip())
    for c, g in new.groupby("country"):
        out = pd.DataFrame({
            "s1_id": g["s1_id"].astype(str).values, "cand_id": g["cand_id"].astype(str).values,
            "src": g["cand_id"].astype(str).str[1].astype(np.int8).values,
            "score": np.float32(0), "mask": np.int16(1 << KT["embed"]), "rank": np.int16(127), "t_rank": np.int16(127),
            "tf_cos": np.float32(np.nan), "tf_rank": np.int8(127), "emb_cos": g["cos"].astype(np.float32).values})
        if truth is not None:
            out["is_true"] = (out["s1_id"] + "|" + out["cand_id"]).isin(truth).astype(np.int8)
        out.to_parquet(cdir / f"part-embed-{c}.parquet", index=False)
        extra = f", {int(out['is_true'].sum()):,} true" if truth is not None else ""
        print(f"[embed] {a.split}/{c}: {len(out):,} new candidate pairs{extra}")


if __name__ == "__main__":
    main()
