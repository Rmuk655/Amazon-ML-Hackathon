"""Build one submission from the saved stack scores with every accepted decision tweak applied together (no retraining).
  python final.py <scores dir> <out dir> '<json params>' [--holdout-only]
params (all optional; defaults = the stack as uploaded, LB 0.983):
  thr      {"India_2": 0.65, "US_3": 0.7, "France": 0.75, ...}  per country[_source] threshold on pst (default 0.675)
  margin   m: drop a record's best S1 if its pst lead over the 2nd-best S1 is < m (ambiguity abstention)
  base_t   threshold for groups not in thr (default 0.675, the pst stack's OOF-tuned value)
  france_file / france_col   parquet (rec, s1, <col>) whose score replaces pst on France S1 pairs
  col      score column in the parquets (default pst; the raw-feature stack uses q)
  t_c, t_e, caps: companion rescue, empty-S1 rescue, generator caps (see postproc.py)
Always prints the locked-holdout macro F0.5 of the India/US part; writes test files unless --holdout-only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "xe_v2code_c8"))
from v2 import common as C                                    # noqa: E402
from v2.model import macro_f05                               # noqa: E402

SC, OUT, P = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])
HO_ONLY = "--holdout-only" in sys.argv
BASE_T = 0.675


def prepare(scores, split):
    d = scores.sort_values(["rec", "pst"], ascending=[True, False], kind="mergesort")
    rank = d.groupby("rec", sort=False).cumcount().values
    sec = pd.Series(d.pst.values[rank == 1], index=d.rec.values[rank == 1])
    b = d[rank == 0].copy()
    b["gap"] = b.pst.values - sec.reindex(b.rec.values).fillna(0.0).values
    b["src"] = (b.rec.values // C.SRC_MULT).astype(np.int8)
    raw = pd.read_parquet(C.WORK / f"raw/{split}.parquet", columns=["id", "country"])
    ctry = pd.Series(raw.country.values, index=raw.id.values)
    b["country"] = ctry.reindex(b.s1.values).values
    return b.reset_index(drop=True)


def decide(b):
    thr = P.get("thr", {})
    t = np.full(len(b), P.get("base_t", BASE_T))
    for c in b.country.unique():
        for s in (2, 3):
            v = thr.get(f"{c}_{s}", thr.get(c))
            if v is not None:
                t[(b.country.values == c) & (b.src.values == s)] = v
    acc = b.pst.values >= t
    if P.get("margin"):
        acc &= b.gap.values >= P["margin"]
    keep = acc.copy()
    pool = b[~acc]
    if P.get("t_c") is not None:
        a = b[acc]
        has2, has3 = set(a.s1.values[a.src.values == 2]), set(a.s1.values[a.src.values == 3])
        top = pool.sort_values("pst", ascending=False).drop_duplicates(["s1", "src"])
        other = np.where(top.src.values == 2, np.isin(top.s1.values, list(has3)), np.isin(top.s1.values, list(has2)))
        keep[top.index[(top.pst.values >= P["t_c"]) & other]] = True
    if P.get("t_e") is not None:
        matched = set(b.s1.values[keep])
        top1 = pool.sort_values("pst", ascending=False).drop_duplicates("s1")
        keep[top1.index[(top1.pst.values >= P["t_e"]) & ~np.isin(top1.s1.values, list(matched))]] = True
    out = b[keep]
    if P.get("caps"):
        out = out.sort_values("pst", ascending=False)
        r = out.groupby(["s1", "src"]).cumcount().values
        out = out[r < np.where(out.src.values == 2, 5, 6)]
    return out[["rec", "s1"]]


def main():
    gt = pd.read_parquet(C.WORK / "raw/gt.parquet")
    s1_all = np.unique(gt.s1.values)
    ho_s1 = s1_all[C.is_holdout(s1_all)]
    col = P.get("col", "pst")
    b = prepare(pd.read_parquet(SC / "train_scores.parquet", columns=["rec", "s1", col]).rename(columns={col: "pst"}), "train")
    r = [macro_f05(decide(b), gt, ho_s1, w) for w in (1.0, 2.0)]
    print(f"params {P}: HOLDOUT macro F0.5 {r[0]['macro_f05']:.5f} (P {r[0]['pair_p']:.4f} R {r[0]['pair_r']:.4f}) "
          f"| x2 decoys {r[1]['macro_f05']:.5f}", flush=True)
    if HO_ONLY:
        return
    ts = pd.read_parquet(SC / "test_scores.parquet", columns=["rec", "s1", col]).rename(columns={col: "pst"})
    if P.get("france_file"):              # France rows scored by a different model (e.g. self-trained)
        fcol = P.get("france_col", "pst")
        fr = pd.read_parquet(P["france_file"], columns=["rec", "s1", fcol]).rename(columns={fcol: "pst_fr"})
        ts = ts.merge(fr, on=["rec", "s1"], how="left")
        raw_c = pd.read_parquet(C.WORK / "raw/test.parquet", columns=["id", "country"])
        isfr = pd.Series(raw_c.country.values, index=raw_c.id.values).reindex(ts.s1.values).values == "France"
        miss = isfr & ts.pst_fr.isna().values
        print(f"France pairs {isfr.sum():,}, missing France-file score {miss.sum():,}", flush=True)
        ts["pst"] = np.where(isfr & ~miss, ts.pst_fr.values, ts.pst.values)
        ts = ts.drop(columns="pst_fr")
    bt = prepare(ts, "test")
    pred = decide(bt)
    print(f"test: {len(pred):,} matches; France {int((bt.loc[bt.rec.isin(pred.rec), 'country'] == 'France').sum()):,}", flush=True)
    raw = pd.read_parquet(C.WORK / "raw/test.parquet", columns=["id", "src"])
    from v2.run import write_outputs
    write_outputs(pred, ts[["rec", "s1"]], raw.id.values[raw.src.values == 1], OUT)


if __name__ == "__main__":
    main()
