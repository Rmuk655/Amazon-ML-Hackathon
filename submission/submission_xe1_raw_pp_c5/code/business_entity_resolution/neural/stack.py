"""Stack v2's pair score (p2) with the cross-encoder score (xe) and measure on the locked holdout.

Run on the AWS instance (needs v2's gt/raw files and v2 code at commit c8dc791 in ~/xe_v2code_c8):
  python stack.py <pairs dir (xe_pkg)> <xe scores dir> <out dir>
Protocol (same as v2): the stacker is trained with 3-fold out-of-fold predictions on NON-holdout pairs, and the
threshold is tuned on those OOF predictions. The locked holdout is scored once with the fold-model average, and
the baseline (p2 alone) goes through the identical procedure, so the two numbers are directly comparable.
Test: fold-model average, tuned threshold, exclusive assignment (v2's assign) -> matching_results.tsv + candidate_pairs.tsv.
"""
import glob
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "xe_v2code_c8"))
from v2 import common as C                                  # noqa: E402
from v2.model import assign, macro_f05, tune_threshold      # noqa: E402

PKG, XE, OUT = map(Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def load_xe(part):
    fs = sorted(glob.glob(str(XE / f"scores_{part}_*.parquet")))
    assert fs, f"no xe scores for {part}"
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def gap_feats(df, col, key, pfx):
    """Rank of the pair's score inside its key group, and its margin over the best OTHER pair of the group."""
    o = df.sort_values([key, col], ascending=[True, False], kind="mergesort")
    rank = o.groupby(key, sort=False).cumcount().values
    top1 = o.groupby(key, sort=False)[col].transform("first").values
    second = o[col].where(rank == 1).groupby(o[key].values).transform("max").reindex(o.index).values
    second = pd.Series(second, index=o.index).fillna(-1e9).values
    other = np.where(rank == 0, second, top1)
    res = pd.DataFrame({f"{pfx}_rank": rank, f"{pfx}_gap": o[col].values - other}, index=o.index)
    return res.reindex(df.index)


def features(df):
    df["lp2"] = np.log(np.clip(df.p2, 1e-6, 1 - 1e-6) / np.clip(1 - df.p2, 1e-6, 1))
    for col in ("xe", "lp2"):
        for key, k in (("rec", "r"), ("s1", "s")):
            g = gap_feats(df, col, key, f"{col}_{k}")
            df[g.columns] = g.values
    df["n_s1_per_rec"] = df.groupby("rec").s1.transform("size").astype(np.int16)
    df["n_rec_per_s1"] = df.groupby("s1").rec.transform("size").astype(np.int16)
    df["xe_max_s1"] = df.groupby("s1").xe.transform("max")
    return df


FEATS = ["lp2", "xe", "xe_r_rank", "xe_r_gap", "xe_s_rank", "xe_s_gap", "lp2_r_rank", "lp2_r_gap",
         "lp2_s_rank", "lp2_s_gap", "n_s1_per_rec", "n_rec_per_s1", "xe_max_s1"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=6)


def fit_stack(df, cols):
    """3-fold OOF on non-holdout rows (folds by S1); holdout rows get the fold-model average."""
    tr = ~df.ho.values
    fold = (pd.util.hash_array(df.s1.values.astype(np.int64), hash_key="xestackfold00001") % 3).astype(int)
    p = np.zeros(len(df)); models = []
    for k in range(3):
        fit = tr & (fold != k)
        m = lgb.train(PARAMS, lgb.Dataset(df.loc[fit, cols], df.y.values[fit]), num_boost_round=400)
        p[tr & (fold == k)] = m.predict(df.loc[tr & (fold == k), cols])
        if (~tr).any():
            p[~tr] += m.predict(df.loc[~tr, cols]) / 3
        models.append(m)
        log(f"  fold {k} done")
    return p, models


def report(df, pcol, gt, label):
    s1_all = np.unique(gt.s1.values)
    ho_s1, tr_s1 = s1_all[C.is_holdout(s1_all)], s1_all[~C.is_holdout(s1_all)]
    t, _ = tune_threshold(df[~df.ho.values], pcol, gt, tr_s1)
    a = assign(df, pcol, t)
    r = [macro_f05(a, gt, ho_s1, w) for w in (1.0, 2.0)]
    log(f"{label}: t={t} HOLDOUT macro F0.5 {r[0]['macro_f05']:.4f} (P {r[0]['pair_p']:.4f} R {r[0]['pair_r']:.4f} "
        f"singletons {r[0]['singleton_acc']:.4f}) | x2 decoys {r[1]['macro_f05']:.4f}")
    return t


def main():
    gt = pd.read_parquet(C.WORK / "raw/gt.parquet")
    df = pd.read_parquet(PKG / "pairs_train.parquet")
    xe = pd.concat([load_xe("oof"), load_xe("holdout")], ignore_index=True)
    df = df.merge(xe, on=["rec", "s1"], how="left")
    log(f"train pairs {len(df):,}, xe missing {df.xe.isna().mean():.4%}")
    df["xe"] = df.xe.fillna(df.xe.min())
    df = features(df)
    report(df, "p2", gt, "BASELINE v2 p2 alone")
    df["pst"], models = fit_stack(df, FEATS)
    imp = pd.Series(models[0].feature_importance("gain"), index=FEATS).sort_values(ascending=False)
    log("stack importance: " + ", ".join(f"{k}={v:.0f}" for k, v in imp.items()))
    t = report(df, "pst", gt, "STACK p2 + cross-encoder")

    te = pd.read_parquet(PKG / "pairs_test.parquet").merge(load_xe("test"), on=["rec", "s1"], how="left")
    log(f"test pairs {len(te):,}, xe missing {te.xe.isna().mean():.4%}")
    te["xe"] = te.xe.fillna(df.xe.min())
    te = features(te)
    te["pst"] = sum(m.predict(te[FEATS]) for m in models) / 3
    te[["rec", "s1", "pst"]].to_parquet(OUT / "test_scores.parquet", index=False)
    df[["rec", "s1", "y", "ho", "pst"]].to_parquet(OUT / "train_scores.parquet", index=False)
    raw = pd.read_parquet(C.WORK / "raw/test.parquet", columns=["id", "src"])
    s1_all = raw.id.values[raw.src.values == 1]
    pred = assign(te, "pst", t)
    log(f"test: {len(pred):,} matches at t={t}")
    from v2.run import write_outputs
    write_outputs(pred, te[["rec", "s1"]], s1_all, OUT)
    log("DONE")


if __name__ == "__main__":
    main()
