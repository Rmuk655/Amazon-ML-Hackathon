"""Pruner + matcher (LightGBM), exclusive assignment and macro-F0.5 evaluation.

Every S2/S3 record belongs to at most one S1 (train truth: 0 of 7.64M ids shared), so the decision is made
per record: take its best-scoring S1 and accept it if the probability clears a threshold. Per-S1 match lists
are then the records assigned to that S1.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import common as C
from .common import log

import os
PRUNE_KEEP = int(os.environ.get("BER_PRUNE_KEEP", "6"))          # S1 candidates kept per record after pruning
PRUNE_MIN_P = float(os.environ.get("BER_PRUNE_MIN_P", "0.002"))


def attach_labels(cand: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """y = 1 for true pairs; grp = the record's true S1 (or the record id if unclaimed); ho = locked holdout."""
    g = gt[gt.rec >= 0][["rec", "s1"]].rename(columns={"s1": "true_s1"})
    cand = cand.merge(g, on="rec", how="left")
    cand["y"] = (cand["true_s1"].values == cand["s1"].values).astype(np.int8)
    claimed = cand["true_s1"].notna().values
    grp = np.where(claimed, cand["true_s1"].fillna(0).values.astype(np.int64), cand["rec"].values)
    cand["grp"] = grp
    ho = np.zeros(len(cand), bool)
    ho[claimed] = C.is_holdout(grp[claimed])
    ho[~claimed] = C.hash_frac(grp[~claimed], 99, C.HOLDOUT_FRAC)
    cand["ho"] = ho
    cand["fold"] = (pd.util.hash_array(grp.astype(np.int64)) % 3).astype(np.int8)
    return cand


def lgb_fit(X, y, params=None, rounds=600, valid=None):
    p = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=50, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=C_THREADS())
    if params:
        p.update(params)
    ds = lgb.Dataset(X, y, free_raw_data=True)
    cb = []
    vs = []
    if valid is not None:
        vs = [lgb.Dataset(valid[0], valid[1], reference=ds)]
        cb = [lgb.early_stopping(40, verbose=False)]
    return lgb.train(p, ds, num_boost_round=rounds, valid_sets=vs, callbacks=cb)


def C_THREADS():
    import os
    return int(os.environ.get("BER_WORKERS", "8"))


def prune(cand: pd.DataFrame, model, feats) -> pd.DataFrame:
    """Score all candidates with the pruner, keep the best PRUNE_KEEP per record above PRUNE_MIN_P."""
    cand["p_prune"] = model.predict(cand[feats].values.astype(np.float32), num_threads=C_THREADS()).astype(np.float32)
    cand = cand.sort_values(["rec", "p_prune"], ascending=[True, False], kind="stable")
    rk = cand.groupby("rec", sort=False).cumcount().values
    keep = (rk < PRUNE_KEEP) & ((cand["p_prune"].values >= PRUNE_MIN_P) | (rk == 0))
    cand = cand[keep].copy()
    g = cand.groupby("rec", sort=False)["p_prune"]
    cand["pr_rank"] = g.cumcount().astype(np.float32)
    cand["pr_margin"] = (cand["p_prune"] - g.transform("max")).astype(np.float32)
    return cand.reset_index(drop=True)


def record_competition(df: pd.DataFrame, pcol: str) -> pd.DataFrame:
    """Per record: margin to the best other candidate; per S1: how many records put it first."""
    g = df.groupby("rec", sort=False)[pcol]
    mx = g.transform("max").values
    second = g.transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else 0.0).values
    p = df[pcol].values
    df[pcol + "_gap"] = np.where(p >= mx, p - second, p - mx).astype(np.float32)
    top = df.loc[p >= mx, "s1"].value_counts()
    df[pcol + "_s1top"] = df["s1"].map(top).fillna(0).values.astype(np.float32)
    return df


# ---------------------------------------------------------------- assignment + metric
def assign(df: pd.DataFrame, pcol: str, t: float) -> pd.DataFrame:
    """Exclusive assignment: each record -> its argmax S1 if p >= t. Returns (rec, s1, p) rows."""
    d = df[["rec", "s1", pcol]]
    idx = d.groupby("rec", sort=False)[pcol].idxmax()
    best = d.loc[idx.values]
    return best[best[pcol].values >= t]


def macro_f05(pred: pd.DataFrame, gt: pd.DataFrame, s1_ids: np.ndarray, fp_weight: float = 1.0) -> dict:
    """Macro F0.5 over s1_ids (singletons included). pred: (rec, s1). fp_weight > 1 inflates false positives
    coming from unclaimed records (simulates the test set's higher decoy density)."""
    s1_ids = np.asarray(s1_ids)
    true = gt[(gt.rec >= 0) & np.isin(gt.s1.values, s1_ids)]
    n_true = true.groupby("s1").size()
    p = pred[np.isin(pred.s1.values, s1_ids)][["rec", "s1"]]
    claimed = set(gt.rec.values[gt.rec.values >= 0])
    p = p.merge(true.rename(columns={"s1": "ts1"}), on="rec", how="left")
    p["tp"] = (p["ts1"].values == p["s1"].values)
    unclaimed = ~np.isin(p.rec.values, np.fromiter(claimed, np.int64, len(claimed))) if fp_weight != 1.0 else None
    w = np.ones(len(p))
    if unclaimed is not None:
        w[unclaimed] = fp_weight
    p["fpw"] = np.where(p["tp"], 0.0, w)
    agg = p.groupby("s1").agg(tp=("tp", "sum"), fp=("fpw", "sum"))
    res = pd.DataFrame(index=pd.Index(s1_ids, name="s1"))
    res["nt"] = n_true.reindex(res.index).fillna(0).values
    res["tp"] = agg["tp"].reindex(res.index).fillna(0).values
    res["fp"] = agg["fp"].reindex(res.index).fillna(0).values
    npred = res.tp + res.fp
    prec = np.where(npred > 0, res.tp / np.maximum(npred, 1e-9), 1.0)
    rec = np.where(res.nt > 0, res.tp / np.maximum(res.nt, 1e-9), 1.0)
    f = np.where((prec + rec) > 0, 1.25 * prec * rec / np.maximum(0.25 * prec + rec, 1e-9), 0.0)
    single = res.nt.values == 0
    f[single] = (npred.values[single] == 0).astype(float)
    f[(~single) & (npred.values == 0)] = 0.0
    return {"macro_f05": float(f.mean()), "pair_p": float(res.tp.sum() / max(npred.sum(), 1)),
            "pair_r": float(res.tp.sum() / max(res.nt.sum(), 1)), "n_s1": len(res),
            "singleton_acc": float(f[single].mean()) if single.any() else float("nan")}


def tune_threshold(df, pcol, gt, s1_ids, fp_weight=1.0, grid=None):
    grid = grid if grid is not None else np.round(np.arange(0.20, 0.96, 0.025), 3)
    best = (None, -1)
    rows = []
    for t in grid:
        m = macro_f05(assign(df, pcol, t), gt, s1_ids, fp_weight)
        rows.append((t, m["macro_f05"]))
        if m["macro_f05"] > best[1]:
            best = (t, m["macro_f05"])
    return best[0], rows
