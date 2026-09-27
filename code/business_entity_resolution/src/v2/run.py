"""v2 pipeline driver.

  python -m v2.run norm   --split train|test [--dev]     lexicon (train only) + normalised records
  python -m v2.run block  --split train|test [--dev]     candidate pairs per country (+ recall on train)
  python -m v2.run feats  --split train|test [--dev]     pair features for the pruned candidates
  python -m v2.run train  [--dev]                        pruner + matcher (OOF), thresholds, holdout report
  python -m v2.run predict [--dev]                       test scores -> assignment -> output TSVs

--dev works on work_v2/dev (10% slice of train); without it on the full data in work_v2.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from . import common as C
from .common import log, read_parquet, write_parquet


def wdir(dev):
    return C.WORK / "dev" if dev else C.WORK


def raw_file(split, dev):
    return (C.WORK / "dev" / "raw" / "train.parquet") if dev else (C.WORK / "raw" / f"{split}.parquet")


def gt_file(dev):
    return (C.WORK / "dev" / "raw" / "gt.parquet") if dev else (C.WORK / "raw" / "gt.parquet")


def stage_norm(split, dev, workers):
    from .normalize import add_french_tokens, learn_lexicon, load_lexicon, normalize_frame, save_lexicon
    W = wdir(dev)
    raw = read_parquet(raw_file(split, dev))
    lex_path = W / "lexicon.pkl"
    if split == "train" or not lex_path.exists():
        rtr = raw if split == "train" else read_parquet(raw_file("train", dev))
        gt = read_parquet(gt_file(dev))
        gt = gt[~C.is_holdout(gt.s1.values)]              # rule inference never sees the locked holdout
        lex = add_french_tokens(learn_lexicon(rtr, gt))
        save_lexicon(lex, lex_path)
        del rtr
    lex = load_lexicon(lex_path)
    nf = normalize_frame(raw, lex, workers=workers)
    write_parquet(nf, W / f"norm_{split}.parquet")
    log("normalised", len(nf))


def stage_block(split, dev, threads):
    from .block import block_country
    W = wdir(dev)
    nf = read_parquet(W / f"norm_{split}.parquet")
    outs = []
    for country, g in nf.groupby("country", sort=False):
        t = time.time()
        s1 = g[g.src == 1].reset_index(drop=True)
        rec = g[g.src > 1].reset_index(drop=True)
        log(f"block {split}/{country}: {len(s1)} S1, {len(rec)} records")
        if len(s1) == 0 or len(rec) == 0:
            continue
        c = block_country(s1, rec, threads=threads)
        log(f"  {country}: {len(c)} pairs ({len(c) / max(len(rec), 1):.1f}/record) in {time.time() - t:.0f}s")
        outs.append(c)
    cand = pd.concat(outs, ignore_index=True)
    write_parquet(cand, W / f"cand_{split}.parquet")
    if split == "train":
        report_recall(cand, dev, "blocking")


def report_recall(cand, dev, label, holdout_only=False):
    gt = read_parquet(gt_file(dev))
    gt = gt[gt.rec >= 0]
    if holdout_only:
        gt = gt[C.is_holdout(gt.s1.values)]
    # true pairs are unique per record: look up each true (rec, s1) among the candidates of its record
    sub = cand.loc[np.isin(cand.rec.values, gt.rec.values), ["rec", "s1"]]
    found = gt.merge(sub, on=["rec", "s1"], how="inner")
    log(f"{label} recall: {len(found) / len(gt):.4f} ({len(found)}/{len(gt)} true pairs), {len(cand)} pairs")


# ---------------------------------------------------------------- pruning
def _load_norm(W, split):
    nf = read_parquet(W / f"norm_{split}.parquet")
    return nf.set_index("id")


def _rec_chunks(rec, n_chunks):
    """Row ranges aligned to record boundaries (cand is grouped by record)."""
    cuts = [0]
    step = max(len(rec) // n_chunks, 1)
    for k in range(1, n_chunks):
        x = k * step
        while x < len(rec) and rec[x] == rec[x - 1]:
            x += 1
        if x > cuts[-1] and x < len(rec):
            cuts.append(x)
    cuts.append(len(rec))
    return list(zip(cuts[:-1], cuts[1:]))


def stage_prune(split, dev):
    """Cheap features + cross-fitted LightGBM pruner, streamed over record-aligned chunks (bounded memory):
    pass A counts S1-side competition and collects a training sample, pass B prunes chunk by chunk."""
    import pickle
    import lightgbm as lgb
    from collections import Counter
    from .block import rank_features
    from .features import CHEAP, cheap_features
    from .model import attach_labels, lgb_fit, prune
    W = wdir(dev)
    norm = _load_norm(W, split)
    cand = read_parquet(W / f"cand_{split}.parquet")
    chunks = _rec_chunks(cand["rec"].values, max(1, len(cand) // int(os.environ.get("BER_CHUNK_ROWS", "12000000"))))
    gt = read_parquet(gt_file(dev)) if split == "train" else None

    def feats(a, b):
        c = cheap_features(rank_features(cand.iloc[a:b].copy()), norm)
        return attach_labels(c, gt) if gt is not None else c

    top = Counter()
    samples = []
    for a, b in chunks:
        c = feats(a, b)
        top.update(c.loc[c["r_comb"].values == 1, "s1"].values.tolist())
        if gt is not None:
            keep = C.hash_frac(c["grp"].values, 5, float(os.environ.get("BER_PRUNE_SAMPLE", "0.06"))) & ~c["ho"].values
            samples.append(c[keep])
    top = pd.Series(top)
    log(f"prune {split}: {len(cand)} candidate pairs in {len(chunks)} chunks")

    def fill_top(c):
        c["s1_top_cnt"] = c["s1"].map(top).fillna(0).values.astype(np.float32)
        return c

    if gt is not None:
        smp = fill_top(pd.concat(samples, ignore_index=True))
        del samples
        half = C.hash_frac(smp["grp"].values, 7, 0.5)
        models = []
        for side in (True, False):
            tr = smp[half == side]
            models.append(lgb_fit(tr[CHEAP].values.astype(np.float32), tr["y"].values,
                                  dict(num_leaves=31, learning_rate=0.15), rounds=150))
        del smp
        with open(W / "pruners.pkl", "wb") as f:
            pickle.dump([m.model_to_string() for m in models], f)
    else:
        with open(W / "pruners.pkl", "rb") as f:
            models = [lgb.Booster(model_str=s_) for s_ in pickle.load(f)]
    out = []
    for a, b in chunks:
        c = fill_top(feats(a, b))
        if gt is not None:
            h = C.hash_frac(c["grp"].values, 7, 0.5)
            ho = c["ho"].values
            out.append(prune(c[~ho & ~h].copy(), models[0], CHEAP))    # model 0 learned on the h == True half
            out.append(prune(c[~ho & h].copy(), models[1], CHEAP))
            out.append(prune(c[ho].copy(), _AvgModel(models), CHEAP))
        else:
            out.append(prune(c, _AvgModel(models), CHEAP))
        del c
    del cand, norm
    pr = pd.concat(out, ignore_index=True)
    if gt is not None:
        tot = (gt.rec >= 0).sum()
        ho_tot = ((gt.rec >= 0) & C.is_holdout(gt.s1.values)).sum()
        log(f"after pruning: {len(pr)} pairs ({len(pr) / pr.rec.nunique():.2f}/record), recall {pr['y'].sum() / tot:.4f}, "
            f"holdout recall {pr.loc[pr.ho, 'y'].sum() / max(ho_tot, 1):.4f}")
    else:
        log(f"after pruning: {len(pr)} pairs ({len(pr) / max(pr.rec.nunique(), 1):.2f}/record)")
    write_parquet(pr, W / f"pruned_{split}.parquet")


class _AvgModel:
    def __init__(self, models):
        self.models = models

    def predict(self, X, **kw):
        return np.mean([m.predict(X, **kw) for m in self.models], 0)


# ---------------------------------------------------------------- features
_G = {}


def _feat_chunk(bounds):
    from .features import pair_features
    i, j = bounds
    return pair_features(_G["cand"].iloc[i:j].reset_index(drop=True), _G["norm"], _G["stats"], workers=1)


def stage_feats(split, dev, workers):
    import pickle
    from multiprocessing import get_context
    from .features import make_stats
    W = wdir(dev)
    stats_path = W / "stats.pkl"
    if split == "train" or not stats_path.exists():
        nt = read_parquet(W / "norm_train.parquet")
        with open(stats_path, "wb") as f:
            pickle.dump(make_stats(nt), f)
        del nt
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    norm = _load_norm(W, split)
    if split == "test":       # name frequencies must describe the split being scored
        stats.update({k: v for k, v in make_stats(norm.reset_index()).items() if k in ("glued_cnt", "first_cnt")})
    cand = read_parquet(W / f"pruned_{split}.parquet")
    _G.update(cand=cand, norm=norm, stats=stats)
    step = 200_000
    bounds = [(i, min(i + step, len(cand))) for i in range(0, len(cand), step)]
    with get_context("fork").Pool(workers) as p:
        parts = list(p.imap(_feat_chunk, bounds))
    F = pd.concat(parts, ignore_index=True)
    keep = [c for c in ["rec", "s1", "y", "grp", "ho", "fold"] if c in cand.columns]
    out = pd.concat([cand[keep].reset_index(drop=True), F], axis=1)
    write_parquet(out, W / f"feats_{split}.parquet")
    log(f"features {split}: {out.shape}")


# ---------------------------------------------------------------- matcher
META = ["rec", "s1", "y", "grp", "ho", "fold"]


def C_WORKERS():
    return int(os.environ.get("BER_WORKERS", "8"))


def feat_cols(df):
    return [c for c in df.columns if c not in META and not c.startswith("p1") and not c.startswith("p2")]


def fit_oof(df, cols, pcol, rounds=int(os.environ.get("BER_ROUNDS", "800"))):
    """3-fold OOF scores on non-holdout rows; holdout / other rows get the fold-model average."""
    from .model import lgb_fit
    X = df[cols].values.astype(np.float32)
    y = df["y"].values
    tr_mask = ~df["ho"].values
    fold = df["fold"].values
    p = np.zeros(len(df), np.float32)
    ho_acc = np.zeros(len(df), np.float32)
    models = []
    frac = float(os.environ.get("BER_TRAIN_FRAC", "1.0"))
    sub = C.hash_frac(df["grp"].values, 21, frac) if frac < 1 else np.ones(len(df), bool)
    for k in range(3):
        tr = tr_mask & (fold != k) & sub
        va = tr_mask & (fold == k)
        vi = np.flatnonzero(va)
        vs = vi[::5] if len(vi) > 200_000 else vi
        m = lgb_fit(X[tr], y[tr], rounds=rounds, valid=(X[vs], y[vs]))
        p[va] = m.predict(X[va])
        ho_acc[~tr_mask] += m.predict(X[~tr_mask]) / 3
        models.append(m)
        log(f"  {pcol} fold {k}: {m.best_iteration or m.current_iteration()} trees")
    p[~tr_mask] = ho_acc[~tr_mask]
    df[pcol] = p
    return models


def evaluate(df, pcol, gt, label):
    from .model import assign, macro_f05, tune_threshold
    s1_all = np.unique(gt.s1.values)
    ho_s1 = s1_all[C.is_holdout(s1_all)]
    tr_s1 = s1_all[~C.is_holdout(s1_all)]
    res = {}
    for w in (1.0, 2.0):
        t, _ = tune_threshold(df[~df.ho.values], pcol, gt, tr_s1, fp_weight=w)
        m_ho = macro_f05(assign(df, pcol, t), gt, ho_s1)
        m_ho_w = macro_f05(assign(df, pcol, t), gt, ho_s1, fp_weight=w)
        log(f"{label} [fp_weight {w}] t={t}: HOLDOUT macro F0.5 {m_ho['macro_f05']:.4f} "
            f"(P {m_ho['pair_p']:.4f} R {m_ho['pair_r']:.4f} singletons {m_ho['singleton_acc']:.4f}); "
            f"under x{w} decoy density {m_ho_w['macro_f05']:.4f}")
        res[w] = (t, m_ho)
    return res


def stage_train(dev):
    import json
    import pickle
    W = wdir(dev)
    df = read_parquet(W / "feats_train.parquet")
    gt = read_parquet(gt_file(dev))
    cols = feat_cols(df)
    log(f"train: {len(df)} pairs, {len(cols)} features, positives {df.y.mean():.3f}")
    m1 = fit_oof(df, cols, "p1")
    ev1 = evaluate(df, "p1", gt, "pass1")
    imp = pd.Series(m1[0].feature_importance("gain"), index=cols).sort_values(ascending=False)
    log("top features: " + ", ".join(f"{k}={v:.0f}" for k, v in imp.head(15).items()))
    # pass 2: collective features from the pass-1 (OOF) assignment
    from .features import cluster_features
    norm = _load_norm(W, "train")
    t1 = ev1[1.0][0]
    cf = cluster_features(df, norm, "p1", t1, workers=C_WORKERS())
    del norm
    for c in cf.columns:
        df[c] = cf[c].values
    cols2 = cols + ["p1"] + list(cf.columns)
    m2 = fit_oof(df, cols2, "p2")
    ev2 = evaluate(df, "p2", gt, "pass2")
    imp = pd.Series(m2[0].feature_importance("gain"), index=cols2).sort_values(ascending=False)
    log("pass2 top features: " + ", ".join(f"{k}={v:.0f}" for k, v in imp.head(15).items()))
    M = {"cols": cols, "cols2": cols2, "p1": [m.model_to_string() for m in m1],
         "p2": [m.model_to_string() for m in m2], "t1": t1, "t2": {str(k): v[0] for k, v in ev2.items()}}
    final = "p2"
    if os.environ.get("BER_PASS3", "0") == "1":        # second collective round on pass-2 scores
        norm = _load_norm(W, "train")
        t2 = ev2[1.0][0]
        cf3 = cluster_features(df, norm, "p2", t2, workers=C_WORKERS(), prefix="k3")
        del norm
        for c in cf3.columns:
            df[c] = cf3[c].values
        cols3 = cols2 + ["p2"] + list(cf3.columns)
        m3 = fit_oof(df, cols3, "p3")
        ev3 = evaluate(df, "p3", gt, "pass3")
        M.update(cols3=cols3, p3=[m.model_to_string() for m in m3], t2_1=t2,
                 t3={str(k): v[0] for k, v in ev3.items()})
        final = "p3"
    M["final"] = final
    with open(W / "matcher.pkl", "wb") as f:
        pickle.dump(M, f)
    write_parquet(df[META + [c for c in ("p1", "p2", "p3") if c in df.columns]], W / "scores_train.parquet")


# ---------------------------------------------------------------- prediction
def stage_predict(dev):
    """Score test pairs, assign each record to its best S1 (if confident), write both submission files."""
    import pickle
    import lightgbm as lgb
    from .features import cluster_features
    from .model import assign
    W = wdir(dev)
    split = "train" if dev else "test"          # dev: exercise the path on the slice
    with open(W / "matcher.pkl", "rb") as f:
        M = pickle.load(f)
    df = read_parquet(W / f"feats_{split}.parquet")
    X = df[M["cols"]].values.astype(np.float32)
    df["p1"] = np.mean([lgb.Booster(model_str=s).predict(X, num_threads=C_WORKERS()) for s in M["p1"]], 0)
    norm = _load_norm(W, split)
    cf = cluster_features(df, norm, "p1", M["t1"], workers=C_WORKERS())
    for c in cf.columns:
        df[c] = cf[c].values
    X2 = df[M["cols2"]].values.astype(np.float32)
    df["p2"] = np.mean([lgb.Booster(model_str=s).predict(X2, num_threads=C_WORKERS()) for s in M["p2"]], 0)
    final = M.get("final", "p2")
    if final == "p3":
        cf3 = cluster_features(df, norm, "p2", M["t2_1"], workers=C_WORKERS(), prefix="k3")
        for c in cf3.columns:
            df[c] = cf3[c].values
        X3 = df[M["cols3"]].values.astype(np.float32)
        df["p3"] = np.mean([lgb.Booster(model_str=s).predict(X3, num_threads=C_WORKERS()) for s in M["p3"]], 0)
    write_parquet(df[["rec", "s1"] + [c for c in ("p1", "p2", "p3") if c in df.columns]], W / f"scores_{split}_pred.parquet")
    tkey = "t3" if final == "p3" else "t2"
    t = float(os.environ.get("BER_T", M[tkey][os.environ.get("BER_FPW", "2.0")]))
    pred = assign(df, final, t)
    s1_all = norm.index.values[norm["src"].values == 1]
    s1_country = norm.loc[s1_all, "country"]
    write_outputs(pred, df[["rec", "s1"]], s1_all, W if dev else C.OUTPUT_DIR)
    # sanity: per country predicted matches per S1, empty share, share of records claimed
    npred = pred.groupby("s1").size().reindex(s1_all).fillna(0)
    recs = norm[norm["src"].values > 1]
    for c in sorted(s1_country.unique()):
        m = (s1_country == c).values
        rc = (recs["country"] == c).values
        claimed = np.isin(recs.index.values[rc], pred.rec.values).mean()
        log(f"sanity {c}: S1 {m.sum()}, matches/S1 {npred.values[m].mean():.2f}, empty {(npred.values[m] == 0).mean():.3f}, "
            f"records claimed {claimed:.3f}  (train truth: 3.46 / 0.056 / 0.74)")
    log(f"threshold {t}, {len(pred)} matches")


def write_outputs(pred, cand, s1_all, out_dir):
    from .common import decode_ids
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, col, pairs in (("matching_results.tsv", "matched_entity_ids", pred[["rec", "s1"]]),
                              ("candidate_pairs.tsv", "candidate_entity_ids", cand)):
        pairs = pairs.drop_duplicates()
        lists = pairs.assign(r=decode_ids(pairs.rec.values)).groupby("s1")["r"].agg(",".join)
        ids = decode_ids(s1_all)
        vals = lists.reindex(s1_all).fillna("").values
        pd.DataFrame({"source1_entity_id": ids, col: vals}).to_csv(out_dir / fname, sep="\t", index=False)
        log(f"wrote {out_dir / fname}: {len(ids)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--split", default="train")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("BER_WORKERS", "8")))
    a = ap.parse_args()
    if a.stage == "norm":
        stage_norm(a.split, a.dev, a.workers)
    elif a.stage == "block":
        stage_block(a.split, a.dev, a.workers)
    elif a.stage == "prune":
        stage_prune(a.split, a.dev)
    elif a.stage == "feats":
        stage_feats(a.split, a.dev, a.workers)
    elif a.stage == "train":
        stage_train(a.dev)
    elif a.stage == "predict":
        stage_predict(a.dev)
    else:
        raise SystemExit(f"unknown stage {a.stage}")
