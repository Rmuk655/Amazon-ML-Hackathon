"""Stage 3b - supervised meta-blocking: shrink each S1 entity's candidate set before matching.

blocking.py proposes ~50 candidates per S1 entity from cheap keys. This stage scores every
candidate with a small LightGBM on the blocking signals (key score, key-type mask, ranks, forward/
reverse margins, TF-IDF cosine) plus three cheap comparisons (core-name token-set similarity,
address token-set similarity, house-number equality; ~14 us/pair, vectorised), and keeps only
pairs above a threshold. The threshold is chosen on out-of-fold predictions so that PRUNE_RECALL
of the true pairs blocking found survive. On a 20k-S1 slice: 57.6 -> 6.8 candidates per S1 at
99.7% of found true pairs kept.

    python prune.py fit                        # train candidates -> models/pruner.joblib, prunes train
    python prune.py apply --split test         # prunes test candidates + writes output/candidate_pairs.tsv
"""
import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import config as C
from matching import addr_parts

BLOCK_FEATS = ["src", "score", "mask", "rank", "tf_cos", "tf_rank", "fwd_n", "fwd_margin", "rev_rank",
               "rev_n", "rev_margin"]
CHEAP_FEATS = ["c_name", "c_addr", "c_house"]
MODEL = os.path.join(C.RUN_MODELS_DIR, "pruner.joblib")
FIT_MAX_PAIRS = int(os.environ.get("BER_PRUNE_FIT_PAIRS", "6000000"))   # training sample (hash-sampled by S1 entity)


def parts(split):
    return sorted(C.candidates_dir(split).glob("*.parquet"))


def load_text(split, ids):
    """entity_id -> (core name, romanised address) for the ids that occur in candidates."""
    cols = ["entity_id", "business_name_c4b", "business_address_rom"]
    out = []
    for n in (1, 2, 3):
        pf = pq.ParquetFile(C.processed_path(split, n))
        for b in pf.iter_batches(columns=cols, batch_size=1_000_000):
            x = b.to_pandas()
            out.append(x[x["entity_id"].isin(ids)])
    t = pd.concat(out, ignore_index=True).drop_duplicates("entity_id").set_index("entity_id").fillna("")
    parts = t["business_address_rom"].map({a: addr_parts(a) for a in t["business_address_rom"].unique()})
    t["_addr0"] = [x[0] for x in parts]         # parsed once per record, not once per pair
    t["_house"] = [x[1] for x in parts]
    return t


def cheap_feats(d, text):
    ia, ib = text.index.get_indexer(d["s1_id"].values), text.index.get_indexer(d["cand_id"].values)
    col = lambda c, i: np.where(i >= 0, text[c].to_numpy(object)[np.maximum(i, 0)], "")
    d["c_name"] = cpdist(col("business_name_c4b", ia), col("business_name_c4b", ib), scorer=fuzz.token_set_ratio,
                         workers=-1, dtype=np.float32) / 100
    d["c_addr"] = cpdist(col("_addr0", ia), col("_addr0", ib), scorer=fuzz.token_set_ratio, workers=-1,
                         dtype=np.float32) / 100
    ha, hb = col("_house", ia), col("_house", ib)
    d["c_house"] = ((ha != "") & (ha == hb)).astype(np.float32)
    return d


def X_of(d, feats):
    X = pd.DataFrame(index=d.index)
    for f in feats:
        X[f] = d[f].astype(np.float32) if f in d else np.float32(np.nan)
    return X


def apply_split(split, bundle, text=None, write_tsv=False):
    t0 = time.time()
    files = parts(split)
    if text is None:
        ids = set()
        for f in files:
            x = pd.read_parquet(f, columns=["s1_id", "cand_id"])
            ids.update(x["s1_id"].astype(str)); ids.update(x["cand_id"].astype(str))
        text = load_text(split, ids)
    before = after = tp_before = tp_after = 0
    thr_split = bundle["thr"]
    budget = os.environ.get("BER_PRUNE_SPLIT_BUDGET")          # optional: hold THIS split to <= budget candidates/S1
    if budget:
        ps = []
        for f in files:
            d0 = cheap_feats(pd.read_parquet(f), text)
            ps.append(bundle["model"].predict_proba(X_of(d0, bundle["feats"]))[:, 1])
        ps = np.sort(np.concatenate(ps))[::-1]
        n_s1_split = len(pd.read_parquet(C.PROCESSED_DIR / f"blocked_s1_{split}.parquet"))
        k = min(int(float(budget) * n_s1_split), len(ps) - 1)
        thr_split = max(thr_split, float(ps[k]))
        print(f"[prune {split}] candidate budget {budget}/S1 -> threshold {thr_split:.4f} (fit threshold {bundle['thr']:.4f})")
    for f in files:
        d = cheap_feats(pd.read_parquet(f), text)
        p = bundle["model"].predict_proba(X_of(d, bundle["feats"]))[:, 1]
        keep = p >= thr_split
        before, after = before + len(d), after + int(keep.sum())
        if "is_true" in d:
            tp_before += int(d["is_true"].sum())
            tp_after += int(d["is_true"][keep].sum())
        d = d[keep].assign(prune_p=p[keep].astype(np.float32))
        d.to_parquet(f, index=False)
    n_s1 = len(pd.read_parquet(C.PROCESSED_DIR / f"blocked_s1_{split}.parquet")) \
        if (C.PROCESSED_DIR / f"blocked_s1_{split}.parquet").exists() else 0
    msg = f"[prune {split}] pairs {before:,} -> {after:,}"
    if n_s1:
        msg += f" | per S1 {before / n_s1:.1f} -> {after / n_s1:.1f}"
    if tp_before:
        msg += f" | found true pairs kept {tp_after / tp_before:.4f}"
    print(msg + f" ({time.time() - t0:.0f}s)")
    if write_tsv:
        from blocking import write_tsv as wt
        wt(split, C.OUTPUT_DIR / "candidate_pairs.tsv")


def cmd_fit(a):
    import lightgbm as lgb
    t0 = time.time()
    files = parts("train")
    frac = 1.0
    n_all = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    if n_all > FIT_MAX_PAIRS:
        frac = FIT_MAX_PAIRS / n_all
    ids, sample = set(), []
    for f in files:                                     # sample per file: never hold every candidate at once
        x = pd.read_parquet(f)
        x = x[~C.is_holdout(x["s1_id"].values)]          # locked holdout never trains the pruner
        ids.update(x["s1_id"].astype(str)); ids.update(x["cand_id"].astype(str))
        if frac < 1.0:
            x = x[pd.util.hash_array(x["s1_id"].values) % 10_000 < int(frac * 10_000)]
        sample.append(x)
    d = pd.concat(sample, ignore_index=True)
    del sample
    text = load_text("train", ids)
    d = cheap_feats(d, text)
    feats = BLOCK_FEATS + CHEAP_FEATS
    X, y = X_of(d, feats), d["is_true"].astype(int).values
    print(f"[prune fit] {len(d):,} pairs, {y.sum():,} true ({time.time() - t0:.0f}s)")
    fold = pd.util.hash_array(d["s1_id"].values) % 3
    oof = np.zeros(len(d))
    mk = lambda: lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.08, max_bin=63,
                                    min_child_samples=50, n_jobs=-1, verbose=-1)
    for k in range(3):
        oof[fold == k] = mk().fit(X[fold != k], y[fold != k]).predict_proba(X[fold == k])[:, 1]
    # highest threshold that keeps PRUNE_RECALL of the true pairs blocking found
    ps = np.sort(oof[y == 1])
    thr_recall = float(ps[int(np.floor((1 - C.PRUNE_RECALL) * len(ps)))]) if len(ps) else 0.0
    n_s1 = d["s1_id"].nunique()
    # candidate budget (smaller candidate sets count in the ranking): lowest threshold giving <= budget per S1
    budget = float(os.environ.get("BER_PRUNE_MAX_CANDS", "5"))
    srt = np.sort(oof)[::-1]
    thr_budget = float(srt[min(int(budget * n_s1), len(srt) - 1)])
    rec_budget = float((oof[y == 1] >= thr_budget).mean()) if len(ps) else 1.0
    if thr_budget <= thr_recall:
        thr = min(thr_recall, max(C.PRUNE_MAX_THR, thr_budget))       # both targets met
    elif rec_budget >= C.PRUNE_RECALL - 0.0005:
        thr = thr_budget
    else:
        thr = min(thr_recall, C.PRUNE_MAX_THR)
        print(f"[prune fit] WARNING: <= {budget:g} candidates/S1 would keep only {rec_budget:.4f} of found true "
              f"pairs (< {C.PRUNE_RECALL}); keeping the recall target (threshold {thr:.4f})")
    print(f"[prune fit] recall-target threshold {thr_recall:.4f} | budget threshold {thr_budget:.4f} "
          f"(keeps {rec_budget:.4f}) -> using {thr:.4f}")
    keep = oof >= thr
    print(f"[prune fit] threshold {thr:.4f}: {len(d) / n_s1:.1f} -> {keep.sum() / n_s1:.1f} candidates per S1, "
          f"true pairs kept {y[keep].sum() / y.sum():.4f} (OOF)")
    bundle = {"model": mk().fit(X, y), "feats": feats, "thr": thr}
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    joblib.dump(bundle, MODEL)
    print(f"[prune fit] saved {MODEL}")
    apply_split("train", bundle, text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    p = sub.add_parser("apply")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--no-tsv", action="store_true")
    a = ap.parse_args()
    if a.cmd == "fit":
        cmd_fit(a)
    else:
        apply_split(a.split, joblib.load(MODEL), write_tsv=(a.split == "test" and not a.no_tsv))


if __name__ == "__main__":
    main()
