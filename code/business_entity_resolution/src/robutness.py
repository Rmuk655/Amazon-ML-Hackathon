"""Stage 6: robustness analysis (slice metrics, leave-country-out test, test-set shift check).

  python robustness.py slices     [--s1-frac 0.2]   # where does the model fail? per country / script pair / address / source
  python robustness.py holdout    [--s1-frac 0.2]   # leave-country-out: train without a country, test on it
  python robustness.py testshift                    # test predictions vs train: unseen countries, cross-country rate

Reuses features/model helpers from matching.py. Outputs CSVs to eda/.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matching as M

EDA = os.path.join(M.ROOT, "eda")


# --------------------------------------------------------------------------- metadata
def load_meta(path, need_ids, id_override=None):
    """id -> country_norm, script (first token of 'scripts'/'script' column if present)."""
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    id_col = M.detect_id(names, id_override)
    sc = next((c for c in ("scripts", "script") if c in names), None)
    want = [id_col] + [c for c in ["country_norm", sc] if c and c in names]
    parts = []
    for b in pf.iter_batches(columns=want, batch_size=500_000):
        d = b.to_pandas()
        d[id_col] = d[id_col].astype(str)
        if need_ids is not None:
            d = d[d[id_col].isin(need_ids)]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True).drop_duplicates(id_col).set_index(id_col)
    out = pd.DataFrame(index=df.index)
    out["country"] = df["country_norm"].fillna("").astype(str) if "country_norm" in df else ""
    if sc:
        out["script"] = df[sc].astype(str).str.extract(r"([A-Za-z]+)")[0].str.lower().fillna("unknown")
    else:
        out["script"] = "unknown"
    return out


def build(a):
    files = M.cand_files("train")
    c = pd.concat([M.read_cand_file(f, a.s1_frac) for f in files], ignore_index=True)
    if "is_true" not in c.columns:
        raise SystemExit("Train candidates lack is_true")
    procs = M.find_processed("train", M.parse_procs(a.proc_file))
    need = set(c["s1_id"]) | set(c["cand_id"])
    tabs = {s: M.load_table(procs[s], need, a.id_col) for s in (1, 2, 3)}
    X = M.compute_features(c, tabs, a.workers)
    meta = pd.concat([load_meta(procs[s], need, a.id_col) for s in (1, 2, 3)])
    meta = meta[~meta.index.duplicated()]
    ok = ~X[M.FEATS].isna().all(axis=1).values
    c, X = c[ok].reset_index(drop=True), X[ok].reset_index(drop=True)
    c["y"] = c["is_true"].astype(bool)
    c["s1_country"] = c["s1_id"].map(meta["country"]).fillna("")
    c["t_country"] = c["cand_id"].map(meta["country"]).fillna("")
    c["s1_script"] = c["s1_id"].map(meta["script"]).fillna("unknown")
    c["t_script"] = c["cand_id"].map(meta["script"]).fillna("unknown")
    c["script_pair"] = c["s1_script"] + ">" + c["t_script"]
    c["addr"] = np.where(X["a_missing"].values == 1, "addr_missing_either", "addr_present")
    c["fold"] = pd.util.hash_array(c["s1_id"].values) % 5
    return c, X


def fit_predict(X, y, tr, te, val_mod=5):
    """Train on rows `tr`, early-stop / threshold on a hashed slice of `tr`, predict rows `te`."""
    model, is_lgb = M.make_model()
    Xtr, ytr = X[tr], y[tr]
    inner = (np.arange(len(Xtr)) % val_mod) == 0
    if is_lgb:
        import lightgbm as lgb
        model.fit(Xtr[~inner], ytr[~inner], eval_set=[(Xtr[inner], ytr[inner])],
                  eval_metric="average_precision", callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(Xtr[~inner], ytr[~inner])
    thr = M.best_threshold(ytr[inner], model.predict_proba(Xtr[inner])[:, 1])
    return model.predict_proba(X[te])[:, 1], thr


def metrics(y, keep):
    p, r, f = M.prf(y, keep)
    return pd.Series({"pos": int(y.sum()), "pairs": len(y), "P": p, "R": r, "F1": f})


# --------------------------------------------------------------------------- commands
def cmd_slices(a):
    c, X = build(a)
    y = c["y"].values
    val = (c["fold"].values == 0)
    p, thr = fit_predict(X, y, ~val, val)
    v = c[val].copy()
    v["keep"] = p >= thr
    print(f"global threshold {thr:.3f}; overall:", metrics(v["y"].values, v["keep"].values).round(4).to_dict())
    rows = []
    for key in ("s1_country", "script_pair", "addr", "srcn"):
        g = v.groupby(key).apply(lambda d: metrics(d["y"].values, d["keep"].values), include_groups=False)
        g.insert(0, "slice", key)
        rows.append(g.reset_index().rename(columns={key: "value"}))
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(EDA, exist_ok=True)
    out.to_csv(os.path.join(EDA, "robustness_slices.csv"), index=False)
    big = out[out["pos"] >= a.min_pos].sort_values("F1")
    print(f"\nweakest slices (>= {a.min_pos} positives):")
    print(big.head(15).round(3).to_string(index=False))
    print("\nsaved eda/robustness_slices.csv")


def cmd_holdout(a):
    c, X = build(a)
    y = c["y"].values
    counts = c[c["y"]].groupby("s1_country").size().sort_values(ascending=False)
    countries = [k for k, n in counts.items() if n >= a.min_pos and k][: a.max_folds]
    if len(countries) < 2:
        raise SystemExit("Need at least 2 countries with enough positives for a holdout test")
    # in-distribution reference: same country rows, model trained on all countries (hashed split)
    val = (c["fold"].values == 0)
    p_id, thr_id = fit_predict(X, y, ~val, val)
    ref = c[val].assign(keep=p_id >= thr_id)
    rows = []
    for k in countries:
        te = (c["s1_country"] == k).values
        tr = ~te
        p, thr = fit_predict(X, y, tr, te)
        ho = metrics(y[te], p >= thr)
        r = ref[ref["s1_country"] == k]
        idf = metrics(r["y"].values, r["keep"].values) if len(r) else pd.Series(dtype=float)
        rows.append({"held_out": k, "pos": ho["pos"], "F1_heldout": ho["F1"], "F1_in_dist": idf.get("F1", np.nan),
                     "P_heldout": ho["P"], "R_heldout": ho["R"]})
    out = pd.DataFrame(rows)
    out["drop"] = out["F1_in_dist"] - out["F1_heldout"]
    os.makedirs(EDA, exist_ok=True)
    out.to_csv(os.path.join(EDA, "robustness_holdout.csv"), index=False)
    print(out.round(3).to_string(index=False))
    print(f"\nmean F1 drop when the country is unseen: {out['drop'].mean():.3f}")
    print("A large drop means country-specific artefacts are leaking into features; prefer script-agnostic ones.")


def cmd_testshift(a):
    sc = os.path.join(M.ROOT, "dataset", "processed", "matches_test_scored.parquet")
    if not os.path.exists(sc):
        raise SystemExit("Run matching.py predict first")
    m = pd.read_parquet(sc)
    m["s1_id"], m["cand_id"] = m["s1_id"].astype(str), m["cand_id"].astype(str)
    tp = M.find_processed("test", M.parse_procs(a.proc_file))
    trp = M.find_processed("train", M.parse_procs(a.proc_file))
    s1_test = load_meta(tp[1], None, a.id_col)
    s1_train = load_meta(trp[1], None, a.id_col)
    need = set(m["s1_id"]) | set(m["cand_id"])
    meta = pd.concat([load_meta(tp[s], need, a.id_col) for s in (1, 2, 3)])
    meta = meta[~meta.index.duplicated()]

    tc, trc = s1_test["country"].value_counts(), s1_train["country"].value_counts()
    unseen = [k for k in tc.index if k not in set(trc.index)]
    print(f"test S1 countries: {len(tc)}, train: {len(trc)}, unseen in train: {unseen or 'none'}")

    m["sc"] = m["s1_id"].map(meta["country"])
    m["tc"] = m["cand_id"].map(meta["country"])
    print(f"predicted cross-country rate on test: {(m['sc'] != m['tc']).mean():.4f}")
    files = M.cand_files("train")
    ref = pd.concat([M.read_cand_file(f, 0.2) for f in files], ignore_index=True)
    if "is_true" in ref.columns:
        trm = load_meta(trp[1], None, a.id_col)
        tm2 = pd.concat([load_meta(trp[s], set(ref["cand_id"]), a.id_col) for s in (2, 3)])
        pos = ref[ref["is_true"].astype(bool)]
        cc = (pos["s1_id"].map(trm["country"]).values != pos["cand_id"].map(tm2["country"]).values).mean()
        print(f"train ground-truth cross-country rate (sample): {cc:.4f}")

    matched = m.groupby("s1_id").size().index
    rate = (s1_test.assign(matched=s1_test.index.isin(matched)).groupby("country")["matched"]
            .agg(["mean", "size"]).rename(columns={"mean": "match_rate", "size": "n_s1"}))
    print("\nper-country S1 match rate on test (compare with train singleton rate):")
    print(rate.sort_values("n_s1", ascending=False).head(15).round(3).to_string())
    print(f"\noverall test match rate: {s1_test.index.isin(matched).mean():.3f}")
    os.makedirs(EDA, exist_ok=True)
    rate.to_csv(os.path.join(EDA, "test_shift_match_rate.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("slices", cmd_slices), ("holdout", cmd_holdout), ("testshift", cmd_testshift)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
        p.add_argument("--proc-file", nargs="*")
        p.add_argument("--id-col", default=None)
        p.add_argument("--s1-frac", type=float, default=0.2)
        p.add_argument("--min-pos", type=int, default=30)
        p.add_argument("--max-folds", type=int, default=6)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()