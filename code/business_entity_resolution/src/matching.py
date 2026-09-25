"""Stage 4: pairwise matching (candidate_pairs -> matching_results.tsv).

Inputs (from earlier stages)
  dataset/processed/*<split>*S1|S2|S3*.parquet   Stage 1 output (name_norm, name_rom, addr_norm, addr_rom, country_norm)
  dataset/processed/candidates_<split>/*.parquet Stage 3 output (s1_id, cand_id, src, score, ..., is_true on train)

Method
  1. Pair features: name / address similarity on normalized AND romanized text, consonant-skeleton
     similarity, postal-code and number overlap, missing flags, plus the blocking features.
  2. Gradient-boosted classifier trained on train candidates (is_true), split by S1 entity.
  3. Per-source probability thresholds tuned for F1 on the validation split.
  4. Decision rules (top-1 fallback, one S1 per S2/S3 entity) evaluated on validation; best combo saved.
  5. Predict on test and write output/matching_results.tsv.

Usage
  python matching.py train   --s1-frac 0.2
  python matching.py predict --format grouped
"""
import argparse
import glob
import os
import re
import sys
import time
from functools import lru_cache
from multiprocessing import Pool

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # .../student_resource
# Short internal feature-column names -> actual column names written by preprocess.py.
TEXT_COLS = ["name_norm", "name_rom", "addr_norm", "addr_rom", "country_norm"]
COL_MAP = {
    "name_norm": "business_name_norm", "name_rom": "business_name_rom",
    "addr_norm": "business_address_norm", "addr_rom": "business_address_rom",
    "country_norm": "country_norm",
}
ID_CANDIDATES = ["entity_id", "id", "source_entity_id", "record_id", "uid"]
RESERVED = {"s1_id", "cand_id", "src", "is_true"}

# --------------------------------------------------------------------------- features
_PIN = re.compile(r"\b\d{5,6}\b")
_NUM = re.compile(r"\b\d{2,}\b")
_VOW = re.compile(r"[aeiouy]")


@lru_cache(maxsize=2_000_000)
def _sk_tok(t):
    if not t:
        return t
    r = t[0] + _VOW.sub("", t[1:])
    return re.sub(r"(.)\1+", r"\1", r)


def skel(s):
    return " ".join(_sk_tok(t) for t in s.split())


def jacc(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


FEATS = [
    "n_norm_ratio", "n_norm_tset", "n_norm_tsort", "n_norm_jw",
    "n_rom_ratio", "n_rom_tset", "n_rom_tsort", "n_rom_jw",
    "n_skel_ratio", "n_skel_tset", "n_best", "n_exact_norm", "n_exact_rom", "n_exact_skel",
    "n_tok_jacc_rom", "n_first_tok_eq", "n_len_diff", "n_tok_cnt_diff", "n_partial_rom",
    "a_norm_tset", "a_rom_tset", "a_rom_partial", "a_tok_jacc_rom", "a_num_jacc", "a_pin_match",
    "a_pin_conflict", "c_eq", "n_missing", "a_missing",
]


def _feats(args):
    an, ar, aa, aar, ac, bn, br, ba, bar, bc = args
    out = np.zeros((len(an), len(FEATS)), dtype=np.float32)
    for k in range(len(an)):
        n1, r1, a1, ar1, c1 = an[k], ar[k], aa[k], aar[k], ac[k]
        n2, r2, a2, ar2, c2 = bn[k], br[k], ba[k], bar[k], bc[k]
        n_miss = float(not ((n1 or r1) and (n2 or r2)))
        a_miss = float(not ((a1 or ar1) and (a2 or ar2)))
        s1, s2 = skel(r1), skel(r2)
        t1, t2 = r1.split(), r2.split()
        f_norm = [fuzz.ratio(n1, n2) / 100, fuzz.token_set_ratio(n1, n2) / 100,
                  fuzz.token_sort_ratio(n1, n2) / 100, JaroWinkler.similarity(n1, n2)]
        f_rom = [fuzz.ratio(r1, r2) / 100, fuzz.token_set_ratio(r1, r2) / 100,
                 fuzz.token_sort_ratio(r1, r2) / 100, JaroWinkler.similarity(r1, r2)]
        sk_r = fuzz.ratio(s1, s2) / 100
        sk_t = fuzz.token_set_ratio(s1, s2) / 100
        best = max(f_norm[1], f_rom[1], sk_t, f_norm[0], f_rom[0])
        p1, p2 = set(_PIN.findall(a1 + " " + ar1)), set(_PIN.findall(a2 + " " + ar2))
        pin_match = float(bool(p1 & p2))
        pin_conf = float(bool(p1) and bool(p2) and not (p1 & p2))
        nums1, nums2 = _NUM.findall(a1 + " " + ar1), _NUM.findall(a2 + " " + ar2)
        row = (
            f_norm + f_rom
            + [sk_r, sk_t, best, float(bool(n1) and n1 == n2), float(bool(r1) and r1 == r2),
               float(bool(s1) and s1 == s2), jacc(t1, t2),
               float(bool(t1) and bool(t2) and t1[0] == t2[0]),
               abs(len(r1) - len(r2)) / max(len(r1), len(r2), 1), abs(len(t1) - len(t2)),
               fuzz.partial_ratio(r1, r2) / 100 if r1 and r2 else 0.0]
            + [fuzz.token_set_ratio(a1, a2) / 100 if a1 and a2 else 0.0,
               fuzz.token_set_ratio(ar1, ar2) / 100 if ar1 and ar2 else 0.0,
               fuzz.partial_ratio(ar1, ar2) / 100 if ar1 and ar2 else 0.0,
               jacc(ar1.split(), ar2.split()), jacc(nums1, nums2), pin_match, pin_conf,
               float(bool(c1) and c1 == c2), n_miss, a_miss]
        )
        out[k] = row
    return out


# --------------------------------------------------------------------------- tables
class Table:
    def __init__(self, df, id_col):
        df = df.drop_duplicates(id_col)
        self.index = pd.Index(df[id_col].astype(str).values)
        self.cols = {c: (df[COL_MAP[c]].fillna("").astype(str).values if COL_MAP[c] in df.columns
                         else np.array([""] * len(df), dtype=object)) for c in TEXT_COLS}

    def pos(self, ids):
        return self.index.get_indexer(pd.Index(ids))


def detect_id(cols, override=None):
    if override:
        return override
    for c in ID_CANDIDATES:
        if c in cols:
            return c
    for c in cols:
        if c.lower().endswith("id"):
            return c
    raise SystemExit(f"Cannot detect ID column in {cols}; pass --id-col")


def find_processed(split, overrides):
    found = dict(overrides)
    pat = re.compile(r"(?:^|[^a-z0-9])(?:s|source_?)([123])(?:[^0-9]|$)")
    for p in glob.glob(os.path.join(ROOT, "dataset", "processed", "**", "*.parquet"), recursive=True):
        rel = os.path.relpath(p, ROOT).lower()
        if "candidates_" in rel or split not in rel:
            continue
        m = pat.search(os.path.basename(p).lower())
        if m and int(m.group(1)) not in found:
            found[int(m.group(1))] = p
    missing = [s for s in (1, 2, 3) if s not in found]
    if missing:
        raise SystemExit(f"Could not locate processed parquet for source(s) {missing}. "
                         f"Use --proc-file 1=path 2=path 3=path")
    return found


def load_table(path, need_ids, id_override=None):
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    id_col = detect_id(names, id_override)
    want = [id_col] + [COL_MAP[c] for c in TEXT_COLS if COL_MAP[c] in names]
    for c in ("name_norm", "name_rom"):
        if COL_MAP[c] not in names:
            print(f"  WARNING {os.path.basename(path)} has no column {COL_MAP[c]}", file=sys.stderr)
    parts = []
    for b in pf.iter_batches(columns=want, batch_size=500_000):
        d = b.to_pandas()
        d[id_col] = d[id_col].astype(str)
        parts.append(d[d[id_col].isin(need_ids)])
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=want)
    print(f"  loaded {len(df):,} rows from {os.path.basename(path)} (id col '{id_col}')")
    return Table(df, id_col)


# --------------------------------------------------------------------------- candidates
def src_num(s):
    return pd.Series(s).astype(str).str.extract(r"(\d)")[0].astype(float).fillna(-1).astype(int).values


def read_cand_file(path, frac=1.0):
    d = pd.read_parquet(path)
    d["s1_id"] = d["s1_id"].astype(str)
    d["cand_id"] = d["cand_id"].astype(str)
    if frac < 1.0:
        h = pd.util.hash_array(d["s1_id"].values) % 10_000
        d = d[h < int(frac * 10_000)]
    d["srcn"] = src_num(d["src"].values)
    return d.reset_index(drop=True)


def cand_files(split):
    files = sorted(glob.glob(os.path.join(ROOT, "dataset", "processed", f"candidates_{split}", "*.parquet")))
    if not files:
        raise SystemExit(f"No candidate parquet files for split '{split}'. Run blocking.py first.")
    return files


def compute_features(c, tabs, workers, chunk=100_000):
    n = len(c)
    X = np.full((n, len(FEATS)), np.nan, dtype=np.float32)
    ia = tabs[1].pos(c["s1_id"].values)
    for s in (2, 3):
        m = np.where(c["srcn"].values == s)[0]
        if len(m) == 0:
            continue
        ib = tabs[s].pos(c["cand_id"].values[m])
        ok = (ia[m] >= 0) & (ib >= 0)
        if not ok.all():
            print(f"  WARNING: {(~ok).sum():,} pairs for S{s} missing from processed tables", file=sys.stderr)
        m, ib = m[ok], ib[ok]
        pa = ia[m]
        T1, TB = tabs[1].cols, tabs[s].cols

        def gen():
            for i in range(0, len(m), chunk):
                sl = slice(i, i + chunk)
                yield tuple([T1[k][pa[sl]].tolist() for k in TEXT_COLS[:4] + ["country_norm"]]
                            + [TB[k][ib[sl]].tolist() for k in TEXT_COLS[:4] + ["country_norm"]])

        if workers > 1:
            with Pool(workers) as p:
                res = list(p.imap(_feats, gen()))
        else:
            res = [_feats(a) for a in gen()]
        if res:
            X[m] = np.vstack(res)
    F = pd.DataFrame(X, columns=FEATS)
    F["srcn"] = c["srcn"].values
    for col in c.columns:
        if col not in RESERVED and col != "srcn" and pd.api.types.is_numeric_dtype(c[col]):
            F["blk_" + col] = c[col].values
    return F


# --------------------------------------------------------------------------- model / decisions
def make_model():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=800, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, min_child_samples=40,
                                  reg_lambda=1.0, verbose=-1), True
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63), False


def best_threshold(y, p):
    from sklearn.metrics import precision_recall_curve
    if y.sum() == 0:
        return 0.5
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-9)
    return float(th[int(np.argmax(f1))])


def decide(d, p, thr, fallback, exclusive, low_frac=0.5):
    """d: DataFrame with s1_id, cand_id, srcn. Returns boolean keep mask (aligned to d)."""
    x = pd.DataFrame({"s1": d["s1_id"].values, "c": d["cand_id"].values, "s": d["srcn"].values, "p": p})
    t = x["s"].map(thr).fillna(0.5).values
    keep = x["p"].values >= t
    x["keep"] = keep
    if fallback:
        g = x.groupby(["s1", "s"])
        best = g["p"].transform("max").values
        anyk = g["keep"].transform("any").values
        x["keep"] = keep | (~anyk & (x["p"].values == best) & (x["p"].values >= t * low_frac))
    if exclusive:
        k = x[x["keep"]].sort_values("p", ascending=False).drop_duplicates(["s", "c"])
        m = np.zeros(len(x), dtype=bool)
        m[k.index.values] = True
        x["keep"] = m
    return x["keep"].values


def prf(y, keep):
    tp = int((y & keep).sum())
    fp = int((~y & keep).sum())
    fn = int((y & ~keep).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


# --------------------------------------------------------------------------- commands
def cmd_train(a):
    t0 = time.time()
    files = cand_files("train")
    c = pd.concat([read_cand_file(f, a.s1_frac) for f in files], ignore_index=True)
    if "is_true" not in c.columns:
        raise SystemExit("Train candidates lack is_true; run blocking.py --split train --eval")
    y = c["is_true"].astype(bool).values
    print(f"train candidates: {len(c):,}, positives: {y.sum():,} ({y.mean():.3%})")
    procs = find_processed("train", parse_procs(a.proc_file))
    need = set(c["s1_id"]) | set(c["cand_id"])
    tabs = {s: load_table(procs[s], need, a.id_col) for s in (1, 2, 3)}
    X = compute_features(c, tabs, a.workers)
    print(f"features done in {time.time() - t0:.0f}s")
    ok = ~X[FEATS].isna().all(axis=1).values
    c, X, y = c[ok].reset_index(drop=True), X[ok].reset_index(drop=True), y[ok]

    val = (pd.util.hash_array(c["s1_id"].values) % 5 == 0)
    model, is_lgb = make_model()
    if is_lgb:
        import lightgbm as lgb
        model.fit(X[~val], y[~val], eval_set=[(X[val], y[val])], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(X[~val], y[~val])
    pv = model.predict_proba(X[val])[:, 1]
    yv, cv = y[val], c[val].reset_index(drop=True)
    thr = {s: best_threshold(yv[cv["srcn"].values == s], pv[cv["srcn"].values == s]) for s in (2, 3)}
    print("thresholds:", thr)

    best = None
    print(f"{'fallback':>9} {'exclusive':>10} {'P':>7} {'R':>7} {'F1':>7}   (within candidate set)")
    for fb in (False, True):
        for ex in (False, True):
            p, r, f = prf(yv, decide(cv, pv, thr, fb, ex))
            print(f"{str(fb):>9} {str(ex):>10} {p:7.4f} {r:7.4f} {f:7.4f}")
            if best is None or f > best[0]:
                best = (f, fb, ex)
    print(f"best rule: fallback={best[1]} exclusive={best[2]} F1={best[0]:.4f}")
    print("NOTE: recall here is over blocking candidates only; multiply by the blocking pair recall for end-to-end.")

    imp = getattr(model, "feature_importances_", None)
    if imp is not None:
        top = sorted(zip(X.columns, imp), key=lambda z: -z[1])[:12]
        print("top features:", ", ".join(f"{k}({v})" for k, v in top))
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    out = os.path.join(ROOT, "models", "stage4_model.joblib")
    joblib.dump({"model": model, "feats": list(X.columns), "thr": thr,
                 "fallback": best[1], "exclusive": best[2]}, out)
    print("saved", out)


def cmd_predict(a):
    bundle = joblib.load(os.path.join(ROOT, "models", "stage4_model.joblib"))
    model, cols, thr = bundle["model"], bundle["feats"], bundle["thr"]
    files = cand_files("test")
    procs = find_processed("test", parse_procs(a.proc_file))
    need = set()
    for f in files:
        d = pd.read_parquet(f, columns=["s1_id", "cand_id"])
        need |= set(d["s1_id"].astype(str)) | set(d["cand_id"].astype(str))
    tabs = {s: load_table(procs[s], need, a.id_col) for s in (1, 2, 3)}
    floor = min(thr.values()) * 0.5
    kept = []
    for i, f in enumerate(files):
        c = read_cand_file(f)
        X = compute_features(c, tabs, a.workers)
        for col in cols:
            if col not in X.columns:
                X[col] = np.nan
        p = model.predict_proba(X[cols])[:, 1]
        m = p >= floor
        kept.append(pd.DataFrame({"s1_id": c["s1_id"].values[m], "cand_id": c["cand_id"].values[m],
                                  "srcn": c["srcn"].values[m], "p": p[m]}))
        print(f"[{i + 1}/{len(files)}] {os.path.basename(f)}: {len(c):,} pairs, {m.sum():,} above floor")
    allp = pd.concat(kept, ignore_index=True)
    keep = decide(allp, allp["p"].values, thr, bundle["fallback"], bundle["exclusive"])
    res = allp[keep]
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    out = os.path.join(ROOT, "output", "matching_results.tsv")
    if a.format == "grouped":
        g = res.groupby("s1_id")["cand_id"].apply(lambda s: ",".join(sorted(s.unique()))).reset_index()
        g.columns = ["source1_entity_id", "matched_entity_ids"]
        g.to_csv(out, sep="\t", index=False)
    else:
        res[["s1_id", "cand_id"]].rename(columns={"s1_id": "source1_entity_id", "cand_id": "matched_entity_id"}) \
            .to_csv(out, sep="\t", index=False)
    res.to_parquet(os.path.join(ROOT, "dataset", "processed", "matches_test_scored.parquet"), index=False)
    print(f"wrote {out}: {res['s1_id'].nunique():,} S1 entities matched, {len(res):,} pairs")


def parse_procs(items):
    o = {}
    for it in items or []:
        k, v = it.split("=", 1)
        o[int(k)] = v
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("train", cmd_train), ("predict", cmd_predict)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
        p.add_argument("--proc-file", nargs="*", help="override processed files, e.g. 1=path 2=path 3=path")
        p.add_argument("--id-col", default=None, help="entity ID column name in processed parquet")
        if name == "train":
            p.add_argument("--s1-frac", type=float, default=1.0, help="hash-sample of S1 entities to train on")
        else:
            p.add_argument("--format", choices=["grouped", "long"], default="grouped")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()