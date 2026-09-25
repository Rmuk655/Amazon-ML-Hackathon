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

import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # .../student_resource
# Short internal feature-column names -> actual column names written by preprocess.py.
TEXT_COLS = ["name_norm", "name_rom", "addr_norm", "addr_rom", "country_norm", "core"]
COL_MAP = {
    "name_norm": "business_name_norm", "name_rom": "business_name_rom",
    "addr_norm": "business_address_norm", "addr_rom": "business_address_rom",
    "country_norm": "country_norm", "core": "business_name_c4b",
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
    for n in (1, 2, 3):
        if n not in found and C.processed_path(split, n).exists():
            found[n] = str(C.processed_path(split, n))
    missing = [s for s in (1, 2, 3) if s not in found]
    if missing:
        raise SystemExit(f"Could not locate processed parquet for source(s) {missing}. "
                         f"Use --proc-file 1=path 2=path 3=path")
    return found


def load_tables(procs, need, id_override=None):
    """S1/S2/S3 tables restricted to `need` ids, plus core-name counts over the WHOLE S1 file
    (chain-name ambiguity: how many S1 entities share a core name)."""
    tabs = {s: load_table(procs[s], need, id_override) for s in (1, 2, 3)}
    col = COL_MAP["core"]
    tabs["core_counts"] = (pd.read_parquet(procs[1], columns=[col])[col].fillna("").value_counts()
                           if col in pq.ParquetFile(procs[1]).schema_arrow.names else pd.Series(dtype=int))
    return tabs


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
    files = sorted(glob.glob(str(C.candidates_dir(split) / "*.parquet")))
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
    # chain-name ambiguity: a target whose core name is shared by many S1 entities is risky
    cc = tabs.get("core_counts")
    if cc is not None and len(cc):
        def core_at(tab, pos):                  # pos -1 = id not loaded -> ""
            v = tab.cols["core"][np.maximum(pos, 0)] if len(tab.index) else np.array([""] * len(pos), dtype=object)
            return np.where(pos >= 0, v, "")
        core_a = pd.Series(core_at(tabs[1], ia), dtype=object)
        core_b = pd.Series([""] * n, dtype=object)
        for s in (2, 3):
            m = np.where(c["srcn"].values == s)[0]
            core_b.iloc[m] = core_at(tabs[s], tabs[s].pos(c["cand_id"].values[m]))
        F["amb_s1_core_freq"] = core_a.map(cc).fillna(0).values.astype(np.float32)
        F["amb_t_core_freq"] = core_b.map(cc).fillna(0).values.astype(np.float32)
        F["amb_core_eq"] = (core_a.values == core_b.values) & (core_a.values != "")
    for col in c.columns:
        if col not in RESERVED and col != "srcn" and pd.api.types.is_numeric_dtype(c[col]):
            F["blk_" + col] = c[col].values
    return F


# --------------------------------------------------------------------------- model / decisions
def make_model():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, min_child_samples=40,
                                  reg_lambda=1.0, max_bin=63, force_col_wise=True, n_jobs=-1,
                                  verbose=-1), True
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63), False


def best_threshold(y, p):
    from sklearn.metrics import precision_recall_curve
    if y.sum() == 0:
        return 0.5
    # leaderboard metric is F0.5 (precision-weighted), so tune for it, not F1
    pr, rc, th = precision_recall_curve(y, p)
    b2 = 0.25
    f = (1 + b2) * pr[:-1] * rc[:-1] / np.maximum(b2 * pr[:-1] + rc[:-1], 1e-9)
    return float(th[int(np.argmax(f))])


def thr_of(thr, s, am):
    """Threshold for source s / address-missing flag am; falls back to the per-source value."""
    return thr.get((int(s), int(am)), thr.get(int(s), 0.5))


class Decider:
    """Decision layer with everything threshold-independent precomputed, so the rule/threshold
    search can evaluate hundreds of settings quickly. Same rules as before:
      thresholds per (source, address-missing) | top-1 fallback per (S1, source) at t*low_frac |
      margin: a pair must lead the best competing S1 for its target by >= margin |
      exclusive: each target keeps only its highest-probability S1."""

    def __init__(self, d, p, low_frac=0.5):
        self.p = np.asarray(p, np.float64)
        self.s = np.asarray(d["srcn"].values, int)
        am = d["a_missing"].fillna(0).values if "a_missing" in d else np.zeros(len(d))
        self.am = (np.asarray(am) > 0).astype(int)
        self.low = low_frac
        s1 = pd.Series(d["s1_id"].values).astype(str)
        tg = pd.Series(d["cand_id"].values).astype(str)
        self.g_ss = pd.factorize(s1 + "|" + self.s.astype(str))[0]
        self.g_t = pd.factorize(tg + "|" + self.s.astype(str))[0]
        x = pd.DataFrame({"g": self.g_ss, "t": self.g_t, "p": self.p})
        self.is_best = self.p == x.groupby("g")["p"].transform("max").values
        top1 = x.groupby("t")["p"].transform("max").values
        n_top = (self.p == top1).astype(int)
        n_top = pd.Series(n_top).groupby(self.g_t).transform("sum").values
        sec = x["p"].where(self.p < top1).groupby(self.g_t).transform("max").fillna(0).values
        self.lead = self.p - np.where((self.p == top1) & (n_top == 1), sec, top1)
        self.perm = np.argsort(-self.p, kind="stable")

    def compress(self, floor):
        """Drop rows that no setting can keep (p < floor); competitor margins were already computed
        on all rows. Calls then return a full-length mask. Makes the search ~n/n_kept faster."""
        idx = np.where(self.p >= floor)[0]
        sub = object.__new__(Decider)
        for k in ("p", "s", "am", "g_ss", "g_t", "is_best", "lead"):
            setattr(sub, k, getattr(self, k)[idx])
        sub.low, sub.perm, sub.n_full, sub.idx = self.low, np.argsort(-sub.p, kind="stable"), len(self.p), idx
        return sub

    def __call__(self, thr, fallback, exclusive, margin=0.0):
        keep = self._keep(thr, fallback, exclusive, margin)
        if getattr(self, "idx", None) is None:
            return keep
        full = np.zeros(self.n_full, bool)
        full[self.idx] = keep
        return full

    def _keep(self, thr, fallback, exclusive, margin=0.0):
        lut = np.array([[thr_of(thr, s_, a_) for a_ in (0, 1)] for s_ in range(4)])
        t = lut[np.clip(self.s, 0, 3), self.am]
        keep = self.p >= t
        if fallback:
            anyk = np.bincount(self.g_ss, weights=keep, minlength=self.g_ss.max() + 1 if len(keep) else 0) > 0
            keep = keep | (~anyk[self.g_ss] & self.is_best & (self.p >= t * self.low))
        if margin > 0:
            keep &= self.lead >= margin
        if exclusive:
            ks = self.perm[keep[self.perm]]
            _, first = np.unique(self.g_t[ks], return_index=True)
            keep = np.zeros(len(keep), bool)
            keep[ks[first]] = True
        return keep


def decide(d, p, thr, fallback, exclusive, margin=0.0, low_frac=0.5):
    """One-off decision (predict); see Decider."""
    if len(d) == 0:
        return np.zeros(0, bool)
    return Decider(d, p, low_frac)(thr, fallback, exclusive, margin)


def fbeta(p, r, b2=0.25):
    return (1 + b2) * p * r / np.maximum(b2 * p + r, 1e-9)


class MacroF05:
    """Leaderboard metric: F0.5 per S1 entity, averaged over S1 entities; a singleton scores 1 for
    an empty prediction and 0 otherwise. Built once, evaluated many times (bincount-fast).
    n_true: optional {s1_id: #true matches} from ground truth, so matches blocking never proposed
    still count as misses. extra_single / extra_other: S1 entities of the population that have
    no candidate at all (they score 1 / 0) - gives the end-to-end estimate."""

    def __init__(self, s1, y, n_true=None, extra_single=0, extra_other=0):
        self.codes, uniq = pd.factorize(pd.Series(s1).astype(str))
        self.G, self.y = len(uniq), np.asarray(y, bool)
        true_c = np.bincount(self.codes, weights=self.y, minlength=self.G)
        if n_true is not None:
            true_c = np.maximum(true_c, pd.Series(uniq).map(n_true).fillna(0).to_numpy())
        self.true, self.extra_single, self.extra_other = true_c, extra_single, extra_other

    def __call__(self, keep):
        keep = np.asarray(keep, bool)
        tp = np.bincount(self.codes, weights=self.y & keep, minlength=self.G)
        pred = np.bincount(self.codes, weights=keep, minlength=self.G)
        per = np.where((pred == 0) & (self.true == 0), 1.0,
                       fbeta(tp / np.maximum(pred, 1), tp / np.maximum(self.true, 1)))
        P = tp.sum() / max(pred.sum(), 1)
        R = tp.sum() / max(self.true.sum(), 1)
        e2e = (per.sum() + self.extra_single) / (self.G + self.extra_single + self.extra_other)
        return {"P": P, "R": R, "F0.5": float(fbeta(P, R)), "macro": float(per.mean()), "e2e": float(e2e)}


def prf(y, keep, s1):
    """(pooled P, R, F0.5, macro F0.5, macro F0.5 over non-singletons) within the candidate set."""
    ev = MacroF05(s1, y)
    m = ev(keep)
    per_has = ev.true > 0
    keep = np.asarray(keep, bool)
    tp = np.bincount(ev.codes, weights=ev.y & keep, minlength=ev.G)
    pred = np.bincount(ev.codes, weights=keep, minlength=ev.G)
    per = fbeta(tp / np.maximum(pred, 1), tp / np.maximum(ev.true, 1))
    mfx = float(per[per_has].mean()) if per_has.any() else float("nan")
    return m["P"], m["R"], m["F0.5"], m["macro"], mfx


# --------------------------------------------------------------------------- commands
def fit_one(X, y, inner, w=None):
    """Fit one model; LightGBM early-stops on the `inner` rows (a split of the training data)."""
    model, is_lgb = make_model()
    w = np.ones(len(y), np.float32) if w is None else w
    if is_lgb:
        import lightgbm as lgb
        model.fit(X[~inner], y[~inner], sample_weight=w[~inner],
                  eval_set=[(X[inner], y[inner])], eval_sample_weight=[w[inner]],
                  eval_metric="average_precision", callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(X, y, sample_weight=w)
    return model


def train_sample(X, y, h):
    """Keep every positive and every hard negative; keep easy negatives at rate NEG_RATE with
    weight 1/NEG_RATE (probabilities stay on the original scale). Deterministic via hash h."""
    hard = y | (X["n_best"].fillna(0).values >= HARD_NEG_SIM)
    if "blk_rank" in X:
        hard |= X["blk_rank"].fillna(127).values < 2        # the S1's own top key-blocking candidates
    easy_keep = ((h // 7) % 1000) < NEG_RATE * 1000
    keep = hard | easy_keep
    w = np.where(hard, 1.0, 1.0 / NEG_RATE).astype(np.float32)
    return keep, w


def tune_thresholds(ev, dec, thr, fb, ex, mg, rounds=2):
    """Coordinate search of every threshold directly on end-to-end macro F0.5."""
    grid = np.round(np.arange(0.20, 0.991, 0.02), 3)
    best = ev(dec(thr, fb, ex, mg))["e2e"]
    for _ in range(rounds):
        for key in list(thr):
            for t in grid:
                trial = dict(thr)
                trial[key] = float(t)
                sc = ev(dec(trial, fb, ex, mg))["e2e"]
                if sc > best + 1e-6:
                    best, thr = sc, trial
    return thr, best


def predict_p(bundle, X):
    """Mean of the fold models, then isotonic calibration."""
    p = np.mean([m.predict_proba(X)[:, 1] for m in bundle["models"]], axis=0)
    cal = bundle.get("calibrator")
    return cal.predict(p) if cal is not None else p


NEG_RATE = 0.2         # share of easy negatives kept for training (rest dropped, kept ones reweighted)
HARD_NEG_SIM = 0.8     # negatives at least this similar (n_best) are always kept


def load_gt_counts(procs, id_override=None):
    """{S1 id: #true matches}, counting only targets present in the processed S2/S3 tables
    (identical to the raw GT on full data; correct for --limit dev slices)."""
    gt = pd.read_csv(C.TRAIN_GT, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    flat = pd.DataFrame({"s1": gt.iloc[:, 0].str.strip(), "t": gt.iloc[:, 1].fillna("").str.split(",")}).explode("t")
    flat["t"] = flat["t"].fillna("").str.strip()
    have = pd.Index([])
    for n in (2, 3):
        col = detect_id(pq.ParquetFile(procs[n]).schema_arrow.names, id_override)
        have = have.append(pd.Index(pd.read_parquet(procs[n], columns=[col]).iloc[:, 0].astype(str)))
    flat["ok"] = flat["t"].isin(have)
    return flat.groupby("s1")["ok"].sum()


def cmd_train(a):
    from sklearn.isotonic import IsotonicRegression
    t0 = time.time()
    files = cand_files("train")
    c = pd.concat([read_cand_file(f, a.s1_frac) for f in files], ignore_index=True)
    if "is_true" not in c.columns:
        raise SystemExit("Train candidates lack is_true; run blocking.py --split train --eval")
    y = c["is_true"].astype(bool).values
    print(f"train candidates: {len(c):,}, positives: {y.sum():,} ({y.mean():.3%})")
    procs = find_processed("train", parse_procs(a.proc_file))
    need = set(c["s1_id"]) | set(c["cand_id"])
    tabs = load_tables(procs, need, a.id_col)
    X = compute_features(c, tabs, a.workers)
    print(f"features done in {time.time() - t0:.0f}s")
    ok = ~X[FEATS].isna().all(axis=1).values
    c, X, y = c[ok].reset_index(drop=True), X[ok].reset_index(drop=True), y[ok]

    # K-fold grouped by S1 entity (all candidates of one S1 stay in one fold) -> out-of-fold preds
    h = pd.util.hash_array(c["s1_id"].values)
    fold = (h % a.folds).astype(int)
    inner_all = (h // a.folds) % 10 == 0          # early-stopping split, independent of the fold id
    hp = pd.util.hash_array((c["s1_id"] + "|" + c["cand_id"]).values)
    samp, wts = train_sample(X, y, hp)
    print(f"training rows after easy-negative sampling: {samp.sum():,} of {len(c):,}")
    oof = np.zeros(len(c), np.float32)
    models = []
    for k in range(a.folds):
        te = fold == k
        tr = ~te & samp
        m = fit_one(X[tr].reset_index(drop=True), y[tr], inner_all[tr], wts[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        models.append(m)
        print(f"  fold {k}: train {tr.sum():,} / held-out {te.sum():,}  "
              f"trees {getattr(m, 'best_iteration_', None)}  ({time.time() - t0:.0f}s)")
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, y)
    pc = cal.predict(oof)
    from sklearn.metrics import average_precision_score, brier_score_loss
    print(f"OOF average precision {average_precision_score(y, oof):.4f} | Brier raw {brier_score_loss(y, oof):.4f}"
          f" -> calibrated {brier_score_loss(y, pc):.4f}")

    # thresholds per (source, address-missing) on calibrated OOF; fall back to per-source if sparse
    am = (X["a_missing"].fillna(0).values > 0).astype(int)
    thr = {}
    for s_ in (2, 3):
        ms = c["srcn"].values == s_
        thr[s_] = best_threshold(y[ms], pc[ms])
        for a_ in (0, 1):
            mm = ms & (am == a_)
            if y[mm].sum() >= 200:
                thr[(s_, a_)] = best_threshold(y[mm], pc[mm])
    print("thresholds:", {str(k): round(v, 4) for k, v in thr.items()})

    # exact leaderboard metric: GT match counts (blocking misses count) + S1 entities without candidates
    n_true = load_gt_counts(procs, a.id_col)
    pop = C.PROCESSED_DIR / "blocked_s1_train.parquet"         # written by blocking.py
    s1_all = (pd.read_parquet(pop).iloc[:, 0] if pop.exists() else pd.read_parquet(
        procs[1], columns=[detect_id(pq.ParquetFile(procs[1]).schema_arrow.names, a.id_col)]).iloc[:, 0]).astype(str)
    if a.s1_frac < 1.0:
        s1_all = s1_all[pd.util.hash_array(s1_all.values) % 10_000 < int(a.s1_frac * 10_000)]
    s1_all = s1_all[s1_all.isin(n_true.index)]
    absent = s1_all[~s1_all.isin(set(c["s1_id"]))]
    n_abs_single = int((absent.map(n_true) == 0).sum())
    ev = MacroF05(c["s1_id"].values, y, n_true.to_dict(), n_abs_single, len(absent) - n_abs_single)
    print(f"S1 population {len(s1_all):,}: {len(absent):,} without candidates "
          f"({n_abs_single:,} singletons score 1, the rest 0)")

    dec = Decider(c.assign(a_missing=X["a_missing"].values), pc).compress(0.2 * 0.5)   # min grid thr x low_frac
    print(f"decision search over {len(dec.p):,} rows with p >= 0.1 (of {len(pc):,})")
    best = None
    print(f"{'fallback':>9} {'exclusive':>10} {'margin':>7} {'P':>7} {'R':>7} {'F0.5':>7} {'macro(cands)':>12} "
          f"{'macro(e2e)':>10}   (OOF; R counts all GT matches)")
    for fb in (False, True):
        for ex in (False, True):
            for mg in (0.0, 0.05, 0.1, 0.2):
                m = ev(dec(thr, fb, ex, mg))
                print(f"{str(fb):>9} {str(ex):>10} {mg:7.2f} {m['P']:7.4f} {m['R']:7.4f} {m['F0.5']:7.4f} "
                      f"{m['macro']:12.4f} {m['e2e']:10.4f}")
                if best is None or m["e2e"] > best[0]:
                    best = (m["e2e"], fb, ex, mg)
    print(f"best rule: fallback={best[1]} exclusive={best[2]} margin={best[3]} macroF0.5(e2e)={best[0]:.4f}")
    thr, sc = tune_thresholds(ev, dec, thr, best[1], best[2], best[3])
    print(f"decision search done ({time.time() - t0:.0f}s)")
    print("thresholds tuned on macro F0.5:", {str(k): round(v, 3) for k, v in thr.items()},
          f"-> macroF0.5(e2e)={sc:.4f}  <- expected leaderboard score (OOF estimate)")
    best = (sc,) + best[1:]

    imp = getattr(models[0], "feature_importances_", None)
    if imp is not None:
        top = sorted(zip(X.columns, imp), key=lambda z: -z[1])[:15]
        print("top features:", ", ".join(f"{k}({v})" for k, v in top))
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    out = os.path.join(ROOT, "models", "stage4_model.joblib")
    joblib.dump({"models": models, "calibrator": cal, "feats": list(X.columns), "thr": thr,
                 "fallback": best[1], "exclusive": best[2], "margin": best[3]}, out)
    print("saved", out)


def cmd_predict(a):
    bundle = joblib.load(os.path.join(ROOT, "models", "stage4_model.joblib"))
    if "models" not in bundle:                         # older single-model bundle
        bundle = dict(bundle, models=[bundle["model"]], calibrator=None, margin=0.0)
    cols, thr = bundle["feats"], bundle["thr"]
    files = cand_files("test")
    procs = find_processed("test", parse_procs(a.proc_file))
    need = set()
    for f in files:
        d = pd.read_parquet(f, columns=["s1_id", "cand_id"])
        need |= set(d["s1_id"].astype(str)) | set(d["cand_id"].astype(str))
    tabs = load_tables(procs, need, a.id_col)
    floor = min(thr.values()) * 0.5
    kept = []
    for i, f in enumerate(files):
        c = read_cand_file(f)
        X = compute_features(c, tabs, a.workers)
        for col in cols:
            if col not in X.columns:
                X[col] = np.nan
        p = predict_p(bundle, X[cols])
        m = p >= floor
        kept.append(pd.DataFrame({"s1_id": c["s1_id"].values[m], "cand_id": c["cand_id"].values[m],
                                  "srcn": c["srcn"].values[m], "a_missing": X["a_missing"].values[m], "p": p[m]}))
        print(f"[{i + 1}/{len(files)}] {os.path.basename(f)}: {len(c):,} pairs, {m.sum():,} above floor")
    allp = pd.concat(kept, ignore_index=True)
    keep = decide(allp, allp["p"].values, thr, bundle["fallback"], bundle["exclusive"], bundle.get("margin", 0.0))
    res = allp[keep]
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    out = os.path.join(ROOT, "output", "matching_results.tsv")
    # validator needs one row per test S1 entity (empty string = no match)
    s1_ids = pd.read_csv(os.path.join(ROOT, "dataset", "test", "test_source1.tsv"), sep="\t", dtype=str,
                         usecols=[0], keep_default_na=False, quoting=3).iloc[:, 0].str.strip()
    g = res.groupby("s1_id")["cand_id"].apply(lambda s: ",".join(sorted(s.unique())))
    g = g.reindex(pd.unique(s1_ids)).fillna("").reset_index()
    g.columns = ["source1_entity_id", "matched_entity_ids"]
    g.to_csv(out, sep="\t", index=False)
    res.to_parquet(C.PROCESSED_DIR / "matches_test_scored.parquet", index=False)
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
            p.add_argument("--folds", type=int, default=5, help="grouped CV folds (by S1 entity)")
        else:
            p.add_argument("--format", choices=["grouped"], default="grouped")  # only format the validator accepts
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()