"""Pair features for (record, S1) candidates.

All string similarities use rapidfuzz's vectorised pairwise `cpdist`; set/number features are plain Python
over pre-split token lists. No country indicator is used (France is unseen in train): only script / field
presence flags, so the model transfers to new countries.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz, process

from .common import log

CHEAP = ["c_name", "c_addr", "c_comb", "c_pair", "r_name", "r_addr", "r_comb", "r_pair", "m_name", "m_addr",
         "m_comb", "m_pair",
         "n_cand", "s1_top_cnt", "hn_eq", "glued_eq", "src"]


def _side(norm: pd.DataFrame, ids: np.ndarray, cols):
    """Column values of `norm` (indexed by id) aligned to ids."""
    pos = norm.index.get_indexer(ids)
    return {c: norm[c].values[pos] for c in cols}


def _first(s):
    return s.split(" ", 1)[0] if s else ""


def _codes(norm: pd.DataFrame):
    """Integer codes for the first house number and glued name of every record (cached on the frame)."""
    if "_hn_code" not in norm.columns:
        hn = norm["anum"].fillna("").astype(str).str.split(" ", n=1).str[0]
        codes, _ = pd.factorize(hn)
        codes[hn.values == ""] = -1
        norm["_hn_code"] = codes.astype(np.int64)
        norm["_glued_code"] = pd.factorize(norm["glued"].fillna("").astype(str))[0].astype(np.int64)
    return norm


def cheap_features(cand: pd.DataFrame, norm: pd.DataFrame) -> pd.DataFrame:
    """Features for the pruner: blocking cosines/ranks + first-house-number and glued-name equality."""
    _codes(norm)
    pa = norm.index.get_indexer(cand.s1.values)
    pb = norm.index.get_indexer(cand.rec.values)
    ha, hb = norm["_hn_code"].values[pa], norm["_hn_code"].values[pb]
    cand["hn_eq"] = np.where((ha < 0) | (hb < 0), 0.5, (ha == hb)).astype(np.float32)
    ga, gb = norm["_glued_code"].values[pa], norm["_glued_code"].values[pb]
    cand["glued_eq"] = (ga == gb).astype(np.float32)
    cand["src"] = (cand.rec.values // 10 ** 10).astype(np.float32)
    return cand


def _cp(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32).astype(np.float32)


def _set_feats(a_list, b_list, idf, default_idf):
    """jaccard, idf-weighted overlap (share of union weight), share of rec weight covered, share of S1 weight covered."""
    n = len(a_list)
    jac = np.zeros(n, np.float32)
    wov = np.zeros(n, np.float32)
    cov_b = np.zeros(n, np.float32)
    cov_a = np.zeros(n, np.float32)
    for i in range(n):
        A = set(a_list[i].split()) if a_list[i] else set()
        B = set(b_list[i].split()) if b_list[i] else set()
        if not A or not B:
            jac[i] = wov[i] = cov_a[i] = cov_b[i] = -1
            continue
        inter = A & B
        uni = A | B
        jac[i] = len(inter) / len(uni)
        wi = sum(idf.get(t, default_idf) for t in inter)
        wu = sum(idf.get(t, default_idf) for t in uni)
        wa = sum(idf.get(t, default_idf) for t in A)
        wb = sum(idf.get(t, default_idf) for t in B)
        wov[i] = wi / wu
        cov_a[i] = wi / wa
        cov_b[i] = wi / wb
    return jac, wov, cov_a, cov_b


def _num_feats(a_list, b_list, pa, pb):
    n = len(a_list)
    out = np.zeros((n, 7), np.float32)
    for i in range(n):
        A = a_list[i].split() if a_list[i] else []
        B = b_list[i].split() if b_list[i] else []
        if not A or not B:
            out[i] = [-1, -1, -1, -1, -1, len(A), len(B)]
        else:
            sa, sb = set(A), set(B)
            first_eq = float(A[0] == B[0])
            b0_in_a = float(B[0] in sa)
            jac = len(sa & sb) / len(sa | sb)
            try:
                d = abs(int(A[0]) - int(B[0]))
                diff = min(d, 1000) / 1000.0
                rel = d / max(int(A[0]), 1)
            except ValueError:
                diff, rel = -1, -1
            out[i] = [first_eq, b0_in_a, jac, diff, min(rel, 10), len(A), len(B)]
    post = np.zeros(n, np.float32)
    for i in range(n):
        x, y = pa[i] or "", pb[i] or ""
        post[i] = -1 if (not x or not y) else float(bool(set(x.split()) & set(y.split())))
    return out, post


def _legal_feats(la, lb):
    n = len(la)
    out = np.zeros((n, 4), np.float32)
    for i in range(n):
        A = set(la[i].split()) if la[i] else set()
        B = set(lb[i].split()) if lb[i] else set()
        out[i] = [float(A == B), len(B - A), len(A - B), float(bool(A) and bool(B) and not (A & B))]
    return out


def build_idf(norm_s1: pd.DataFrame, col):
    df = Counter()
    for s in norm_s1[col].values:
        if s:
            df.update(set(s.split()))
    n = max(len(norm_s1), 1)
    return {t: math.log(1 + n / c) for t, c in df.items()}, math.log(1 + n)


def pair_features(cand: pd.DataFrame, norm: pd.DataFrame, stats: dict, workers=8) -> pd.DataFrame:
    """Full feature matrix for pruned candidates. `stats` holds idf tables and name-frequency counts."""
    cols = ["core", "legal", "glued", "atok", "anum", "post", "indic"]
    A = _side(norm, cand.s1.values, cols)
    B = _side(norm, cand.rec.values, cols)
    for d in (A, B):
        for c in cols[:-1]:
            d[c] = np.array([x if isinstance(x, str) else "" for x in d[c]], dtype=object)
    f = {}
    ca, cb = list(A["core"]), list(B["core"])
    f["n_ratio"] = _cp(ca, cb, fuzz.ratio, workers)
    f["n_tset"] = _cp(ca, cb, fuzz.token_set_ratio, workers)
    f["n_tsort"] = _cp(ca, cb, fuzz.token_sort_ratio, workers)
    f["n_partial"] = _cp(ca, cb, fuzz.partial_ratio, workers)
    ga, gb = list(A["glued"]), list(B["glued"])
    f["g_jw"] = _cp(ga, gb, distance.JaroWinkler.normalized_similarity, workers)
    f["g_lev"] = _cp(ga, gb, distance.Levenshtein.distance, workers)
    f["g_eq"] = (A["glued"] == B["glued"]).astype(np.float32)
    f["g_contain"] = np.array([float(bool(x) and bool(y) and (x in y or y in x)) for x, y in zip(ga, gb)], np.float32)
    jac, wov, cov_a, cov_b = _set_feats(ca, cb, stats["idf_name"], stats["idf_name_def"])
    f.update(n_jac=jac, n_wov=wov, n_cov_s1=cov_a, n_cov_rec=cov_b)
    f["n_ntok_s1"] = np.array([len(x.split()) for x in ca], np.float32)
    f["n_ntok_rec"] = np.array([len(x.split()) for x in cb], np.float32)
    lg = _legal_feats(A["legal"], B["legal"])
    f.update(l_eq=lg[:, 0], l_added=lg[:, 1], l_dropped=lg[:, 2], l_conflict=lg[:, 3])
    aa, ab = list(A["atok"]), list(B["atok"])
    f["a_tset"] = _cp(aa, ab, fuzz.token_set_ratio, workers)
    f["a_ratio"] = _cp(aa, ab, fuzz.ratio, workers)
    f["a_partial"] = _cp(aa, ab, fuzz.partial_token_set_ratio, workers)
    jac, wov, cov_a, cov_b = _set_feats(aa, ab, stats["idf_addr"], stats["idf_addr_def"])
    f.update(a_jac=jac, a_wov=wov, a_cov_s1=cov_a, a_cov_rec=cov_b)
    nf, post = _num_feats(A["anum"], B["anum"], A["post"], B["post"])
    for j, k in enumerate(["h_first_eq", "h_b0_in_a", "h_jac", "h_diff", "h_rel", "h_n_s1", "h_n_rec"]):
        f[k] = nf[:, j]
    f["post_eq"] = post
    f["a_empty_rec"] = np.array([float(not x and not y) for x, y in zip(ab, B["anum"])], np.float32)
    f["indic_rec"] = B["indic"].astype(np.float32)
    # name frequency: how many S1 (same country) share the glued core name / its first word
    f["freq_glued"] = stats["glued_cnt"].reindex(A["glued"]).fillna(0).values.astype(np.float32)
    f["freq_first"] = stats["first_cnt"].reindex([_first(x) for x in ca]).fillna(0).values.astype(np.float32)
    out = pd.DataFrame(f)
    for c in CHEAP + ["p_prune", "pr_rank", "pr_margin"]:
        if c in cand.columns:
            out[c] = cand[c].values.astype(np.float32)
    return out


def make_stats(norm: pd.DataFrame) -> dict:
    s1 = norm[norm.src == 1]
    idf_n, dn = build_idf(s1, "core")
    idf_a, da = build_idf(s1, "atok")
    key = s1["country"].astype(str) + "|"
    return {"idf_name": idf_n, "idf_name_def": dn, "idf_addr": idf_a, "idf_addr_def": da,
            "glued_cnt": s1["glued"].astype(str).value_counts(),
            "first_cnt": s1["core"].astype(str).str.split(" ", n=1).str[0].value_counts()}


def cluster_features(df: pd.DataFrame, norm: pd.DataFrame, pcol: str, t: float, workers=8, prefix="cl") -> pd.DataFrame:
    """Pass-2 (collective) features: compare each candidate record with the other records that pass 1 assigned
    to the same S1 (its tentative cluster). Replaced names / missing addresses are supported by the cluster;
    decoys (changed house number, added legal form) disagree with it."""
    p = df[pcol].values
    g = df.groupby("rec", sort=False)[pcol]
    mx = g.transform("max").values
    second = np.zeros(len(df), np.float32)
    srt = df[["rec", pcol]].sort_values(["rec", pcol], ascending=[True, False], kind="stable")
    rk = srt.groupby("rec", sort=False).cumcount().values
    sec = pd.Series(srt[pcol].values[rk == 1], index=srt["rec"].values[rk == 1])
    second = sec.reindex(df["rec"].values).fillna(0).values.astype(np.float32)
    out = pd.DataFrame(index=df.index)
    out[pcol + "_gap"] = np.where(p >= mx, p - second, p - mx).astype(np.float32)
    is_top = p >= mx
    members = df.loc[is_top & (p >= t), ["rec", "s1", pcol]].rename(columns={"rec": "m", pcol: "pm"})
    top_cnt = df.loc[is_top, "s1"].value_counts()
    out[pcol + "_s1top"] = df["s1"].map(top_cnt).fillna(0).values.astype(np.float32)
    out[pcol + "_s1n"] = df["s1"].map(members["s1"].value_counts()).fillna(0).values.astype(np.float32)
    # pairs (candidate rec, other member of that S1's cluster)
    pr = df[["rec", "s1"]].reset_index().merge(members, on="s1", how="inner")
    pr = pr[pr["m"].values != pr["rec"].values]
    k = pr.groupby("index").size()
    out["cl_n"] = k.reindex(df.index).fillna(0).values.astype(np.float32)
    out["cl_same_src"] = pr[(pr.m.values // 10 ** 10) == (pr.rec.values // 10 ** 10)].groupby("index").size() \
        .reindex(df.index).fillna(0).values.astype(np.float32)
    out["cl_pmean"] = pr.groupby("index")["pm"].mean().reindex(df.index).fillna(-1).values.astype(np.float32)
    cols = ["core", "glued", "atok", "anum", "legal"]
    A = _side(norm, pr["rec"].values, cols)
    B = _side(norm, pr["m"].values, cols)
    for d in (A, B):
        for c in cols:
            d[c] = [x if isinstance(x, str) else "" for x in d[c]]
    sims = pd.DataFrame({
        "index": pr["index"].values,
        "cn": _cp(A["core"], B["core"], fuzz.token_set_ratio, workers),
        "cg": _cp(A["glued"], B["glued"], distance.JaroWinkler.normalized_similarity, workers),
        "ca": _cp(A["atok"], B["atok"], fuzz.token_set_ratio, workers),
        "ch": np.array([(-1.0 if (not x or not y) else float(x.split(" ", 1)[0] == y.split(" ", 1)[0]))
                        for x, y in zip(A["anum"], B["anum"])], np.float32),
        "cl": np.array([float(x == y) for x, y in zip(A["legal"], B["legal"])], np.float32),
    })
    agg = sims.groupby("index").agg(cl_name_max=("cn", "max"), cl_name_mean=("cn", "mean"),
                                    cl_glued_max=("cg", "max"), cl_addr_max=("ca", "max"),
                                    cl_addr_mean=("ca", "mean"), cl_hn_agree=("ch", "mean"),
                                    cl_hn_any=("ch", "max"), cl_legal_agree=("cl", "mean"))
    for c in agg.columns:
        out[c] = agg[c].reindex(df.index).fillna(-2).values.astype(np.float32)
    if prefix != "cl":
        out.columns = [prefix + c[2:] if c.startswith("cl_") else c for c in out.columns]
    return out
