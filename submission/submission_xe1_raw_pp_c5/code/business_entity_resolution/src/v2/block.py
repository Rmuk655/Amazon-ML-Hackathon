"""Record-centric blocking: for every S2/S3 record, retrieve S1 candidates of the same country.

Three TF-IDF channels, each a sparse cosine top-K against the S1 matrix of the country:
  name  core-name words + character 3-grams of the glued core name (typos, glued words, website forms)
  addr  address words + house/unit numbers + postcodes (catches replaced names)
  comb  name words + address words/numbers together (common names disambiguated by the address)
Features whose S1 document frequency exceeds a cap are dropped (uninformative and expensive).
The union of the three top-K lists is scored by all three cosines; `prune` later keeps the best few.
"""
import math
import os
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from .common import log

K_NAME, K_ADDR, K_COMB = int(os.environ.get("BER_K_NAME", "10")), int(os.environ.get("BER_K_ADDR", "8")), \
    int(os.environ.get("BER_K_COMB", "10"))
K_KEY = int(os.environ.get("BER_K_KEY", "15"))
K_PAIR = int(os.environ.get("BER_K_PAIR", "10"))


def name_feats(core, glued):
    f = ["w:" + t for t in core.split()]
    g = "#" + glued + "#"
    f += ["g:" + g[i:i + 3] for i in range(len(g) - 2)]
    return f


def addr_feats(atok, anum, post):
    f = ["a:" + t for t in atok.split()] + ["p:" + t for t in post.split()]
    for t in anum.split():
        f.append("n:" + t)
        if len(t) >= 3:                        # digit dropped / changed at either end still shares a key
            f.append("x:" + t[:3])
            f.append("y:" + t[-3:])
    return f


def comb_feats(core, atok, anum, post):
    return ["w:" + t for t in core.split()] + addr_feats(atok, anum, post)


def _build(docs_s1, docs_q, max_df, weight=None):
    """TF-IDF (binary tf) matrices for S1 docs and query docs over the S1 vocabulary, rows L2-normalised."""
    df = Counter()
    for d in docs_s1:
        df.update(set(d))
    n = len(docs_s1)
    vocab = {}
    idf = []
    for f, c in df.items():
        if c <= max_df:
            vocab[f] = len(vocab)
            w = math.log(1 + n / c)
            if weight:
                w *= weight(f)
            idf.append(w)
    idf = np.asarray(idf, np.float32)

    def mat(docs):
        indptr = [0]
        indices = []
        for d in docs:
            ids = {vocab[f] for f in d if f in vocab}
            indices.extend(ids)
            indptr.append(len(indices))
        indices = np.asarray(indices, np.int32)
        m = sp.csr_matrix((idf[indices], indices, np.asarray(indptr, np.int64)), shape=(len(docs), len(vocab)))
        norms = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
        norms[norms == 0] = 1
        return sp.diags(1 / norms).dot(m).astype(np.float32).tocsr()

    return mat(docs_s1), mat(docs_q)


def pair_feats(core, atok, anum, post):
    """Composite name-word x address-item features: rare even when both parts are common."""
    ws = core.split()[:4]
    xs = atok.split()[-6:] + anum.split()[:3] + post.split()[:1]
    return ["q:" + w + "|" + x for w in ws for x in xs]


def _weight(f):
    return {"g": 0.6, "n": 1.5, "p": 1.2, "x": 0.5, "y": 0.5}.get(f[0], 1.0)


def _topk(Q, S, k, threads, chunk=500_000):
    rows, cols, vals = [], [], []
    ST = S.T.tocsr()
    for i in range(0, Q.shape[0], chunk):
        C = sp_matmul_topn(Q[i:i + chunk], ST, top_n=k, threshold=0.05, n_threads=threads).tocoo()
        rows.append(C.row + i)
        cols.append(C.col)
        vals.append(C.data)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def _rowdot(A, B, ia, ib, chunk=2_000_000):
    out = np.empty(len(ia), np.float32)
    for i in range(0, len(ia), chunk):
        out[i:i + chunk] = np.asarray(A[ia[i:i + chunk]].multiply(B[ib[i:i + chunk]]).sum(1)).ravel()
    return out


def block_country(s1: pd.DataFrame, rec: pd.DataFrame, threads=8) -> pd.DataFrame:
    """Candidate pairs (rec, s1) with the three channel cosines, for one country partition."""
    s1_core, s1_glued = s1.core.fillna("").values, s1.glued.fillna("").values
    q_core, q_glued = rec.core.fillna("").values, rec.glued.fillna("").values
    s1_a = (s1.atok.fillna("").values, s1.anum.fillna("").values, s1.post.fillna("").values)
    q_a = (rec.atok.fillna("").values, rec.anum.fillna("").values, rec.post.fillna("").values)
    n1 = len(s1)
    # absolute document-frequency caps keep the sparse products linear in the number of records
    cap_small = int(os.environ.get("BER_CAP_SMALL", "3000"))
    cap_big = int(os.environ.get("BER_CAP_BIG", "12000"))

    mats = {}
    mats["name"] = _build([name_feats(c, g) for c, g in zip(s1_core, s1_glued)],
                          [name_feats(c, g) for c, g in zip(q_core, q_glued)], cap_small, _weight)
    mats["addr"] = _build([addr_feats(*x) for x in zip(*s1_a)], [addr_feats(*x) for x in zip(*q_a)], cap_small, _weight)
    mats["comb"] = _build([comb_feats(c, *x) for c, *x in zip(s1_core, *s1_a)],
                          [comb_feats(c, *x) for c, *x in zip(q_core, *q_a)], cap_big, _weight)
    mats["pair"] = _build([pair_feats(c, *x) for c, *x in zip(s1_core, *s1_a)],
                          [pair_feats(c, *x) for c, *x in zip(q_core, *q_a)], cap_small, _weight)
    log(f"  matrices built: {n1} S1, {len(rec)} records")

    parts = []
    for ch, k in (("name", K_NAME), ("addr", K_ADDR), ("comb", K_COMB), ("pair", K_PAIR)):
        S, Q = mats[ch]
        r, c, _ = _topk(Q, S, k, threads)
        parts.append(np.stack([r, c], 1))
        log(f"  {ch}: {len(r)} hits")
    # same-name key channel: all S1 sharing the glued (or sorted-token) core name, ranked by address cosine
    parts.append(_key_channel(s1, rec, mats["addr"]))
    log(f"  key: {len(parts[-1])} hits")
    rc = np.unique(np.concatenate(parts).astype(np.int64), axis=0)
    ir, ic = rc[:, 0], rc[:, 1]
    out = pd.DataFrame({"rec": rec.id.values[ir], "s1": s1.id.values[ic]})
    for ch in ("name", "addr", "comb", "pair"):
        S, Q = mats[ch]
        out["c_" + ch] = _rowdot(Q, S, ir, ic)
    return out


def _sorted_key(core):
    return "".join(sorted(core.split()))


def _key_channel(s1, rec, addr_mats, max_group=500, keep=K_KEY):
    S, Q = addr_mats
    out = []
    for key_fn in (lambda d: d.glued.fillna("").astype(str).values,
                   lambda d: np.array([_sorted_key(c) for c in d.core.fillna("").astype(str).values], dtype=object)):
        ks, kq = key_fn(s1), key_fn(rec)
        a = pd.DataFrame({"k": ks, "i": np.arange(len(s1))})
        a = a[a.k != ""]
        sz = a.groupby("k")["i"].transform("size")
        a = a[sz.values <= max_group]
        b = pd.DataFrame({"k": kq, "r": np.arange(len(rec))})
        b = b[b.k != ""]
        # size-bounded join in record chunks
        for lo in range(0, len(b), 2_000_000):
            j = b.iloc[lo:lo + 2_000_000].merge(a, on="k", how="inner")
            if not len(j):
                continue
            ir, ic = j["r"].values, j["i"].values
            sc = _rowdot(Q, S, ir, ic)
            d = pd.DataFrame({"r": ir, "i": ic, "s": sc}).sort_values(["r", "s"], ascending=[True, False])
            d = d[d.groupby("r").cumcount().values < keep]
            out.append(np.stack([d.r.values, d.i.values], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), np.int64)


def rank_features(c: pd.DataFrame) -> pd.DataFrame:
    """Within-record ranks / margins of the channel scores (cheap competition features), numpy only."""
    c = c.reset_index(drop=True)                          # rows of a record are contiguous (blocking output)
    rec = c["rec"].values
    starts = np.flatnonzero(np.r_[True, rec[1:] != rec[:-1]])
    sizes = np.diff(np.r_[starts, len(rec)])
    gid = np.repeat(np.arange(len(starts)), sizes)
    for ch in ("name", "addr", "comb", "pair"):
        v = c["c_" + ch].values
        order = np.lexsort((-v, gid))                     # by record, score descending
        pos = np.empty(len(v), np.int64)
        pos[order] = np.arange(len(v))
        c["r_" + ch] = (pos - starts[gid] + 1).astype(np.float32)
        mx = np.maximum.reduceat(v, starts)
        c["m_" + ch] = (v - mx[gid]).astype(np.float32)
    c["n_cand"] = sizes[gid].astype(np.float32)
    top = pd.Series(c["s1"].values[c["r_comb"].values == 1]).value_counts()
    c["s1_top_cnt"] = c["s1"].map(top).fillna(0).astype(np.float32)
    return c
