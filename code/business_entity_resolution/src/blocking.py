"""Stage 3 - blocking / candidate generation.

For every S1 entity, find a small set of S2/S3 candidates. Recall here is the ceiling for
everything downstream, so we use several complementary keys built from the Stage-2 columns:

  core    exact c4b (core name)
  tok     each core-name token (len>=2)
  skelbi  adjacent-token pairs of a consonant skeleton (vowel/typo/transliteration variants:
          raam~ram, praivet~private)
  pre     4-char prefixes of the first two core tokens (typos late in a word)
  addrpc  postal-like number (5-6 digits) in the address + skeleton of the first name token
  nameaddr skeleton of the first name token + a specific address token (house number / street
          word) - keeps chain names ("global trust", 300+ targets) blockable after core is capped
  addrbi  adjacent pairs of the first address tokens ("2621|cotten") - finds pairs whose names
          differ entirely
  join    glued tokens: adjacent core tokens concatenated, and single long tokens
          ("ants technology" ~ "antstechnology com")

Second channel (similarity, not keys): char n-gram TF-IDF on core names. Per S1, the top
TFIDF_K targets per source by cosine are unioned in (mask bit 'tfidf'); every candidate
gets its TF-IDF cosine as column tf_cos (a matching feature).

Keys that are shared by more targets than BLOCK_CAPS[type] are dropped (too common to help).
Each (S1, target) pair is scored by the sum of idf weights of the keys it shares; the top
K_PER_SOURCE per S1 and per source (S2 / S3) are kept.

    python blocking.py --split train --eval             # candidate parquet + recall report
    python blocking.py --split test --write-tsv         # + output/candidate_pairs.tsv

Partitions by country_norm (generic string label, nothing hard-coded); --cross-country to disable.
"""
import argparse
import gc
import os
import re
import time
from functools import lru_cache

import numpy as np
import pandas as pd

import config as C

KT = {k: i for i, k in enumerate(C.KTYPES)}
_VOWELS = re.compile(r"[aeiou]")
_REPEAT = re.compile(r"(.)\1+")
_LEET = str.maketrans("0134578", "oieastb")  # digit-for-letter noise: va1ley, 5ervices
_SKEL_SUB = (("ph", "f"), ("ck", "k"), ("w", "v"), ("c", "k"), ("q", "k"), ("z", "s"), ("x", "ks"), ("y", "i"))


@lru_cache(maxsize=2_000_000)
def skel(tok: str) -> str:
    """Consonant skeleton: 'raam'->'rm', 'praivet'->'prvt', 'private'->'prvt'."""
    if tok.isdigit():
        return tok
    t = tok.translate(_LEET) if any(ch.isdigit() for ch in tok) else tok
    for a, b in _SKEL_SUB:
        t = t.replace(a, b)
    s = _REPEAT.sub(r"\1", t[0] + _VOWELS.sub("", t[1:]))
    return s or tok


_STREET_WORDS = {"road", "street", "lane", "avenue", "nagar", "floor", "building", "near", "opposite", "sector",
                 "plot", "flat", "house", "shop", "office", "block", "phase", "main", "cross", "colony", "drive",
                 "court", "place", "suite", "apartment", "complex", "tower", "rue", "boulevard", "chemin", "allée"}


def record_keys(name: str, addr: str):
    out = []
    t = name.split() if name else []
    if not t:
        return out
    out.append((KT["core"], "C|" + name))
    for w in set(t):
        if len(w) >= 2:
            out.append((KT["tok"], "T|" + w))
    sk = [skel(w) for w in t]
    if len(t) == 1:
        out.append((KT["skelbi"], "P|" + sk[0]))
    else:
        for i in range(min(len(t) - 1, 6)):
            out.append((KT["skelbi"], f"P|{sk[i]}|{sk[i + 1]}"))
    out.append((KT["pre"], "F|" + "|".join(w[:4] for w in t[:2])))
    for i in range(min(len(t) - 1, 3)):
        out.append((KT["join"], "J|" + t[i] + t[i + 1]))
    for i in range(min(len(t) - 2, 2)):                    # 3-word glue (generator glues 2-3 words)
        out.append((KT["join"], "J|" + t[i] + t[i + 1] + t[i + 2]))
    if 2 <= len(t) <= 5:
        out.append((KT["join"], "J|" + "".join(t)))        # whole name glued ('agroshreedevelopers')
    for w in t:
        if len(w) >= 8:
            out.append((KT["join"], "J|" + w))
    if len(t) >= 2:                                        # word order invariant (reordered names)
        out.append((KT["sorted"], "S|" + " ".join(sorted(set(t)))))
        out.append((KT["acro"], "I|" + "".join(w[0] for w in t)))   # acronym side: initials of the name
    elif 2 <= len(t[0]) <= 5 and t[0].isalpha():
        out.append((KT["acro"], "I|" + t[0]))              # a short one-word name may be an acronym
    if addr:
        a = addr.split()
        for pc in [w for w in a if w.isdigit() and 5 <= len(w) <= 6][:2]:
            out.append((KT["addrpc"], f"A|{pc}|{sk[0][:3]}"))
        spec = [w for w in a if w.isdigit() and len(w) <= 4][:2] + [w for w in a if len(w) >= 4 and w.isalpha()][:2]
        for w in spec:
            out.append((KT["nameaddr"], f"N|{sk[0]}|{w}"))
        for i in range(min(len(a) - 1, 2)):
            out.append((KT["addrbi"], f"B|{a[i]}|{a[i + 1]}"))
        # address-only key (house number + first street word, any order): finds records whose name was
        # replaced entirely ('rose infocom partners' -> 'fayelum') when the address parts are reordered
        house = next((w for w in a if w[:1].isdigit() and len(w) <= 6), None)
        street = next((w for w in a if w.isalpha() and len(w) >= 4 and w not in _STREET_WORDS), None)
        if house and street:
            out.append((KT["addrbi"], f"H|{house}|{street}"))
    return out


class TfidfChannel:
    """Char n-gram TF-IDF over one partition's target core names."""

    def __init__(self, names, tsrc):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=C.TFIDF_NGRAM, max_df=C.TFIDF_MAX_DF,
                                   dtype=np.float32, sublinear_tf=True)
        self.B = self.vec.fit_transform(names).tocsr()
        self.by_src = {}
        for s in (2, 3):
            idx = np.where(tsrc == s)[0]
            self.by_src[s] = (self.B[idx].T.tocsr(), idx)

    def transform(self, names):
        return self.vec.transform(names).tocsr()

    def topk(self, A, k, offset=0):
        """-> DataFrame(s, t, tf_rank) for the top-k targets per S1 row and source."""
        try:                                   # C++ multithreaded sparse top-n (~300x faster)
            from sparse_dot_topn import sp_matmul_topn
        except ImportError:
            sp_matmul_topn = None
        outs = []
        for a in range(0, A.shape[0], C.TFIDF_BLOCK):
            Ab = A[a:a + C.TFIDF_BLOCK]
            for s, (BT, idx) in self.by_src.items():
                if sp_matmul_topn is not None:
                    P = sp_matmul_topn(Ab, BT, top_n=k, sort=True, n_threads=os.cpu_count() or 1)
                    if not P.nnz:
                        continue
                    rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
                    rank = np.arange(P.nnz) - P.indptr[rows]      # sort=True -> row-wise descending
                    outs.append(pd.DataFrame({"s": rows.astype(np.int64) + a + offset,
                                              "t": idx[P.indices].astype(np.int64), "tf_rank": rank}))
                    continue
                P = (Ab @ BT).tocoo()
                if not P.nnz:
                    continue
                d = pd.DataFrame({"s": P.row.astype(np.int64) + a + offset, "t": idx[P.col].astype(np.int64),
                                  "v": P.data}).sort_values(["s", "v"], ascending=[True, False])
                d["tf_rank"] = d.groupby("s").cumcount()
                outs.append(d[d["tf_rank"] < k][["s", "t", "tf_rank"]])
        return pd.concat(outs, ignore_index=True) if outs else None

    def cos(self, A, s_local, t):
        """Row-wise cosine between A[s_local] and B[t] (both L2-normalised)."""
        out = np.zeros(len(t), np.float32)
        for a in range(0, len(t), 1_000_000):
            sl = slice(a, a + 1_000_000)
            out[sl] = np.asarray(A[s_local[sl]].multiply(self.B[t[sl]]).sum(axis=1)).ravel()
        return out


def merge_tfidf(cand, tf, n_t, tsrc):
    """Union key candidates with TF-IDF top-k; TF-IDF-only pairs get score 0, rank 127."""
    tf = tf.assign(pid=tf["s"] * n_t + tf["t"])[["pid", "tf_rank"]]
    if cand is None:
        cand = pd.DataFrame({"pid": tf["pid"], "score": np.float32(0), "mask": np.int16(0), "rank": np.int16(127)})
    else:
        cand = cand.merge(tf, on="pid", how="outer")
        cand["score"] = cand["score"].fillna(0).astype(np.float32)
        cand["mask"] = cand["mask"].fillna(0).astype(np.int16)
        cand["rank"] = cand["rank"].fillna(127).astype(np.int16)
    cand["tf_rank"] = cand.get("tf_rank", pd.Series(np.nan, index=cand.index)).fillna(127).astype(np.int16)
    hit = cand["tf_rank"].to_numpy() < 127
    cand.loc[hit, "mask"] = (cand.loc[hit, "mask"] | np.int16(1 << KT["tfidf"])).astype(np.int16)
    cand["s"] = (cand["pid"] // n_t).astype(np.int64)
    cand["t"] = (cand["pid"] % n_t).astype(np.int64)
    cand["src"] = tsrc[cand["t"].to_numpy()]
    return cand


def group_ctx(key, score):
    """Per-group ambiguity context for a cheap first-pass score.
    -> rank within group (0 = best), group size, margin = score - best *other* score in group
    (top-1: lead over #2; others: <= 0; singleton group: the score itself)."""
    d = pd.DataFrame({"k": key, "v": score}).sort_values(["k", "v"], ascending=[True, False], kind="stable")
    g = d.groupby("k", sort=False)["v"]
    r = g.cumcount().to_numpy()
    n = g.transform("size").to_numpy()
    top1 = g.transform("first").to_numpy()
    sec = d["v"].where(r == 1).groupby(d["k"], sort=False).transform("max").fillna(0).to_numpy()
    margin = d["v"].to_numpy() - np.where(r == 0, sec, top1)
    out = [np.empty(len(d), a.dtype) for a in (r, n, margin)]
    for o, a in zip(out, (r, n, margin)):
        o[d.index.to_numpy()] = a
    return out


def add_context(files, parts):
    """Reverse ambiguity features over a whole country partition, written back into its part files:
    rev_*: among all S1 entities that proposed this target (only visible once every S1 chunk is done).
    The forward ones (fwd_*: among this S1's candidates from the same source) are computed per chunk."""
    if not parts:
        return
    t = np.concatenate([p[0] for p in parts]); v = np.concatenate([p[1] for p in parts])
    rr, rn, rm = group_ctx(t, v)
    del t, v
    a = 0
    for f, p in zip(files, parts):
        b = a + len(p[0])
        d = pd.read_parquet(f)
        d["rev_rank"], d["rev_n"], d["rev_margin"] = (rr[a:b].astype(np.int16), rn[a:b].astype(np.int32),
                                                      rm[a:b].astype(np.float32))
        d.to_parquet(f, index=False)
        a = b


def build_keys(names, addrs) -> pd.DataFrame:
    """-> DataFrame(idx int32, kt int8, key uint64), de-duplicated per (idx, key)."""
    parts = []
    for start in range(0, len(names), C.KEYGEN_BLOCK):
        ii, kk, ss = [], [], []
        for i in range(start, min(start + C.KEYGEN_BLOCK, len(names))):
            for kt, s in record_keys(names[i], addrs[i]):
                ii.append(i), kk.append(kt), ss.append(s)
        if ss:
            parts.append(pd.DataFrame({
                "idx": np.array(ii, dtype=np.int32),
                "kt": np.array(kk, dtype=np.int8),
                "key": pd.util.hash_array(np.array(ss, dtype=object)),
            }))
    if not parts:
        return pd.DataFrame({"idx": np.array([], np.int32), "kt": np.array([], np.int8), "key": np.array([], np.uint64)})
    return pd.concat(parts, ignore_index=True).drop_duplicates(["idx", "key"])


def build_index(tkeys: pd.DataFrame, n_targets: int) -> pd.DataFrame:
    counts = tkeys["key"].value_counts()
    df = tkeys["key"].map(counts).to_numpy()
    caps = np.array([C.BLOCK_CAPS[k] for k in C.KTYPES])[tkeys["kt"].to_numpy()]
    keep = df <= caps
    idx = tkeys[keep][["key", "idx"]].copy()
    idx["w"] = np.log2((n_targets + 1) / df[keep]).astype(np.float32)
    return idx


def chunk_candidates(skeys, index, n_t, tsrc, k):
    m = skeys.merge(index, on="key", suffixes=("_s", "_t"))
    if m.empty:
        return None
    m["pid"] = m["idx_s"].to_numpy(np.int64) * n_t + m["idx_t"].to_numpy(np.int64)
    score = m.groupby("pid")["w"].sum()
    mk = m[["pid", "kt"]].drop_duplicates()
    mk["b"] = np.left_shift(1, mk["kt"].to_numpy(np.int16)).astype(np.int16)
    mask = mk.groupby("pid")["b"].sum()
    out = pd.DataFrame({"score": score.astype(np.float32), "mask": mask}).reset_index()
    out["s"] = (out["pid"] // n_t).astype(np.int64)
    out["t"] = (out["pid"] % n_t).astype(np.int64)
    out["src"] = tsrc[out["t"].to_numpy()]
    out = out.sort_values(["s", "src", "score"], ascending=[True, True, False])
    out["rank"] = out.groupby(["s", "src"]).cumcount().astype(np.int16)
    return out


def load(split, n):
    cols = ["entity_id", "country_norm", "business_name_c4b", "business_address_rom"]
    return pd.read_parquet(C.processed_path(split, n), columns=cols)


def load_gt(ids_s1: pd.Index, ids_t: pd.Index):
    gt = pd.read_csv(C.TRAIN_GT, sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=3)
    gt["m"] = gt["matched_entity_ids"].fillna("").str.split(",").apply(lambda x: [i.strip() for i in x if i.strip()])
    flat = gt[["source1_entity_id", "m"]].explode("m").dropna(subset=["m"]).rename(
        columns={"source1_entity_id": "s1", "m": "t"}).reset_index(drop=True)
    ok = flat.s1.isin(ids_s1) & flat.t.isin(ids_t)
    return flat[ok].reset_index(drop=True), len(flat)


class Stats:
    KS = (1, 5, 10, 20, 30, 50)

    def __init__(self, k):
        self.k, self.n_true_total = k, 0
        self.reach = {2: 0, 3: 0}                       # true pairs inside a partition
        self.pre_trunc = {2: 0, 3: 0}                   # found before top-K truncation
        self.kept = {2: 0, 3: 0}
        self.by_rank = {(s, kk): 0 for s in (2, 3) for kk in self.KS}
        self.by_type = {kt: 0 for kt in C.KTYPES}
        self.n_pairs = self.n_s1 = self.tf_only = self.tf_only_true = 0
        self.denom = 0
        self.s1_with = self.s1_any = self.s1_all = 0

    def report(self):
        p = print
        p("\n=== BLOCKING RECALL (train, vs ground truth) ===")
        p(f"true pairs (ids present): {self.n_true_total:,} | reachable within country partition: "
          f"{self.reach[2] + self.reach[3]:,}")
        for s in (2, 3):
            r = max(self.reach[s], 1)
            p(f"S{s}: n_true={self.reach[s]:,} | key-coverage ceiling (before top-K)={self.pre_trunc[s] / r:.4f} | "
              f"kept @K={self.k}: {self.kept[s] / r:.4f}")
            p("   recall@rank<k: " + "  ".join(f"k={kk}:{self.by_rank[(s, kk)] / r:.4f}" for kk in self.KS if kk <= self.k))
        tot = max(self.n_true_total, 1)
        p(f"overall pair recall @K (incl. cross-country loss): {(self.kept[2] + self.kept[3]) / tot:.4f}")
        r = max(self.kept[2] + self.kept[3], 1)
        p("share of found true pairs hit by key type: " + "  ".join(f"{k}={v / r:.3f}" for k, v in self.by_type.items()))
        w = max(self.s1_with, 1)
        p(f"S1 with >=1 match: {self.s1_with:,} | any match found: {self.s1_any / w:.4f} | ALL matches found: {self.s1_all / w:.4f}")
        p(f"pairs found only by the tfidf channel: {self.tf_only:,} (true: {self.tf_only_true:,})")
        p(f"candidate pairs: {self.n_pairs:,} ({self.n_pairs / max(self.n_s1, 1):.1f} per S1) | "
          f"reduction ratio vs full cross-join: {1 - self.n_pairs / max(self.denom, 1):.8f}")


def run(args):
    t0 = time.time()
    S1 = load(args.split, 1)
    if args.limit_s1:
        S1 = S1.head(args.limit_s1)
    if args.s1_address_tokens:     # dev: only S1 whose address contains one of these tokens (e.g. states)
        want = set(args.s1_address_tokens)
        S1 = S1[S1["business_address_rom"].fillna("").map(lambda a: not want.isdisjoint(a.split()))]
    T = pd.concat([load(args.split, 2).assign(src=2), load(args.split, 3).assign(src=3)], ignore_index=True)
    for df in (S1, T):
        df["business_name_c4b"] = df["business_name_c4b"].fillna("")
        df["business_address_rom"] = df["business_address_rom"].fillna("")
    print(f"[load] S1={len(S1):,} targets={len(T):,} ({time.time() - t0:.0f}s)")

    flat = None
    st = Stats(args.k)
    if args.eval:
        flat, _ = load_gt(pd.Index(S1.entity_id), pd.Index(T.entity_id))
        st.n_true_total = len(flat)
        print(f"[eval] {len(flat):,} ground-truth pairs with both ids loaded")

    outdir = C.candidates_dir(args.split)
    outdir.mkdir(parents=True, exist_ok=True)
    # the S1 population this run covers (matching.py scores entities without candidates too)
    blocked = S1 if not args.countries else S1[S1.country_norm.isin(args.countries)]
    blocked[["entity_id"]].to_parquet(C.PROCESSED_DIR / f"blocked_s1_{args.split}.parquet", index=False)
    for f in outdir.glob("*.parquet"):
        f.unlink()

    if C.WITHIN_COUNTRY and not args.cross_country:
        # built one country at a time (generator) so only the current partition is held in memory
        parts = ((c, S1[S1.country_norm == c].reset_index(drop=True), T[T.country_norm == c].reset_index(drop=True))
                 for c in sorted(S1.country_norm.unique()) if not args.countries or c in args.countries)
    else:
        parts = [("all", S1.reset_index(drop=True), T)]

    missed_pool = []
    for pname, s1c, tc in parts:
        if not len(tc):
            print(f"[{pname}] no targets, skipped")
            continue
        n_t = len(tc)
        tsrc = tc["src"].to_numpy()
        print(f"[{pname}] S1={len(s1c):,} targets={n_t:,}")
        index = build_index(build_keys(tc.business_name_c4b.tolist(), tc.business_address_rom.tolist()), n_t)
        print(f"   target index rows={len(index):,} ({time.time() - t0:.0f}s)")
        tfc = TfidfChannel(tc.business_name_c4b.tolist(), tsrc) if C.TFIDF_K > 0 else None
        if tfc is not None:
            print(f"   tfidf: {tfc.B.shape[1]:,} n-grams, nnz={tfc.B.nnz:,} ({time.time() - t0:.0f}s)")
        s1_ids, t_ids = s1c.entity_id.to_numpy(), tc.entity_id.to_numpy()

        codes = None
        if flat is not None:
            sp = pd.Index(s1c.entity_id).get_indexer(flat.s1)
            tp = pd.Index(tc.entity_id).get_indexer(flat.t)
            ok = (sp >= 0) & (tp >= 0)
            codes = np.sort(sp[ok].astype(np.int64) * n_t + tp[ok])
            tsrc_true = tsrc[tp[ok]]
            for s in (2, 3):
                st.reach[s] += int((tsrc_true == s).sum())
            true_per_s1 = np.bincount(sp[ok], minlength=len(s1c))
            found_per_s1 = np.zeros(len(s1c), np.int64)
            found_codes = []

        ctx_files, ctx_parts = [], []
        for a in range(0, len(s1c), C.S1_CHUNK):
            sub = s1c.iloc[a:a + C.S1_CHUNK]
            sk = build_keys(sub.business_name_c4b.tolist(), sub.business_address_rom.tolist())
            sk["idx"] = sk["idx"] + a
            cand = chunk_candidates(sk, index, n_t, tsrc, args.k)
            A = None
            if tfc is not None:
                A = tfc.transform(sub.business_name_c4b.tolist())
                tf = tfc.topk(A, C.TFIDF_K, offset=a)
                if tf is not None and len(tf):
                    cand = merge_tfidf(cand, tf, n_t, tsrc)
            if cand is None:
                continue
            if "tf_rank" not in cand.columns:
                cand["tf_rank"] = np.int16(127)
            if codes is not None:
                pos = np.minimum(np.searchsorted(codes, cand["pid"].to_numpy()), len(codes) - 1)
                cand["is_true"] = (codes[pos] == cand["pid"].to_numpy()) if len(codes) else False
                for s in (2, 3):
                    st.pre_trunc[s] += int(((cand.is_true) & (cand.src == s)).sum())
            # reverse top-K_REV: this target's best S1 entities in this chunk (a superset of its global top)
            cand["t_rank"] = (cand.groupby("t")["score"].rank(method="first", ascending=False) - 1).to_numpy(np.int16)
            keep = (cand["rank"] < args.k) | (cand["tf_rank"] < C.TFIDF_K) | (cand["t_rank"] < C.K_REV)
            if C.BYPASS_KEYS:                                # exact core name + an address key: never cut
                mk = cand["mask"].to_numpy().astype(np.int64)
                core_hit = (mk >> KT["core"]) & 1 == 1
                addr_hit = ((mk >> KT["nameaddr"]) & 1 == 1) | ((mk >> KT["addrbi"]) & 1 == 1)
                keep |= core_hit & addr_hit
            cand = cand[keep]
            cand = cand.reset_index(drop=True)
            if A is not None:
                cand["tf_cos"] = tfc.cos(A, cand["s"].to_numpy() - a, cand["t"].to_numpy())
            st.n_pairs += len(cand)
            only_tf = cand["mask"].to_numpy() == (1 << KT["tfidf"])
            st.tf_only += int(only_tf.sum())
            if codes is not None:
                st.tf_only_true += int((only_tf & cand["is_true"].to_numpy().astype(bool)).sum())
            if codes is not None:
                tr = cand[cand.is_true]
                for s in (2, 3):
                    ts = tr[tr.src == s]
                    st.kept[s] += len(ts)
                    for kk in Stats.KS:
                        st.by_rank[(s, kk)] += int((ts["rank"] < kk).sum())
                for kt, i in KT.items():
                    st.by_type[kt] += int(((tr["mask"].to_numpy() >> i) & 1).sum())
                found_per_s1 += np.bincount(tr["s"].to_numpy(), minlength=len(s1c))
                found_codes.append(tr["pid"].to_numpy())
            out = pd.DataFrame({
                "s1_id": s1_ids[cand["s"].to_numpy()], "cand_id": t_ids[cand["t"].to_numpy()],
                "src": cand["src"].to_numpy(np.int8), "score": cand["score"].to_numpy(),
                "mask": cand["mask"].to_numpy(np.int16), "rank": cand["rank"].to_numpy(np.int16),
                "t_rank": cand["t_rank"].to_numpy(np.int16)})
            if "tf_cos" in cand.columns:
                out["tf_cos"] = cand["tf_cos"].to_numpy(np.float32)
                out["tf_rank"] = cand["tf_rank"].to_numpy(np.int8)
            if codes is not None:
                out["is_true"] = cand["is_true"].to_numpy(np.int8)
            # forward context: an S1's candidates all sit in this chunk, so compute it here
            _, fn, fm = group_ctx(cand["s"].to_numpy(np.int64) * 4 + cand["src"].to_numpy(np.int64),
                                  cand["score"].to_numpy(np.float32))
            out["fwd_n"], out["fwd_margin"] = fn.astype(np.int16), fm.astype(np.float32)
            f = outdir / f"part-{pname}-{a // C.S1_CHUNK:04d}.parquet"
            out.to_parquet(f, index=False)
            ctx_files.append(f)
            ctx_parts.append((cand["t"].to_numpy(np.int32), cand["score"].to_numpy(np.float32)))
            print(f"   chunk {a // C.S1_CHUNK}: {len(out):,} pairs ({time.time() - t0:.0f}s)")
        index = tfc = A = None                     # free the target index before the reverse pass
        gc.collect()
        add_context(ctx_files, ctx_parts)
        del ctx_parts

        st.n_s1 += len(s1c)
        st.denom += len(s1c) * n_t
        if codes is not None:
            w = true_per_s1 > 0
            st.s1_with += int(w.sum())
            st.s1_any += int((w & (found_per_s1 > 0)).sum())
            st.s1_all += int((w & (found_per_s1 >= true_per_s1)).sum())
            fc = np.concatenate(found_codes) if found_codes else np.array([], np.int64)
            miss = codes[~np.isin(codes, fc)]
            if len(miss):
                miss = np.random.default_rng(0).choice(miss, min(200, len(miss)), replace=False)
                missed_pool.append(pd.DataFrame({
                    "s1_id": s1_ids[miss // n_t], "t_id": t_ids[miss % n_t],
                    "s1_name": s1c.business_name_c4b.to_numpy()[miss // n_t],
                    "t_name": tc.business_name_c4b.to_numpy()[miss % n_t]}))

    if flat is not None:
        st.report()
        if missed_pool:
            p = C.PROCESSED_DIR / "eda" / "missed_true_pairs.tsv"
            p.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(missed_pool).head(400).to_csv(p, sep="\t", index=False)
            print(f"missed true pairs sample -> {p} (inspect: what do the misses have in common?)")
    print(f"[done] {time.time() - t0:.0f}s -> {outdir}")
    if args.write_tsv:
        write_tsv(args.split, C.OUTPUT_DIR / "candidate_pairs.tsv", S1["entity_id"])


def write_tsv(split, path, all_s1_ids=None):
    """output/candidate_pairs.tsv - grouped format matching utils/validate_submission.py:
    one row per S1 entity, header 'source1_entity_id<TAB>candidate_entity_ids', candidates
    comma-joined (empty string if none). `all_s1_ids` should be every S1 entity_id in the
    split so entities with zero blocking candidates still get a (empty) row, since the
    validator requires a row for every entity in test_source1.tsv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [pd.read_parquet(part, columns=["s1_id", "cand_id"])
             for part in sorted(C.candidates_dir(split).glob("*.parquet"))]
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["s1_id", "cand_id"])
    n_pairs = len(d)
    grouped = d.groupby("s1_id")["cand_id"].apply(lambda s: ",".join(sorted(s.unique())))
    # every S1 of the raw source file gets a row (validator rule), also for --limit / --countries runs
    raw = pd.read_csv(C.source_path(split, 1), sep="\t", usecols=[0], dtype=str, keep_default_na=False,
                      quoting=3).iloc[:, 0].str.strip()
    grouped = grouped.reindex(pd.unique(pd.Index(raw))).fillna("")
    grouped = grouped.reset_index()
    grouped.columns = ["source1_entity_id", "candidate_entity_ids"]
    grouped.to_csv(path, sep="\t", index=False)
    print(f"[tsv] {n_pairs:,} pairs across {len(grouped):,} S1 rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--k", type=int, default=C.K_PER_SOURCE, help="candidates kept per S1 per source")
    ap.add_argument("--eval", action="store_true", help="train only: recall vs ground truth")
    ap.add_argument("--countries", nargs="*", help="only these country_norm values (memory control)")
    ap.add_argument("--cross-country", action="store_true")
    ap.add_argument("--limit-s1", type=int, default=None, help="dev: first N S1 rows")
    ap.add_argument("--s1-address-tokens", nargs="*", help="dev: keep S1 whose address has one of these tokens")
    ap.add_argument("--write-tsv", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()