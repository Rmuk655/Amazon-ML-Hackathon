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
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import OSA, JaroWinkler, LCSseq, Levenshtein
from rapidfuzz.process import cpdist
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

import config as C
from text_utils import ADDRESS_ABBR, normalize_address

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # .../student_resource
# Short internal feature-column names -> actual column names written by preprocess.py.
TEXT_COLS = ["name_norm", "name_rom", "addr_norm", "addr_rom", "country_norm", "core", "addr_raw", "legal"]
COL_MAP = {
    "name_norm": "business_name_norm", "name_rom": "business_name_rom",
    "addr_norm": "business_address_norm", "addr_rom": "business_address_rom",
    "country_norm": "country_norm", "core": "business_name_c4b", "addr_raw": "business_address",
    "legal": "business_name_legal",
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
    # address signals that separate same-name chain branches (profile_data.py P4)
    "a_house_eq", "a_house_conflict", "a_street_overlap", "a_abbr_tset", "a_abbr_jacc",
]
_HOUSE = re.compile(r"\d{1,4}[a-z]?")
STREET_TYPES = set(ADDRESS_ABBR.values()) | {"no", "near", "opposite", "floor", "building", "sector", "plot"}


@lru_cache(maxsize=1_000_000)
def addr_parts(addr):
    """-> (abbreviation-unified address, house number or '', set of street words)."""
    t = [ADDRESS_ABBR.get(w, w) for w in addr.split()]
    house = next((w for w in t[:3] if _HOUSE.fullmatch(w)), "")
    street = frozenset(w for w in t[:6] if w.isalpha() and len(w) >= 4 and w not in STREET_TYPES)
    return " ".join(t), house, street


def addr_feats(x, y):
    if not x or not y:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    (ax, hx, sx), (ay, hy, sy) = addr_parts(x), addr_parts(y)
    return [float(bool(hx) and hx == hy), float(bool(hx) and bool(hy) and hx != hy),
            float(bool(sx & sy)), fuzz.token_set_ratio(ax, ay) / 100, jacc(ax.split(), ay.split())]


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
            + addr_feats(ar1, ar2)
        )
        out[k] = row
    return out



def _per_unique(arr, fn):
    """fn applied once per distinct value -> list aligned with arr (strings repeat a lot)."""
    codes, uniq = pd.factorize(pd.Series(arr, dtype=object), sort=False)
    vals = np.empty(len(uniq), dtype=object)
    vals[:] = [fn(u) for u in uniq]
    return vals[codes]


def _cp(a, b, scorer, workers, scale=100.0):
    return cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float64) / scale


def _jacc_pairs(xs, ys):
    """Set Jaccard of token lists per pair (0 if either is empty), via sparse binary rows."""
    xs, ys = list(xs), list(ys)
    n = len(xs)
    toks = pd.Series(xs + ys, dtype=object).explode()
    toks = toks[toks.notna()]
    codes, _ = pd.factorize(toks.values)
    rows = toks.index.to_numpy()
    if len(codes) == 0:
        return np.zeros(n)
    Mx = sparse.csr_matrix((np.ones(len(codes)), (rows, codes)), shape=(2 * n, codes.max() + 1))
    Mx.data[:] = 1.0                          # duplicates summed -> back to a set (binary)
    Mx.sum_duplicates(); Mx.data[:] = 1.0
    A, B = Mx[:n], Mx[n:]
    inter = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    na, nb = np.asarray(A.sum(axis=1)).ravel(), np.asarray(B.sum(axis=1)).ravel()
    union = na + nb - inter
    return np.where((na > 0) & (nb > 0), inter / np.maximum(union, 1), 0.0)


def _feats_vec_1(args):
    return _feats_vec(args, 1)


def _feats_vec(args, workers=-1):
    """Same values as _feats (checked by test_feats_equal), but the string similarities run in
    rapidfuzz's C batch API (cpdist, all cores) and per-string work is done once per distinct string."""
    an, ar, aa, aar, ac, bn, br, ba, bar, bc = [np.asarray(x, dtype=object) for x in args]
    n = len(an)
    out = np.zeros((n, len(FEATS)), dtype=np.float32)
    if n == 0:
        return out
    ne = lambda x: x != ""
    w = workers
    sk1, sk2 = _per_unique(ar, skel), _per_unique(br, skel)
    t1, t2 = _per_unique(ar, str.split), _per_unique(br, str.split)
    col = {f: None for f in FEATS}
    col["n_norm_ratio"] = _cp(an, bn, fuzz.ratio, w)
    col["n_norm_tset"] = _cp(an, bn, fuzz.token_set_ratio, w)
    col["n_norm_tsort"] = _cp(an, bn, fuzz.token_sort_ratio, w)
    col["n_norm_jw"] = _cp(an, bn, JaroWinkler.similarity, w, 1.0)
    col["n_rom_ratio"] = _cp(ar, br, fuzz.ratio, w)
    col["n_rom_tset"] = _cp(ar, br, fuzz.token_set_ratio, w)
    col["n_rom_tsort"] = _cp(ar, br, fuzz.token_sort_ratio, w)
    col["n_rom_jw"] = _cp(ar, br, JaroWinkler.similarity, w, 1.0)
    col["n_skel_ratio"] = _cp(sk1, sk2, fuzz.ratio, w)
    col["n_skel_tset"] = _cp(sk1, sk2, fuzz.token_set_ratio, w)
    col["n_best"] = np.maximum.reduce([col["n_norm_tset"], col["n_rom_tset"], col["n_skel_tset"],
                                       col["n_norm_ratio"], col["n_rom_ratio"]])
    col["n_exact_norm"] = (ne(an) & (an == bn)).astype(np.float64)
    col["n_exact_rom"] = (ne(ar) & (ar == br)).astype(np.float64)
    s1a, s2a = np.asarray(sk1, dtype=object), np.asarray(sk2, dtype=object)
    col["n_exact_skel"] = (ne(s1a) & (s1a == s2a)).astype(np.float64)
    col["n_tok_jacc_rom"] = _jacc_pairs(t1, t2)
    col["n_first_tok_eq"] = np.array([float(bool(x) and bool(y) and x[0] == y[0]) for x, y in zip(t1, t2)])
    l1, l2 = np.array([len(x) for x in ar]), np.array([len(x) for x in br])
    col["n_len_diff"] = np.abs(l1 - l2) / np.maximum(np.maximum(l1, l2), 1)
    col["n_tok_cnt_diff"] = np.abs(np.array([len(x) for x in t1]) - np.array([len(x) for x in t2])).astype(np.float64)
    both_r = ne(ar) & ne(br)
    col["n_partial_rom"] = np.where(both_r, _cp(ar, br, fuzz.partial_ratio, w), 0.0)
    col["a_norm_tset"] = np.where(ne(aa) & ne(ba), _cp(aa, ba, fuzz.token_set_ratio, w), 0.0)
    both_ar = ne(aar) & ne(bar)
    col["a_rom_tset"] = np.where(both_ar, _cp(aar, bar, fuzz.token_set_ratio, w), 0.0)
    col["a_rom_partial"] = np.where(both_ar, _cp(aar, bar, fuzz.partial_ratio, w), 0.0)
    col["a_tok_jacc_rom"] = _jacc_pairs(_per_unique(aar, str.split), _per_unique(bar, str.split))
    j1 = [a + " " + b for a, b in zip(aa, aar)]
    j2 = [a + " " + b for a, b in zip(ba, bar)]
    col["a_num_jacc"] = _jacc_pairs(_per_unique(j1, _NUM.findall), _per_unique(j2, _NUM.findall))
    p1, p2 = _per_unique(j1, lambda x: frozenset(_PIN.findall(x))), _per_unique(j2, lambda x: frozenset(_PIN.findall(x)))
    col["a_pin_match"] = np.array([float(bool(x & y)) for x, y in zip(p1, p2)])
    col["a_pin_conflict"] = np.array([float(bool(x) and bool(y) and not (x & y)) for x, y in zip(p1, p2)])
    col["c_eq"] = (ne(ac) & (ac == bc)).astype(np.float64)
    col["n_missing"] = (~((ne(an) | ne(ar)) & (ne(bn) | ne(br)))).astype(np.float64)
    col["a_missing"] = (~((ne(aa) | ne(aar)) & (ne(ba) | ne(bar)))).astype(np.float64)
    q1, q2 = _per_unique(aar, addr_parts), _per_unique(bar, addr_parts)
    hx, hy = np.array([x[1] for x in q1], dtype=object), np.array([y[1] for y in q2], dtype=object)
    col["a_house_eq"] = np.where(both_ar, ne(hx) & (hx == hy), False).astype(np.float64)
    col["a_house_conflict"] = np.where(both_ar, ne(hx) & ne(hy) & (hx != hy), False).astype(np.float64)
    col["a_street_overlap"] = np.where(both_ar, [bool(x[2] & y[2]) for x, y in zip(q1, q2)], False).astype(np.float64)
    ab1, ab2 = [x[0] for x in q1], [y[0] for y in q2]
    col["a_abbr_tset"] = np.where(both_ar, _cp(ab1, ab2, fuzz.token_set_ratio, w), 0.0)
    col["a_abbr_jacc"] = np.where(both_ar, _jacc_pairs([x.split() for x in ab1], [y.split() for y in ab2]), 0.0)
    for i, f in enumerate(FEATS):
        out[:, i] = col[f]
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
    tabs["procs"] = procs
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


# --------------------------------------------------------------------------- extra similarity features
# Levenshtein / Damerau (OSA) / LCS edit similarities, character-trigram Jaccard, TF-IDF cosine
# (character and word) and IDF-weighted token Jaccard for names and addresses, plus address
# comparisons on a country-aware canonical form. TF-IDF weights are fitted per country without
# labels: token rarity differs by country ('road' is common in India, 'rue' in France, 'street' in
# the US). BER_EXTRA_FEATS=0 switches the block off (A/B runs).
EXTRA = os.environ.get("BER_EXTRA_FEATS", "1") != "0"
EXTRA_FEATS = [
    "n_lev_rom", "n_lcs_rom", "n_osa_core", "n_jw_core", "n_tset_core",
    "n_char3_jacc", "n_tfidf_char", "n_tfidf_word", "n_idf_jacc",
    "a_lev_canon", "a_tset_canon", "a_char3_jacc", "a_tfidf_char", "a_tfidf_word", "a_idf_jacc",
    "a_comp_jacc", "a_comp_cov",
]
_ORDINALS = {w: f"{i}{'st' if i % 10 == 1 and i != 11 else 'nd' if i % 10 == 2 and i != 12 else 'rd' if i % 10 == 3 and i != 13 else 'th'}"
             for i, w in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
                                   "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth "
                                   "twentieth".split(), 1)}
_DIRECTIONS = {"n": "north", "s": "south", "e": "east", "w": "west",
               "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest"}
_US_UNIT = {"suite", "apt", "apartment", "unit", "fl", "floor", "pmb", "room", "rm", "bldg", "building", "lot"}
_IN_NOISE = {"no", "number", "hno"}           # 'H.No. 12', 'No.4/2', 'Plot No 7'
_FR_NOISE = {"n", "bis", "ter"}               # 'N° 21 Rue ...', '53 Bis Rue ...'


@lru_cache(maxsize=2_000_000)
def addr_canon(addr, country):
    """Country-aware canonical address (on the romanised, abbreviation-unified text):
    US: ordinal words -> '15th', N/S/E/W -> north/..., unit designators (suite 200, fl 3, pmb 51)
    dropped (Source 1 keeps them, S2 mostly does not); India: ordinals, 'no'/'h no' fillers dropped;
    France: 'n°', 'bis', 'ter' dropped."""
    out, t, skip = [], [ADDRESS_ABBR.get(w, w) for w in addr.split()], False
    for i, w in enumerate(t):
        if skip:
            skip = False
            continue
        if country == "us":
            w = _ORDINALS.get(w, w)
            if w in _US_UNIT and i + 1 < len(t) and (any(ch.isdigit() for ch in t[i + 1]) or len(t[i + 1]) == 1):
                skip = True
                continue
            w = _DIRECTIONS.get(w, w)
        elif country == "india":
            w = _ORDINALS.get(w, w)
            if w in _IN_NOISE or (w == "h" and i + 1 < len(t) and t[i + 1] == "no"):
                continue
        elif country == "france" and w in _FR_NOISE:
            continue
        out.append(w)
    return " ".join(out)


@lru_cache(maxsize=2_000_000)
def addr_components(raw, country):
    """Comma-separated address parts (street line, locality, city, district, state/region), each
    canonicalised; compared as a set because sources reorder them ('Lille, Hauts-de-France' vs
    'Hauts-de-France, Lille')."""
    return frozenset(x for x in (addr_canon(normalize_address(p), country) for p in raw.split(",")) if x)


def _rowdot(A, B):
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)


def _new_vectorizer(char):
    return (TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), binary=True, norm=None, dtype=np.float32)
            if char else TfidfVectorizer(token_pattern=r"\S+", binary=True, norm=None, dtype=np.float32))


def fitted_vectorizers(procs, fit_max=3_000_000):
    """{(country, field, char): TfidfVectorizer} fitted once per split on ALL records of that country
    (S1+S2+S3 names / canonical addresses; no labels), cached next to the processed data. The same
    pair therefore gets the same value in training, evaluation and prediction, whatever the batch."""
    import hashlib, json
    key = hashlib.md5(json.dumps(_file_sig([procs[n] for n in (1, 2, 3)])).encode()).hexdigest()[:12]
    path = C.PROCESSED_DIR / f"tfidf_vectorizers_{key}.joblib"
    if path.exists():
        return joblib.load(path)
    cols = ["business_name_rom", "business_address_rom", "country_norm"]
    df = pd.concat([pd.read_parquet(procs[n], columns=cols) for n in (1, 2, 3)], ignore_index=True).fillna("")
    out = {}
    rng = np.random.default_rng(0)
    for k, g in df.groupby("country_norm"):
        names = pd.unique(g["business_name_rom"].to_numpy(object))
        addrs = pd.unique(np.array([addr_canon(x, k) for x in pd.unique(g["business_address_rom"].to_numpy(object))],
                                   dtype=object))
        for field, texts in (("n", names), ("a", addrs)):
            texts = texts[texts != ""]
            if len(texts) > fit_max:
                texts = texts[rng.choice(len(texts), fit_max, replace=False)]
            for char in (True, False):
                v = _new_vectorizer(char)
                try:
                    out[(k, field, char)] = v.fit(texts)
                except ValueError:
                    pass
    joblib.dump(out, path)
    print(f"  fitted TF-IDF vectorizers -> {path.name}")
    return out


def _vector_sims(a, b, char, vec, chunk=500_000):
    """TF-IDF cosine and Jaccard for aligned text arrays a[i] vs b[i] with a prefitted vectorizer.
    char=True: character trigrams (within words) -> (cosine, plain trigram Jaccard); char=False:
    words -> (cosine, IDF-weighted Jaccard). Words unseen by the vectorizer are ignored."""
    codes, uniq = pd.factorize(np.concatenate([a, b]))
    n = len(a)
    if len(uniq) == 0 or vec is None:
        return np.zeros(n), np.zeros(n)
    M = sparse.vstack([vec.transform(uniq[i:i + chunk]) for i in range(0, len(uniq), chunk)]).tocsr()
    B = M.copy()
    B.data[:] = 1.0
    sq, s_idf, cnt = _rowdot(M, M), np.asarray(M.sum(axis=1)).ravel(), np.asarray(B.sum(axis=1)).ravel()
    ca, cb = codes[:n], codes[n:]
    cos, jac = np.zeros(n), np.zeros(n)
    for i in range(0, n, chunk):
        x, y = ca[i:i + chunk], cb[i:i + chunk]
        dot = _rowdot(M[x], M[y])
        cos[i:i + chunk] = _safe_div(dot, np.sqrt(sq[x] * sq[y]))
        if char:
            inter = _rowdot(B[x], B[y])
            jac[i:i + chunk] = _safe_div(inter, cnt[x] + cnt[y] - inter)
        else:
            inter = _rowdot(M[x], B[y])          # sum of IDF over shared words
            jac[i:i + chunk] = _safe_div(inter, s_idf[x] + s_idf[y] - inter)
    return cos, jac


def _pair_texts(c, tabs, ia, col):
    """Aligned S1 / candidate text arrays for one column ('' where an id is missing)."""
    n = len(c)
    t1 = tabs[1].cols[col]
    a = np.where(ia >= 0, t1[np.maximum(ia, 0)] if len(t1) else "", "").astype(object)
    b = np.full(n, "", dtype=object)
    for s in (2, 3):
        m = np.where(c["srcn"].values == s)[0]
        if len(m) == 0 or not len(tabs[s].index):
            continue
        ib = tabs[s].pos(c["cand_id"].values[m])
        b[m] = np.where(ib >= 0, tabs[s].cols[col][np.maximum(ib, 0)], "")
    return a, b


def _edit_sim(a, b, scorer, workers):
    v = cpdist(list(a), list(b), scorer=scorer, workers=workers, dtype=np.float32)
    return np.where((a != "") & (b != ""), v, 0.0)


def extra_features(c, tabs, ia, workers):
    n = len(c)
    F = pd.DataFrame(0.0, index=range(n), columns=EXTRA_FEATS, dtype=np.float32)
    w = -1 if workers > 1 else 1
    ctry, _ = _pair_texts(c, tabs, ia, "country_norm")
    nr_a, nr_b = _pair_texts(c, tabs, ia, "name_rom")
    co_a, co_b = _pair_texts(c, tabs, ia, "core")
    ar_a, ar_b = _pair_texts(c, tabs, ia, "addr_rom")
    raw_a, raw_b = _pair_texts(c, tabs, ia, "addr_raw")
    ca_a = np.array([addr_canon(x, k) for x, k in zip(ar_a, ctry)], dtype=object)
    ca_b = np.array([addr_canon(x, k) for x, k in zip(ar_b, ctry)], dtype=object)
    F["n_lev_rom"] = _edit_sim(nr_a, nr_b, Levenshtein.normalized_similarity, w)
    F["n_lcs_rom"] = _edit_sim(nr_a, nr_b, LCSseq.normalized_similarity, w)
    F["n_osa_core"] = _edit_sim(co_a, co_b, OSA.normalized_similarity, w)
    F["n_jw_core"] = _edit_sim(co_a, co_b, JaroWinkler.similarity, w)
    F["n_tset_core"] = _edit_sim(co_a, co_b, fuzz.token_set_ratio, w) / 100
    F["a_lev_canon"] = _edit_sim(ca_a, ca_b, Levenshtein.normalized_similarity, w)
    F["a_tset_canon"] = _edit_sim(ca_a, ca_b, fuzz.token_set_ratio, w) / 100
    vecs = fitted_vectorizers(tabs["procs"]) if "procs" in tabs else {}
    for k in pd.unique(ctry):                   # TF-IDF weights fitted per country, once per split
        g = np.where(ctry == k)[0]
        for pre, (x, y) in (("n", (nr_a, nr_b)), ("a", (ca_a, ca_b))):
            F.loc[g, f"{pre}_tfidf_char"], F.loc[g, f"{pre}_char3_jacc"] = _vector_sims(
                x[g], y[g], True, vecs.get((k, pre, True)))
            F.loc[g, f"{pre}_tfidf_word"], F.loc[g, f"{pre}_idf_jacc"] = _vector_sims(
                x[g], y[g], False, vecs.get((k, pre, False)))
    jac, cov = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for i, (x, y, k) in enumerate(zip(raw_a, raw_b, ctry)):
        if x and y:
            p, q = addr_components(x, k), addr_components(y, k)
            if p and q:
                inter = len(p & q)
                jac[i], cov[i] = inter / len(p | q), inter / len(p)
    F["a_comp_jacc"], F["a_comp_cov"] = jac, cov
    return F


# Feature store: each group's columns are cached per candidate file and reused while the group's
# code version and its inputs (candidate file + processed tables) are unchanged. Bump a group's
# version when its code changes; a new group is computed alone and appended to the cache.
FEATURE_GROUPS = {"base": 1, "amb": 1, "extra": 2, "decoy": 2}
DECOY = os.environ.get("BER_DECOY_FEATS", "1") != "0"
DECOY_FEATS = ["d_house_delta", "d_house_off_small", "d_num_mismatch", "d_one_word_swap", "d_swap_real_words",
               "d_legal_conflict", "d_extra_word", "d_missing_word", "d_core_tok_diff", "d_legal_added",
               "d_legal_dropped"]
_HN = re.compile(r"\d+(?:[/-]\d+)*[a-z]?")
_LEGAL_CANON = {"pvt": "private", "ltd": "limited", "co": "company", "corp": "corporation", "inc": "incorporated"}


@lru_cache(maxsize=1)
def _name_vocab():
    try:
        with open(C.NAME_VOCAB, encoding="utf-8") as f:
            return {w: int(c) for w, c in (l.rstrip("\n").split("\t") for l in f) if c.isdigit()}
    except OSError:
        return {}


def decoy_pair(core1, core2, legal1, legal2, addr1, addr2, vocab):
    """Signals of the decoy generator (a decoy is an S1 record with ONE detail changed): house number off by a
    small amount, numbers disagreeing, exactly one real word swapped for another real word, legal form changed,
    one word added / dropped."""
    out = [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    h1, h2 = _HN.findall(addr1), _HN.findall(addr2)
    if h1 and h2:
        a_, b_ = re.findall(r"\d+", h1[0]), re.findall(r"\d+", h2[0])
        if len(a_) == len(b_) and a_[:-1] == b_[:-1]:
            out[0] = float(abs(int(a_[-1]) - int(b_[-1])))
            out[1] = float(0 < out[0] <= 10)
        n1, n2 = Counter(re.findall(r"\d+", addr1)), Counter(re.findall(r"\d+", addr2))
        out[2] = float(sum(((n1 - n2) + (n2 - n1)).values()))
    t1, t2 = core1.split(), core2.split()
    if len(t1) == len(t2) and len(t1) >= 2:
        diff = [(x, y) for x, y in zip(t1, t2) if x != y]
        if len(diff) == 1:
            x, y = diff[0]
            out[3] = 1.0
            out[4] = float(vocab.get(x, 0) >= 20 and vocab.get(y, 0) >= 20 and Levenshtein.normalized_similarity(x, y) < 0.6)
    l1 = {_LEGAL_CANON.get(w, w) for w in legal1.split()}
    l2 = {_LEGAL_CANON.get(w, w) for w in legal2.split()}
    out[5] = float(bool(l1) and bool(l2) and l1 != l2)
    s1, s2 = set(t1), set(t2)
    out[6] = float(len(t2) == len(t1) + 1 and s1 <= s2)
    out[7] = float(len(t1) == len(t2) + 1 and s2 <= s1)
    out[8] = float(abs(len(t1) - len(t2)))
    out[9] = float(not l1 and bool(l2))       # top decoy fingerprint: legal form ADDED (+ house number changed)
    out[10] = float(bool(l1) and not l2)
    return out


def decoy_features(c, tabs, ia):
    vocab = _name_vocab()
    cols = {}
    for k in ("core", "legal", "addr_rom"):
        cols[k] = _pair_texts(c, tabs, ia, k)
    rows = [decoy_pair(a1, b1, a2, b2, a3, b3, vocab) for a1, b1, a2, b2, a3, b3 in
            zip(*cols["core"], *cols["legal"], *cols["addr_rom"])] if len(c) else []
    return pd.DataFrame(np.asarray(rows, dtype=np.float32).reshape(-1, len(DECOY_FEATS)), columns=DECOY_FEATS)


def _base_group(c, tabs, workers, ia, chunk=25_000):
    n = len(c)
    X = np.full((n, len(FEATS)), np.nan, dtype=np.float32)
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

        if workers > 1:                  # chunks in parallel processes; cpdist single-threaded inside each
            with Pool(workers) as p:
                res = list(p.imap(_feats_vec_1, gen()))
        else:
            res = [_feats_vec_1(a) for a in gen()]
        if res:
            X[m] = np.vstack(res)
    return pd.DataFrame(X, columns=FEATS)


def _amb_group(c, tabs, ia):
    """Chain-name ambiguity: a target whose core name is shared by many S1 entities is risky."""
    n = len(c)
    cc = tabs.get("core_counts")
    if cc is None or not len(cc):
        return pd.DataFrame(index=range(n))

    def core_at(tab, pos):                  # pos -1 = id not loaded -> ""
        v = tab.cols["core"][np.maximum(pos, 0)] if len(tab.index) else np.array([""] * len(pos), dtype=object)
        return np.where(pos >= 0, v, "")
    core_a = pd.Series(core_at(tabs[1], ia), dtype=object)
    core_b = pd.Series([""] * n, dtype=object)
    for s in (2, 3):
        m = np.where(c["srcn"].values == s)[0]
        core_b.iloc[m] = core_at(tabs[s], tabs[s].pos(c["cand_id"].values[m]))
    return pd.DataFrame({"amb_s1_core_freq": core_a.map(cc).fillna(0).values.astype(np.float32),
                         "amb_t_core_freq": core_b.map(cc).fillna(0).values.astype(np.float32),
                         "amb_core_eq": (core_a.values == core_b.values) & (core_a.values != "")})


def _file_sig(paths):
    out = []
    for p in paths:
        if os.path.exists(p):
            st = os.stat(p)
            out.append([str(p), st.st_size, st.st_mtime_ns])
    return out


def feature_cache(split, tag, cand_paths, procs):
    """Cache spec for compute_features (None when BER_FEATURE_CACHE=0)."""
    if os.environ.get("BER_FEATURE_CACHE", "1") == "0":
        return None
    d = C.PROCESSED_DIR / f"features_{split}"
    d.mkdir(parents=True, exist_ok=True)
    return {"path": d / f"{tag}.parquet", "inputs": _file_sig(list(cand_paths) + [procs[n] for n in (1, 2, 3)])}


def compute_features(c, tabs, workers, cache=None):
    import json
    ia = tabs[1].pos(c["s1_id"].values)
    groups = {"base": lambda: _base_group(c, tabs, workers, ia), "amb": lambda: _amb_group(c, tabs, ia)}
    if EXTRA:
        groups["extra"] = lambda: extra_features(c, tabs, ia, workers)
    if DECOY:
        groups["decoy"] = lambda: decoy_features(c, tabs, ia)
    rowkey = str(int(pd.util.hash_array((c["s1_id"].astype(str) + "|" + c["cand_id"].astype(str)).values).sum()))
    old, meta = None, {}
    if cache is not None and cache["path"].exists() and cache["path"].with_suffix(".json").exists():
        meta = json.loads(cache["path"].with_suffix(".json").read_text())
        if meta.get("rows") == len(c) and meta.get("rowkey") == rowkey:
            old = pd.read_parquet(cache["path"])
        else:
            meta = {}
    parts, done, fresh = {}, [], []
    for g, fn in groups.items():
        sig = [FEATURE_GROUPS[g], cache["inputs"]] if cache is not None else None
        m = meta.get("groups", {}).get(g)
        if old is not None and m is not None and m["sig"] == sig and all(col in old for col in m["cols"]):
            parts[g] = old[m["cols"]].reset_index(drop=True)
            done.append(g)
        else:
            parts[g] = fn().reset_index(drop=True)
            fresh.append(g)
    if cache is not None and fresh:
        keep = {g: {"sig": [FEATURE_GROUPS[g], cache["inputs"]], "cols": list(parts[g].columns)} for g in parts}
        pd.concat(list(parts.values()), axis=1).to_parquet(cache["path"], index=False)
        cache["path"].with_suffix(".json").write_text(json.dumps({"rows": len(c), "rowkey": rowkey, "groups": keep}))
    if cache is not None:
        print(f"  features: reused {done or '-'}, computed {fresh or '-'} ({cache['path'].name})")
    F = parts["base"]
    F["srcn"] = c["srcn"].values
    F = pd.concat([F] + [parts[g] for g in groups if g != "base"], axis=1)
    for col in c.columns:
        if col not in RESERVED and col != "srcn" and pd.api.types.is_numeric_dtype(c[col]):
            F["blk_" + col] = c[col].values
    return F


# --------------------------------------------------------------------------- model / decisions
def make_model():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=1500, learning_rate=C.LGB_LEARNING_RATE, num_leaves=63, subsample=0.8,
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


SAVE_FLOOR = 0.05     # predict keeps every test pair scored at least this (for `decide`)
# saved with every scored test pair: key similarities, conflict flags and the blocking evidence
TRACE_COLS = ["n_best", "n_tok_jacc_rom", "n_rom_tset", "a_tok_jacc_rom", "a_rom_tset", "a_num_jacc",
              "a_house_eq", "a_house_conflict", "a_pin_conflict", "amb_s1_core_freq", "blk_prune_p", "blk_mask",
              "blk_rank", "blk_t_rank", "blk_tf_cos", "blk_rev_n", "blk_rev_margin"]
BLOCK_KEY_NAMES = ["core", "tok", "skelbi", "pre", "addrpc", "nameaddr", "addrbi", "join", "tfidf", "sorted", "acro", "subset"]


def blocking_keys(mask):
    """blk_mask bits -> 'core+nameaddr+...' (which blocking keys proposed the pair)."""
    m = np.asarray(mask, dtype=np.int64)
    out = np.full(len(m), "", dtype=object)
    for i, k in enumerate(BLOCK_KEY_NAMES):
        hit = (m >> i) & 1 == 1
        out[hit] = np.where(out[hit] == "", k, out[hit] + "+" + k)
    return out
MAX_PER_SOURCE = 6    # no train S1 has more than 5 S2 / 6 S3 matches (profile_data.py P5)


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

    def __call__(self, thr, fallback, exclusive, margin=0.0, cap=None):
        keep = self._keep(thr, fallback, exclusive, margin, MAX_PER_SOURCE if cap is None else cap)
        if getattr(self, "idx", None) is None:
            return keep
        full = np.zeros(self.n_full, bool)
        full[self.idx] = keep
        return full

    def _keep(self, thr, fallback, exclusive, margin=0.0, cap=MAX_PER_SOURCE):
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
        if cap:
            ks = self.perm[keep[self.perm]]
            over = ks[pd.Series(self.g_ss[ks]).groupby(self.g_ss[ks]).cumcount().to_numpy() >= cap]
            keep[over] = False
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

    def __init__(self, s1, y, n_true=None, extra_single=0, extra_other=0, fp_weight=1.0):
        self.fp_weight = fp_weight
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
        pred = tp + self.fp_weight * (pred - tp)          # fp_weight > 1: stress-test precision
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
    """Mean of the fold models (stacked with Extra-Trees + logistic regression when the bundle has a
    kept ensemble), then isotonic calibration."""
    p = np.mean([m.predict_proba(X)[:, 1] for m in bundle["models"]], axis=0)
    if bundle.get("ensemble"):
        import ensemble as E
        srcn = X["srcn"].values if "srcn" in X else np.full(len(X), 2)
        am = X["a_missing"].fillna(0).values if "a_missing" in X else np.zeros(len(X))
        p = E.predict_ensemble(bundle["ensemble"], X, p, srcn, am)
    cal = bundle.get("calibrator")
    return cal.predict(p) if cal is not None else p


NEG_RATE = 0.2         # share of easy negatives kept for training (rest dropped, kept ones reweighted)
HARD_NEG_SIM = 0.8     # negatives at least this similar (n_best) are always kept


def decoy_ratio(n_true):
    """How many more decoy targets per S1 test has than train, from row counts only (no test labels),
    assuming test S1 entities have as many true matches as train ones. Used to up-weight false
    positives when tuning thresholds, so train-tuned thresholds are not too lenient for test."""
    try:
        rows = lambda sp, n: pq.ParquetFile(C.processed_path(sp, n)).metadata.num_rows
        m = float(n_true.mean())
        per = {sp: (rows(sp, 2) + rows(sp, 3)) / rows(sp, 1) for sp in ("train", "test")}
        r = (per["test"] - m) / max(per["train"] - m, 1e-6)
        r = float(np.clip(r, 1.0, 3.0))
        print(f"targets per S1: train {per['train']:.2f}, test {per['test']:.2f}; mean true matches {m:.2f} "
              f"-> false-positive weight {r:.2f}")
        return r
    except OSError:
        return 1.0


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
    ho = C.is_holdout(c["s1_id"].values)
    print(f"locked holdout: {ho.sum():,} candidate rows of {c.loc[ho, 's1_id'].nunique():,} S1 excluded from training/tuning")
    use_ens = os.environ.get("BER_ENSEMBLE", "0") == "1"
    ch = c[ho].reset_index(drop=True) if use_ens else None     # scored only for the ensemble gate
    c = c[~ho].reset_index(drop=True)
    y = c["is_true"].astype(bool).values
    print(f"train candidates: {len(c):,}, positives: {y.sum():,} ({y.mean():.3%})")
    procs = find_processed("train", parse_procs(a.proc_file))
    need = set(c["s1_id"]) | set(c["cand_id"])
    tabs = load_tables(procs, need, a.id_col)
    if ch is not None:
        need |= set(ch["s1_id"]) | set(ch["cand_id"])
        tabs = load_tables(procs, need, a.id_col)
    X = compute_features(c, tabs, a.workers, feature_cache("train", f"train_frac{a.s1_frac}", files, procs))
    Xh = compute_features(ch, tabs, a.workers, feature_cache("train", f"holdout_frac{a.s1_frac}", files, procs)) \
        if ch is not None and len(ch) else None
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
    oof_et, oof_lr, ens_et, ens_lr = np.zeros(len(c)), np.zeros(len(c)), [], []
    for k in range(a.folds):
        te = fold == k
        tr = ~te & samp
        m = fit_one(X[tr].reset_index(drop=True), y[tr], inner_all[tr], wts[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        models.append(m)
        print(f"  fold {k}: train {tr.sum():,} / held-out {te.sum():,}  "
              f"trees {getattr(m, 'best_iteration_', None)}  ({time.time() - t0:.0f}s)")
        if use_ens:
            import ensemble as E
            Xtr = X[tr].reset_index(drop=True)
            imp = m.booster_.feature_importance("gain") if hasattr(m, "booster_") else np.ones(X.shape[1])
            et = E.fit_extra_trees(Xtr, y[tr], wts[tr], imp, seed=k)
            lr = E.fit_logreg(Xtr, y[tr], wts[tr], seed=k)
            oof_et[te], oof_lr[te] = E.predict_extra_trees(et, X[te]), E.predict_logreg(lr, X[te])
            ens_et.append(et); ens_lr.append(lr)
            print(f"  fold {k}: extra-trees + logistic regression ({time.time() - t0:.0f}s)")
    ensemble = None
    if use_ens:
        # gate on the LOCKED HOLDOUT (never trained on): stacked ensemble must beat LightGBM alone in macro F0.5;
        # calibration and per-source thresholds for both come from the out-of-fold predictions
        import ensemble as E
        am0 = (X["a_missing"].fillna(0).values > 0).astype(int)
        Z = E.stack_matrix(oof, oof_et, oof_lr, c["srcn"].values, am0)
        oof_stack = E.stacked_oof(Z, y, fold)
        stacker = E.fit_stacker(Z, y)
        ensemble_cand = {"et": ens_et, "lr": ens_lr, "stacker": stacker}
        kept = False
        if Xh is not None:
            cols_h = list(X.columns)
            for col in cols_h:
                if col not in Xh:
                    Xh[col] = np.nan
            Xh = Xh[cols_h]
            yh = ch["is_true"].astype(bool).values
            amh = (Xh["a_missing"].fillna(0).values > 0).astype(int)
            ph_lgb = np.mean([m_.predict_proba(Xh)[:, 1] for m_ in models], axis=0)
            ph_stack = E.predict_ensemble(ensemble_cand, Xh, ph_lgb, ch["srcn"].values, amh)
            n_true0 = load_gt_counts(procs, a.id_col)
            evh = MacroF05(ch["s1_id"].values, yh, n_true0.to_dict())
            dh = ch.assign(a_missing=amh)

            def holdout_macro(p_oof, p_h):
                cal0 = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_oof, y)
                q, qh = cal0.predict(p_oof), cal0.predict(p_h)
                t_ = {s_: best_threshold(y[c["srcn"].values == s_], q[c["srcn"].values == s_]) for s_ in (2, 3)}
                return evh(decide(dh, qh, t_, False, False, 0.05))["e2e"]
            m_lgb, m_stack = holdout_macro(oof, ph_lgb), holdout_macro(oof_stack, ph_stack)
            print(f"ensemble gate (LOCKED HOLDOUT, {ch['s1_id'].nunique():,} S1): LightGBM alone {m_lgb:.4f} | "
                  f"stacked LGB + Extra-Trees + RBF {m_stack:.4f}")
            kept = m_stack > m_lgb + 0.0005
        if kept:
            ensemble = ensemble_cand
            oof = oof_stack.astype(np.float32)
            print("ensemble KEPT")
        else:
            print("ensemble NOT kept (no locked-holdout gain)")
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
    s1_all = s1_all[s1_all.isin(n_true.index) & ~C.is_holdout(s1_all.values)]
    absent = s1_all[~s1_all.isin(set(c["s1_id"]))]
    n_abs_single = int((absent.map(n_true) == 0).sum())
    fpw = decoy_ratio(n_true)
    ev = MacroF05(c["s1_id"].values, y, n_true.to_dict(), n_abs_single, len(absent) - n_abs_single, fpw)
    ev_plain = MacroF05(c["s1_id"].values, y, n_true.to_dict(), n_abs_single, len(absent) - n_abs_single)
    print(f"S1 population {len(s1_all):,}: {len(absent):,} without candidates "
          f"({n_abs_single:,} singletons score 1, the rest 0)")

    pd.DataFrame({"s1_id": c["s1_id"].values, "cand_id": c["cand_id"].values, "srcn": c["srcn"].values,
                  "a_missing": X["a_missing"].values, "p": pc, "y": y}).to_parquet(C.PROCESSED_DIR / "oof_train.parquet")
    joblib.dump({"n_abs_single": n_abs_single, "n_abs_other": len(absent) - n_abs_single, "fp_weight": fpw,
                 "n_true": n_true}, C.PROCESSED_DIR / "oof_context.joblib")
    print(f"saved out-of-fold predictions -> {C.PROCESSED_DIR / 'oof_train.parquet'} (for `matching.py retune`)")
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
    plain = ev_plain(dec(thr, best[1], best[2], best[3]))
    print("thresholds tuned on decoy-weighted macro F0.5:", {str(k): round(v, 3) for k, v in thr.items()},
          f"-> weighted {sc:.4f} | unweighted macroF0.5(e2e)={plain['e2e']:.4f} (P {plain['P']:.4f}, R {plain['R']:.4f})")
    best = (sc,) + best[1:]

    imp = getattr(models[0], "feature_importances_", None)
    if imp is not None:
        top = sorted(zip(X.columns, imp), key=lambda z: -z[1])[:15]
        print("top features:", ", ".join(f"{k}({v})" for k, v in top))
    os.makedirs(C.RUN_MODELS_DIR, exist_ok=True)
    out = os.path.join(C.RUN_MODELS_DIR, "stage4_model.joblib")
    joblib.dump({"models": models, "calibrator": cal, "feats": list(X.columns), "thr": thr, "ensemble": ensemble,
                 "fallback": best[1], "exclusive": best[2], "margin": best[3]}, out)
    print("saved", out)


def cmd_predict(a):
    bundle = joblib.load(os.path.join(C.RUN_MODELS_DIR, "stage4_model.joblib"))
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
    floor = min(SAVE_FLOOR, min(thr.values()) * 0.5)
    kept = []
    for i, f in enumerate(files):
        c = read_cand_file(f)
        X = compute_features(c, tabs, a.workers, feature_cache("test", os.path.basename(f)[:-8], [f], procs))
        for col in cols:
            if col not in X.columns:
                X[col] = np.nan
        p = predict_p(bundle, X[cols])
        m = p >= floor
        kd = pd.DataFrame({"s1_id": c["s1_id"].values[m], "cand_id": c["cand_id"].values[m],
                           "srcn": c["srcn"].values[m], "a_missing": X["a_missing"].values[m], "p": p[m]})
        for col in TRACE_COLS:               # why the pair scored what it did (decision trace)
            if col in X.columns:
                kd[col] = X[col].values[m]
        kept.append(kd)
        print(f"[{i + 1}/{len(files)}] {os.path.basename(f)}: {len(c):,} pairs, {m.sum():,} above floor")
    allp = pd.concat(kept, ignore_index=True)
    allp.to_parquet(C.PROCESSED_DIR / "scored_test.parquet", index=False)
    print(f"saved {len(allp):,} scored test pairs (p >= {floor:.2f}) -> scored_test.parquet (for `matching.py decide`)")
    write_results(allp, bundle)


def write_results(allp, bundle):
    """Decision layer on scored test pairs -> output/matching_results.tsv (one row per test S1)."""
    thr = bundle["thr"]
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
    trace = allp.assign(decision=np.where(keep, "match", "reject"), source="model")
    if "blk_mask" in trace:
        trace["blocking_keys"] = blocking_keys(trace["blk_mask"].fillna(0).values)
    trace.to_parquet(C.PROCESSED_DIR / "decision_trace_test.parquet", index=False)
    print(f"wrote {out}: {res['s1_id'].nunique():,} S1 entities matched, {len(res):,} pairs")


def load_bundle():
    return joblib.load(os.path.join(C.RUN_MODELS_DIR, "stage4_model.joblib"))


def cmd_retune(a):
    """Re-tune thresholds + rule from saved out-of-fold predictions (no feature computation, no retraining)."""
    t0 = time.time()
    oof = pd.read_parquet(C.PROCESSED_DIR / "oof_train.parquet")
    ctx = joblib.load(C.PROCESSED_DIR / "oof_context.joblib")
    fpw = ctx["fp_weight"] if a.fp_weight is None else a.fp_weight
    ev = MacroF05(oof["s1_id"].values, oof["y"].values.astype(bool), ctx["n_true"].to_dict(),
                  ctx["n_abs_single"], ctx["n_abs_other"], fpw)
    ev_plain = MacroF05(oof["s1_id"].values, oof["y"].values.astype(bool), ctx["n_true"].to_dict(),
                        ctx["n_abs_single"], ctx["n_abs_other"])
    bundle = load_bundle()
    dec = Decider(oof, oof["p"].values).compress(0.1)
    best = None
    for fb in (False, True):
        for ex in (False, True):
            for mg in (0.0, 0.05, 0.1, 0.2):
                m = ev(dec(bundle["thr"], fb, ex, mg))
                if best is None or m["e2e"] > best[0]:
                    best = (m["e2e"], fb, ex, mg)
    thr, sc = tune_thresholds(ev, dec, dict(bundle["thr"]), best[1], best[2], best[3])
    plain = ev_plain(dec(thr, best[1], best[2], best[3]))
    print(f"fp weight {fpw:.2f}: rule fallback={best[1]} exclusive={best[2]} margin={best[3]} | "
          f"weighted {sc:.4f} | unweighted macroF0.5 {plain['e2e']:.4f} (P {plain['P']:.4f}, R {plain['R']:.4f}) "
          f"({time.time() - t0:.0f}s)")
    print("thresholds:", {str(k): round(v, 3) for k, v in thr.items()})
    if a.save:
        bundle.update(thr=thr, fallback=best[1], exclusive=best[2], margin=best[3])
        joblib.dump(bundle, os.path.join(C.RUN_MODELS_DIR, "stage4_model.joblib"))
        print("updated models/stage4_model.joblib (run `matching.py decide` to rewrite the results)")


def cmd_decide(a):
    """Rewrite output/matching_results.tsv from saved test scores with the bundle's (or scaled) thresholds."""
    allp = pd.read_parquet(C.PROCESSED_DIR / "scored_test.parquet")
    bundle = load_bundle()
    if a.thr_scale != 1.0:
        bundle["thr"] = {k: min(0.99, v * a.thr_scale) for k, v in bundle["thr"].items()}
        print("thresholds scaled x", a.thr_scale, "->", {str(k): round(v, 3) for k, v in bundle["thr"].items()})
    write_results(allp, bundle)


def cmd_evaluate(a):
    """Score the saved model on train S1 entities it never saw (hash >= --train-frac, the fraction
    `train --s1-frac` used) with the exact leaderboard metric - no training. Works on a blocking
    slice (blocking.py --split train --eval --limit-s1 N): the population is the blocked S1s."""
    bundle = load_bundle()
    cols = bundle["feats"]
    procs = find_processed("train", parse_procs(a.proc_file))
    cut = int(a.train_frac * 10_000)
    files = cand_files("train")
    c = pd.concat([read_cand_file(f) for f in files], ignore_index=True)

    def in_pop(ids):
        """held-out: never in the model's training sample; with --population holdout also in the locked holdout"""
        keep = pd.util.hash_array(pd.Series(ids).astype(str).values) % 10_000 >= cut
        return keep & C.is_holdout(ids) if a.population == "holdout" else keep
    c = c[in_pop(c["s1_id"].values)].reset_index(drop=True)
    print(f"evaluation population: {a.population} (train-frac cut {a.train_frac})")
    tabs = load_tables(procs, set(c["s1_id"]) | set(c["cand_id"]), a.id_col)
    X = compute_features(c, tabs, a.workers, feature_cache("train", f"evaluate_{a.train_frac}", files, procs))
    for col in cols:
        if col not in X.columns:
            X[col] = np.nan
    p = predict_p(bundle, X[cols])
    keep = decide(c.assign(a_missing=X["a_missing"].values), p, bundle["thr"], bundle["fallback"],
                  bundle["exclusive"], bundle.get("margin", 0.0))
    y = c["is_true"].astype(bool).values
    n_true = load_gt_counts(procs, a.id_col)
    s1_all = pd.read_parquet(C.PROCESSED_DIR / "blocked_s1_train.parquet").iloc[:, 0].astype(str)
    s1_all = s1_all[in_pop(s1_all.values) & s1_all.isin(n_true.index)]
    ctry = pd.Series(tabs[1].cols["country_norm"], index=tabs[1].index)
    s1_ctry = pd.read_parquet(procs[1], columns=["entity_id", "country_norm"]).set_index("entity_id")["country_norm"]
    print(f"held-out S1 entities: {len(s1_all):,} | candidate pairs {len(c):,} ({len(c) / max(len(s1_all), 1):.1f} per S1)")
    print(f"{'slice':10}{'S1':>10}{'P':>8}{'R':>8}{'F0.5':>8}{'macro F0.5':>12}")
    for name, s1s in [("all", s1_all)] + [(k, s1_all[s1_all.map(s1_ctry) == k]) for k in sorted(ctry.unique())]:
        m = c["s1_id"].isin(set(s1s)).values
        absent = s1s[~s1s.isin(set(c["s1_id"][m]))]
        n_single = int((absent.map(n_true) == 0).sum())
        r = MacroF05(c["s1_id"].values[m], y[m], n_true.to_dict(), n_single, len(absent) - n_single)(keep[m])
        print(f"{name:10}{len(s1s):>10,}{r['P']:8.4f}{r['R']:8.4f}{r['F0.5']:8.4f}{r['e2e']:12.4f}")
    breakdown_report(a, c, X, p, keep, y, s1_all, procs, tabs)
    if a.permute:
        permutation_report(bundle, c, X[cols], y, s1_all, n_true)


def loss_decomposition(g, c, s1_all):
    """Macro F0.5 lost to each cause: the score if ONLY that loss were fixed, minus the actual score.
    g: every true pair of the population with its outcome; c: scored candidates with keep / is_true."""
    b2 = 0.25
    pop = pd.Index(pd.unique(s1_all.astype(str)))
    true_n = g.groupby("s1_id").size().reindex(pop).fillna(0)
    oc = g.groupby(["s1_id", "outcome"]).size().unstack(fill_value=0).reindex(pop).fillna(0)
    for col in ("TP", "missed_by_blocking", "removed_by_pruning", "rejected_by_model"):
        if col not in oc:
            oc[col] = 0
    fp = c[c["keep"] & ~c["is_true"]].groupby("s1_id").size().reindex(pop).fillna(0)

    def macro(tp, fpn):
        pred = tp + fpn
        f = np.where((pred == 0) & (true_n == 0), 1.0,
                     np.where(pred == 0, 0.0, (1 + b2) * tp / np.maximum((1 + b2) * tp + b2 * (true_n - tp) + fpn, 1e-9)))
        return float(np.mean(f))
    base = macro(oc["TP"], fp)
    rows = [("actual", base, 0.0)]
    for name, add in (("fix blocking misses", oc["missed_by_blocking"]), ("fix pruning losses", oc["removed_by_pruning"]),
                      ("fix model rejections", oc["rejected_by_model"])):
        m = macro(oc["TP"] + add, fp); rows.append((name, m, m - base))
    m = macro(oc["TP"], fp * 0); rows.append(("remove all false positives", m, m - base))
    single_fp = int(((true_n == 0) & (fp > 0)).sum())
    m = macro(oc["TP"], fp.where(true_n > 0, 0)); rows.append(("empty prediction for singletons", m, m - base))
    m = macro(true_n, fp * 0); rows.append(("perfect", m, m - base))
    print("\nloss decomposition (macro F0.5 if only that loss were fixed)")
    for name, m, d in rows:
        print(f"  {name:34} {m:.4f}  (+{d:.4f})")
    print(f"  singletons wrongly given a match: {single_fp:,} of {int((true_n == 0).sum()):,}")
    lp = C.PROCESSED_DIR / "lost_pairs_classified.parquet"
    pd.DataFrame(rows, columns=["scenario", "macro_f05", "gain"]).to_csv(C.PROCESSED_DIR / "eval_loss_decomposition.csv", index=False)


def breakdown_report(a, c, X, p, keep, y, s1_all, procs, tabs):
    """Pair-level confusion matrix per target source x address present/missing, where every
    ground-truth pair of the held-out S1 population ends in exactly one bucket: missed by blocking,
    removed by pruning, rejected by the model (p below threshold / rules) or matched (TP).
    Raw counts + % of the slice's true pairs, P/R/F0.5, and mean probability per outcome.
    Saves every scored pair (eval_pairs.parquet) and the table (eval_breakdown.csv)."""
    gt = pd.read_csv(C.TRAIN_GT, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    g = pd.DataFrame({"s1_id": gt.iloc[:, 0].str.strip(), "cand_id": gt.iloc[:, 1].str.split(",")}).explode("cand_id")
    g["cand_id"] = g["cand_id"].fillna("").str.strip()
    g = g[(g["cand_id"] != "") & g["s1_id"].isin(set(s1_all))].drop_duplicates().reset_index(drop=True)
    g["srcn"] = src_num(g["cand_id"].values)
    tg = load_tables(procs, set(g["s1_id"]) | set(g["cand_id"]), a.id_col)
    ia = tg[1].pos(g["s1_id"].values)
    have = np.zeros(len(g), bool)
    for s_ in (2, 3):
        m = np.where(g["srcn"].values == s_)[0]
        have[m] = tg[s_].pos(g["cand_id"].values[m]) >= 0
    g = g[have & (ia >= 0)].reset_index(drop=True)
    ia = tg[1].pos(g["s1_id"].values)
    miss = np.zeros(len(g), bool)
    for side in (0, 1):
        present = np.zeros(len(g), bool)
        for col in ("addr_norm", "addr_rom"):
            t = _pair_texts(g, tg, ia, col)[side]
            present |= t != ""
        miss |= ~present
    g["a_missing"] = miss.astype(int)
    key = lambda d: d["s1_id"].astype(str) + "|" + d["cand_id"].astype(str)
    blocked = set()
    if a.blocked_dir:
        for f in sorted(glob.glob(os.path.join(a.blocked_dir, "*.parquet"))):
            b = pd.read_parquet(f, columns=["s1_id", "cand_id", "is_true"])
            b = b[b["is_true"].astype(bool)]
            blocked |= set(key(b))
    c = c.assign(a_missing=(X["a_missing"].fillna(0).values > 0).astype(int), p=p, keep=keep, is_true=y)
    feats = X.drop(columns=[f for f in ("a_missing", "srcn") if f in X.columns]).reset_index(drop=True)
    pd.concat([c.assign(country=pd.Series(tabs[1].cols["country_norm"], index=tabs[1].index).reindex(c["s1_id"]).values)[
        ["s1_id", "cand_id", "srcn", "a_missing", "country", "p", "keep", "is_true"]].reset_index(drop=True), feats],
        axis=1).to_parquet(C.PROCESSED_DIR / "eval_pairs.parquet", index=False)
    ck = key(c)
    in_c = set(ck[c["is_true"].values])
    tp_set = set(ck[c["is_true"].values & c["keep"].values])
    gk = key(g)
    g["outcome"] = np.where(gk.isin(tp_set), "TP", np.where(gk.isin(in_c), "rejected_by_model",
                            np.where(gk.isin(blocked), "removed_by_pruning", "missed_by_blocking")))
    if not a.blocked_dir:
        g.loc[g["outcome"] == "removed_by_pruning", "outcome"] = "missed_by_blocking"
    g.to_parquet(C.PROCESSED_DIR / "eval_gt_outcomes.parquet", index=False)
    loss_decomposition(g, c, s1_all)
    thr = load_bundle()["thr"]
    rows = []
    for s_ in (2, 3, None):
        for am in (0, 1, None):
            gm = np.ones(len(g), bool) if s_ is None else g["srcn"].values == s_
            cm = np.ones(len(c), bool) if s_ is None else c["srcn"].values == s_
            if am is not None:
                gm &= g["a_missing"].values == am
                cm &= c["a_missing"].values == am
            d, o = c[cm], g[gm]["outcome"].value_counts()
            n_gt = int(gm.sum())
            tp = int(o.get("TP", 0))
            fp = int((d["keep"] & ~d["is_true"]).sum())
            tn = int((~d["keep"] & ~d["is_true"]).sum())
            P, R = tp / max(tp + fp, 1), tp / max(n_gt, 1)
            mp = lambda m: float(d["p"][m].mean()) if m.any() else float("nan")
            rows.append({
                "source": "all" if s_ is None else f"S{s_}",
                "address": "all" if am is None else ("missing" if am else "present"),
                "threshold": "" if s_ is None or am is None else round(thr_of(thr, s_, am), 3),
                "true_pairs": n_gt,
                "missed_by_blocking": int(o.get("missed_by_blocking", 0)),
                "removed_by_pruning": int(o.get("removed_by_pruning", 0)),
                "rejected_by_model": int(o.get("rejected_by_model", 0)),
                "TP": tp, "FP": fp, "FN": n_gt - tp, "TN_candidates": tn,
                "P": P, "R": R, "F05": float(fbeta(P, R)),
                "p_TP": mp(d["keep"] & d["is_true"]), "p_FP": mp(d["keep"] & ~d["is_true"]),
                "p_rejected_true": mp(~d["keep"] & d["is_true"]), "p_TN": mp(~d["keep"] & ~d["is_true"]),
            })
    t = pd.DataFrame(rows)
    t.to_csv(C.PROCESSED_DIR / "eval_breakdown.csv", index=False)
    pct = lambda n, tot: f"{n:>10,} {100 * n / max(tot, 1):5.1f}%"
    print("\nconfusion matrix by target source x address (pairs; % of the slice's true pairs)")
    for r in t.itertuples():
        print(f"\n[{r.source} | address {r.address}] threshold {r.threshold or '-'} | true pairs {r.true_pairs:,}")
        print(f"   TP (matched)             {pct(r.TP, r.true_pairs)}   mean p {r.p_TP:.3f}")
        print(f"   FN total                 {pct(r.FN, r.true_pairs)}")
        print(f"     missed by blocking     {pct(r.missed_by_blocking, r.true_pairs)}")
        print(f"     removed by pruning     {pct(r.removed_by_pruning, r.true_pairs)}")
        print(f"     rejected by model      {pct(r.rejected_by_model, r.true_pairs)}   mean p {r.p_rejected_true:.3f}")
        print(f"   FP (wrong matches)       {r.FP:>10,} {100 * r.FP / max(r.TP + r.FP, 1):5.2f}% of predictions   mean p {r.p_FP:.3f}")
        print(f"   TN (candidates rejected) {r.TN_candidates:>10,}         mean p {r.p_TN:.3f}")
        print(f"   P {r.P:.4f}  R {r.R:.4f}  F0.5 {r.F05:.4f}")


def permutation_report(bundle, c, X, y, s1_all, n_true, seed=0, max_pairs=500_000):
    """Effect of each feature on held-out P / R / F0.5: shuffle that column (breaks its link to the
    label), re-score, re-decide, compare. No retraining. Correlated features cover for each other,
    so groups are shuffled together too."""
    if len(c) > max_pairs:                          # hash-subsample S1 entities: ~1 min per feature
        keep_s1 = pd.util.hash_array(s1_all.values, hash_key="permute000000000") % 10_000 < int(1e4 * max_pairs / len(c))
        s1_all = s1_all[keep_s1]
        m = c["s1_id"].isin(set(s1_all)).values
        c, X, y = c[m].reset_index(drop=True), X[m].reset_index(drop=True), y[m]
        print(f"\npermutation on a subsample: {len(s1_all):,} S1 entities, {len(c):,} pairs")
    absent = s1_all[~s1_all.isin(set(c["s1_id"]))]
    n_single = int((absent.map(n_true) == 0).sum())
    ev = MacroF05(c["s1_id"].values, y, n_true.to_dict(), n_single, len(absent) - n_single)
    d = c.assign(a_missing=X["a_missing"].values)

    def score(Xp):
        keep = decide(d, predict_p(bundle, Xp), bundle["thr"], bundle["fallback"], bundle["exclusive"],
                      bundle.get("margin", 0.0))
        return ev(keep)

    base = score(X)
    rng = np.random.default_rng(seed)
    groups = {"ALL name similarity": [f for f in X.columns if f.startswith("n_")],
              "ALL address similarity": [f for f in X.columns if f.startswith("a_") and f != "a_missing"],
              "ALL blocking signals": [f for f in X.columns if f.startswith("blk_")],
              "ALL chain ambiguity": [f for f in X.columns if f.startswith("amb_")]}
    rows = [(f, [f]) for f in X.columns] + list(groups.items())
    print(f"\npermutation effect on held-out (base P {base['P']:.4f} R {base['R']:.4f} macro F0.5 {base['e2e']:.4f})")
    print(f"{'feature':26}{'dP':>9}{'dR':>9}{'dF0.5':>9}")
    out = []
    for name, fs in rows:
        Xp = X.copy()
        idx = rng.permutation(len(X))
        for f in fs:
            Xp[f] = X[f].values[idx]
        r = score(Xp)
        out.append((name, r["P"] - base["P"], r["R"] - base["R"], r["e2e"] - base["e2e"]))
    for name, dp, dr, df in sorted(out, key=lambda t: t[3]):
        print(f"{name:26}{dp:+9.4f}{dr:+9.4f}{df:+9.4f}")


def parse_procs(items):
    o = {}
    for it in items or []:
        k, v = it.split("=", 1)
        o[int(k)] = v
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("retune", cmd_retune), ("decide", cmd_decide)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if name == "retune":
            p.add_argument("--fp-weight", type=float, default=None, help="false-positive weight (default: from training)")
            p.add_argument("--save", action="store_true", help="write the re-tuned thresholds into the model bundle")
        else:
            p.add_argument("--thr-scale", type=float, default=1.0, help="multiply every threshold (<1: more recall)")
    for name, fn in (("train", cmd_train), ("predict", cmd_predict), ("evaluate", cmd_evaluate)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
        p.add_argument("--proc-file", nargs="*", help="override processed files, e.g. 1=path 2=path 3=path")
        p.add_argument("--id-col", default=None, help="entity ID column name in processed parquet")
        if name == "train":
            p.add_argument("--s1-frac", type=float, default=1.0, help="hash-sample of S1 entities to train on")
            p.add_argument("--folds", type=int, default=5, help="grouped CV folds (by S1 entity)")
        elif name == "evaluate":
            p.add_argument("--train-frac", type=float, default=0.5, help="the --s1-frac the saved model was trained with")
            p.add_argument("--population", choices=["heldout", "holdout"], default="heldout",
                           help="holdout = the locked 10%% holdout (final reporting); heldout = every untrained S1")
            p.add_argument("--permute", action="store_true", help="also report each feature's effect on P / R / F0.5")
            p.add_argument("--blocked-dir", default=None, help="copy of the candidates before pruning (splits FN into blocking vs pruning)")
        else:
            p.add_argument("--format", choices=["grouped"], default="grouped")  # only format the validator accepts
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()