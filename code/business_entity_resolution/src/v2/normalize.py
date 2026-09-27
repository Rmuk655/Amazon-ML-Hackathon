"""Text normalisation for names and addresses, with a lexicon learned from train truth pairs.

Lexicon (learned, no external data):
  indic_tok  Indic-script token -> Latin token (names are transliterated from a closed vocabulary; aligned
             position by position on pairs whose token counts agree)
  comp       address component -> canonical S1 component (state abbreviations / regional-script states,
             e.g. "tn" -> "tamil nadu", "महाराष्ट्र" -> "maharashtra")
  tok        address token -> canonical token ("st" -> "street", "ave" -> "avenue", "r" -> "rue")

Per record we produce:
  core    name tokens without legal forms / alias prefixes / website wrapping (space separated)
  legal   canonical legal-form tokens (sorted)
  glued   core without spaces
  atok    address word tokens (canonicalised, placeholders removed)
  anum    address numbers in order of appearance (leading zeros stripped; >= 5 digit postcodes kept apart)
  post    postcode-like numbers (5-6 digits)
"""
import html
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from unidecode import unidecode

from .common import WORK, log

INDIC_RE = re.compile(r"[ऀ-෿]")
LEGAL_CANON = {
    "inc": "inc", "incorporated": "inc", "llc": "llc", "ltd": "ltd", "limited": "ltd", "pvt": "pvt",
    "private": "pvt", "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "llp": "llp",
    "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc", "sa": "sa", "sas": "sas", "sasu": "sasu",
    "sarl": "sarl", "eurl": "eurl", "sci": "sci", "snc": "snc", "gmbh": "gmbh", "cie": "cie", "opc": "opc",
    "public": "public",
}
ALIAS_RE = re.compile(r"^(.*?)\b(?:dba|d b a|aka|a k a|doing business as|trading as|t a|also known as|formerly)\b(.*)$")
WEB_RE = re.compile(r"^(?:https?://)?(?:www\.)?([\w-]+)\.(?:c[o0]m|co\.in|in|net|org|fr|biz|us|info|co)$", re.I)
PLACEHOLDER = {"null", "none", "na", "nan", "n a", "nil", "not available", "unknown", "-"}
ORD_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
LZ_RE = re.compile(r"\b0+(\d)")
NONAN_RE = re.compile(r"[^a-z0-9 ]+")
SINGLE_RUN_RE = re.compile(r"\b(?:[a-z] ){1,}[a-z]\b")
NUM_RE = re.compile(r"\d+")

_LEX = None


def fix_encoding(s: str) -> str:
    if "&" in s and ";" in s:
        s = html.unescape(s)
    if "Ã" in s or "â€" in s:
        try:
            s = s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return s


def basic(s: str, indic_tok=None) -> str:
    """Lowercase ASCII text with punctuation removed and letter runs joined ('l.l.c' -> 'llc')."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", fix_encoding(s))
    if indic_tok is not None and INDIC_RE.search(s):
        s = " ".join(indic_tok.get(t, t) for t in s.split())
    s = unidecode(s).lower().replace("&", " and ").replace("'", "")
    s = NONAN_RE.sub(" ", s)
    s = SINGLE_RUN_RE.sub(lambda m: m.group(0).replace(" ", ""), s)
    s = ORD_RE.sub(r"\1", s)
    s = LZ_RE.sub(r"\1", s)
    return " ".join(s.split())


def norm_name(raw: str, lex) -> tuple:
    s = raw or ""
    m = WEB_RE.match(s.strip())
    if m:
        s = m.group(1).replace("-", " ")
    b = basic(s, lex["indic_tok"])
    m = ALIAS_RE.match(b)
    if m and m.group(2).strip():
        b = m.group(2).strip()
    toks = b.split()
    core = [t for t in toks if t not in LEGAL_CANON]
    while core and core[-1] == "and":
        core.pop()
    while core and core[0] == "and":
        core.pop(0)
    if not core:                       # name made only of legal words: keep them as the core
        core = toks
    legal = sorted({LEGAL_CANON[t] for t in toks if t in LEGAL_CANON})
    core_s = " ".join(core)
    return core_s, " ".join(legal), core_s.replace(" ", "")


def norm_addr(raw: str, lex) -> tuple:
    if not raw:
        return "", "", ""
    s = unicodedata.normalize("NFKC", fix_encoding(raw))
    comps = []
    for c in s.split(","):
        c = c.strip()
        if not c:
            continue
        key = c.lower() if not INDIC_RE.search(c) else c
        if key in lex["comp_raw"]:
            c = lex["comp_raw"][key]
        b = basic(c, lex["indic_tok"])
        if not b or b in PLACEHOLDER:
            continue
        b = lex["comp"].get(b, b)
        comps.append(b)
    toks, nums, post = [], [], []
    tokmap = lex["tok"]
    for c in comps:
        for t in c.split():
            if t.isdigit():
                (post if len(t) >= 6 else nums).append(t)
            elif any(ch.isdigit() for ch in t):          # 1056c, g3, af684
                d = NUM_RE.findall(t)
                nums.extend(x.lstrip("0") or "0" for x in d if len(x) < 6)
                toks.append(t)
            elif t not in PLACEHOLDER:
                toks.append(tokmap.get(t, t))
    return " ".join(toks), " ".join(nums), " ".join(post)


# ---------------------------------------------------------------- lexicon learning
def _pairs_frame(raw, gt, max_pairs=600_000):
    g = gt[gt.rec >= 0]
    if len(g) > max_pairs:
        g = g.sample(max_pairs, random_state=0)
    d = raw.set_index("id")
    a = d.loc[g.s1.values, ["name", "addr"]].reset_index(drop=True)
    b = d.loc[g.rec.values, ["name", "addr", "country"]].reset_index(drop=True)
    return a, b


def learn_lexicon(raw, gt):
    """Learn the Indic token dictionary, address component map and address token map from true pairs."""
    a, b = _pairs_frame(raw, gt)
    log("lexicon from", len(a), "pairs")
    # 1) Indic name tokens, aligned position-wise
    cnt = defaultdict(Counter)
    for sn, rn in zip(a["name"].values, b["name"].values):
        if not rn or not INDIC_RE.search(rn):
            continue
        rt = rn.split()
        st = basic(sn).split()
        if len(rt) != len(st):
            continue
        for x, y in zip(rt, st):
            if INDIC_RE.search(x):
                cnt[x][y] += 1
    indic_tok = {}
    for x, c in cnt.items():
        y, n = c.most_common(1)[0]
        if n >= 2 and n >= 0.5 * sum(c.values()):
            indic_tok[x] = y
    log("indic tokens:", len(indic_tok))

    # 2) address components: raw rec component (lowercased / Indic) -> S1 component; learned on unmatched ones
    ccnt = defaultdict(Counter)
    ctot = Counter()
    for sa, ra in zip(a["addr"].values, b["addr"].values):
        if not sa or not ra:
            continue
        sc = [basic(c) for c in sa.split(",") if c.strip()]
        scs = set(sc)
        for c in ra.split(","):
            c = c.strip()
            if not c:
                continue
            key = c if INDIC_RE.search(c) else c.lower()
            if basic(c) in scs:
                continue
            if len(key.split()) > 3:
                continue
            ctot[key] += 1
            for x in sc[-3:]:                       # states / cities sit at the end of S1 addresses
                if x not in scs or True:
                    ccnt[key][x] += 1
    last = Counter()
    anyc = Counter()
    for sa in raw.loc[raw.src == 1, "addr"].values[:400_000]:
        cs = [basic(c) for c in sa.split(",") if c.strip()]
        if cs:
            last[cs[-1]] += 1
            anyc.update(cs)
    states = {k for k, v in last.items() if v >= 300}
    comp_raw = {}
    for k, c in ccnt.items():
        y, n = c.most_common(1)[0]
        bk = basic(k)
        if (n >= 20 and n >= 0.6 * ctot[k] and bk != y and y in states
                and anyc[bk] < 0.2 * ctot[k]):
            comp_raw[k] = y
    log("component map:", len(comp_raw))

    # 3) address tokens: unmatched rec token -> unmatched S1 token with the same first letter
    lex0 = {"indic_tok": indic_tok, "comp_raw": comp_raw, "comp": {}, "tok": {}}
    tcnt = defaultdict(Counter)
    ttot = Counter()
    s1_vocab = Counter()
    for sa, ra in zip(a["addr"].values, b["addr"].values):
        st = set(norm_addr(sa, lex0)[0].split())
        s1_vocab.update(st)
        rt = set(norm_addr(ra, lex0)[0].split())
        ro, so = rt - st, st - rt
        for x in ro:
            ttot[x] += 1
            for y in so:
                if y[0] == x[0] and len(y) > len(x):
                    tcnt[x][y] += 1
    tok = {}
    for x, c in tcnt.items():
        y, n = c.most_common(1)[0]
        if n >= 30 and n >= 0.4 * ttot[x] and (len(x) >= 2 or x in "nsew") and s1_vocab[x] < 0.2 * ttot[x]:
            tok[x] = y
    log("token map:", len(tok), dict(list(sorted(tok.items(), key=lambda kv: -tcnt[kv[0]][kv[1]]))[:40]))
    return {"indic_tok": indic_tok, "comp_raw": comp_raw, "comp": {}, "tok": tok}


def add_french_tokens(lex):
    """French street-type abbreviations (domain knowledge, like US 'St'/'Ave'); France is absent from train."""
    fr = {"r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard", "blvd": "boulevard",
          "imp": "impasse", "ch": "chemin", "che": "chemin", "chem": "chemin", "pl": "place", "all": "allee",
          "rte": "route", "sq": "square", "crs": "cours", "pass": "passage", "qu": "quai", "fg": "faubourg",
          "fbg": "faubourg", "res": "residence", "lot": "lotissement", "st": "saint", "ste": "sainte",
          "prom": "promenade", "cite": "cite", "hameau": "hameau", "sent": "sentier", "voie": "voie"}
    lex["tok_fr"] = fr
    return lex


def save_lexicon(lex, path=None):
    path = path or WORK / "lexicon.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(lex, f)


def load_lexicon(path=None):
    with open(path or WORK / "lexicon.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- bulk normalisation
def _init(lex):
    global _LEX
    _LEX = lex


def _norm_chunk(args):
    names, addrs, countries = args
    out = []
    base_tok = _LEX["tok"]
    fr_lex = dict(_LEX, tok=_LEX.get("tok_fr", {}), comp={}, comp_raw={})
    for n, a, c in zip(names, addrs, countries):
        lx = fr_lex if c == "France" else _LEX
        core, legal, glued = norm_name(n, lx)
        atok, anum, post = norm_addr(a, lx)
        out.append((core, legal, glued, atok, anum, post, bool(n and INDIC_RE.search(n))))
    cols = list(zip(*out)) if out else [()] * 7
    return pd.DataFrame({k: (np.array(v, dtype=bool) if k == "indic" else pd.array(list(v), dtype="string"))
                         for k, v in zip(["core", "legal", "glued", "atok", "anum", "post", "indic"], cols)})


def normalize_frame(df, lex, workers=8, chunk=50_000):
    """Add normalised columns to a raw frame (id, src, name, addr, country)."""
    jobs = [(df["name"].values[i:i + chunk], df["addr"].values[i:i + chunk], df["country"].values[i:i + chunk])
            for i in range(0, len(df), chunk)]
    with Pool(workers, initializer=_init, initargs=(lex,)) as p:
        parts = list(p.imap(_norm_chunk, jobs))
    out = pd.concat(parts, ignore_index=True)
    base = df[["id", "src", "country"]].reset_index(drop=True)
    return pd.concat([base, out], axis=1)
