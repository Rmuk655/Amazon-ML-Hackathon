"""Canonical name-key derivation (Stage 2, c4).

    name_keys(s) -> (c4a, c4b, legal)

`s` is expected to be an already-normalized/romanized business name (i.e. the value in
business_name_rom from preprocess.py) — lowercase, punctuation collapsed to spaces.

    c4a   canonical root form: abbreviation expansion + '&'/and folding, suffixes KEPT.
    c4b   c4a with legal suffixes and generic stopwords stripped (the "core" name used
          for matching and for the ambiguity/chain-risk check in eda.py 1.7).
    legal space-joined legal-suffix tokens found in the name (empty string if none) — a
          useful flag/feature on its own (e.g. "has a legal suffix" or "which one").

Tune LEGAL_SUFFIXES / STOPWORDS / ABBR_EXPAND from eda/eda_token_df.csv (Stage 1, check 1.10):
the most frequent name tokens are exactly the suffix/stopword candidates.
"""
import csv
import re

# Legal-entity suffixes across common jurisdictions (India, US, UK, EU). Multi-word entries
# listed as tuples of the tokens they occupy so they're matched as a unit from the tail.
LEGAL_SUFFIXES = {
    "pvt", "private", "ltd", "limited", "llp", "llc", "inc", "incorporated", "corp",
    "corporation", "co", "company", "plc", "pty", "gmbh", "ag", "kg", "ug", "sa", "sarl",
    "srl", "sl", "slu", "nv", "bv", "oy", "ab", "spa", "kft", "zrt", "oyj", "asa", "aps",
    "sas", "sasu", "eurl", "sce", "scop", "sro", "kk", "gk", "yugen", "opc",
    "pllc", "lp", "llp", "pc", "ei", "snc", "sci", "scp", "scm", "sca", "gie", "selarl", "ets",
}

# Generic stopwords / conjunctions folded during canonicalization, not part of the brand.
STOPWORDS = {"the", "of", "and", "&", "ms", "dba"}   # ms: Indian "M/s." prefix; dba: doing business as

# Common abbreviation -> expanded form, applied token-by-token before suffix detection so
# "pvt ltd" and "private limited" canonicalize identically.
ABBR_EXPAND = {
    "pvt": "private",
    "ltd": "limited",
    "co": "company",
    "corp": "corporation",
    "intl": "international",
    "mfg": "manufacturing",
    "bros": "brothers",
    "assoc": "associates",
    "&": "and",
}

def _load_learned():
    """Union in legal forms / stopwords learned from the data by learn_suffixes.py (if run)."""
    try:
        import config as C
        with open(C.LEARNED_TOKENS, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                (LEGAL_SUFFIXES if r["kind"] == "legal" else STOPWORDS).add(r["token"])
    except (ImportError, OSError, KeyError):
        pass


_load_learned()

_WS_RE = re.compile(r"\s+")


def join_initials(tokens):
    """Re-join dotted acronyms split by punctuation removal: 'p c' -> 'pc', 'l l c' -> 'llc'."""
    out, run = [], []
    for t in tokens + [""]:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        if t:
            out.append(t)
    return out


def _tokens(s):
    return join_initials([t for t in _WS_RE.split(s.strip()) if t])


def _expand(tok):
    return ABBR_EXPAND.get(tok, tok)


def _strip_trailing_suffixes(tokens):
    """Strip a run of legal-suffix tokens from the end (a name can have more than one,
    e.g. 'x private limited co'). Returns (remaining_tokens, removed_tokens_in_order)."""
    end = len(tokens)
    while end > 0 and tokens[end - 1] in LEGAL_SUFFIXES:
        end -= 1
    return tokens[:end], tokens[end:]


def _drop_id_noise(tokens):
    """Drop record-id noise: long pure-digit tokens ('13896') and an 'id'/'no' tag before one."""
    out = []
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if (t.isdigit() and len(t) >= 4) or (t in ("id", "no") and nxt.isdigit() and len(nxt) >= 4):
            continue
        out.append(t)
    return out or tokens


def name_keys(s):
    if not isinstance(s, str) or not s.strip():
        return "", "", ""

    expanded = _drop_id_noise([_expand(t) for t in _tokens(s)])
    c4a = " ".join(expanded)

    core, removed = _strip_trailing_suffixes(expanded)
    core = [t for t in core if t not in STOPWORDS]
    # legal suffixes can also appear mid-name for chains ("x limited east"); drop those too
    # for c4b, but only after the trailing pass above already captured the common case.
    core_no_legal = [t for t in core if t not in LEGAL_SUFFIXES]
    if core_no_legal:
        core = core_no_legal
    c4b = " ".join(core) if core else c4a  # never return an empty core; fall back to c4a

    legal = " ".join(removed)
    return c4a, c4b, legal