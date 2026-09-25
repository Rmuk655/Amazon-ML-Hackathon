"""Unicode normalization, script detection and accent folding.

Nothing here is destructive: callers keep the raw column and add derived columns.
"""
import html
import re
import unicodedata
from functools import lru_cache
from itertools import groupby

from unidecode import unidecode

from config import SCRIPT_DEFAULT_LANG

# Unicode name prefix -> ISO 15924 script code (only scripts IndicXlit can read)
INDIC_SCRIPTS = {
    "DEVANAGARI": "Deva", "BENGALI": "Beng", "GURMUKHI": "Guru", "GUJARATI": "Gujr",
    "ORIYA": "Orya", "TAMIL": "Taml", "TELUGU": "Telu", "KANNADA": "Knda",
    "MALAYALAM": "Mlym", "SINHALA": "Sinh", "ARABIC": "Arab",
}
# Any char in Arabic / Indic blocks (cheap pre-filter before the slower per-char logic)
INDIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u0900-\u0DFF]")

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_ASCII_NON_ALNUM = re.compile(r"[^0-9a-z]+")
ADDRESS_NOISE_TOKENS = {"null", "nan"}


@lru_cache(maxsize=None)
def script_of(ch: str) -> str:
    """'Latn', an Indic script code, or 'Othr' (Cyrillic, CJK, ...)."""
    if ch.isascii():
        return "Latn"
    first = unicodedata.name(ch, "").split(" ")[0]
    if first in INDIC_SCRIPTS:
        return INDIC_SCRIPTS[first]
    if first == "LATIN" or (first == "COMBINING" and unicodedata.category(ch) == "Mn"):
        return "Latn"
    return "Othr"


# Mojibake: UTF-8 bytes decoded as Latin-1/cp1252 ('â\x80\x99' for "'", 'Ã©' for 'é'), sometimes
# uppercased afterwards ('Â\x80\x93' for '–'). A lead char + 1-2 continuation chars is re-encoded
# to bytes and decoded as UTF-8; unrecoverable runs become a space (never a stray letter).
_CP1252_CONT = "€‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ"
_MOJIBAKE = re.compile(f"[ÂâÃã][\u0080-\u00bf{_CP1252_CONT}]{{1,2}}")
_LEADS = {"Â": (0xE2, 0xC2), "â": (0xE2, 0xC2), "Ã": (0xC3, 0xE3), "ã": (0xC3, 0xE3)}


def _byte(ch):
    o = ord(ch)
    return o if o < 256 else ch.encode("cp1252")[0]


def _unmojibake(m):
    run = m.group(0)
    cont = bytes(_byte(c) for c in run[1:])
    for lead in _LEADS[run[0]]:
        try:
            return (bytes([lead]) + cont).decode("utf-8")
        except UnicodeDecodeError:
            continue
    return " "


def repair_text(t: str) -> str:
    """HTML entities and mojibake -> the intended characters (no-op on clean text)."""
    if "&" in t and ";" in t:
        t = html.unescape(t)
    if not t.isascii() and _MOJIBAKE.search(t):
        t = _MOJIBAKE.sub(_unmojibake, t)
    return t


def normalize(text) -> str:
    """Repair (HTML entities, mojibake), NFKC, casefold, punctuation/symbols -> space,
    native digits -> ASCII, invisible format characters (Cf: soft hyphen, bidi marks) deleted.

    Keeps letters and combining marks (needed for Indic scripts) and all digits.
    """
    if text is None or text != text:  # None / NaN
        return ""
    t = unicodedata.normalize("NFKC", repair_text(str(text))).translate(_ZERO_WIDTH)
    if t.isascii():  # fast path: most US rows
        return _ASCII_NON_ALNUM.sub(" ", t.lower().replace("&", " and ")).strip()
    out = []
    for ch in t:
        cat = unicodedata.category(ch)
        if ch == "&":
            out.append(" and ")
        elif cat == "Nd":
            out.append(str(unicodedata.decimal(ch)))
        elif cat == "Cf":                 # invisible format chars: delete, don't split the word
            continue
        elif cat[0] in "PSZC":
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).casefold().split())


# Street-type abbreviations -> one form. Applied identically to every source, so an imperfect
# expansion ('st' -> 'street' even for 'saint') is harmless; it only has to be consistent.
ADDRESS_ABBR = {
    "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard",
    "blvd": "boulevard", "all": "allée", "imp": "impasse", "pl": "place", "rte": "route",
    "ch": "chemin", "chem": "chemin", "fg": "faubourg", "sq": "square", "st": "street",
    "rd": "road", "dr": "drive", "ln": "lane", "ct": "court", "hwy": "highway", "pkwy": "parkway",
    "ste": "suite", "nr": "near", "opp": "opposite",
}


def normalize_address(text) -> str:
    """normalize() + drop literal 'null'/'nan' tokens + unify street-type abbreviations."""
    return " ".join(ADDRESS_ABBR.get(t, t) for t in normalize(text).split() if t not in ADDRESS_NOISE_TOKENS)


def fold_latin(seg: str) -> str:
    """Accent folding for Latin text: 'allée' -> 'allee', 'ø' -> 'o'."""
    return seg if seg.isascii() else unidecode(seg).lower()


def split_scripts(token: str):
    """[('Latn','tech'), ('Gujr','પ્રાઇવેટ')] for a token mixing scripts."""
    return [(k, "".join(g)) for k, g in groupby(token, key=script_of)]


def scripts_present(s: str) -> str:
    """'Latn', 'Deva', 'Deva+Latn', ... ('' if no letters). Digits ignored."""
    if s.isascii():
        return "Latn" if any(c.isalpha() for c in s) else ""
    return "+".join(sorted({
        script_of(c) for c in s if c.isalpha() or unicodedata.category(c)[0] == "M"
    }))


def indic_part(norm: str) -> str:
    """Tokens of a normalized string that contain Indic-script characters."""
    return " ".join(t for t in norm.split() if INDIC_RE.search(t))


def dominant_indic_script(text: str):
    counts = {}
    for ch in text:
        s = script_of(ch)
        if s in SCRIPT_DEFAULT_LANG:
            counts[s] = counts.get(s, 0) + 1
    return max(counts, key=counts.get) if counts else None


def tokenize(norm: str):
    """Tokens of an already-normalized string."""
    return norm.split()