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


# Placeholders injected into otherwise valid fields ('NULL', '<NULL>', 'N/A' ~3% of S2/S3 addresses).
_ADDR_PLACEHOLDER = re.compile(r"<\s*null\s*>|#?\bn\s*/\s*a\b|\bnot\s+available\b|\bnot\s+applicable\b", re.I)
_WHOLE_PLACEHOLDER = re.compile(r"^\s*(?:null|nan|none|n\s*/?\s*a|nil|#n/a|unknown|undefined|-+|\.+|\?+)\s*$", re.I)
_OCR_NUM = re.compile(r"(?=[0-9oil]*[0-9])(?=[0-9oil]*[oil])[0-9oil]{2,}")
_OCR_NUM_MAP = str.maketrans("oil", "011")


def _fix_number(tok):
    """Letter-for-digit OCR in numbers: '3o5' -> '305', 'i30' -> '130' (token must be all 0-9/o/i/l)."""
    return tok.translate(_OCR_NUM_MAP) if _OCR_NUM.fullmatch(tok) else tok


def normalize_name(text) -> str:
    """normalize() for names; a whole-field placeholder ('NA', 'NULL', 'N/A') means no name."""
    if text is None or text != text or _WHOLE_PLACEHOLDER.match(str(text)):
        return ""
    return normalize(text)


def normalize_address(text) -> str:
    """normalize() + placeholder removal ('NULL', '<NULL>', 'N/A', 'nan') + street-type abbreviations
    unified + letter-for-digit OCR repair inside numbers."""
    if text is None or text != text or _WHOLE_PLACEHOLDER.match(str(text)):
        return ""
    t = _ADDR_PLACEHOLDER.sub(" ", str(text))
    return " ".join(_fix_number(ADDRESS_ABBR.get(w, w)) for w in normalize(t).split() if w not in ADDRESS_NOISE_TOKENS)


_DIGIT_RUN = re.compile(r"\d\d")


class OcrFixer:
    """Digit-for-letter OCR noise in names ('5ervices', 'c0m', 'Br0thers', 'Denta1', 'lnfra').
    A token is rewritten only if it is NOT a known word and a variant IS: the vocabulary is the
    clean Source-1 name tokens (train + test, no labels), so '3eme', '4x4', '3m' are left alone."""
    OPTIONS = {"0": "o", "1": "li", "2": "z", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b",
               "9": "g", "l": "i", "i": "l"}

    def __init__(self, vocab):
        self.vocab = vocab                      # {token: count}
        self.cache = {}

    def _variants(self, tok):
        out = [""]
        for ch in tok:
            opts = ch + self.OPTIONS.get(ch, "")
            out = [o + c for o in out for c in opts]
            if len(out) > 256:
                return []
        return out[1:]                           # first is the token itself

    def fix_token(self, tok):
        if tok in self.vocab or len(tok) < 3 or tok.isdigit() or _DIGIT_RUN.search(tok):
            return tok                           # known word, too short, or a model number ('i10', 'a320')
        if not (any(c.isdigit() for c in tok) or "l" in tok or "i" in tok):
            return tok
        if tok not in self.cache:
            cand = [v for v in self._variants(tok) if v in self.vocab]
            self.cache[tok] = max(cand, key=self.vocab.get) if cand else tok
        return self.cache[tok]

    def __call__(self, s):
        return " ".join(self.fix_token(t) for t in s.split()) if s else s


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