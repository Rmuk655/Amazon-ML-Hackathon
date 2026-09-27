"""Undo the rule-based name noise found in true pairs blocking missed (analyze_missed.py):
aliases ('onyxyumasyn doing business as baba properties'), website forms ('alphabuilders com',
'www kaiser com'), title prefixes ('smt global care', 'mr indo finance') and glued words
('agroshreedevelopers'). Applied identically to every source on the romanised name, before the
canonical keys are built. Vocabulary = clean Source-1 name tokens (no labels)."""
import math
import re

ALIAS = re.compile(r"\b(?:a k a|aka|also known as|d b a|dba|doing business as|trading as)\b")
TITLES = {"mr", "mrs", "ms", "dr", "shri", "sri", "smt", "prof"}
DOMAIN_TAIL = {"com", "c0m", "net", "org"}


class NameCleaner:
    def __init__(self, vocab, min_count=20):
        self.vocab = vocab
        self.seg = {w: c for w, c in vocab.items() if c >= min_count and len(w) >= 2 and w.isalpha()}
        self.log_n = math.log(max(sum(self.seg.values()), 1))
        self.cache = {}

    def segment(self, tok):
        """'agroshreedevelopers' -> ['agro', 'shree', 'developers'] (max unigram likelihood); unchanged
        unless every piece is a known word."""
        if tok in self.cache:
            return self.cache[tok]
        n = len(tok)
        best = [(-1e18, None)] * (n + 1)
        best[0] = (0.0, None)
        for i in range(1, n + 1):
            for j in range(max(0, i - 20), i - 1):
                w = tok[j:i]
                c = self.seg.get(w)
                if c and best[j][0] > -1e17:
                    s = best[j][0] + math.log(c) - self.log_n
                    if s > best[i][0]:
                        best[i] = (s, j)
        out = [tok]
        if best[n][1] is not None:
            parts, i = [], n
            while i > 0:
                j = best[i][1]
                parts.append(tok[j:i])
                i = j
            if len(parts) >= 2:
                out = parts[::-1]
        self.cache[tok] = out
        return out

    def __call__(self, s):
        if not s:
            return s
        parts = [p.strip() for p in ALIAS.split(s) if p.strip()]
        if len(parts) > 1:
            s = parts[-1]                                   # the real name follows the alias marker
        t = [w for w in s.split() if w != "www"]
        while len(t) > 1 and t[-1] in DOMAIN_TAIL:
            t.pop()
        if len(t) > 1 and t[0] in TITLES:
            t = t[1:]
        if len(t) > 2 and t[0] == "m" and t[1] == "s":        # 'M/s.' prefix
            t = t[2:]
        out = []
        for w in t:
            out += self.segment(w) if len(w) >= 8 and w.isalpha() and w not in self.vocab else [w]
        return " ".join(out) or s


# ---------------------------------------------------------------- noisy-channel token repair
# Noise model (edit costs) from the aligned train true pairs: doubled / undoubled letters, OCR digit-letter
# swaps (o-0 s-5 b-8 l-1 g-6), adjacent swaps are cheap; other edits cost 1. Clean model: Source-1 token
# frequencies (no labels). A token that is not a Source-1 word is replaced by argmin over vocabulary words w of
#   cost(token | w) - LAMBDA * log(count(w))
# when w is clearly the best explanation; otherwise it is left as is.
OCR_PAIRS = {("o", "0"), ("s", "5"), ("b", "8"), ("l", "1"), ("i", "1"), ("g", "6"), ("e", "3"), ("a", "4"), ("t", "7"), ("z", "2")}
OCR_PAIRS |= {(b, a) for a, b in OCR_PAIRS}


def _undouble(s):
    return re.sub(r"(.)\1+", r"\1", s)


def noise_cost(noisy, clean):
    """Cheap edit cost of producing `noisy` from `clean` under the inferred noise operations."""
    if noisy == clean:
        return 0.0
    if _undouble(noisy) == _undouble(clean):
        return 0.3                                           # doubled / undoubled letters
    if len(noisy) == len(clean):
        diff = [(a, b) for a, b in zip(noisy, clean) if a != b]
        if all(p in OCR_PAIRS for p in diff) and len(diff) <= 2:
            return 0.3 * len(diff)                           # OCR digit-letter swaps
        if len(diff) == 2 and sorted(noisy) == sorted(clean):
            i = next(k for k in range(len(noisy)) if noisy[k] != clean[k])
            if noisy[i:i + 2] == clean[i:i + 2][::-1]:
                return 0.5                                   # adjacent swap
    from rapidfuzz.distance import Levenshtein
    return float(Levenshtein.distance(noisy, clean))


class TokenRepair:
    LAMBDA = 0.15

    def __init__(self, vocab, min_count=5):
        from collections import defaultdict
        self.vocab = vocab
        self.words = {w: c for w, c in vocab.items() if c >= min_count and len(w) >= 3 and w.isalpha()}
        self.buckets = defaultdict(list)                     # (first letter, length) -> words
        for w in self.words:
            self.buckets[(w[0], len(w))].append(w)
        self.cache = {}

    def repair_token(self, t):
        if t in self.cache:
            return self.cache[t]
        out = t
        if len(t) >= 4 and t not in self.vocab and not t.isdigit():
            from rapidfuzz import process
            from rapidfuzz.distance import Levenshtein
            tl = _undouble(t) if t[0].isalpha() else t
            firsts = {t[0]} | {b for a, b in OCR_PAIRS if a == t[0]}
            pool = [w for f in firsts for L in range(len(t) - 2, len(t) + 3) for w in self.buckets.get((f, L), ())]
            if pool:
                maxd = 1 if len(t) < 8 else 2
                cands = process.extract(t, pool, scorer=Levenshtein.distance, score_cutoff=maxd + 1, limit=8)
                cands += [(w, 0, 0) for w in pool if _undouble(w) == tl]
                scored = sorted({(noise_cost(t, w) - self.LAMBDA * math.log(self.words[w]), w) for w, _, _ in cands})
                if scored:
                    best_s, best_w = scored[0]
                    ok_cost = noise_cost(t, best_w) <= (1.0 if len(t) < 8 else 2.0)
                    margin = scored[1][0] - best_s if len(scored) > 1 else 9.0
                    if ok_cost and margin >= 0.3:
                        out = best_w
        self.cache[t] = out
        return out

    def __call__(self, s):
        return " ".join(self.repair_token(w) for w in s.split()) if s else s
