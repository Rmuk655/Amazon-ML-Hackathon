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
