"""Learn legal-form tokens and stopwords from the data (no labels, no hard-coded language).

Legal forms sit at a name boundary and are short: 'sci', 'ei', 'ets' (France), 'pllc', 'pc'
(US). Per country, a token is learned as a legal form when it is the first or last token of
>= MIN_BOUNDARY_SHARE of that country's names and has <= MAX_LEN characters; descriptive
words that are equally common ('group', 'center', 'trust') are longer and so are kept.
A short token found anywhere in >= MIN_STOP_SHARE of names is learned as a stopword
('de', 'la', 'des').

Reads business_name + country from all six raw source files (train and test), so legal
forms that only appear in test (e.g. a country absent from train) are covered.
canonical.py loads the output file if it exists; re-run preprocess.py afterwards.

    python learn_suffixes.py            # -> dataset/processed/learned_tokens.tsv
"""
import collections

import pandas as pd

import config as C
from canonical import LEGAL_SUFFIXES, STOPWORDS, join_initials
from text_utils import fold_latin, normalize

SAMPLE_PER_FILE_COUNTRY = 150_000
MIN_BOUNDARY_SHARE = 0.015
MIN_STOP_SHARE = 0.03
MAX_LEN = 4
MAX_STOP_LEN = 3


def tokens(s):
    return join_initials(fold_latin(normalize(s)).split())


def main():
    first, last, anywhere = (collections.defaultdict(collections.Counter) for _ in range(3))
    n = collections.Counter()
    for split in ("train", "test"):
        for k in (1, 2, 3):
            path = C.source_path(split, k)
            d = pd.read_csv(path, sep="\t", usecols=["business_name", "country"], dtype=str,
                            keep_default_na=False, quoting=3)
            for ctry, g in d.groupby("country"):
                names = g["business_name"]
                names = names.sample(min(len(names), SAMPLE_PER_FILE_COUNTRY), random_state=0)
                for s in names:
                    t = tokens(s)
                    if len(t) < 2:
                        continue
                    n[ctry] += 1
                    first[ctry][t[0]] += 1
                    last[ctry][t[-1]] += 1
                    anywhere[ctry].update(set(t))
            print(f"[{path.name}] done")

    rows = []
    for ctry in n:
        for pos, cnt in (("first", first[ctry]), ("last", last[ctry])):
            for w, c in cnt.items():
                share = c / n[ctry]
                if share >= MIN_BOUNDARY_SHARE and len(w) <= MAX_LEN and w.isalpha() and w not in LEGAL_SUFFIXES:
                    rows.append(("legal", w, ctry, pos, round(share, 4)))
        for w, c in anywhere[ctry].items():
            share = c / n[ctry]
            if share >= MIN_STOP_SHARE and len(w) <= MAX_STOP_LEN and w.isalpha() \
                    and w not in STOPWORDS and w not in LEGAL_SUFFIXES:
                rows.append(("stop", w, ctry, "any", round(share, 4)))
    out = pd.DataFrame(rows, columns=["kind", "token", "country", "position", "share"])
    out = out.sort_values(["kind", "share"], ascending=[True, False]).drop_duplicates(["kind", "token"])
    C.LEARNED_TOKENS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.LEARNED_TOKENS, sep="\t", index=False)
    print(out.to_string(index=False))
    print(f"-> {C.LEARNED_TOKENS}  (now re-run preprocess.py --overwrite so c4b uses them)")


if __name__ == "__main__":
    main()
