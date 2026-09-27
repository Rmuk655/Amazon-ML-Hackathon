"""Analysis only (not part of the submission). Step 1 toward 0.98+: classify every true pair that never reached the model (missed by blocking or
removed by pruning) by the noise pattern that separates the two records. Multi-label, plus one
primary label by priority. Reads eval_gt_outcomes.parquet (matching.py evaluate)."""
import re
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

import config as C
from text_utils import INDIC_RE

pd.set_option("display.width", 250, "display.max_colwidth", 45)
D = C.PROCESSED_DIR
g = pd.read_parquet(D / "eval_gt_outcomes.parquet")
lost = g[g.outcome.isin(["missed_by_blocking", "removed_by_pruning"])].copy()
found = g[~g.outcome.isin(["missed_by_blocking", "removed_by_pruning"])].sample(min(50_000, len(g)), random_state=0)

vocab = {}
for line in open(C.NAME_VOCAB, encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 2 and p[1].isdigit():
        vocab[p[0]] = int(p[1])

need = set(lost.s1_id) | set(lost.cand_id) | set(found.s1_id) | set(found.cand_id)
cols = ["entity_id", "business_name", "business_name_rom", "business_name_c4b", "business_address_rom"]
parts = []
for n in (1, 2, 3):
    for b in pq.ParquetFile(C.processed_path("train", n)).iter_batches(columns=cols, batch_size=1_000_000):
        x = b.to_pandas()
        parts.append(x[x.entity_id.isin(need)])
T = pd.concat(parts).drop_duplicates("entity_id").set_index("entity_id").fillna("")

TITLES = {"mr", "mrs", "dr", "shri", "sri", "smt", "ms", "m", "sh", "prof", "er", "adv", "ca"}
DOMAIN = {"com", "c0m", "co", "in", "net", "org", "www", "coin", "ltd"}
ALIAS = re.compile(r"\b(a k a|aka|dba|d b a|alias|t a|trading as)\b")
DIGIT_MIX = re.compile(r"(?=\w*\d)(?=\w*[a-z])\w+")


def dedup(t):
    return re.sub(r"(.)\1+", r"\1", t)


def labels(s1_core, s1_rom, s1_addr, t_raw, t_rom, t_core, t_addr):
    L = []
    st, tt = s1_core.split(), t_rom.split()
    if not t_rom.strip():
        return ["target name empty"]
    if INDIC_RE.search(t_raw):
        L.append("target in Indic script")
    if tt and tt[0] in TITLES and (not st or st[0] != tt[0]):
        L.append("title prefix (mr/dr/shri..)")
    if ALIAS.search(t_rom):
        L.append("alias (a k a / dba)")
    if set(tt) & DOMAIN - set(st):
        L.append("domain / glued suffix (com, c0m..)")
    glued = "".join(st)
    if len(st) >= 2 and any(len(w) >= 8 and w not in vocab and sum(x in w for x in st if len(x) >= 3) >= 2 for w in tt):
        L.append("words glued together")
    if any(DIGIT_MIX.fullmatch(w) for w in tt):
        L.append("digits inside words (OCR)")
    if s1_core and s1_core == t_core:
        L.append("core names EQUAL (lost to K / caps)")
    tset = fuzz.token_set_ratio(s1_core, t_core) / 100 if s1_core and t_core else 0.0
    dd = fuzz.token_set_ratio(" ".join(dedup(w) for w in st), " ".join(dedup(w) for w in t_core.split())) / 100 \
        if s1_core and t_core else 0.0
    if 0.5 <= tset < 0.9 and dd >= 0.9:
        L.append("doubled letters / typos (fixable by spell-correct)")
    elif 0.5 <= tset < 0.9:
        L.append("partial name overlap")
    part = fuzz.partial_ratio(glued, t_rom.replace(" ", "")) / 100 if glued else 0.0
    if tset < 0.3 and part < 0.6:
        L.append("NAME REPLACED / unrelated")
    at, bt = set(s1_addr.split()), set(t_addr.split())
    aj = len(at & bt) / len(at | bt) if at and bt else 0.0
    if aj >= 0.5:
        L.append("address matches well (address-only key would find it)")
    elif not bt:
        L.append("target address empty")
    if "NAME REPLACED / unrelated" in L and aj < 0.3:
        L.append("UNRECOVERABLE? (name AND address differ)")
    return L or ["other"]


def classify(d):
    a, b = T.reindex(d.s1_id.values), T.reindex(d.cand_id.values)
    return [labels(*r) for r in zip(a.business_name_c4b.fillna(""), a.business_name_rom.fillna(""),
                                    a.business_address_rom.fillna(""), b.business_name.fillna(""),
                                    b.business_name_rom.fillna(""), b.business_name_c4b.fillna(""),
                                    b.business_address_rom.fillna(""))]


lost["labels"] = classify(lost)
found["labels"] = classify(found)
N_all = len(g)
print(f"true pairs {N_all:,} | lost before the model {len(lost):,} ({100 * len(lost) / N_all:.2f}%) "
      f"[missed by blocking {int((lost.outcome == 'missed_by_blocking').sum()):,}, "
      f"removed by pruning {int((lost.outcome == 'removed_by_pruning').sum()):,}]")
allL = sorted({l for L in lost.labels for l in L})
rows = []
for l in allL:
    n = int(sum(l in L for L in lost.labels))
    f = float(np.mean([l in L for L in found.labels]))
    rows.append((l, n, 100 * n / len(lost), 100 * n / N_all, 100 * f))
t = pd.DataFrame(rows, columns=["pattern", "lost pairs", "% of lost", "% of ALL true pairs", "% among FOUND pairs"])
print("\nmulti-label (a pair can have several):")
print(t.sort_values("lost pairs", ascending=False).round(2).to_string(index=False))

PRIORITY = ["target name empty", "core names EQUAL (lost to K / caps)", "alias (a k a / dba)",
            "words glued together", "domain / glued suffix (com, c0m..)", "title prefix (mr/dr/shri..)",
            "digits inside words (OCR)", "doubled letters / typos (fixable by spell-correct)", "target in Indic script",
            "partial name overlap", "UNRECOVERABLE? (name AND address differ)", "NAME REPLACED / unrelated", "other"]
lost["primary"] = [next((p for p in PRIORITY if p in L), L[0]) for L in lost.labels]
pc = lost.primary.value_counts()
print("\nprimary label (one per pair):")
for k, v in pc.items():
    print(f"  {k:55} {v:8,} {100 * v / len(lost):5.1f}% of lost  {100 * v / N_all:5.2f}% of all")
print("\nexamples per primary label:")
for k in pc.index:
    ex = lost[lost.primary == k].sample(min(5, int(pc[k])), random_state=0)
    a, b = T.reindex(ex.s1_id.values), T.reindex(ex.cand_id.values)
    print(f"-- {k}")
    for x, y, xa, ya in zip(a.business_name_rom, b.business_name_rom, a.business_address_rom, b.business_address_rom):
        print(f"   S1: {x[:40]:40} | T: {y[:45]:45} || {xa[:35]:35} | {ya[:35]}")
lost.to_parquet(D / "lost_pairs_classified.parquet", index=False)
