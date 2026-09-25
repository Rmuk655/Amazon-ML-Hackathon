"""Analysis only (not part of the submission). Failure analysis of a held-out evaluation (reads eval_pairs.parquet + eval_gt_outcomes.parquet)."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

import config as C
from text_utils import INDIC_RE

pd.set_option("display.width", 250, "display.max_colwidth", 48, "display.max_columns", 30)
D = C.PROCESSED_DIR
pairs = pd.read_parquet(D / "eval_pairs.parquet")
gt = pd.read_parquet(D / "eval_gt_outcomes.parquet")
KT = C.KTYPES

need = set(gt.s1_id) | set(gt.cand_id) | set(pairs.s1_id) | set(pairs.cand_id)
cols = ["entity_id", "business_name", "business_address", "business_name_c4b", "business_name_rom"]
txt = []
for n in (1, 2, 3):
    for b in pq.ParquetFile(C.processed_path("train", n)).iter_batches(columns=cols, batch_size=1_000_000):
        x = b.to_pandas()
        txt.append(x[x.entity_id.isin(need)])
txt = pd.concat(txt).drop_duplicates("entity_id").set_index("entity_id").fillna("")


def attach(d):
    a, b = txt.reindex(d.s1_id.values), txt.reindex(d.cand_id.values)
    d = d.copy()
    d["s1_name"], d["t_name"] = a.business_name.values, b.business_name.values
    d["s1_addr"], d["t_addr"] = a.business_address.values, b.business_address.values
    d["s1_core"], d["t_core"] = a.business_name_c4b.values, b.business_name_c4b.values
    d["t_indic"] = b.business_name.map(lambda s: bool(INDIC_RE.search(s)) if isinstance(s, str) else False).values
    d["core_tset"] = [fuzz.token_set_ratio(x, y) / 100 for x, y in zip(d.s1_core, d.t_core)]
    return d


def pct(n, tot):
    return f"{n:>9,} ({100 * n / max(tot, 1):5.1f}%)"


def keys(mask):
    return "+".join(k for i, k in enumerate(KT) if (int(mask) >> i) & 1) or "-"


tp = pairs[pairs.keep & pairs.is_true]
fp = attach(pairs[pairs.keep & ~pairs.is_true])
rej = attach(pairs[~pairs.keep & pairs.is_true])
miss = attach(gt[gt.outcome == "missed_by_blocking"])
tpa = attach(tp.sample(min(len(tp), 50_000), random_state=0))
N = len(gt)
print(f"India held-out: {N:,} true pairs | TP {len(tp):,} | FP {len(fp):,} | rejected by model {len(rej):,} | "
      f"removed by pruning {(gt.outcome == 'removed_by_pruning').sum():,} | missed by blocking {len(miss):,}")

# ---------------------------------------------------------------- 1. missed by blocking
print("\n=== 1. MISSED BY BLOCKING (never became candidates) ===")
m = miss
for lab, cond in [("target name in Indic script", m.t_indic),
                  ("core names share NO word", m.core_tset == 0),
                  ("core-name token-set < 0.5", m.core_tset < 0.5),
                  ("core-name token-set >= 0.9 (names match; lost to top-K / key caps)", m.core_tset >= 0.9),
                  ("target name empty", m.t_name.str.strip() == "")]:
    print(f"  {lab:70} {pct(int(cond.sum()), len(m))}")
print("  same slices among TRUE MATCHES FOUND (TP sample), for comparison:")
for lab, cond in [("target name in Indic script", tpa.t_indic), ("core names share NO word", tpa.core_tset == 0)]:
    print(f"    {lab:68} {pct(int(cond.sum()), len(tpa))}")
print(m.sample(min(12, len(m)), random_state=1)[["s1_name", "t_name", "t_core", "core_tset"]].to_string(index=False))

# ---------------------------------------------------------------- 2/3. rejected + FP: feature profile
F = ["p", "blk_prune_p", "n_best", "n_tok_jacc_rom", "n_partial_rom", "a_num_jacc", "a_tok_jacc_rom",
     "a_house_conflict", "a_pin_conflict", "amb_s1_core_freq", "amb_t_core_freq"]
F = [f for f in F if f in pairs.columns]
prof = pd.DataFrame({"TP (correct)": tp[F].mean(), "rejected true": rej[F].mean(), "FP (wrong)": fp[F].mean(),
                     "TN (rejected false)": pairs[~pairs.keep & ~pairs.is_true][F].mean()}).round(3)
print("\n=== 2. FEATURE PROFILE (mean values) ===")
print(prof.to_string())

print("\n=== 3. REJECTED BY MODEL (true pair, p below threshold) ===")
r = rej
for lab, cond in [("target name in Indic script", r.t_indic),
                  ("name similarity low (n_best < 0.8)", r.n_best < 0.8),
                  ("address differs (a_tok_jacc_rom < 0.3)", r.a_tok_jacc_rom < 0.3),
                  ("house number conflict", r.a_house_conflict > 0),
                  ("chain name (core shared by >= 3 S1)", r.amb_s1_core_freq >= 3),
                  ("p in [0.5, threshold) - near miss", (r.p >= 0.5))]:
    print(f"  {lab:55} {pct(int(cond.sum()), len(r))}")
print(r.sort_values("p", ascending=False).head(10)[["p", "s1_name", "t_name", "s1_addr", "t_addr"]].to_string(index=False))

print("\n=== 4. FALSE POSITIVES (predicted match, not in ground truth) ===")
f = fp
owner = gt.set_index("cand_id")["s1_id"]
f["t_true_owner"] = owner.reindex(f.cand_id).values
full = pd.read_csv(C.TRAIN_GT, sep="\t", dtype=str, keep_default_na=False, quoting=3)
full = pd.DataFrame({"s1": full.iloc[:, 0].str.strip(), "t": full.iloc[:, 1].str.split(",")}).explode("t")
full["t"] = full["t"].fillna("").str.strip()
full_owner = full[full.t != ""].drop_duplicates("t").set_index("t")["s1"]
pop = set(pd.read_parquet(D / "blocked_s1_train.parquet").iloc[:, 0].astype(str))
f["owner_any"] = full_owner.reindex(f.cand_id).values
art = f.owner_any.notna() & ~f.owner_any.isin(pop)
print(f"  FPs whose true owner is an S1 OUTSIDE the evaluated slice (slice artifact): {pct(int(art.sum()), len(f))}")
print(f"  -> analysis below uses the remaining {int((~art).sum()):,} FPs")
f = f[~art]
fp = f
hi_name = f.n_best >= 0.9
addr_diff = (f.a_tok_jacc_rom < 0.3) | (f.a_house_conflict > 0) | (f.a_pin_conflict > 0)
for lab, cond in [("name similarity high (n_best >= 0.9)", hi_name),
                  ("address differs (low overlap / house or pin conflict)", addr_diff),
                  ("BOTH: same name, different address  <- your hypothesis", hi_name & addr_diff),
                  ("same name, SAME address (likely duplicate S1 / GT gap)", hi_name & ~addr_diff),
                  ("target belongs to ANOTHER held-out S1 (chain sibling took it)", f.t_true_owner.notna()),
                  ("chain name (core shared by >= 3 S1)", f.amb_s1_core_freq >= 3),
                  ("target name in Indic script", f.t_indic)]:
    print(f"  {lab:65} {pct(int(cond.sum()), len(f))}")
f["keys"] = f.blk_mask.map(keys) if "blk_mask" in f else ""
tpv = tpa[(tpa.n_best >= 0.9) & ((tpa.a_tok_jacc_rom < 0.3) | (tpa.a_house_conflict > 0) | (tpa.a_pin_conflict > 0))]
print(f"  CORRECT matches with the same pattern (a veto would kill them): {len(tpv) * len(tp) / len(tpa):,.0f} "
      f"({100 * len(tpv) / len(tpa):.1f}% of TP)  vs FPs with it: {int((hi_name & addr_diff).sum()):,}")
print("  mean p of FPs:", round(f.p.mean(), 3), "| share with p >= 0.95:", round((f.p >= 0.95).mean(), 3))
print(f.sample(min(12, len(f)), random_state=2)[["p", "s1_name", "t_name", "s1_addr", "t_addr"]].to_string(index=False))

print("\n=== 5. WHICH BLOCKING KEYS FOUND THE PAIR (share of pairs whose mask includes the key) ===")
rows = {}
for lab, d in [("TP", tp), ("FP", fp), ("rejected true", rej)]:
    mk = d.blk_mask.fillna(0).astype(int).values
    rows[lab] = {k: round(100 * ((mk >> i) & 1).mean(), 1) for i, k in enumerate(KT)}
print(pd.DataFrame(rows).to_string())
