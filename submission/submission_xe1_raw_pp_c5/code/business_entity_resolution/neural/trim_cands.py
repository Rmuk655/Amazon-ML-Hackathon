"""Trim candidate_pairs to <= 5 per S1 (top by stack score), always keeping every predicted match.
  python trim_cands.py <test_scores.parquet> <matching_results.tsv> <out candidate_pairs.tsv>"""
import sys
sys.path.insert(0, "/home/ubuntu/xe_v2code_c8")
import numpy as np, pandas as pd
from v2 import common as C
sc, mr, out = sys.argv[1:4]
ts = pd.read_parquet(sc, columns=["rec", "s1", "pst"])
m = pd.read_csv(mr, sep="\t", dtype=str, keep_default_na=False)
m = m.assign(r=m.matched_entity_ids.str.split(",")).explode("r")
m = m[m.r.fillna("") != ""]
matched = set(zip(C.encode_ids(m.source1_entity_id.values), C.encode_ids(m.r.values)))
ts["m"] = [(s, r) in matched for s, r in zip(ts.s1.values, ts.rec.values)]
ts = ts.sort_values(["s1", "m", "pst"], ascending=[True, False, False])
rank = ts.groupby("s1").cumcount().values
nm = ts.groupby("s1").m.transform("sum").values
keep = ts[(rank < np.maximum(5, nm))]
print("matched pairs", len(matched), "| all kept:", int(keep.m.sum()) == len(matched))
print(f"candidates {len(ts):,} -> {len(keep):,}; per S1 {len(ts)/ts.s1.nunique():.2f} -> {len(keep)/ts.s1.nunique():.2f}")
raw = pd.read_parquet(C.WORK / "raw/test.parquet", columns=["id", "src"])
s1_all = raw.id.values[raw.src.values == 1]
lists = keep.assign(r=C.decode_ids(keep.rec.values)).groupby("s1")["r"].agg(",".join)
pd.DataFrame({"source1_entity_id": C.decode_ids(s1_all), "candidate_entity_ids": lists.reindex(s1_all).fillna("").values}).to_csv(out, sep="\t", index=False)
print("wrote", out)
