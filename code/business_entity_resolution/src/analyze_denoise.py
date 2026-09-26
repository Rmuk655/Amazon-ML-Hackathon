"""Analysis only (not part of the submission). Denoiser report on the LOCKED HOLDOUT: for holdout true pairs,
share whose target core name EQUALS the S1 core name, and share sharing at least one blocking key, before vs after
the noisy-channel token repair. Run on processed data that does NOT yet contain the repair."""
import sys
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from blocking import record_keys  # noqa: E402
from canonical import name_keys  # noqa: E402
from name_clean import TokenRepair  # noqa: E402

gt = pd.read_csv(C.TRAIN_GT, sep="\t", dtype=str, keep_default_na=False, quoting=3)
flat = pd.DataFrame({"s1_id": gt.iloc[:, 0].str.strip(), "cand_id": gt.iloc[:, 1].str.split(",")}).explode("cand_id")
flat["cand_id"] = flat["cand_id"].fillna("").str.strip()
flat = flat[(flat.cand_id != "") & C.is_holdout(flat.s1_id.values)].sample(100_000, random_state=0)
need = set(flat.s1_id) | set(flat.cand_id)
cols = ["entity_id", "business_name_rom", "business_name_c4b", "business_address_rom"]
parts = []
for n in (1, 2, 3):
    for b in pq.ParquetFile(C.processed_path("train", n)).iter_batches(columns=cols, batch_size=1_000_000):
        x = b.to_pandas()
        parts.append(x[x.entity_id.isin(need)])
T = pd.concat(parts).drop_duplicates("entity_id").set_index("entity_id").fillna("")
vocab = {}
for line in open(C.NAME_VOCAB, encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 2 and p[1].isdigit():
        vocab[p[0]] = int(p[1])
rep = TokenRepair(vocab)
a, b = T.reindex(flat.s1_id.values), T.reindex(flat.cand_id.values)
s_core, t_core = a.business_name_c4b.values, b.business_name_c4b.values
t_rom_new = [rep(x) for x in b.business_name_rom.values]
t_core_new = [name_keys(x)[1] for x in t_rom_new]
changed = np.mean([x != y for x, y in zip(b.business_name_rom.values, t_rom_new)])
eq_before = np.mean([x == y and x != "" for x, y in zip(s_core, t_core)])
eq_after = np.mean([x == y and x != "" for x, y in zip(s_core, t_core_new)])


def share_key(sn, sa, tn, ta):
    return bool({k for _, k in record_keys(sn, sa)} & {k for _, k in record_keys(tn, ta)})


kb = np.mean([share_key(x, y, z, w) for x, y, z, w in zip(s_core, a.business_address_rom.values, t_core, b.business_address_rom.values)])
ka = np.mean([share_key(x, y, z, w) for x, y, z, w in zip(s_core, a.business_address_rom.values, t_core_new, b.business_address_rom.values)])
print(f"LOCKED HOLDOUT true pairs sampled: {len(flat):,} | target names changed by token repair: {100 * changed:.2f}%")
print(f"core name EQUAL to S1:         before {100 * eq_before:.2f}%  after {100 * eq_after:.2f}%  ({100 * (eq_after - eq_before):+.2f} pts)")
print(f"share >= 1 blocking key w/ S1: before {100 * kb:.2f}%  after {100 * ka:.2f}%  ({100 * (ka - kb):+.2f} pts)")
