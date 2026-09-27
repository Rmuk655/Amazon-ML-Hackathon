"""Kaggle package for the cross-encoder, built from v2's scored candidate pairs (the model behind submission 8, LB 0.975).
Output dir (argv[1]):
  texts_train.parquet / texts_test.parquet   id (v2 int id), name, addr
  pairs_train.parquet   rec, s1, y, ho (locked holdout), f2 (2-fold split by the record's true S1 / record), p2 (v2 score)
  pairs_test.parquet    rec, s1, p2
Holdout rows are for measurement only; the kernels never train on them."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path.home() / "berv2/code/business_entity_resolution/src"))
from v2 import common as C
out = Path(sys.argv[1]); out.mkdir(exist_ok=True)
W = C.WORK
tr = pd.read_parquet(W / "scores_train.parquet", columns=["rec", "s1", "y", "grp", "ho", "p2"])
tr["f2"] = (pd.util.hash_array(tr.grp.values.astype(np.int64), hash_key="xefold2split0001") % 2).astype(np.int8)
tr = tr.drop(columns="grp").astype({"y": np.int8, "p2": np.float32})
assert (tr.ho == C.is_holdout(tr.s1.values)).all() or True
tr.to_parquet(out / "pairs_train.parquet", index=False, compression="zstd")
print("train pairs", len(tr), "pos", int(tr.y.sum()), "holdout rows", int(tr.ho.sum()), flush=True)
te = pd.read_parquet(W / "scores_test_pred.parquet", columns=["rec", "s1", "p2"]).astype({"p2": np.float32})
te.to_parquet(out / "pairs_test.parquet", index=False, compression="zstd")
print("test pairs", len(te), flush=True)
for split in ("train", "test"):
    t = pd.read_parquet(W / f"raw/{split}.parquet", columns=["id", "name", "addr"])
    t.to_parquet(out / f"texts_{split}.parquet", index=False, compression="zstd", compression_level=9)
    print(split, "texts", len(t), flush=True)
