"""Does romanization make TRUE matched pairs more similar? (name and address token overlap)

Run after preprocess.py on the train split (full files so ids line up).
"""
import argparse
import csv
import re

import pandas as pd

import config as C

NAME, ADDR = "business_name", "business_address"
COLS = ["entity_id", f"{NAME}_norm", f"{NAME}_rom", f"{NAME}_c4b", f"{NAME}_script", f"{ADDR}_norm", f"{ADDR}_rom"]


def parse_matched_ids(x: str):
    # Challenge format is comma-separated; also tolerate ; | and whitespace while inspecting.
    return [t for t in re.split(r"[,;|\s]+", x) if t]


def jac(a, b):
    A, B = set(a.split()), set(b.split())
    return len(A & B) / len(A | B) if A | B else 0.0


def report(label, sub, field, kinds=("norm", "rom")):
    if not len(sub):
        return
    res = {k: [jac(x, y) for x, y in zip(sub[f"{field}_{k}_a"], sub[f"{field}_{k}_b"])] for k in kinds}
    a = "  ".join(f"{k}={pd.Series(v).mean():.3f}" for k, v in res.items())
    b = "  ".join(f"{k}={(pd.Series(v) >= .5).mean():.3f}" for k, v in res.items())
    print(f"  {field:17} {label:22} n={len(sub):>7,}  mean J {a} | J>=0.5 {b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="S1 entities to sample")
    n = ap.parse_args().n

    gt = pd.read_csv(C.TRAIN_GT, sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE)
    print("sample matched_entity_ids:", gt["matched_entity_ids"].dropna().head(3).tolist())
    gt = gt.dropna(subset=["matched_entity_ids"]).sample(n, random_state=0)
    gt["m"] = gt["matched_entity_ids"].apply(parse_matched_ids)
    pairs = gt.explode("m").rename(columns={"source1_entity_id": "s1"}).dropna(subset=["m"])[["s1", "m"]]

    def load(src, ids):  # read only the needed rows
        return pd.read_parquet(C.processed_path("train", src), columns=COLS,
                               filters=[("entity_id", "in", list(ids))])

    a = load(1, pairs.s1.unique()).add_suffix("_a")
    b = pd.concat([load(k, pairs.m.unique()) for k in (2, 3)]).add_suffix("_b")
    df = pairs.merge(a, left_on="s1", right_on="entity_id_a").merge(b, left_on="m", right_on="entity_id_b")
    print(f"{len(df):,} matched pairs evaluated\n")

    nonlatin = ~df[f"{NAME}_script_a"].isin(["Latn", ""]) | ~df[f"{NAME}_script_b"].isin(["Latn", ""])
    for label, sub in (("all pairs", df), ("non-Latin name pairs", df[nonlatin])):
        report(label, sub, NAME, ("norm", "rom", "c4b"))
        both = sub[(sub[f"{ADDR}_norm_a"] != "") & (sub[f"{ADDR}_norm_b"] != "")]
        report(label + " (both addr)", both, ADDR)

    print("\nExamples (S1 romanized | matched romanized):")
    for _, r in df[nonlatin].head(10).iterrows():
        print(f"  {r[f'{NAME}_rom_a']!r:45} | {r[f'{NAME}_rom_b']!r}")


if __name__ == "__main__":
    main()