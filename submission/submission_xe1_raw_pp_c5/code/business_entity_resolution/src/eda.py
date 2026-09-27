"""Stage 1 EDA: answers the design questions (checks 1.1-1.10) on the raw TRAIN data.

  python eda.py                       # full run
  python eda.py --nrows 300000        # dev run (reads the first N rows of each file; GT is filtered to match)
  python eda.py --s1 path --s2 path --s3 path --gt path

Writes to eda/: eda_findings.md (numbers + auto-derived decisions), eda_token_df.csv, eda_worst_pairs.tsv,
eda_script_crosstab.csv, eda_ambiguous_names.csv
"""
import argparse
import csv
import glob
import os
import re
import sys
import unicodedata
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "eda")
FINDINGS = []
DECISIONS = []


def log(*a):
    print(*a)


def note(check, text):
    FINDINGS.append((check, text))


def decide(finding, decision):
    DECISIONS.append((finding, decision))


# --------------------------------------------------------------------------- loading
def discover(split, overrides):
    found = dict(overrides)
    pat = re.compile(r"(?:^|[^a-z0-9])(?:s|source_?)([123])(?:[^0-9]|$)")
    tsvs = [p for p in glob.glob(os.path.join(ROOT, "dataset", "**", "*.tsv"), recursive=True)
            if not any(x in os.path.relpath(p, ROOT).lower().split(os.sep) for x in ("processed", "eda", "output"))]
    for p in tsvs:
        b = os.path.basename(p).lower()
        if split not in os.path.relpath(p, ROOT).lower():   # relative path: the absolute path may contain anything
            continue
        m = pat.search(b)
        if m and int(m.group(1)) not in found:
            found[int(m.group(1))] = p
        elif any(k in b for k in ("ground", "truth", "label", "gt", "match")) and "gt" not in found:
            found["gt"] = p
    missing = [k for k in (1, 2, 3, "gt") if k not in found]
    if missing:
        raise SystemExit(f"Could not locate {missing} among TSVs: {[os.path.relpath(p, ROOT) for p in tsvs]}\n"
                         f"Pass --s1/--s2/--s3/--gt")
    return found


def read_tsv(path, nrows=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""],
                       quoting=csv.QUOTE_NONE, nrows=nrows, encoding="utf-8", on_bad_lines="warn")


def pick(df, keys, what):
    for k in keys:
        for c in df.columns:
            if c.lower() == k:
                return c
    for k in keys:
        for c in df.columns:
            if k in c.lower():
                return c
    raise SystemExit(f"Cannot find {what} column in {list(df.columns)}")


# --------------------------------------------------------------------------- helpers
@lru_cache(maxsize=None)
def _char_script(ch):
    try:
        return unicodedata.name(ch).split()[0].lower()
    except ValueError:
        return "other"


def script_of(s):
    if not isinstance(s, str):
        return "none"
    c = Counter(_char_script(ch) for ch in s if ch.isalpha())
    if not c:
        return "none"
    top, n = c.most_common(1)[0]
    return top if n / sum(c.values()) >= 0.8 else "mixed"


STOP = {"private", "pvt", "ltd", "limited", "llp", "inc", "incorporated", "corp", "corporation", "co", "company",
        "llc", "the", "and", "of", "pty", "gmbh", "sa", "sarl", "plc", "&"}


def norm(s):
    s = unicodedata.normalize("NFKC", s if isinstance(s, str) else "").casefold()
    # drop only punctuation/symbols; \w would also strip Indic vowel signs (combining marks) and split words
    s = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def core(s):
    return " ".join(t for t in norm(s).split() if t not in STOP)


def pct(x):
    return f"{100 * x:.2f}%"


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--pair-sample", type=int, default=20000)
    ap.add_argument("--script-sample", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=0)
    for k in ("s1", "s2", "s3", "gt"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ov = {}
    for k, key in (("s1", 1), ("s2", 2), ("s3", 3), ("gt", "gt")):
        if getattr(a, k):
            ov[key] = getattr(a, k)
    files = discover(a.split, ov)
    for k, v in files.items():
        log(f"  {k}: {os.path.relpath(v, ROOT)}")

    # ---------------- 1.1 load + sanity
    log("\n== 1.1 load + sanity ==")
    S = {k: read_tsv(files[k], a.nrows) for k in (1, 2, 3)}
    GT = read_tsv(files["gt"])
    cols = {}
    for k, df in S.items():
        cols[k] = dict(id=pick(df, ["entity_id", "id"], "id"), name=pick(df, ["business_name", "name"], "name"),
                       addr=pick(df, ["business_address", "address"], "address"),
                       country=pick(df, ["country"], "country"))
        log(f"S{k}: shape {df.shape}, columns {list(df.columns)}")
        log("  null rate:", df.isna().mean().round(4).to_dict())
        idc = cols[k]["id"]
        lit = df[cols[k]["addr"]].fillna("").str.strip().str.lower().isin(["null", "nan", "none"]).mean()
        log(f"  id unique: {df[idc].is_unique}; id prefixes: {df[idc].str[:3].value_counts().head(3).to_dict()}; "
            f"literal null/nan/none addresses: {pct(lit)}")
        note("1.1", f"S{k}: {len(df):,} rows; address null {pct(df[cols[k]['addr']].isna().mean())}; "
                    f"literal 'null' strings {pct(lit)}; id unique={df[idc].is_unique}")
        log(df[cols[k]["country"]].value_counts(dropna=False).head(15).to_string())
    gid = pick(GT, ["source1_entity_id", "entity_id"], "GT source1 id")
    gm = pick(GT, ["matched_entity_ids", "matched"], "GT matched ids")

    # ---------------- 1.2 GT structure
    log("\n== 1.2 ground truth structure ==")
    sample_vals = GT[gm].dropna().head(2000).astype(str)
    for sep in (";", "|"):
        if sample_vals.str.contains(re.escape(sep)).any():
            log(f"  WARNING: separator '{sep}' seen in {gm}; this script splits on commas only")
    if a.nrows:
        GT = GT[GT[gid].isin(set(S[1][cols[1]["id"]]))]
    GT = GT.copy()
    GT["matches"] = GT[gm].fillna("").astype(str).str.split(",").apply(lambda x: [i.strip() for i in x if i.strip()])
    GT["n_match"] = GT["matches"].str.len()
    ids2, ids3 = set(S[2][cols[2]["id"]]), set(S[3][cols[3]["id"]])
    GT["n_s2"] = GT["matches"].apply(lambda m: sum(i in ids2 for i in m))
    GT["n_s3"] = GT["matches"].apply(lambda m: sum(i in ids3 for i in m))
    single = (GT["n_match"] == 0).mean()
    log("singleton rate:", pct(single))
    log(GT["n_match"].describe().round(2).to_string())
    log(GT["n_match"].value_counts().head(10).to_string())
    flat = GT[["matches", gid]].explode("matches").dropna(subset=["matches"]).rename(columns={"matches": "target"}).reset_index(drop=True)
    claimed = flat["target"].value_counts()
    multi = int((claimed > 1).sum())
    log("targets claimed by >1 S1:", multi)
    m2 = S[2][cols[2]["id"]].isin(set(flat["target"])).mean()
    m3 = S[3][cols[3]["id"]].isin(set(flat["target"])).mean()
    log(f"S2 matched share: {pct(m2)}; S3 matched share: {pct(m3)}")
    note("1.2", f"singleton rate {pct(single)}; matches per S1 mean {GT['n_match'].mean():.2f} max {GT['n_match'].max()}; "
                f"targets claimed by >1 S1: {multi}; S2 matched share {pct(m2)}, S3 {pct(m3)}; "
                f"targets outside S2/S3: {int((~flat['target'].isin(ids2 | ids3)).sum())}")
    decide(f"Targets claimed by >1 S1 = {multi}",
           "Enforce one-to-one assignment (matching.py exclusive rule ON)" if multi == 0 else
           "Do NOT force one-to-one; keep multi-claim targets")
    decide(f"Singleton rate {pct(single)}",
           "High: precision/thresholding dominate, tune conservatively" if single > 0.3 else
           "Low: recall matters most, singletons are a minor factor")

    # ---------------- lookup tables
    allr = pd.concat([S[k][[cols[k]["id"], cols[k]["name"], cols[k]["addr"], cols[k]["country"]]]
                      .set_axis(["id", "name", "addr", "country"], axis=1) for k in (1, 2, 3)], ignore_index=True)
    allr = allr.drop_duplicates("id").set_index("id")

    # ---------------- 1.3 country
    log("\n== 1.3 country consistency ==")
    flat["s1_country"] = flat[gid].map(allr["country"])
    flat["t_country"] = flat["target"].map(allr["country"])
    have13 = flat[gid].isin(allr.index) & flat["target"].isin(allr.index)
    if not have13.all():
        log(f"  note: {int((~have13).sum()):,} of {len(flat):,} GT pairs reference an id outside the "
            f"loaded rows (likely due to --nrows); excluding them from the cross-country rate")
    cross = (flat.loc[have13, "s1_country"] != flat.loc[have13, "t_country"]).mean()
    labels = sorted(allr["country"].dropna().unique())
    log("cross-country matches:", pct(cross))
    log(f"{len(labels)} country labels:", labels[:60])
    note("1.3", f"cross-country matches {pct(cross)}; {len(labels)} distinct labels "
                f"({', '.join(map(str, labels[:15]))}{'...' if len(labels) > 15 else ''})")
    decide(f"Cross-country matches {pct(cross)}",
           "Block within country (WITHIN_COUNTRY=True)" if cross < 0.005 else
           "Cross-country matches exist: use --cross-country in blocking or a country-group map")

    # ---------------- 1.4 script
    log("\n== 1.4 script profile ==")
    rng = np.random.RandomState(a.seed)
    for k in (1, 2, 3):
        d = S[k]
        sub = d.sample(min(a.script_sample, len(d)), random_state=a.seed)
        sc = sub[cols[k]["name"]].map(script_of)
        tab = pd.crosstab(sub[cols[k]["country"]], sc)
        log(f"S{k} (sample {len(sub):,}) country x script:\n{tab.head(15).to_string()}")
    have14 = flat[gid].isin(allr.index) & flat["target"].isin(allr.index)
    fp = flat[have14]
    ps = fp.sample(min(a.pair_sample * 5, len(fp)), random_state=a.seed).copy()
    ps["s1_script"] = ps[gid].map(allr["name"]).map(script_of)
    ps["t_script"] = ps["target"].map(allr["name"]).map(script_of)
    ct = pd.crosstab(ps["s1_script"], ps["t_script"])
    ct.to_csv(os.path.join(OUT, "eda_script_crosstab.csv"))
    log("true pairs: S1 script x target script\n" + ct.to_string())
    nonlat = ps[(ps["s1_script"] != "latin") | (ps["t_script"] != "latin")]
    cross_script = (ps["s1_script"] != ps["t_script"]).mean()
    note("1.4", f"true pairs with different S1/target script: {pct(cross_script)}; "
                f"pairs involving a non-Latin name: {pct(len(nonlat) / max(len(ps), 1))}")
    decide(f"Cross-script true pairs {pct(cross_script)}",
           "Transliteration + phonetic/embedding channel are essential" if cross_script > 0.05 else
           "Mostly same-script: transliteration is a modest gain")

    # ---------------- 1.5 missing address
    log("\n== 1.5 missing address ==")
    for k in (1, 2, 3):
        log(f"S{k} address missing: {pct(S[k][cols[k]['addr']].isna().mean())}")
    tgt_missing = allr["addr"].isna()
    flat["a1"] = flat[gid].map(tgt_missing)
    flat["a2"] = flat["target"].map(tgt_missing)
    both = (flat["a1"] & flat["a2"]).mean()
    either = (flat["a1"] | flat["a2"]).mean()
    matched_ids = set(flat["target"]) | set(flat[gid])
    miss_m = tgt_missing[tgt_missing.index.isin(matched_ids)].mean()
    miss_u = tgt_missing[~tgt_missing.index.isin(matched_ids)].mean()
    log(f"true pairs with both addresses missing: {pct(both)}; either: {pct(either)}")
    log(f"address missing among matched records {pct(miss_m)} vs unmatched {pct(miss_u)}")
    note("1.5", f"true pairs both-missing {pct(both)}, either-missing {pct(either)}; address missing matched "
                f"{pct(miss_m)} vs unmatched {pct(miss_u)}")
    decide(f"Both addresses missing on {pct(both)} of true pairs",
           "Use a separate (stricter) threshold for name-only pairs" if both > 0.05 else
           "Name-only pairs are rare; a single threshold is fine")

    # ---------------- 1.6 name noise
    log("\n== 1.6 name noise on true pairs ==")
    have = flat[gid].isin(allr.index) & flat["target"].isin(allr.index)
    if not have.all():
        log(f"  note: {int((~have).sum()):,} of {len(flat):,} GT pairs reference an id outside the loaded "
            f"rows (likely due to --nrows); excluding them from 1.6/1.7 sampling")
    fm = flat[have]
    sm = fm.sample(min(a.pair_sample, len(fm)), random_state=a.seed).copy()
    n1 = sm[gid].map(allr["name"]).fillna("").astype(str).str.casefold()
    n2 = sm["target"].map(allr["name"]).fillna("").astype(str).str.casefold()
    sm["name_sim"] = [JaroWinkler.similarity(x, y) for x, y in zip(n1, n2)]
    sm["s1_name"], sm["t_name"] = allr.loc[sm[gid], "name"].values, allr.loc[sm["target"], "name"].values
    sm[[gid, "target", "s1_name", "t_name", "name_sim"]].sort_values("name_sim").head(50) \
        .to_csv(os.path.join(OUT, "eda_worst_pairs.tsv"), sep="\t", index=False)
    d = sm["name_sim"].describe()
    log(d.round(3).to_string())
    log("share of true pairs with JW < 0.7:", pct((sm["name_sim"] < 0.7).mean()))
    note("1.6", f"Jaro-Winkler on true pairs: median {d['50%']:.3f}, p10 {sm['name_sim'].quantile(.1):.3f}; "
                f"share below 0.7 = {pct((sm['name_sim'] < 0.7).mean())} (eyeball eda_worst_pairs.tsv)")
    decide(f"{pct((sm['name_sim'] < 0.7).mean())} of true pairs have JW < 0.7",
           "String similarity alone is insufficient: keep phonetic/skeleton features (and consider embeddings)"
           if (sm["name_sim"] < 0.7).mean() > 0.15 else "String similarity carries most positives")

    # ---------------- 1.7 ambiguity
    log("\n== 1.7 name ambiguity in S1 ==")
    c1 = S[1][cols[1]["name"]].map(core)
    vc = c1[c1 != ""].value_counts()
    dup_names = int((vc > 1).sum())
    dup_share = float(vc[vc > 1].sum() / max(len(c1), 1))
    vc.head(200).rename("n_s1").to_csv(os.path.join(OUT, "eda_ambiguous_names.csv"))
    log(f"S1 core names shared by >1 entity: {dup_names:,} (covering {pct(dup_share)} of S1)")
    log(vc.head(20).to_string())
    note("1.7", f"S1 core names shared by >1 entity: {dup_names:,}, covering {pct(dup_share)} of S1 rows "
                f"(top: {', '.join(vc.index[:5])})")
    decide(f"{pct(dup_share)} of S1 rows share a core name",
           "Chain risk: ambiguity features (name frequency, top-2 margin) are mandatory; cap key sizes in blocking"
           if dup_share > 0.03 else "Name ambiguity is low")

    # ---------------- 1.8 duplicates
    log("\n== 1.8 group sizes ==")
    log(f"max S2 per S1: {GT['n_s2'].max()}, max S3 per S1: {GT['n_s3'].max()}")
    log(GT[["n_s2", "n_s3"]].describe().round(2).to_string())
    note("1.8", f"matches per S1: S2 mean {GT['n_s2'].mean():.2f} max {GT['n_s2'].max()}; "
                f"S3 mean {GT['n_s3'].mean():.2f} max {GT['n_s3'].max()}; exact duplicate core names in S1 = {dup_names:,}")

    # ---------------- 1.9 lengths
    log("\n== 1.9 text length ==")
    for k in (1, 2, 3):
        nl = S[k][cols[k]["name"]].fillna("").str.split().str.len()
        al = S[k][cols[k]["addr"]].dropna().str.split().str.len()
        log(f"S{k} name tokens: mean {nl.mean():.2f} p99 {nl.quantile(.99):.0f}; "
            f"address tokens: mean {al.mean():.2f} p99 {al.quantile(.99):.0f}")
        note("1.9", f"S{k}: name tokens mean {nl.mean():.1f} (p99 {nl.quantile(.99):.0f}); "
                    f"address tokens mean {al.mean():.1f} (p99 {al.quantile(.99):.0f})")

    # ---------------- 1.10 token df
    log("\n== 1.10 token frequency ==")
    df_name, df_addr = Counter(), Counter()
    for k in (1, 2, 3):
        for s in S[k][cols[k]["name"]].dropna():
            df_name.update(set(norm(s).split()))
        for s in S[k][cols[k]["addr"]].dropna():
            df_addr.update(set(norm(s).split()))
    top = pd.DataFrame(df_name.most_common(300), columns=["token", "df"])
    top.to_csv(os.path.join(OUT, "eda_token_df.csv"), index=False)
    log("top name tokens (suffix/stopword candidates):", [t for t, _ in df_name.most_common(40)])
    rare = sum(1 for v in df_addr.values() if v < 50)
    log(f"address tokens with df<50: {rare:,} of {len(df_addr):,}")
    note("1.10", f"top name tokens: {', '.join(t for t, _ in df_name.most_common(20))}; "
                 f"address tokens with df<50: {rare:,} of {len(df_addr):,} (rare-token blocking keys)")

    # ---------------- findings note
    with open(os.path.join(OUT, "eda_findings.md"), "w", encoding="utf-8") as f:
        f.write("# EDA findings (auto-generated)\n\n")
        for chk, txt in FINDINGS:
            f.write(f"- **{chk}**: {txt}\n")
        f.write("\n## What each finding decides\n\n| Finding | Decision |\n|---|---|\n")
        for fnd, dec in DECISIONS:
            f.write(f"| {fnd} | {dec} |\n")
    log("\n== decisions ==")
    for fnd, dec in DECISIONS:
        log(f"- {fnd}: {dec}")
    log(f"\nwrote {os.path.relpath(OUT, ROOT)}/eda_findings.md and CSVs")


if __name__ == "__main__":
    main()