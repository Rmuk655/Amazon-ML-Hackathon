"""Stage 5b - second look at near-miss rejections with a small LLM (Qwen2.5-Instruct via llama.cpp, CPU).

Recall is the weak side, so rejected candidates just below the decision threshold get a second look:
  gray zone   p in [GRAY_FRAC * threshold, threshold), threshold per (source, address-missing) as in matching.py
  per S1      one prompt with the S1 record, its ACCEPTED matches (as examples of how this business looks in
              the noisy sources) and its gray-zone REJECTED candidates (numbered); Qwen names the numbers
              that are the same business (same branch / address) as S1
  guard       a gray candidate is shown to an S1 only if that S1 is the candidate's best claimant over ALL S1
              entities (highest model probability) and no S1 has accepted it - a record is never handed to
              S1a when S1b fits it better, and one record never goes to two S1 entities
  merge       Qwen's picks are added to the accepted matches (per-source cap as in matching.py); the decision
              trace records source = qwen

    python second_look.py validate --pairs <eval_pairs.parquet> --split train   # labelled held-out: does it help?
    python second_look.py apply --split test                                   # rewrite output/matching_results.tsv

Both stop at --time-budget seconds (S1 entities ordered by their best gray-zone probability).
validate writes models/second_look.json {"apply": bool, ...}; apply refuses to change the output unless
validate found a gain (override with --force).
"""
import argparse
import json
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C
from matching import MAX_PER_SOURCE, MacroF05, load_gt_counts, thr_of

GRAY_FRAC = 0.8
MAX_ACCEPTED_SHOWN = 3
MAX_REJECTED_SHOWN = 6
MODEL_PATH = os.environ.get("BER_LLM", str(C.RUN_MODELS_DIR / "llm" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"))
DECISION = C.RUN_MODELS_DIR / "second_look.json"

MASTER_PROMPT = """You check business records for an entity-resolution system.
You get one REFERENCE business (clean record), some ACCEPTED records already confirmed to be this same
business in noisy sources, and some REJECTED records that the system was unsure about.
Noisy records may differ from the reference by: transliteration into Indian scripts or back, typos, missing or
extra words, legal-form changes written differently (Pvt Ltd / Private Limited), words glued together, a website
form (name.com), titles (Mr/Shri), abbreviations, reordered address parts, missing address parts.
A REJECTED record is the SAME business only if its name is plausibly the same name AND its address is compatible
(same building/house number and street/locality, or address missing). Different house/plot numbers, a different
business word (e.g. Foods vs Trading), or a different locality mean a DIFFERENT business, even with a similar name.
Answer with the numbers of the REJECTED records that are the same business, comma-separated, or NONE."""


def load_texts(split, ids):
    cols = ["entity_id", "business_name", "business_name_rom", "business_address"]
    out = []
    for n in (1, 2, 3):
        pf = pq.ParquetFile(C.processed_path(split, n))
        for b in pf.iter_batches(columns=cols, batch_size=1_000_000):
            x = b.to_pandas()
            out.append(x[x["entity_id"].isin(ids)])
    return pd.concat(out, ignore_index=True).drop_duplicates("entity_id").set_index("entity_id").fillna("")


def show(t, eid):
    if eid not in t.index:
        return "(missing)"
    r = t.loc[eid]
    name = r.business_name if r.business_name == r.business_name_rom or not r.business_name_rom \
        else f"{r.business_name} (romanised: {r.business_name_rom})"
    return f"name: {name[:120]} | address: {r.business_address[:160] or '(none)'}"


def gray_pool(d, thr):
    """d: s1_id, cand_id, srcn, a_missing, p, keep. -> gray-zone rows that pass the target-centric guard."""
    d = d.copy()
    d["thr"] = [thr_of(thr, s, a) for s, a in zip(d["srcn"], d["a_missing"].fillna(0).astype(int))]
    accepted_t = set(d.loc[d["keep"], "cand_id"])
    best = d.sort_values("p", ascending=False).drop_duplicates("cand_id").set_index("cand_id")["s1_id"]
    g = d[~d["keep"] & (d["p"] >= GRAY_FRAC * d["thr"]) & (d["p"] < d["thr"])]
    g = g[~g["cand_id"].isin(accepted_t)]
    g = g[g["s1_id"].values == best.reindex(g["cand_id"]).values]          # S1 must be the target's best claimant
    return g


class Judge:
    def __init__(self, path=MODEL_PATH, threads=None):
        from llama_cpp import Llama
        self.llm = Llama(model_path=path, n_ctx=2048, n_threads=threads or os.cpu_count(), verbose=False)

    def __call__(self, ref, accepted, rejected):
        lines = [f"REFERENCE: {ref}"]
        lines += [f"ACCEPTED: {a}" for a in accepted] or ["ACCEPTED: (none yet)"]
        lines += [f"REJECTED {i + 1}: {r}" for i, r in enumerate(rejected)]
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": MASTER_PROMPT}, {"role": "user", "content": "\n".join(lines)}],
            temperature=0.0, max_tokens=24)
        text = out["choices"][0]["message"]["content"]
        return sorted({int(x) - 1 for x in re.findall(r"\d+", text) if 0 < int(x) <= len(rejected)}), text


def run_judge(d, pool, texts, budget, judge):
    """-> DataFrame(s1_id, cand_id) picked by the judge, and the number of S1 entities judged."""
    t0 = time.time()
    order = pool.groupby("s1_id")["p"].max().sort_values(ascending=False).index
    acc = d[d["keep"]].sort_values("p", ascending=False).groupby("s1_id")
    picked, n = [], 0
    for s1 in order:
        if time.time() - t0 > budget:
            break
        rej = pool[pool["s1_id"] == s1].sort_values("p", ascending=False).head(MAX_REJECTED_SHOWN)
        a_ids = acc.get_group(s1)["cand_id"].head(MAX_ACCEPTED_SHOWN).tolist() if s1 in acc.groups else []
        sel, _ = judge(show(texts, s1), [show(texts, x) for x in a_ids], [show(texts, x) for x in rej["cand_id"]])
        picked += [(s1, rej["cand_id"].iloc[i], rej["srcn"].iloc[i]) for i in sel]
        n += 1
    return pd.DataFrame(picked, columns=["s1_id", "cand_id", "srcn"]), n


def merge(d, add):
    """keep + judge additions, one S1 per target, per-source cap per S1."""
    keep = d["keep"].to_numpy().copy()
    key = d["s1_id"].astype(str) + "|" + d["cand_id"].astype(str)
    addk = set(add["s1_id"].astype(str) + "|" + add["cand_id"].astype(str))
    new = key.isin(addk).to_numpy() & ~keep
    cnt = d[keep].groupby(["s1_id", "srcn"]).size()
    room = {k: MAX_PER_SOURCE - v for k, v in cnt.items()}
    for i in np.where(new)[0]:
        k = (d["s1_id"].iat[i], d["srcn"].iat[i])
        if room.get(k, MAX_PER_SOURCE) > 0:
            keep[i] = True
            room[k] = room.get(k, MAX_PER_SOURCE) - 1
    return keep


def cmd_validate(a):
    bundle = joblib.load(C.RUN_MODELS_DIR / "stage4_model.joblib")
    d = pd.read_parquet(a.pairs, columns=["s1_id", "cand_id", "srcn", "a_missing", "p", "keep", "is_true"])
    pool = gray_pool(d, bundle["thr"])
    s1s = pool["s1_id"].unique()
    print(f"[validate] gray-zone pairs {len(pool):,} over {len(s1s):,} S1 entities "
          f"({int(pool['is_true'].sum()):,} true)")
    texts = load_texts(a.split, set(pool["s1_id"]) | set(pool["cand_id"]) | set(d.loc[d["keep"], "cand_id"]))
    add, n = run_judge(d, pool, texts, a.time_budget, Judge())
    judged = pool.drop_duplicates("s1_id").sort_values("p", ascending=False)["s1_id"].head(n)
    keep2 = merge(d, add)
    gained = keep2 & ~d["keep"].to_numpy()
    tp = int((gained & d["is_true"].to_numpy()).sum())
    # macro F0.5 on the S1 entities that were judged (before vs after); GT counts include blocking misses
    n_true = load_gt_counts({n: C.processed_path(a.split, n) for n in (1, 2, 3)})
    m = d["s1_id"].isin(set(judged)).to_numpy()
    ev = MacroF05(d["s1_id"].values[m], d["is_true"].values[m], n_true.to_dict())
    before, after = ev(d["keep"].values[m]), ev(keep2[m])
    res = {"judged_s1": int(n), "added": int(gained.sum()), "added_true": tp,
           "precision_of_additions": tp / max(int(gained.sum()), 1),
           "macro_before": before["macro"], "macro_after": after["macro"],
           "delta_macro_on_judged": after["macro"] - before["macro"],
           "gray_s1_total": int(len(s1s)), "apply": bool(after["macro"] > before["macro"])}
    print(json.dumps(res, indent=1))
    DECISION.parent.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(res))


def cmd_apply(a):
    dec = json.loads(DECISION.read_text()) if DECISION.exists() else {"apply": False}
    if not dec.get("apply") and not a.force:
        print(f"[apply] validation found no gain ({dec}); output left unchanged")
        return
    bundle = joblib.load(C.RUN_MODELS_DIR / "stage4_model.joblib")
    tr_path = C.PROCESSED_DIR / "decision_trace_test.parquet"
    d = pd.read_parquet(tr_path)
    d["keep"] = d["decision"].eq("match")
    pool = gray_pool(d, bundle["thr"])
    print(f"[apply] gray-zone pairs {len(pool):,} over {pool['s1_id'].nunique():,} S1 entities")
    texts = load_texts("test", set(pool["s1_id"]) | set(pool["cand_id"]) | set(d.loc[d["keep"], "cand_id"]))
    add, n = run_judge(d, pool, texts, a.time_budget, Judge())
    keep2 = merge(d, add)
    gained = keep2 & ~d["keep"].to_numpy()
    print(f"[apply] judged {n:,} of {pool['s1_id'].nunique():,} S1 entities; added {int(gained.sum()):,} matches")
    d.loc[gained, ["decision", "source"]] = ["match", "qwen"]
    d.drop(columns=["keep"]).to_parquet(tr_path, index=False)
    res = d[d["decision"] == "match"]
    s1_ids = pd.read_csv(C.source_path("test", 1), sep="\t", dtype=str, usecols=[0], keep_default_na=False,
                         quoting=3).iloc[:, 0].str.strip()
    g = res.groupby("s1_id")["cand_id"].apply(lambda s: ",".join(sorted(s.unique())))
    g = g.reindex(pd.unique(s1_ids)).fillna("").reset_index()
    g.columns = ["source1_entity_id", "matched_entity_ids"]
    g.to_csv(C.OUTPUT_DIR / "matching_results.tsv", sep="\t", index=False)
    print(f"[apply] wrote {C.OUTPUT_DIR / 'matching_results.tsv'}: {res['s1_id'].nunique():,} S1 matched")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--pairs", default=str(C.PROCESSED_DIR / "eval_pairs.parquet"))
    v.add_argument("--split", default="train")
    v.add_argument("--time-budget", type=float, default=1800)
    p = sub.add_parser("apply")
    p.add_argument("--time-budget", type=float, default=5400)
    p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cmd_validate(a) if a.cmd == "validate" else cmd_apply(a)


if __name__ == "__main__":
    main()
