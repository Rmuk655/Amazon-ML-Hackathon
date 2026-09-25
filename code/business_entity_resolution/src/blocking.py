"""Stage 3 - blocking / candidate generation.

For every S1 entity, find a small set of S2/S3 candidates. Recall here is the ceiling for
everything downstream, so we use several complementary keys built from the Stage-2 columns:

  core    exact c4b (core name)
  tok     each core-name token (len>=2)
  skelbi  adjacent-token pairs of a consonant skeleton (vowel/typo/transliteration variants:
          raam~ram, praivet~private)
  pre     4-char prefixes of the first two core tokens (typos late in a word)
  addrpc  postal-like number (5-6 digits) in the address + skeleton of the first name token

Keys that are shared by more targets than BLOCK_CAPS[type] are dropped (too common to help).
Each (S1, target) pair is scored by the sum of idf weights of the keys it shares; the top
K_PER_SOURCE per S1 and per source (S2 / S3) are kept.

    python blocking.py --split train --eval             # candidate parquet + recall report
    python blocking.py --split test --write-tsv         # + output/candidate_pairs.tsv

Partitions by country_norm (generic string label, nothing hard-coded); --cross-country to disable.
"""
import argparse
import re
import time
from functools import lru_cache

import numpy as np
import pandas as pd

import config as C

KT = {k: i for i, k in enumerate(C.KTYPES)}
_VOWELS = re.compile(r"[aeiou]")
_REPEAT = re.compile(r"(.)\1+")
_SKEL_SUB = (("ph", "f"), ("ck", "k"), ("w", "v"), ("c", "k"), ("q", "k"), ("z", "s"), ("x", "ks"), ("y", "i"))


@lru_cache(maxsize=2_000_000)
def skel(tok: str) -> str:
    """Consonant skeleton: 'raam'->'rm', 'praivet'->'prvt', 'private'->'prvt'."""
    if any(ch.isdigit() for ch in tok):
        return tok
    t = tok
    for a, b in _SKEL_SUB:
        t = t.replace(a, b)
    s = _REPEAT.sub(r"\1", t[0] + _VOWELS.sub("", t[1:]))
    return s or tok


def record_keys(name: str, addr: str):
    out = []
    t = name.split() if name else []
    if not t:
        return out
    out.append((KT["core"], "C|" + name))
    for w in set(t):
        if len(w) >= 2:
            out.append((KT["tok"], "T|" + w))
    sk = [skel(w) for w in t]
    if len(t) == 1:
        out.append((KT["skelbi"], "P|" + sk[0]))
    else:
        for i in range(min(len(t) - 1, 6)):
            out.append((KT["skelbi"], f"P|{sk[i]}|{sk[i + 1]}"))
    out.append((KT["pre"], "F|" + "|".join(w[:4] for w in t[:2])))
    if addr:
        for pc in [w for w in addr.split() if w.isdigit() and 5 <= len(w) <= 6][:2]:
            out.append((KT["addrpc"], f"A|{pc}|{sk[0][:3]}"))
    return out


def build_keys(names, addrs) -> pd.DataFrame:
    """-> DataFrame(idx int32, kt int8, key uint64), de-duplicated per (idx, key)."""
    parts = []
    for start in range(0, len(names), C.KEYGEN_BLOCK):
        ii, kk, ss = [], [], []
        for i in range(start, min(start + C.KEYGEN_BLOCK, len(names))):
            for kt, s in record_keys(names[i], addrs[i]):
                ii.append(i), kk.append(kt), ss.append(s)
        if ss:
            parts.append(pd.DataFrame({
                "idx": np.array(ii, dtype=np.int32),
                "kt": np.array(kk, dtype=np.int8),
                "key": pd.util.hash_array(np.array(ss, dtype=object)),
            }))
    if not parts:
        return pd.DataFrame({"idx": np.array([], np.int32), "kt": np.array([], np.int8), "key": np.array([], np.uint64)})
    return pd.concat(parts, ignore_index=True).drop_duplicates(["idx", "key"])


def build_index(tkeys: pd.DataFrame, n_targets: int) -> pd.DataFrame:
    counts = tkeys["key"].value_counts()
    df = tkeys["key"].map(counts).to_numpy()
    caps = np.array([C.BLOCK_CAPS[k] for k in C.KTYPES])[tkeys["kt"].to_numpy()]
    keep = df <= caps
    idx = tkeys[keep][["key", "idx"]].copy()
    idx["w"] = np.log2((n_targets + 1) / df[keep]).astype(np.float32)
    return idx


def chunk_candidates(skeys, index, n_t, tsrc, k):
    m = skeys.merge(index, on="key", suffixes=("_s", "_t"))
    if m.empty:
        return None
    m["pid"] = m["idx_s"].to_numpy(np.int64) * n_t + m["idx_t"].to_numpy(np.int64)
    score = m.groupby("pid")["w"].sum()
    mk = m[["pid", "kt"]].drop_duplicates()
    mk["b"] = np.left_shift(1, mk["kt"].to_numpy(np.int16)).astype(np.int16)
    mask = mk.groupby("pid")["b"].sum()
    out = pd.DataFrame({"score": score.astype(np.float32), "mask": mask}).reset_index()
    out["s"] = (out["pid"] // n_t).astype(np.int64)
    out["t"] = (out["pid"] % n_t).astype(np.int64)
    out["src"] = tsrc[out["t"].to_numpy()]
    out = out.sort_values(["s", "src", "score"], ascending=[True, True, False])
    out["rank"] = out.groupby(["s", "src"]).cumcount().astype(np.int16)
    return out


def load(split, n):
    cols = ["entity_id", "country_norm", "business_name_c4b", "business_address_rom"]
    return pd.read_parquet(C.processed_path(split, n), columns=cols)


def load_gt(ids_s1: pd.Index, ids_t: pd.Index):
    gt = pd.read_csv(C.TRAIN_GT, sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=3)
    gt["m"] = gt["matched_entity_ids"].fillna("").str.split(",").apply(lambda x: [i.strip() for i in x if i.strip()])
    flat = gt[["source1_entity_id", "m"]].explode("m").dropna(subset=["m"]).rename(
        columns={"source1_entity_id": "s1", "m": "t"}).reset_index(drop=True)
    ok = flat.s1.isin(ids_s1) & flat.t.isin(ids_t)
    return flat[ok].reset_index(drop=True), len(flat)


class Stats:
    KS = (1, 5, 10, 20, 30, 50)

    def __init__(self, k):
        self.k, self.n_true_total = k, 0
        self.reach = {2: 0, 3: 0}                       # true pairs inside a partition
        self.pre_trunc = {2: 0, 3: 0}                   # found before top-K truncation
        self.kept = {2: 0, 3: 0}
        self.by_rank = {(s, kk): 0 for s in (2, 3) for kk in self.KS}
        self.by_type = {kt: 0 for kt in C.KTYPES}
        self.n_pairs = self.n_s1 = 0
        self.denom = 0
        self.s1_with = self.s1_any = self.s1_all = 0

    def report(self):
        p = print
        p("\n=== BLOCKING RECALL (train, vs ground truth) ===")
        p(f"true pairs (ids present): {self.n_true_total:,} | reachable within country partition: "
          f"{self.reach[2] + self.reach[3]:,}")
        for s in (2, 3):
            r = max(self.reach[s], 1)
            p(f"S{s}: n_true={self.reach[s]:,} | key-coverage ceiling (before top-K)={self.pre_trunc[s] / r:.4f} | "
              f"kept @K={self.k}: {self.kept[s] / r:.4f}")
            p("   recall@rank<k: " + "  ".join(f"k={kk}:{self.by_rank[(s, kk)] / r:.4f}" for kk in self.KS if kk <= self.k))
        tot = max(self.n_true_total, 1)
        p(f"overall pair recall @K (incl. cross-country loss): {(self.kept[2] + self.kept[3]) / tot:.4f}")
        r = max(self.kept[2] + self.kept[3], 1)
        p("share of found true pairs hit by key type: " + "  ".join(f"{k}={v / r:.3f}" for k, v in self.by_type.items()))
        w = max(self.s1_with, 1)
        p(f"S1 with >=1 match: {self.s1_with:,} | any match found: {self.s1_any / w:.4f} | ALL matches found: {self.s1_all / w:.4f}")
        p(f"candidate pairs: {self.n_pairs:,} ({self.n_pairs / max(self.n_s1, 1):.1f} per S1) | "
          f"reduction ratio vs full cross-join: {1 - self.n_pairs / max(self.denom, 1):.8f}")


def run(args):
    t0 = time.time()
    S1 = load(args.split, 1)
    if args.limit_s1:
        S1 = S1.head(args.limit_s1)
    T = pd.concat([load(args.split, 2).assign(src=2), load(args.split, 3).assign(src=3)], ignore_index=True)
    for df in (S1, T):
        df["business_name_c4b"] = df["business_name_c4b"].fillna("")
        df["business_address_rom"] = df["business_address_rom"].fillna("")
    print(f"[load] S1={len(S1):,} targets={len(T):,} ({time.time() - t0:.0f}s)")

    flat = None
    st = Stats(args.k)
    if args.eval:
        flat, _ = load_gt(pd.Index(S1.entity_id), pd.Index(T.entity_id))
        st.n_true_total = len(flat)
        print(f"[eval] {len(flat):,} ground-truth pairs with both ids loaded")

    outdir = C.candidates_dir(args.split)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("*.parquet"):
        f.unlink()

    if C.WITHIN_COUNTRY and not args.cross_country:
        parts = [(c, S1[S1.country_norm == c].reset_index(drop=True), T[T.country_norm == c].reset_index(drop=True))
                 for c in sorted(S1.country_norm.unique()) if not args.countries or c in args.countries]
    else:
        parts = [("all", S1.reset_index(drop=True), T)]

    missed_pool = []
    for pname, s1c, tc in parts:
        if not len(tc):
            print(f"[{pname}] no targets, skipped")
            continue
        n_t = len(tc)
        tsrc = tc["src"].to_numpy()
        print(f"[{pname}] S1={len(s1c):,} targets={n_t:,}")
        index = build_index(build_keys(tc.business_name_c4b.tolist(), tc.business_address_rom.tolist()), n_t)
        print(f"   target index rows={len(index):,} ({time.time() - t0:.0f}s)")
        s1_ids, t_ids = s1c.entity_id.to_numpy(), tc.entity_id.to_numpy()

        codes = None
        if flat is not None:
            sp = pd.Index(s1c.entity_id).get_indexer(flat.s1)
            tp = pd.Index(tc.entity_id).get_indexer(flat.t)
            ok = (sp >= 0) & (tp >= 0)
            codes = np.sort(sp[ok].astype(np.int64) * n_t + tp[ok])
            tsrc_true = tsrc[tp[ok]]
            for s in (2, 3):
                st.reach[s] += int((tsrc_true == s).sum())
            true_per_s1 = np.bincount(sp[ok], minlength=len(s1c))
            found_per_s1 = np.zeros(len(s1c), np.int64)
            found_codes = []

        for a in range(0, len(s1c), C.S1_CHUNK):
            sub = s1c.iloc[a:a + C.S1_CHUNK]
            sk = build_keys(sub.business_name_c4b.tolist(), sub.business_address_rom.tolist())
            sk["idx"] = sk["idx"] + a
            cand = chunk_candidates(sk, index, n_t, tsrc, args.k)
            if cand is None:
                continue
            if codes is not None:
                pos = np.minimum(np.searchsorted(codes, cand["pid"].to_numpy()), len(codes) - 1)
                cand["is_true"] = (codes[pos] == cand["pid"].to_numpy()) if len(codes) else False
                for s in (2, 3):
                    st.pre_trunc[s] += int(((cand.is_true) & (cand.src == s)).sum())
            cand = cand[cand["rank"] < args.k]
            st.n_pairs += len(cand)
            if codes is not None:
                tr = cand[cand.is_true]
                for s in (2, 3):
                    ts = tr[tr.src == s]
                    st.kept[s] += len(ts)
                    for kk in Stats.KS:
                        st.by_rank[(s, kk)] += int((ts["rank"] < kk).sum())
                for kt, i in KT.items():
                    st.by_type[kt] += int(((tr["mask"].to_numpy() >> i) & 1).sum())
                found_per_s1 += np.bincount(tr["s"].to_numpy(), minlength=len(s1c))
                found_codes.append(tr["pid"].to_numpy())
            out = pd.DataFrame({
                "s1_id": s1_ids[cand["s"].to_numpy()], "cand_id": t_ids[cand["t"].to_numpy()],
                "src": cand["src"].to_numpy(np.int8), "score": cand["score"].to_numpy(),
                "mask": cand["mask"].to_numpy(), "rank": cand["rank"].to_numpy(np.int8)})
            if codes is not None:
                out["is_true"] = cand["is_true"].to_numpy(np.int8)
            out.to_parquet(outdir / f"part-{pname}-{a // C.S1_CHUNK:04d}.parquet", index=False)
            print(f"   chunk {a // C.S1_CHUNK}: {len(out):,} pairs ({time.time() - t0:.0f}s)")

        st.n_s1 += len(s1c)
        st.denom += len(s1c) * n_t
        if codes is not None:
            w = true_per_s1 > 0
            st.s1_with += int(w.sum())
            st.s1_any += int((w & (found_per_s1 > 0)).sum())
            st.s1_all += int((w & (found_per_s1 >= true_per_s1)).sum())
            fc = np.concatenate(found_codes) if found_codes else np.array([], np.int64)
            miss = codes[~np.isin(codes, fc)]
            if len(miss):
                miss = np.random.default_rng(0).choice(miss, min(200, len(miss)), replace=False)
                missed_pool.append(pd.DataFrame({
                    "s1_id": s1_ids[miss // n_t], "t_id": t_ids[miss % n_t],
                    "s1_name": s1c.business_name_c4b.to_numpy()[miss // n_t],
                    "t_name": tc.business_name_c4b.to_numpy()[miss % n_t]}))

    if flat is not None:
        st.report()
        if missed_pool:
            p = C.PROCESSED_DIR / "eda" / "missed_true_pairs.tsv"
            p.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(missed_pool).head(400).to_csv(p, sep="\t", index=False)
            print(f"missed true pairs sample -> {p} (inspect: what do the misses have in common?)")
    print(f"[done] {time.time() - t0:.0f}s -> {outdir}")
    if args.write_tsv:
        write_tsv(args.split, C.OUTPUT_DIR / "candidate_pairs.tsv", S1["entity_id"])


def write_tsv(split, path, all_s1_ids=None):
    """output/candidate_pairs.tsv - grouped format matching utils/validate_submission.py:
    one row per S1 entity, header 'source1_entity_id<TAB>candidate_entity_ids', candidates
    comma-joined (empty string if none). `all_s1_ids` should be every S1 entity_id in the
    split so entities with zero blocking candidates still get a (empty) row, since the
    validator requires a row for every entity in test_source1.tsv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [pd.read_parquet(part, columns=["s1_id", "cand_id"])
             for part in sorted(C.candidates_dir(split).glob("*.parquet"))]
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["s1_id", "cand_id"])
    n_pairs = len(d)
    grouped = d.groupby("s1_id")["cand_id"].apply(lambda s: ",".join(sorted(s.unique())))
    if all_s1_ids is not None:
        grouped = grouped.reindex(pd.unique(pd.Index(all_s1_ids))).fillna("")
    grouped = grouped.reset_index()
    grouped.columns = ["source1_entity_id", "candidate_entity_ids"]
    grouped.to_csv(path, sep="\t", index=False)
    print(f"[tsv] {n_pairs:,} pairs across {len(grouped):,} S1 rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--k", type=int, default=C.K_PER_SOURCE, help="candidates kept per S1 per source")
    ap.add_argument("--eval", action="store_true", help="train only: recall vs ground truth")
    ap.add_argument("--countries", nargs="*", help="only these country_norm values (memory control)")
    ap.add_argument("--cross-country", action="store_true")
    ap.add_argument("--limit-s1", type=int, default=None, help="dev: first N S1 rows")
    ap.add_argument("--write-tsv", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()