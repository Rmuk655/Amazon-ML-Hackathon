"""High-variance noisy training samples for fine-tuning (analysis / training aid, not part of the submission path).

Each sample is a mini train split in the official schema (train_source{1,2,3}.tsv + train_ground_truth.tsv):
  * real   : non-holdout train S1s with their real S2/S3 matches, plus extra noised copies of the S1 and of the
             real matches (noise on noise)
  * france : pseudo-entities built from clean *test* S1 France rows (no labels used) - France is absent from train
  * decoys : generator rule (house number changed + legal form added) and sibling S1s (same name, other address)
Noise level grows per sample; the last sample has ~2x decoy density (as in test).

  python augment_noise.py --out ../../../dataset/augmented --samples 3 --n-s1 12000 --n-fr 12000
"""
import argparse
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import config

DS = Path(__file__).resolve().parents[3] / "dataset"
COLS = ["entity_id", "business_name", "business_address", "country"]

LEGAL = {
    "US": ["Inc", "LLC", "Corp", "Co", "Ltd", "Inc.", "L.L.C.", "Corporation", "Company", "LLP", "PLLC"],
    "India": ["Pvt Ltd", "Private Limited", "Ltd", "LLP", "Pvt. Ltd.", "Limited", "Private Ltd", "OPC Pvt Ltd", "Co"],
    "France": ["SARL", "SAS", "SA", "EURL", "SASU", "SNC", "S.A.R.L.", "S.A.S."],
}
FILLERS = {
    "US": ["The", "Services", "Group", "Solutions", "Enterprises", "Holdings", "& Co", "Shop", "Store", "Center"],
    "India": ["Shri", "Sri", "New", "Enterprises", "Traders", "Industries", "Agency", "& Sons", "Stores", "Services"],
    "France": ["Le", "La", "Les", "Société", "Groupe", "Services", "Atelier", "Maison", "et Fils", "Cie"],
}
ABBR = [("Private Limited", "Pvt Ltd"), ("Limited", "Ltd"), ("Company", "Co"), ("Corporation", "Corp"),
        ("Incorporated", "Inc"), (" and ", " & "), ("Saint", "St"), ("Brothers", "Bros"), ("International", "Intl"),
        ("Services", "Svcs"), ("Enterprises", "Ent"), ("Société", "Ste"), ("Compagnie", "Cie")]
STREET = [("Road", "Rd"), ("Street", "St"), ("Avenue", "Ave"), ("Drive", "Dr"), ("Boulevard", "Blvd"),
          ("Lane", "Ln"), ("Court", "Ct"), ("Highway", "Hwy"), ("Nagar", "Ngr"), ("Colony", "Col"),
          ("rue", "R."), ("avenue", "Av."), ("boulevard", "Bd"), ("place", "Pl."), ("chemin", "Ch."),
          ("Rue", "R."), ("Avenue", "AV"), ("Boulevard", "BD"), ("North", "N"), ("South", "S"), ("East", "E"),
          ("West", "W")]
OCR = [("0", "O"), ("O", "0"), ("1", "l"), ("l", "1"), ("rn", "m"), ("m", "rn"), ("5", "S"), ("S", "5"),
       ("8", "B"), ("e", "c"), ("i", "l"), ("vv", "w"), ("cl", "d")]
KEYB = "qwertyuiopasdfghjklzxcvbnm"
NEIGH = {c: set() for c in KEYB}
_rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
for r, row in enumerate(_rows):
    for i, c in enumerate(row):
        for rr in (r - 1, r, r + 1):
            if 0 <= rr < 3:
                for j in (i - 1, i, i + 1):
                    if 0 <= j < len(_rows[rr]) and _rows[rr][j] != c:
                        NEIGH[c].add(_rows[rr][j])
NEIGH = {c: sorted(v) for c, v in NEIGH.items()}
HOUSE = re.compile(r"\b\d+[A-Za-z]?\b")


# ---------------------------------------------------------------- noise primitives
def typo(w, rng):
    if len(w) < 3:
        return w
    i = rng.randrange(len(w))
    op = rng.random()
    c = w[i].lower()
    if op < 0.3 and c in NEIGH:
        return w[:i] + rng.choice(NEIGH[c]) + w[i + 1:]
    if op < 0.55:
        return w[:i] + w[i + 1:]
    if op < 0.8 and i < len(w) - 1:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return w[:i] + w[i] + w[i:]


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


FR_ART = re.compile(r"\b(le|la|les|du|de|des|d'|l'|et|au|aux)\s*", re.I)
FR_SWAP = [("Saint", "St"), ("Sainte", "Ste"), ("Société", "Sté"), ("Etablissements", "Ets"),
           ("Établissements", "Ets"), ("Boulangerie", "Boulang."), ("Pharmacie", "Pharm."), ("Cabinet", "Cab."),
           ("Restaurant", "Resto"), ("Garage", "Gge"), (" et ", " & "), ("Frères", "Freres")]


def french_noise(s, lvl, rng):
    p = lambda base: rng.random() < min(0.95, base * lvl)
    if p(0.35):
        s = strip_accents(s)
    if "'" in s and p(0.3):
        s = s.replace("'", rng.choice([" ", "", "’"]))
    if p(0.12):
        s = FR_ART.sub("", s, count=1)
    elif p(0.08):
        s = rng.choice(["Le ", "La ", "L'", "Les "]) + s
    for a, b in FR_SWAP:
        if a in s and p(0.35):
            s = s.replace(a, b)
    if "-" in s and p(0.2):
        s = s.replace("-", rng.choice([" ", ""]))
    return s


def split_legal(name, country):
    toks = name.split()
    forms = sorted(LEGAL[country] + LEGAL["US"], key=lambda f: -len(f.split()))
    for f in forms:
        k = len(f.split())
        if len(toks) > k and " ".join(toks[-k:]).lower() == f.lower():
            return " ".join(toks[:-k]), " ".join(toks[-k:])
    return name, ""


def noise_name(name, country, lvl, rng, translit):
    core, legal = split_legal(name, country)
    toks = core.split() or [core]
    p = lambda base: rng.random() < min(0.95, base * lvl)
    if p(0.25):
        toks = [typo(t, rng) if rng.random() < 0.5 else t for t in toks]
    if len(toks) > 1 and p(0.12):
        del toks[rng.randrange(len(toks))]
    if p(0.12):
        toks.insert(rng.choice([0, len(toks)]), rng.choice(FILLERS[country]))
    if len(toks) > 1 and p(0.06):
        i = rng.randrange(len(toks) - 1)
        toks[i], toks[i + 1] = toks[i + 1], toks[i]
    if len(toks) > 1 and p(0.06):
        i = rng.randrange(len(toks) - 1)
        toks[i:i + 2] = [toks[i] + toks[i + 1]]
    if p(0.05):
        t = rng.randrange(len(toks))
        if len(toks[t]) > 7:
            k = rng.randrange(3, len(toks[t]) - 3)
            toks[t:t + 1] = [toks[t][:k], toks[t][k:]]
    s = " ".join(toks)
    if p(0.2):
        legal = rng.choice(LEGAL[country]) if rng.random() < 0.7 else ""
    elif not legal and p(0.1):
        legal = rng.choice(LEGAL[country])
    if legal:
        s = f"{s} {legal}"
    for a, b in ABBR:
        if a in s and p(0.3):
            s = s.replace(a, b)
        elif b in s and p(0.1):
            s = s.replace(b, a)
    if p(0.05):
        a, b = rng.choice(OCR)
        s = s.replace(a, b, 1)
    if country == "France":
        s = french_noise(s, lvl, rng)
    if country == "India" and translit and p(0.15):
        tt = [translit[w.lower()] if w.lower() in translit and rng.random() < 0.8 else w for w in s.split()]
        s = " ".join(tt)
    r = rng.random()
    if p(0.04):
        s = re.sub(r"[^0-9A-Za-z]", "", core).lower() + rng.choice([".com", ".in" if country == "India" else ".fr"
                                                                        if country == "France" else ".net", ""])
    elif r < 0.12 * lvl:
        s = s.upper()
    elif r < 0.18 * lvl:
        s = s.lower()
    if p(0.03):
        s = rng.choice(["-- ", "* ", "'", "\""]) + s
    if p(0.03):
        s = s + " " + rng.choice(["(Head Office)", "- Branch", "#2", "(Closed)", "aka " + toks[0]])
    return s.strip() or name


def change_house(addr, rng):
    m = list(HOUSE.finditer(addr))
    if not m:
        return f"{rng.randint(1, 9999)} {addr}"
    g = rng.choice(m)
    num = re.sub(r"\D", "", g.group()) or "1"
    new = str(max(1, int(num) + rng.choice([-1, 1]) * rng.randint(1, 40)))
    if rng.random() < 0.4:
        new = num[:-1] + str((int(num[-1]) + rng.randint(1, 9)) % 10) if len(num) > 1 else new
    return addr[:g.start()] + new + addr[g.end():]


def noise_addr(addr, country, lvl, rng):
    if not addr:
        return addr
    p = lambda base: rng.random() < min(0.95, base * lvl)
    if p(0.05):
        return ""
    parts = [x.strip() for x in addr.split(",") if x.strip()]
    if len(parts) > 2 and p(0.12):
        del parts[rng.randrange(1, len(parts))]
    if len(parts) > 1 and p(0.08):
        rng.shuffle(parts)
    if len(parts) > 1 and p(0.05):
        parts = parts[1:] + parts[:1]
    s = ", ".join(parts)
    for a, b in STREET:
        if re.search(rf"\b{re.escape(a)}\b", s) and p(0.35):
            s = re.sub(rf"\b{re.escape(a)}\b", b, s)
    if p(0.15):
        s = " ".join(typo(w, rng) if rng.random() < 0.2 else w for w in s.split(" "))
    if p(0.04):
        s = change_house(s, rng)          # digit noise on a true match (hard positive)
    if p(0.05):
        s += rng.choice([", Suite ", ", Unit ", ", Apt ", ", Bât. ", ", Floor "]) + str(rng.randint(1, 400))
    if country == "France" and p(0.3):
        s = strip_accents(s)
    r = rng.random()
    if r < 0.25 * lvl:
        s = s.upper()
    elif r < 0.3 * lvl:
        s = s.lower()
    return s


def make_decoy(name, addr, country, rng):
    core, legal = split_legal(name, country)
    forms = [f for f in LEGAL[country] if f.lower() != legal.lower()]
    return f"{core} {rng.choice(forms)}", change_house(addr or "", rng)


# ---------------------------------------------------------------- data loading
def load_tsv(path, **kw):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3, **kw)


def learn_translit(s1, recs):
    """Latin token -> Devanagari token from aligned matched pairs of equal token count."""
    votes = defaultdict(Counter)
    for n1, n2 in zip(s1, recs):
        a, b = n1.split(), n2.split()
        if len(a) == len(b) and any("ऀ" <= ch <= "ॿ" for ch in n2):
            for x, y in zip(a, b):
                if any("ऀ" <= ch <= "ॿ" for ch in y) and x.isalpha():
                    votes[x.lower()][y] += 1
    return {k: c.most_common(1)[0][0] for k, c in votes.items() if c.most_common(1)[0][1] >= 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DS / "augmented"))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--n-s1", type=int, default=12000, help="real train S1 per sample")
    ap.add_argument("--n-fr", type=int, default=12000, help="France pseudo S1 per sample (from test S1)")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    print("loading train S1 + GT ...", flush=True)
    s1 = load_tsv(DS / "train/train_source1.tsv")
    gt = load_tsv(DS / "train/train_ground_truth.tsv")
    gt = gt[~config.is_holdout(gt.source1_entity_id.values)]
    s1 = s1[~config.is_holdout(s1.entity_id.values)].set_index("entity_id")
    rng0 = random.Random(a.seed)
    picks = [gt.sample(a.n_s1, random_state=a.seed + k) for k in range(a.samples)]
    need = set()
    for pk in picks:
        for m in pk.matched_entity_ids:
            need.update(x for x in m.split(",") if x)
    recs = {}
    for n in (2, 3):
        for ch in load_tsv(DS / f"train/train_source{n}.tsv", chunksize=500_000):
            ch = ch[ch.entity_id.isin(need)]
            recs.update(zip(ch.entity_id, zip(ch.business_name, ch.business_address, ch.country)))
    print(f"real matched records loaded: {len(recs):,}", flush=True)

    al1, al2 = [], []
    for pk in picks[:1]:
        for sid, m in zip(pk.source1_entity_id, pk.matched_entity_ids):
            for r in m.split(","):
                if r in recs and sid in s1.index:
                    al1.append(s1.at[sid, "business_name"]); al2.append(recs[r][0])
    translit = learn_translit(al1, al2)
    print(f"translit dictionary: {len(translit):,} tokens", flush=True)

    fr = load_tsv(DS / "test/test_source1.tsv")
    fr = fr[fr.country == "France"].reset_index(drop=True)

    uid = iter(range(10 ** 9 + 1, 2 * 10 ** 9))
    out = Path(a.out)
    for k in range(a.samples):
        rng = random.Random(a.seed * 31 + k)
        lvl = [1.0, 1.6, 2.3][k] if k < 3 else 1.0 + 0.6 * k
        decoy_rate = [0.35, 0.6, 1.0][k] if k < 3 else 1.0
        rows1, rows2, rows3, gts = [], [], [], []
        stats = Counter()

        def add_rec(name, addr, country):
            rid = next(uid)
            src = 2 if rng.random() < 0.5 else 3
            (rows2 if src == 2 else rows3).append((f"S{src}-{rid}", name, addr, country))
            return f"S{src}-{rid}"

        def entity(name, addr, country, real_matches):
            sid = f"S1-{next(uid)}"
            rows1.append((sid, name, addr, country))
            matched = []
            for rn, ra, rc in real_matches:              # real noisy records (+ noise on noise)
                matched.append(add_rec(rn, ra, rc)); stats["real_pos"] += 1
                if rng.random() < 0.3 * lvl:
                    matched.append(add_rec(noise_name(rn, rc, lvl * 0.6, rng, translit),
                                           noise_addr(ra, rc, lvl * 0.6, rng), rc)); stats["noise_on_noise"] += 1
            if real_matches:
                n_syn = rng.choice([0, 1, 1, 2, 2, 3, 4])
            elif country == "France":
                n_syn = rng.choice([0, 2, 3, 3, 4, 5, 6, 7, 8])
            else:
                n_syn = rng.choice([0, 1, 2, 3, 3, 4, 5, 6])
            for _ in range(n_syn):
                matched.append(add_rec(noise_name(name, country, lvl, rng, translit),
                                       noise_addr(addr, country, lvl, rng), country)); stats["syn_pos"] += 1
            for _ in range(int(decoy_rate * 3) + (rng.random() < (decoy_rate * 3) % 1)):
                dn, da = make_decoy(name, addr, country, rng)
                add_rec(noise_name(dn, country, lvl * 0.5, rng, translit), noise_addr(da, country, lvl * 0.3, rng),
                        country); stats["decoy"] += 1
            gts.append((sid, ",".join(matched)))
            stats["s1_singleton"] += not matched

        for sid, m in zip(picks[k].source1_entity_id, picks[k].matched_entity_ids):
            if sid not in s1.index:
                continue
            n, ad, c = s1.loc[sid, ["business_name", "business_address", "country"]]
            real = [recs[r] for r in m.split(",") if r in recs]
            entity(n, ad, c, real)
            if rng.random() < 0.08 * decoy_rate:        # sibling S1: same name, different location, own matches
                entity(n, noise_addr(change_house(ad, rng), c, 0.3, rng), c, []); stats["sibling_s1"] += 1
        for _, r in fr.sample(min(a.n_fr, len(fr)), random_state=a.seed + k).iterrows():
            entity(r.business_name, r.business_address, "France", []); stats["france_s1"] += 1
            if rng.random() < 0.15 * decoy_rate:
                entity(r.business_name, noise_addr(change_house(r.business_address, rng), "France", 0.3, rng),
                       "France", []); stats["france_sibling_s1"] += 1

        d = out / f"aug{k + 1}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows1, columns=COLS).to_csv(d / "train_source1.tsv", sep="\t", index=False)
        for n, rows in ((2, rows2), (3, rows3)):
            df = pd.DataFrame(rows, columns=COLS).sample(frac=1, random_state=k)
            df.to_csv(d / f"train_source{n}.tsv", sep="\t", index=False)
        pd.DataFrame(gts, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
            d / "train_ground_truth.tsv", sep="\t", index=False)
        print(f"aug{k + 1}: noise x{lvl} decoy_rate {decoy_rate}: S1 {len(rows1):,}  S2 {len(rows2):,}  "
              f"S3 {len(rows3):,}  {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
