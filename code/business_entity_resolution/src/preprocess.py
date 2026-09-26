"""Stage 1: safe load -> normalize -> script/language -> transliterate -> processed parquet.

    python preprocess.py --splits train --backend unidecode --limit 200000      # dev
    python preprocess.py --splits train test --backend indicxlit --dump-vocab   # collect vocab only
    python preprocess.py --splits train test --backend indicxlit                # full (cache must be complete)

Output columns (raw columns untouched):
    entity_id, business_name, business_address, country, country_norm,
    business_name_{norm,rom,script,lang,c4a,c4b,legal}, business_address_{norm,rom,script,lang}
Existing outputs are skipped unless --overwrite (resumable at file level; writes are atomic).
"""
import argparse
import csv
import os

import pandas as pd
from tqdm import tqdm

import config as C
from canonical import name_keys
from name_clean import NameCleaner, TokenRepair
from lang_id import NativeLID
from text_utils import (INDIC_RE, OcrFixer, dominant_indic_script, fold_latin, indic_part, normalize,
                        normalize_address, normalize_name, scripts_present, split_scripts, tokenize)
from transliterate import Transliterator, write_pending

FIELDS = (("business_name", normalize_name), ("business_address", normalize_address))


def load_ocr_fixer():
    """OCR repair for names needs the clean S1 vocabulary written by learn_suffixes.py."""
    try:
        with open(C.NAME_VOCAB, encoding="utf-8") as f:
            vocab = {w: int(c) for w, c in (l.rstrip("\n").split("\t") for l in f)}
        print(f"[ocr] name vocabulary: {len(vocab):,} tokens")
        return OcrFixer(vocab)
    except OSError:
        print("[ocr] no name vocabulary (run learn_suffixes.py) -> OCR repair off")
        return None


def read_source(path, limit=None) -> pd.DataFrame:
    # Only truly empty cells become NaN; a literal "null" stays a string.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""],
                       quoting=csv.QUOTE_NONE, encoding="utf-8", nrows=limit)


def segments(token, dom, lang):
    """Yield (script, segment, xlit_lang or None). Only Indic segments get a language."""
    for script, seg in split_scripts(token):
        if script in C.SCRIPT_DEFAULT_LANG:
            yield script, seg, (lang if (script == dom and lang) else C.SCRIPT_DEFAULT_LANG[script])
        else:
            yield script, seg, None


def build_lang_info(uniq_norms, lid: NativeLID):
    """norm string -> (dominant Indic script, xlit lang). LID only for ambiguous scripts."""
    info, to_lid = {}, []
    for s in uniq_norms:
        if not INDIC_RE.search(s):
            continue
        part = indic_part(s)
        dom = dominant_indic_script(part)
        if dom is None:
            continue
        info[s] = (dom, C.SCRIPT_DEFAULT_LANG[dom])          # unambiguous / fallback
        if dom in C.AMBIGUOUS_SCRIPTS and len(part) >= C.LID_MIN_CHARS:
            to_lid.append((s, dom, part))
    changed = 0
    if to_lid and lid.available:
        for (s, dom, _), (label, conf) in zip(to_lid, lid.predict([p for _, _, p in to_lid])):
            lang = lid.resolve(label, conf, dom)
            if lang and lang != info[s][1]:
                info[s] = (dom, lang)
                changed += 1
    print(f"[lang] indic strings={len(info):,} | sent to IndicLID={len(to_lid):,} | "
          f"differ from script default={changed:,}")
    return info


def romanize_string(s, dom, lang, cache) -> str:
    if s.isascii():
        return s
    out = []
    for tok in tokenize(s):
        if tok.isascii():
            out.append(tok)
            continue
        pieces = []
        for script, seg, l in segments(tok, dom, lang):
            if l is not None:
                pieces.append(cache.get((l, seg), seg))
            elif script == "Latn":
                pieces.append(fold_latin(seg))
            else:
                pieces.append(seg)                            # other scripts preserved
        out.append("".join(pieces))
    return " ".join(out)


def process_file(src, dst, lid, xlit, limit=None, dump_vocab=False):
    df = read_source(src, limit)
    if os.environ.get("BER_ONLY_COUNTRY"):             # dev: one country only (fast A/B runs)
        df = df[df["country"].fillna("").str.lower() == os.environ["BER_ONLY_COUNTRY"].lower()].reset_index(drop=True)
    print(f"[{src.name}] {len(df):,} rows")
    for col, fn in FIELDS:
        m = {s: fn(s) for s in df[col].dropna().unique()}
        df[f"{col}_norm"] = df[col].map(m).fillna("")
    df["country_norm"] = df["country"].map({s: normalize(s) for s in df["country"].dropna().unique()}).fillna("")

    uniq = pd.unique(pd.concat([df[f"{c}_norm"] for c, _ in FIELDS], ignore_index=True))
    lang_info = build_lang_info(uniq, lid)
    need = set()
    for s, (dom, lang) in lang_info.items():
        for tok in tokenize(s):
            if INDIC_RE.search(tok):
                need.update((l, seg) for _, seg, l in segments(tok, dom, lang) if l is not None)

    if dump_vocab:
        n = write_pending(xlit.missing(need), C.PENDING_VOCAB)
        print(f"[{src.name}] pending vocab now {n:,} pairs -> {C.PENDING_VOCAB}")
        return
    cache = xlit.romanize(need)

    rom = {s: romanize_string(s, *lang_info.get(s, (None, None)), cache) for s in tqdm(uniq, desc="romanize")}
    scr = {s: scripts_present(s) for s in uniq}
    lng = {s: lang_info[s][1] if s in lang_info else "" for s in uniq}
    cols = ["entity_id", "business_name", "business_address", "country", "country_norm"]
    for col, _ in FIELDS:
        n = df[f"{col}_norm"]
        df[f"{col}_rom"], df[f"{col}_script"], df[f"{col}_lang"] = n.map(rom), n.map(scr), n.map(lng)
        cols += [f"{col}_{k}" for k in ("norm", "rom", "script", "lang")]

    if OCR is not None:                               # '5ervices' -> 'services' before canonical keys
        u = df["business_name_rom"].unique()
        df["business_name_rom"] = df["business_name_rom"].map(dict(zip(u, map(OCR, u))))
        if os.environ.get("BER_NAME_CLEAN", "1") != "0":  # aliases, website forms, titles, glued words
            cleaner = NameCleaner(OCR.vocab)
            u = df["business_name_rom"].unique()
            df["business_name_rom"] = df["business_name_rom"].map(dict(zip(u, map(cleaner, u))))
        if os.environ.get("BER_TOKEN_REPAIR", "1") != "0":   # noisy-channel repair of non-vocabulary tokens
            rep = TokenRepair(OCR.vocab)
            u = df["business_name_rom"].unique()
            df["business_name_rom"] = df["business_name_rom"].map(dict(zip(u, map(rep, u))))
    rn = df["business_name_rom"]
    keys = {s: name_keys(s) for s in rn.unique()}
    for i, k in enumerate(("c4a", "c4b", "legal")):
        df[f"business_name_{k}"] = rn.map({s: v[i] for s, v in keys.items()})
    cols += ["business_name_c4a", "business_name_c4b", "business_name_legal"]

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".parquet.tmp")
    df[cols].to_parquet(tmp, index=False)
    os.replace(tmp, dst)                                      # atomic: no half-written outputs
    ind = df["business_name_script"].str.contains("|".join(C.SCRIPT_DEFAULT_LANG), regex=True).sum()
    print(f"[{src.name}] -> {dst.name} | names with Indic script: {ind:,} ({ind / len(df):.2%})")


OCR = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--sources", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--backend", choices=["indicxlit", "unidecode"], default="unidecode")
    ap.add_argument("--beam-width", type=int, default=C.BEAM_WIDTH)
    ap.add_argument("--limit", type=int, default=None, help="first N rows per file (dev runs)")
    ap.add_argument("--dump-vocab", action="store_true", help="only write pending_vocab.tsv (no transliteration)")
    ap.add_argument("--cache-only", action="store_true", help="never call IndicXlit; unknown tokens use unidecode")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--jobs", type=int, default=3, help="files processed in parallel (~10 GB RAM each)")
    a = ap.parse_args()

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    todo = []
    for split in a.splits:
        for n in a.sources:
            dst = C.processed_path(split, n)
            if dst.exists() and not a.overwrite and not a.dump_vocab:
                print(f"[skip] {dst.name} exists (use --overwrite)")
                continue
            todo.append((split, n))
    # files are independent -> run them in parallel (each ~10 GB peak); vocab dumping and
    # engine transliteration write shared files, so those stay sequential
    jobs = 1 if (a.dump_vocab or not a.cache_only and a.backend == "indicxlit") else max(1, a.jobs)
    if jobs > 1 and len(todo) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(jobs, len(todo))) as pool:
            pool.starmap(_run_one, [(sp, n, a) for sp, n in todo])
    else:
        for sp, n in todo:
            _run_one(sp, n, a)


def _run_one(split, n, a):
    global OCR
    if OCR is None:
        OCR = load_ocr_fixer()
    process_file(C.source_path(split, n), C.processed_path(split, n), NativeLID(),
                 Transliterator(a.backend, a.beam_width, a.cache_only), a.limit, a.dump_vocab)


if __name__ == "__main__":
    main()