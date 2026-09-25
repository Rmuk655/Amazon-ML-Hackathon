"""Native-script -> Roman transliteration at the *vocabulary* level.

Each unique (lang, token) is transliterated once and cached on disk in
dataset/processed/translit_vocab_<backend>.tsv (lang <TAB> token <TAB> roman).

Cache properties: persistent, resumable (append-only, flushed every N entries),
safe to interrupt (a truncated last line is ignored on load), portable (plain TSV).

The IndicXlit engine (fairseq) is created LAZILY, only if there are pairs missing
from the cache. With a complete cache, fairseq is never imported.

Windows/Colab workflow
  1. local:  python preprocess.py --splits train test --backend indicxlit --dump-vocab
             -> dataset/processed/pending_vocab.tsv   (small file, no fairseq needed)
  2. Colab/WSL/Kaggle:  python transliterate.py --fill pending_vocab.tsv --backend indicxlit
             -> translit_vocab_indicxlit.tsv
  3. copy that TSV back to dataset/processed/ and run preprocess.py normally.
"""
import argparse
import re
from pathlib import Path

from tqdm import tqdm
from unidecode import unidecode

import config as C

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _fallback(text: str) -> str:
    return _NON_ALNUM.sub("", unidecode(text).lower()) or text


class Transliterator:
    def __init__(self, backend="unidecode", beam_width=C.BEAM_WIDTH, cache_only=False):
        if backend not in ("unidecode", "indicxlit"):
            raise ValueError(f"unknown backend {backend}")
        self.backend, self.beam_width, self.cache_only = backend, beam_width, cache_only
        self.cache_path = Path(C.PROCESSED_DIR) / f"translit_vocab_{backend}.tsv"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache, self.n_fail, self._engine = {}, 0, None
        self._load_cache()

    @property
    def engine(self):
        if self._engine is None:
            try:
                from ai4bharat.transliteration import XlitEngine
            except ImportError as e:
                raise RuntimeError(
                    "IndicXlit is not installed here and the cache is missing entries. "
                    "Generate them elsewhere (see module docstring) or pass --cache-only."
                ) from e
            self._engine = XlitEngine(src_script_type="indic", beam_width=self.beam_width, rescore=False)
        return self._engine

    def _load_cache(self):
        if not self.cache_path.exists():
            return
        with open(self.cache_path, encoding="utf-8") as f:
            for line in f:
                if not line.endswith("\n"):      # truncated by an interrupt
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    self.cache[(parts[0], parts[1])] = parts[2]
        print(f"[translit] cache: {len(self.cache):,} entries from {self.cache_path.name}")

    def _one(self, lang, text):
        """-> (romanized, cacheable). Engine failures are not cached, so they retry later."""
        if self.backend == "indicxlit":
            try:
                out = self.engine.translit_word(text, lang_code=lang, topk=1)
                if isinstance(out, dict):
                    out = next(iter(out.values()))
                if isinstance(out, (list, tuple)):
                    out = out[0]
                r = _NON_ALNUM.sub("", str(out).lower())
                if r:
                    return r, True
            except RuntimeError:
                raise
            except Exception:
                pass
            self.n_fail += 1
            return _fallback(text), False
        return _fallback(text), True

    def missing(self, pairs):
        return {p for p in pairs if p not in self.cache}

    def romanize(self, pairs):
        """Fill the cache for `pairs` [(lang, native_token)] and return it."""
        todo = sorted(self.missing(pairs))
        if not todo:
            return self.cache
        if self.backend == "indicxlit" and self.cache_only:
            print(f"[translit] WARNING: {len(todo):,} pairs not in cache; using unidecode fallback (--cache-only)")
            for p in todo:
                self.cache[p] = _fallback(p[1])
            return self.cache
        print(f"[translit] {len(todo):,} new (lang, token) pairs [{self.backend}]")
        with open(self.cache_path, "a", encoding="utf-8") as f:
            try:
                for i, (lang, text) in enumerate(tqdm(todo, unit="tok")):
                    r, cacheable = self._one(lang, text)
                    self.cache[(lang, text)] = r
                    if cacheable:
                        f.write(f"{lang}\t{text}\t{r}\n")
                    if i % C.CACHE_FLUSH_EVERY == 0:
                        f.flush()
            finally:
                f.flush()
        if self.n_fail:
            print(f"[translit] {self.n_fail:,} engine failures -> unidecode fallback (not cached)")
        return self.cache


def read_pending(path):
    with open(path, encoding="utf-8") as f:
        return {tuple(l.rstrip("\n").split("\t")[:2]) for l in f if l.endswith("\n") and "\t" in l}


def write_pending(pairs, path):
    path = Path(path)
    new = (read_pending(path) if path.exists() else set()) | set(pairs)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for lang, tok in sorted(new):
            f.write(f"{lang}\t{tok}\n")
    tmp.replace(path)
    return len(new)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fill the transliteration cache from a pending-vocab TSV")
    ap.add_argument("--fill", required=True, help="TSV of 'lang<TAB>token' (from preprocess.py --dump-vocab)")
    ap.add_argument("--backend", default="indicxlit", choices=["indicxlit", "unidecode"])
    ap.add_argument("--beam-width", type=int, default=C.BEAM_WIDTH)
    a = ap.parse_args()
    Transliterator(a.backend, a.beam_width).romanize(read_pending(a.fill))