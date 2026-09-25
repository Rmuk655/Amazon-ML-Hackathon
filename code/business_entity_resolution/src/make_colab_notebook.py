"""Build a self-contained Colab notebook that fills the IndicXlit transliteration cache.

The notebook embeds config.py, transliterate.py and pending_vocab.tsv, so the only
upload is the .ipynb itself; its only output is translit_vocab_indicxlit.tsv, which
goes back into dataset/processed/.

  python preprocess.py --splits train test --backend indicxlit --dump-vocab
  python make_colab_notebook.py        # -> code/business_entity_resolution/colab/translit_colab.ipynb
"""
import base64
import json
from pathlib import Path

import config as C

OUT = C.CODE_DIR / "colab" / "translit_colab.ipynb"

INSTALL = r'''import subprocess, sys

def sh(cmd):
    print("$", cmd)
    return subprocess.run(cmd, shell=True).returncode

# IndicXlit needs fairseq 0.12.x, which only ships wheels up to Python 3.9 and fails to
# build on Colab's newer Python -> run it from a separate Python 3.9 venv made with uv.
PY = "/content/xl/bin/python"
TEST = f'{PY} -c "from ai4bharat.transliteration import XlitEngine"'
if sh(TEST) != 0:
    sh("pip install -q uv")
    assert sh("uv venv -q --seed -p 3.9 /content/xl") == 0
    sh(f'{PY} -m pip install -q "pip<24.1"')          # newer pip rejects omegaconf 2.0.x metadata
    sh(f"{PY} -m pip install -q torch==1.13.1 --index-url https://download.pytorch.org/whl/cpu")
    sh(f'{PY} -m pip install -q "numpy<1.24" ai4bharat-transliteration tqdm unidecode')
    assert sh(TEST) == 0, "IndicXlit import failed - see pip output above"
print("IndicXlit ready")
'''

RUN = r'''import os, subprocess
os.makedirs("/content/out", exist_ok=True)
env = dict(os.environ, BER_PROCESSED_DIR="/content/out")
# first call downloads the IndicXlit model (~few hundred MB); resumable if interrupted
subprocess.run(f"{PY} transliterate.py --fill pending_vocab.tsv --backend indicxlit",
               shell=True, env=env, check=True, cwd="/content/work")
'''

CHECK = r'''out = "/content/out/translit_vocab_indicxlit.tsv"
pend = {tuple(l.rstrip("\n").split("\t")[:2]) for l in open("/content/work/pending_vocab.tsv", encoding="utf-8")}
done = {}
for l in open(out, encoding="utf-8"):
    p = l.rstrip("\n").split("\t")
    if len(p) == 3:
        done[(p[0], p[1])] = p[2]
missing = pend - done.keys()
print(f"{len(done):,} cached / {len(pend):,} pending; missing {len(missing):,}")
for k in list(done)[:15]:
    print(k, "->", done[k])
if missing:
    print("re-run the previous cell to retry missing tokens:", list(missing)[:10])
'''

DOWNLOAD = r'''from google.colab import files
files.download("/content/out/translit_vocab_indicxlit.tsv")
'''


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def embed_cell():
    files = {"config.py": C.SRC_DIR / "config.py",
             "transliterate.py": C.SRC_DIR / "transliterate.py",
             "pending_vocab.tsv": C.PENDING_VOCAB}
    lines = ["import base64, os", 'os.makedirs("/content/work", exist_ok=True)', "FILES = {"]
    for name, p in files.items():
        lines.append(f'    "{name}": "{base64.b64encode(p.read_bytes()).decode()}",')
    lines += ["}", "for name, b in FILES.items():",
              '    open(f"/content/work/{name}", "wb").write(base64.b64decode(b))',
              'print(sorted(os.listdir("/content/work")))']
    return cell("code", "\n".join(lines) + "\n")


def main():
    n = sum(1 for _ in open(C.PENDING_VOCAB, encoding="utf-8"))
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": [
            cell("markdown", f"# IndicXlit vocabulary fill\n\nTransliterates the {n:,} pending "
                 "(lang, token) pairs with IndicXlit (CPU is fine) and downloads "
                 "`translit_vocab_indicxlit.tsv`.\n\nRuntime -> Run all, then copy the downloaded "
                 "file to `dataset/processed/` on the laptop and run "
                 "`python preprocess.py --splits train test --backend indicxlit`.\n"),
            embed_cell(),
            cell("code", INSTALL), cell("code", RUN), cell("code", CHECK), cell("code", DOWNLOAD),
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB, {n:,} pending pairs embedded)")


if __name__ == "__main__":
    main()
