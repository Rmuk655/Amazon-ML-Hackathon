#!/usr/bin/env bash
# Setup for Ubuntu/WSL: build tools, Python 3.12 venv, pinned pipeline deps.
# Run:  bash setup.sh          (WITH_MODELS=1 also fetches the IndicLID model; FULL=1 bash setup.sh also installs the root requirements.txt: torch, faiss, llama-cpp, ...)
# Then (from README.md):
#   cd code/business_entity_resolution && PYTHON=../../.venv/bin/python BER_WORKERS=8 ./run_v2.sh   # about 4 h on 8 vCPU / 64 GB
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

sudo apt-get update
sudo apt-get install -y build-essential cmake python3 python3-venv python3-pip

python3 -m venv .venv      # README pins Python 3.12
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
if [ "${FULL:-0}" = "1" ]; then
  .venv/bin/pip install -r requirements.txt
fi

# Dataset: NOT public in this repo. It is the Amazon ML Challenge 2026 (Business Entity Resolution) release.
# TODO: no public mirror found (Hugging Face/Kaggle search by name gave nothing official); download it from the
# challenge portal you registered on and place it as dataset/train/*.tsv and dataset/test/*.tsv.

# Optional IndicLID language-ID model (native-script fastText "FTN", 160 MB), official GitHub release; the pipeline
# falls back to script-default languages without it. Needs: pip install fasttext-wheel "numpy<2" (separate venv).
IDIR=code/business_entity_resolution/models/indiclid-ftn
if [ "${WITH_MODELS:-0}" = "1" ] && [ ! -d "$IDIR" ]; then
  mkdir -p "$IDIR"
  .venv/bin/python - "$IDIR" <<'PY'
import sys, urllib.request, zipfile, io
u = "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftn.zip"
zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(u).read())).extractall(sys.argv[1])
PY
fi
# Optional Qwen2.5-1.5B-Instruct Q4_K_M GGUF (about 1.1 GB) for src/second_look.py (path via BER_LLM):
#   .venv/bin/pip install huggingface_hub
#   .venv/bin/huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models/llm
# Optional IndicXlit transliteration (ai4bharat-transliteration is on PyPI, v1.1.3, but needs Python 3.9 and numpy<1.24;
# the pipeline runs it separately via code/business_entity_resolution/colab/translit_colab.ipynb):
#   pip install "numpy<1.24" ai4bharat-transliteration    (source: https://github.com/AI4Bharat/IndicXlit)
# WITH_MODELS=1 bash setup.sh downloads only the IndicLID model above.
[ -d dataset/train ] || echo "WARNING: dataset/train missing (see TODO above)."
echo "Done. Activate with: source .venv/bin/activate"
