"""Skip pipeline steps whose code, settings and inputs have not changed since their last success.

run_pipeline.sh calls
    python stages.py fresh <step>    exit 0 if the step can be skipped (outputs exist, fingerprint unchanged)
    python stages.py stamp <step>    after a successful run: record the fingerprint
FORCE=1 ./run_pipeline.sh ... ignores the stamps. Stamps live in dataset/processed/.stamps/.

Fingerprint = sha1 of the step's code files + relevant env settings + the command + (size, mtime) of its
inputs; the outputs must exist.
Steps that rewrite their inputs in place (prune) record the post-run state, so a rerun of the step
before them (new candidates) makes them stale again.
"""
import glob
import hashlib
import json
import os
import sys

import config as C

D = str(C.PROCESSED_DIR)
RAW = [str(C.source_path(sp, n)) for sp in ("train", "test") for n in (1, 2, 3)]
PROC = lambda sp: [str(C.processed_path(sp, n)) for n in (1, 2, 3)]
CAND = lambda sp: [os.path.join(D, f"candidates_{sp}", "*.parquet")]
MODELS = str(C.RUN_MODELS_DIR)
ENV = ["BER_EXTRA_FEATS", "BER_LR", "S1_FRAC", "FOLDS", "BER_K", "BER_K_REV", "BER_PRUNE_FIT_PAIRS", "BER_ONLY_COUNTRY", "BER_PROCESSED_DIR",
       "BER_MODELS_DIR"]
TEXT = ["text_utils.py", "canonical.py", "config.py"]

# step -> (code files, inputs, outputs); globs allowed
STAGES = {
    "learn": (["learn_suffixes.py"] + TEXT, RAW, [str(C.LEARNED_TOKENS), str(C.NAME_VOCAB)]),
    "preprocess": (["preprocess.py", "transliterate.py", "lang_id.py"] + TEXT,
                   RAW + [str(C.LEARNED_TOKENS), str(C.NAME_VOCAB), os.path.join(D, "translit_vocab_*.tsv")],
                   PROC("train") + PROC("test")),
    "preprocess_train": (["preprocess.py", "transliterate.py", "lang_id.py"] + TEXT,
                         RAW[:3] + [str(C.LEARNED_TOKENS), str(C.NAME_VOCAB), os.path.join(D, "translit_vocab_*.tsv")],
                         PROC("train")),
    "block_train": (["blocking.py"] + TEXT, PROC("train") + [str(C.TRAIN_GT)],
                    CAND("train") + [os.path.join(D, "blocked_s1_train.parquet")]),
    "block_dev": (["blocking.py"] + TEXT, PROC("train") + [str(C.TRAIN_GT)], CAND("train")),
    "prune_train": (["prune.py", "matching.py"] + TEXT, CAND("train"), CAND("train") + [os.path.join(MODELS, "pruner.joblib")]),
    "match_train": (["matching.py"] + TEXT, CAND("train") + PROC("train"), [os.path.join(MODELS, "stage4_model.joblib")]),
    "match_dev": (["matching.py"] + TEXT, CAND("train") + PROC("train"), [os.path.join(MODELS, "stage4_model.joblib")]),
    "block_test": (["blocking.py"] + TEXT, PROC("test"), CAND("test") + [os.path.join(D, "blocked_s1_test.parquet")]),
    "prune_test": (["prune.py", "matching.py"] + TEXT, CAND("test") + [os.path.join(MODELS, "pruner.joblib")],
                   CAND("test") + [str(C.OUTPUT_DIR / "candidate_pairs.tsv")]),
    "predict": (["matching.py"] + TEXT, CAND("test") + PROC("test") + [os.path.join(MODELS, "stage4_model.joblib")],
                [str(C.OUTPUT_DIR / "matching_results.tsv"), os.path.join(D, "scored_test.parquet")]),
    "package": (["package_submission.py"], [str(C.OUTPUT_DIR / "matching_results.tsv"), str(C.OUTPUT_DIR / "candidate_pairs.tsv")],
                [str(C.STUDENT_RESOURCE / "submission.zip")]),
}


def _files(patterns):
    out = []
    for p in patterns:
        out += sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
    return out


def _sig(paths):
    return [[p, os.path.getsize(p), os.stat(p).st_mtime_ns] if os.path.exists(p) else [p, None] for p in _files(paths)]


def fingerprint(step):
    code, inputs, outputs = STAGES[step]
    h = hashlib.sha1()
    for f in code:
        with open(os.path.join(C.SRC_DIR, f), "rb") as fh:
            h.update(fh.read())
    h.update(json.dumps({k: os.environ.get(k) for k in ENV}, sort_keys=True).encode())
    h.update(json.dumps(sys.argv[3:]).encode())                 # extra args passed by the runner
    # outputs only need to exist: later steps may rewrite them in place (prune rewrites candidates)
    return {"code_env": h.hexdigest(), "inputs": _sig(inputs)}


def stamp_path(step):
    return os.path.join(D, ".stamps", f"{step}.json")


def fresh(step):
    if step not in STAGES or not os.path.exists(stamp_path(step)):
        return False
    outputs = _files(STAGES[step][2])
    if not outputs or not all(os.path.exists(p) for p in outputs):
        return False
    with open(stamp_path(step)) as f:
        old = json.load(f)
    return old == fingerprint(step)


def stamp(step):
    if step not in STAGES:
        return
    os.makedirs(os.path.dirname(stamp_path(step)), exist_ok=True)
    with open(stamp_path(step), "w") as f:
        json.dump(fingerprint(step), f)


if __name__ == "__main__":
    cmd, step = sys.argv[1], sys.argv[2]
    if cmd == "fresh":
        sys.exit(0 if fresh(step) else 1)
    stamp(step)
