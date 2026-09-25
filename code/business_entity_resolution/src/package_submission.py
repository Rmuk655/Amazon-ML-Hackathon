"""Stage 5: validate outputs and build submission.zip in the required layout.

submission.zip
  output/matching_results.tsv
  output/candidate_pairs.tsv
  code/business_entity_resolution/{src/, README.md, requirements.txt}
  Documentation_template.md

Usage:
  python package_submission.py                       # validate + zip
  python package_submission.py --check-only
  python package_submission.py --doc path/to/Documentation_template.md
"""
import argparse
import os
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CODE = os.path.join(ROOT, "code", "business_entity_resolution")
SKIP_FILES = {"profile_data.py"}   # analysis-only, not part of the submission
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", "venv", ".venv", "logs"}
SKIP_EXT = {".pyc", ".parquet", ".joblib", ".bin", ".zip", ".pkl"}
MAX_CODE_FILE_MB = 5

problems, warnings = [], []


def err(m):
    problems.append(m)
    print("  ERROR:", m)


def warn(m):
    warnings.append(m)
    print("  WARN: ", m)


def ok(m):
    print("  ok:   ", m)


def check_matching(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    print(f"matching_results.tsv: {len(df):,} rows, columns {list(df.columns)}")
    if len(df) == 0:
        err("matching_results.tsv is empty")
        return set()
    if df.iloc[:, 0].eq("").any():
        err("blank source1 IDs present")
    pairs = set()
    if df.shape[1] == 2 and "matched_entity_ids" in df.columns:
        if df.iloc[:, 0].duplicated().any():
            err("duplicate source1_entity_id rows in grouped format")
        for a, b in zip(df.iloc[:, 0], df["matched_entity_ids"]):
            for x in b.split(","):
                if x:
                    pairs.add((a, x))
    elif df.shape[1] == 2:
        pairs = set(zip(df.iloc[:, 0], df.iloc[:, 1]))
        if len(pairs) < len(df):
            err(f"{len(df) - len(pairs):,} duplicate pairs in long format")
    else:
        warn("unrecognised matching_results.tsv layout; skipped pair checks")
    ok(f"{len(pairs):,} distinct matched pairs, {df.iloc[:, 0].nunique():,} S1 entities")
    return pairs


def check_candidates(path, match_pairs):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    print(f"candidate_pairs.tsv: {len(df):,} rows, columns {list(df.columns)}")
    if df.shape[1] < 2:
        err("candidate_pairs.tsv needs at least two columns")
        return
    cand = {(a, x) for a, b in zip(df.iloc[:, 0], df.iloc[:, 1]) for x in b.split(",") if x}
    ok(f"{len(cand):,} distinct candidate pairs")
    if match_pairs:
        missing = match_pairs - cand
        if missing:
            warn(f"{len(missing):,} matched pairs are not in candidate_pairs.tsv "
                 f"(e.g. {next(iter(missing))}); the two files should be consistent")
        else:
            ok("every matched pair appears in candidate_pairs.tsv")
    size = os.path.getsize(path) / 1e6
    if size > 500:
        warn(f"candidate_pairs.tsv is {size:.0f} MB; check the upload size limit (lower --k in blocking)")


def collect_code_files():
    files = []
    for d, dirs, fs in os.walk(CODE):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in fs:
            p = os.path.join(d, f)
            if f in SKIP_FILES or os.path.splitext(f)[1].lower() in SKIP_EXT:
                continue
            if os.path.getsize(p) > MAX_CODE_FILE_MB * 1e6:
                warn(f"skipping large file {os.path.relpath(p, ROOT)}")
                continue
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=os.path.join(ROOT, "Documentation_template.md"))
    ap.add_argument("--out", default=os.path.join(ROOT, "submission.zip"))
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    mr = os.path.join(ROOT, "output", "matching_results.tsv")
    cp = os.path.join(ROOT, "output", "candidate_pairs.tsv")
    print("== outputs ==")
    pairs = set()
    for p in (mr, cp):
        if not os.path.exists(p):
            err(f"missing {os.path.relpath(p, ROOT)}")
    if os.path.exists(mr):
        pairs = check_matching(mr)
    if os.path.exists(cp):
        check_candidates(cp, pairs)

    print("== official validator (utils/validate_submission.py) ==")
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(ROOT, "utils", "validate_submission.py"),
                        "--matching", mr, "--candidate", cp,
                        "--test-dir", os.path.join(ROOT, "dataset", "test")], cwd=ROOT)
    if r.returncode != 0:
        err("official validator FAILED (see its output above)")

    print("== code ==")
    for rel in ("README.md", "requirements.txt", "src"):
        if not os.path.exists(os.path.join(CODE, rel)):
            err(f"missing code/business_entity_resolution/{rel}")
    code_files = collect_code_files() if os.path.isdir(CODE) else []
    ok(f"{len(code_files)} code files to include")
    req = os.path.join(CODE, "requirements.txt")
    if os.path.exists(req):
        lines = [l.strip() for l in open(req) if l.strip() and not l.startswith("#")]
        unpinned = [l for l in lines if "==" not in l]
        if unpinned:
            warn(f"unpinned requirements (brief asks for pinned): {', '.join(unpinned[:8])}")

    print("== documentation ==")
    if not os.path.exists(a.doc):
        err(f"missing documentation file {a.doc}")
    else:
        txt = open(a.doc, encoding="utf-8", errors="ignore").read()
        if any(t in txt for t in ("TODO", "<fill", "[fill", "TBD")):
            warn("documentation still contains TODO/placeholder markers")
        ok(f"{os.path.basename(a.doc)} found")

    print(f"\n{len(problems)} error(s), {len(warnings)} warning(s)")
    if problems:
        sys.exit(1)
    if a.check_only:
        return

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in (mr, cp):
            z.write(p, "output/" + os.path.basename(p))
        for p in code_files:
            z.write(p, os.path.relpath(p, ROOT).replace(os.sep, "/"))
        z.write(a.doc, "Documentation_template.md")
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
    with zipfile.ZipFile(a.out) as z:
        for n in z.namelist():
            print("  ", n)


if __name__ == "__main__":
    main()