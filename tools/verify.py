#!/usr/bin/env python3
"""
verify.py — reproduce every published score anchor in this suite.

Each benchmark in `benchmarks/` publishes a ladder of measured scores in its
`config.yaml`. This script re-runs the shipped graders against the shipped
data and checks that the published numbers still hold, so a reviewer never has
to take a README table on trust.

    python tools/verify.py                 # floors only  (seconds)
    python tools/verify.py --reference     # floors + reference solutions (~15 min, CPU)
    python tools/verify.py --only accent-transfer

Exit code is 0 when every checked anchor reproduces within tolerance, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"

# Every entry is (expected, tolerance). Tolerance is 0.001 for deterministic
# scorers; the one loose tolerance is documented inline and explained in
# docs/BENCHMARKS.md.
SUITE = [
    {
        "slug": "accent-transfer",
        "title": "The Accent Translator",
        "metric": "change_segment_accuracy",
        "floor": {"name": "copy source IPA (sample_submission)", "expected": 0.112, "tol": 0.001},
        "grade": ["grade.py", "{submission}", "dataset/private/answers.csv"],
        "sample": "dataset/public/sample_submission.csv",
        "reference": {
            "name": "per-segment context rewrite table (CPU)",
            "cmd": ["solution.py", "--out", "{out}"],
            "expected": 0.768, "tol": 0.001,
        },
    },
    {
        "slug": "prompt-edit-attribution",
        "title": "The Edit That Moved the Answer",
        "metric": "positional_credit",
        "floor": {"name": "unranked candidates (sample_submission)", "expected": 0.337, "tol": 0.001},
        "grade": ["grade.py", "{submission}", "--answers", "dataset/private/answers.csv"],
        "sample": "dataset/public/sample_submission.csv",
        "reference": {
            "name": "ridge regression over behaviour fingerprints (CPU)",
            "notebook": "solution.ipynb",   # writes working/submission.csv
            "writes": "working/submission.csv",
            "expected": 0.493, "tol": 0.001,
        },
    },
    {
        "slug": "pipeline-attribution",
        "title": "Recovering OCR Batch Origin from Character Noise",
        "metric": "mean_per_bag_adjusted_rand_index",
        "floor": {"name": "every snippet its own group (sample_submission)", "expected": 0.000, "tol": 0.001},
        "grade": ["grade.py", "{submission}", "dataset/private/answers.csv"],
        "sample": "dataset/public/sample_submission.csv",
        "reference": {
            "name": "RandomForest-probability embedding + per-bag agglomerative (CPU)",
            "cmd": ["reference_solution.py", "{out}"],
            # RandomForest(n_jobs=-1) + AgglomerativeClustering drift by ~0.01
            # across sklearn versions and core counts; published value 0.342.
            "expected": 0.342, "tol": 0.015,
        },
    },
    {
        "slug": "experimental-order",
        "title": "The Order of Discovery",
        "metric": "mean_per_paper_position_accuracy",
        "floor": {"name": "shipped shuffled order (sample_submission)", "expected": 0.285, "tol": 0.001},
        "grade": ["grade.py", "{submission}", "dataset/private/answers.csv"],
        "sample": "dataset/public/sample_submission.csv",
        "reference": {
            "name": "pairwise TF-IDF logistic + aggregate-win decode (CPU)",
            "cmd": ["solution.py", "--out", "{out}"],
            "expected": 0.358, "tol": 0.001,
        },
    },
]

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def parse_score(stdout: str) -> float:
    """Graders print either a bare float or a one-key JSON object."""
    text = stdout.strip().splitlines()[-1].strip()
    if text.startswith("{"):
        return float(next(iter(json.loads(text).values())))
    return float(text)


def run_notebook(pkg: Path, name: str) -> None:
    """Execute a reference notebook's code cells in order, in the package dir.

    The notebooks are linear and side-effect-only at the end (they write a
    submission), so concatenating their code cells reproduces a Run All without
    pulling in a jupyter dependency.
    """
    cells = json.loads((pkg / name).read_text())["cells"]
    source = "\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    script = pkg / ".verify_notebook.py"
    script.write_text(source)
    try:
        run(pkg, [script.name])
    finally:
        script.unlink(missing_ok=True)


def run(pkg: Path, argv: list[str]) -> str:
    proc = subprocess.run([sys.executable, *argv], cwd=pkg, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} failed in {pkg.name}:\n{proc.stderr[-2000:]}")
    return proc.stdout


def check(label: str, got: float, expected: float, tol: float, seconds: float) -> bool:
    ok = abs(got - expected) <= tol
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label:<52} {got:.4f}  (published {expected:.3f}, tol {tol}) {DIM}{seconds:5.1f}s{RESET}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", action="store_true",
                    help="also retrain and re-score each CPU reference solution (slower)")
    ap.add_argument("--only", metavar="SLUG", help="verify a single benchmark")
    args = ap.parse_args()

    entries = [e for e in SUITE if not args.only or e["slug"] == args.only]
    if not entries:
        print(f"no benchmark named {args.only!r}; known: {', '.join(e['slug'] for e in SUITE)}")
        return 1

    results = []
    for entry in entries:
        pkg = BENCH / entry["slug"]
        print(f"\n{entry['slug']}  {DIM}{entry['title']} — {entry['metric']}{RESET}")

        t0 = time.time()
        argv = [a.format(submission=entry["sample"]) for a in entry["grade"]]
        results.append(check(entry["floor"]["name"], parse_score(run(pkg, argv)),
                             entry["floor"]["expected"], entry["floor"]["tol"], time.time() - t0))

        ref = entry["reference"]
        if not args.reference:
            continue
        t0 = time.time()
        if "notebook" in ref:
            run_notebook(pkg, ref["notebook"])
            out = pkg / ref["writes"]
        else:
            out = pkg / "submission.verify.csv"
            run(pkg, [a.format(out=out.name) for a in ref["cmd"]])
        argv = [a.format(submission=str(out.relative_to(pkg))) for a in entry["grade"]]
        results.append(check(ref["name"], parse_score(run(pkg, argv)),
                             ref["expected"], ref["tol"], time.time() - t0))
        if "notebook" not in ref:
            out.unlink(missing_ok=True)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} anchors reproduced.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
