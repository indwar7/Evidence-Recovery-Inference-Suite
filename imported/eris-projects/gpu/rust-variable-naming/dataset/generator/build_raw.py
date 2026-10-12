"""Reproduces dataset/raw from rust-lang/rust at the pinned commit.

Run from the dataset/generator directory:
    python3 build_raw.py --out ../raw
    python3 build_raw.py --out ../raw --from-clone /path/to/existing/checkout

Pinned commit: 76c90957b7e422c4b9c45192b0197214d7de5a54 (2026-10-09).

Subtrees, copied to the same repo-relative paths under --out:
  library/{core,alloc,std,coretests,alloctests}  the standard library and its
                                                 test crates
  compiler                                       rustc
  src/tools/rust-analyzer/crates                 rust-analyzer

A sparse, blobless, depth-1 clone downloads only these subtrees (~70 MB),
not the full ~1 GB repository. Deliberately excluded: library/stdarch
(macro-generated SIMD intrinsics whose locals are near-exclusively single
letters — see dataset-description.md Limitations) and the other src/tools/*
crates, kept out so the snapshot is three well-known codebases, not a grab-bag.

Post-processing, applied identically on every build:
  * every directory named `test_data` is removed — rust-analyzer's parser and
    syntax fixture corpora are synthetic snippets, not production code;
  * later byte-identical copies of a file are dropped (first in sorted-path
    order kept), so the snapshot contains no duplicate contents. Upstream has
    a few such pairs (os/nuttx/mod.rs == os/rtems/mod.rs, ...).
"""
import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/rust-lang/rust.git"
PINNED_COMMIT = "76c90957b7e422c4b9c45192b0197214d7de5a54"
SUBTREES = [
    "library/core",
    "library/alloc",
    "library/std",
    "library/coretests",
    "library/alloctests",
    "compiler",
    "src/tools/rust-analyzer/crates",
]
FIXTURE_DIR_NAMES = {"test_data"}


def _clone(into: Path) -> Path:
    clone_dir = into / "rust"
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--sparse", REPO_URL, str(clone_dir)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(clone_dir), "fetch", "--depth", "1", "origin", PINNED_COMMIT],
        check=True,
    )
    subprocess.run(["git", "-C", str(clone_dir), "checkout", PINNED_COMMIT], check=True)
    subprocess.run(
        ["git", "-C", str(clone_dir), "sparse-checkout", "set", *SUBTREES], check=True
    )
    return clone_dir


def _copy(clone_dir: Path, out_root: Path) -> None:
    for subtree in SUBTREES:
        src = clone_dir / subtree
        if not src.is_dir():
            raise FileNotFoundError(f"{src} missing from checkout")
        dst = out_root / subtree
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    for lic in ("LICENSE-APACHE", "LICENSE-MIT"):
        shutil.copy(clone_dir / lic, out_root / lic)


def _postprocess(out_root: Path) -> None:
    for d in sorted(p for p in out_root.rglob("*") if p.is_dir() and p.name in FIXTURE_DIR_NAMES):
        if d.exists():
            shutil.rmtree(d)
    seen = set()
    for f in sorted(p for p in out_root.rglob("*") if p.is_file()):
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        if digest in seen:
            f.unlink()
        else:
            seen.add(digest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir (dataset/raw)")
    ap.add_argument("--from-clone", help="existing checkout of the pinned commit with the subtrees present; skips the network clone")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.from_clone:
        _copy(Path(args.from_clone), out_root)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            _copy(_clone(Path(tmp)), out_root)
    _postprocess(out_root)
    print("done:", out_root)


if __name__ == "__main__":
    main()
