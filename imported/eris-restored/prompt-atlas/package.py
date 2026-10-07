#!/usr/bin/env python3
"""package.py -- assemble the Desktop upload folder.

Writes ~/Desktop/prompt-atlas-upload/ containing

    prompt-atlas-dataset.zip    raw files at the TOP LEVEL of the zip, plus
                                the reference notebook
    prepare.py, grade.py
    problem-description.md, dataset-description.md, rubrics.md, config.yaml
    solution.ipynb
    UPLOAD-GUIDE.md

and checks every pasteable file for the characters that the web form mangles:
backticks, en/em dashes and any other non-ASCII.
"""

import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.expanduser("~/Desktop/prompt-atlas-upload")

ZIP_MEMBERS = [
    ("dataset/raw/atlas.csv", "atlas.csv"),
    ("dataset/raw/articles.csv", "articles.csv"),
    ("dataset/raw/components.csv", "components.csv"),
    ("dataset/raw/LICENSE_DATA.txt", "LICENSE_DATA.txt"),
    ("solution.ipynb", "solution.ipynb"),
]

COPY = ["prepare.py", "grade.py", "reference_solution.py", "solution.ipynb",
        "problem-description.md", "dataset-description.md", "rubrics.md",
        "config.yaml"]

PASTEABLE = ["problem-description.md", "dataset-description.md", "rubrics.md"]


def check_ascii():
    bad = []
    for f in PASTEABLE:
        text = open(os.path.join(HERE, f), encoding="utf-8").read()
        for i, ch in enumerate(text):
            if ord(ch) > 126 or ch == "`":
                line = text[:i].count("\n") + 1
                bad.append(f"{f}:{line} {ch!r} (U+{ord(ch):04X})")
    return bad


def main():
    missing = [s for s, _ in ZIP_MEMBERS
               if not os.path.isfile(os.path.join(HERE, s))]
    if missing:
        sys.exit("missing raw files: " + ", ".join(missing))

    bad = check_ascii()
    if bad:
        print("FORM-UNSAFE CHARACTERS:")
        for b in bad:
            print("  " + b)
        sys.exit(1)

    os.makedirs(DEST, exist_ok=True)
    zpath = os.path.join(DEST, "prompt-atlas-dataset.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in ZIP_MEMBERS:
            z.write(os.path.join(HERE, src), arc)      # top level, no folder

    for f in COPY:
        shutil.copy2(os.path.join(HERE, f), os.path.join(DEST, f))

    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    print(f"zip members (all top level): {names}")
    print(f"zip size: {os.path.getsize(zpath) / 1e6:.1f} MB")
    print(f"delivered -> {DEST}")


if __name__ == "__main__":
    main()
