"""
Reference solution for Digitization Pipeline Attribution.

Approach: train a RandomForest classifier on hand-crafted OCR-noise features
over the 19 labeled TRAIN batches. For each TEST snippet (from one of 16
batches never seen in training), use the classifier's predicted probability
vector over the 19 known train classes as a learned embedding -- two
snippets from the same unseen batch should land on a similar distribution
over known fingerprint "neighbourhoods" even though the exact class is
novel. Within each bag, agglomerative-cluster those embeddings into a guessed
number of groups (~1 group per 6 snippets, swept and fixed on a held-out
slice of TRAIN, not on test).

Measured on the shipped test set: 0.316 mean per-bag Adjusted Rand Index,
against an oracle ceiling of 1.0 and a strongest-measured-adversary of 0.139
(see DESIGN.md). This is a reference, not a ceiling -- a from-scratch neural
encoder (work/char_encoder.py) was also tried and measured WEAKER (0.133) at
this data scale, which is reported honestly in DESIGN.md as real headroom
above this reference, not evidence the task is solved.
"""
import re
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier

COMMON_ENGLISH = set(
    "the of and to a in that is was he for it with as his on be at by i this had not "
    "are but from or have an they which one you were her all she there would their we "
    "him been has when who will more no if out so said what up its about into than them "
    "can only other new some could time these two may then do first any my now such like "
    "our over man me even most made after also did many before must through back years "
    "where much your way well down should because each just those people mr how too little "
    "state good very make world still see own men work long here get both between city "
    "under never day same another know while last might us great old year come since "
    "against go came right used take three".split()
)
CONFUSION_PAIRS = ["rn", "cl", "li", "ii", "vv", "nn", "ri"]

FEATURE_NAMES = [
    "len", "non_ascii_ratio", "digit_ratio", "punct_ratio", "alpha_ratio",
    "upper_mid_word_ratio", "oov_word_ratio", "single_char_token_ratio",
    "repeated_char_run_rate", "question_mark_rate", "confusion_pair_rate",
    "digit_letter_mix_rate", "avg_word_len", "n_words",
]


def extract_features(text: str) -> dict:
    if not text:
        text = ""
    n = len(text)
    words = text.split()
    n_words = max(1, len(words))
    alpha = sum(c.isalpha() for c in text)
    digit = sum(c.isdigit() for c in text)
    punct = sum(c in string.punctuation for c in text)
    non_ascii = sum(ord(c) > 127 for c in text)
    upper_mid_word = sum(1 for w in words if len(w) > 2 and any(c.isupper() for c in w[1:-1]))
    lower_words = [w.strip(string.punctuation).lower() for w in words]
    oov = sum(1 for w in lower_words if w and w.isalpha() and w not in COMMON_ENGLISH)
    single_char_tokens = sum(1 for w in words if len(w) == 1 and w.isalpha())
    repeated_char_runs = len(re.findall(r"(.)\1{2,}", text))
    question_marks = text.count("?")
    confusion_hits = sum(text.count(p) for p in CONFUSION_PAIRS)
    digit_letter_mix = len(re.findall(r"[a-zA-Z]\d|\d[a-zA-Z]", text))
    avg_word_len = sum(len(w) for w in words) / n_words
    return {
        "len": n, "non_ascii_ratio": non_ascii / max(1, n), "digit_ratio": digit / max(1, n),
        "punct_ratio": punct / max(1, n), "alpha_ratio": alpha / max(1, n),
        "upper_mid_word_ratio": upper_mid_word / n_words, "oov_word_ratio": oov / n_words,
        "single_char_token_ratio": single_char_tokens / n_words,
        "repeated_char_run_rate": repeated_char_runs / max(1, n),
        "question_mark_rate": question_marks / max(1, n),
        "confusion_pair_rate": confusion_hits / max(1, n),
        "digit_letter_mix_rate": digit_letter_mix / max(1, n),
        "avg_word_len": avg_word_len, "n_words": n_words,
    }


def vector(text: str) -> list:
    f = extract_features(text)
    return [f[k] for k in FEATURE_NAMES]


def guess_k(n_items: int) -> int:
    return max(2, round(n_items / 6))


def solve(data_dir: Path, out_path: Path) -> None:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    X_train = np.array([vector(t) for t in train["text"]])
    y_train = train["batch_id"].values
    clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)

    rows = []
    for bag_id, group in test.groupby("bag_id"):
        sids = list(group["snippet_id"])
        texts = list(group["text"])
        vals = np.array([vector(t) for t in texts])
        proba = clf.predict_proba(vals)
        k = max(1, min(guess_k(len(sids)), len(sids)))
        if k == 1:
            labels = [0] * len(sids)
        else:
            labels = AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average"
            ).fit_predict(proba)
        for sid, lab in zip(sids, labels):
            rows.append({"row_id": f"{bag_id}::{sid}", "bag_id": bag_id, "snippet_id": sid, "group_label": int(lab)})

    pd.DataFrame(rows).to_csv(out_path, index=False)


if __name__ == "__main__":
    data_dir = next(
        p for p in (Path("/data"), Path("dataset/public"), Path("./dataset/public"))
        if (p / "test.csv").exists()
    )
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("working/submission.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    solve(data_dir, out)
    print(f"wrote {out}")
