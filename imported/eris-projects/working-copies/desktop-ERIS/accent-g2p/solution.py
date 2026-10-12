"""
Reference solution for "The Accent Translator" -- CPU baseline.

The source transcription is hidden, so the model works from the spelling and
the target accent: a letter-context (prev, letter, next) table per target
accent, with backoff to a context-free per-letter table.
"""
import argparse
import collections
import csv
import sys
import unicodedata

_STRIP = set("/ˈˌ.ːˑ ()[]")


def norm(s):
    return "".join(c for c in unicodedata.normalize("NFC", str(s)) if c not in _STRIP)


def align(a, b):
    """Levenshtein traceback -> list of (a_char|'', b_char|'') ops."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, ops = n, m, []
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append((a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            ops.append((a[i - 1], "")); i -= 1
        else:
            ops.append(("", b[j - 1])); j -= 1
    return ops[::-1]


def build_rules(rows):
    """Letter-context table per target accent: tgt_accent -> {(prev,letter,next): Counter(phonemes)}."""
    R = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    B = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for word, t, tp in rows:
        w = "#" + word.lower() + "#"
        tp = norm(tp)
        # proportional alignment: letter k -> slice of the phoneme string
        n, m = len(w) - 2, len(tp)
        for k in range(n):
            seg = tp[round(k * m / n):round((k + 1) * m / n)]
            R[t][w[k] + w[k + 1] + w[k + 2]][seg] += 1
            B[t][w[k + 1]][seg] += 1
    return R, B


def apply_rules(R, B, word, t):
    w = "#" + word.lower() + "#"
    out = []
    for k in range(len(w) - 2):
        c = R[t].get(w[k] + w[k + 1] + w[k + 2]) or B[t].get(w[k + 1])
        out.append(c.most_common(1)[0][0] if c else "")
    return "".join(out)


def read_csv(path):
    with open(path, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="dataset/public/train.csv")
    ap.add_argument("--test", default="dataset/public/test.csv")
    ap.add_argument("--answers", default="dataset/private/answers.csv",
                     help="optional; if present, prints the local score before writing")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    train = read_csv(args.train)
    test = read_csv(args.test)

    R, B = build_rules([(r["word"], r["tgt_accent"], r["phonemes"]) for r in train])

    preds = {}
    for r in test:
        preds[r["row_id"]] = apply_rules(R, B, r["word"], r["tgt_accent"])

    with open(args.out, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "phonemes"])
        w.writeheader()
        for row_id, ph in preds.items():
            w.writerow({"row_id": row_id, "phonemes": ph})
    print(f"wrote {args.out}: {len(preds)} rows", file=sys.stderr)

    try:
        from grade import grade
        import pandas as pd
        sub = pd.read_csv(args.out, keep_default_na=False)
        ans = pd.read_csv(args.answers, keep_default_na=False)
        print(f"local score: {grade(sub, ans):.4f}", file=sys.stderr)
    except Exception as e:
        print(f"(skipped local scoring: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
