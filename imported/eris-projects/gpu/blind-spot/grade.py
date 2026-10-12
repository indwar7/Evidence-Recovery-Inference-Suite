"""grade.py -- Blind Spot.

Score = mean over test rows of the row-normalised response the submitted
operation pair elicits from the hidden model:

    idx  = op1 * 20 + op2
    s_i  = (v_i[idx] - min(v_i)) / (max(v_i) - min(v_i))
    score = mean(s_i)

v_i is the length-400 vector of p_H(target_class | op2(op1(image))) for row i,
held in private/answers.csv as columns r0..r399. Per-row normalisation makes
rows of differing attainability commensurable: 1.0 is the best pair available
for that row, 0.0 the worst.

Raises only for genuinely invalid submissions -- missing columns, missing or
duplicated row ids, non-integer or out-of-range op indices. Anything that is a
well-formed choice gets scored, however poor.
"""
import numpy as np
import pandas as pd

N_OPS = 20


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    required = {"row_id", "op1", "op2"}
    missing = required - set(submission.columns)
    if missing:
        raise ValueError(f"submission missing column(s): {sorted(missing)}")

    sub = submission[["row_id", "op1", "op2"]].copy()

    if sub["row_id"].duplicated().any():
        dup = sub.loc[sub["row_id"].duplicated(), "row_id"].unique()[:5]
        raise ValueError(f"duplicate row_id(s) in submission, e.g. {list(dup)}")

    want = answers["row_id"].tolist()
    have = set(sub["row_id"])
    absent = [r for r in want if r not in have]
    if absent:
        raise ValueError(
            f"submission is missing {len(absent)} of {len(want)} required "
            f"row_id(s), e.g. {absent[:5]}")

    for col in ("op1", "op2"):
        vals = pd.to_numeric(sub[col], errors="coerce")
        if vals.isna().any():
            raise ValueError(f"column {col} contains non-numeric values")
        if (vals != vals.round()).any():
            raise ValueError(f"column {col} must contain integers")
        vals = vals.astype(int)
        if vals.min() < 0 or vals.max() >= N_OPS:
            raise ValueError(
                f"column {col} has values outside [0, {N_OPS - 1}]")
        sub[col] = vals

    sub = sub.set_index("row_id").loc[want]          # align to answer order
    idx = (sub["op1"].to_numpy() * N_OPS + sub["op2"].to_numpy())

    rcols = [f"r{k}" for k in range(N_OPS * N_OPS)]
    V = answers[rcols].to_numpy(dtype=np.float64)

    chosen = V[np.arange(len(V)), idx]
    lo = V.min(axis=1)
    hi = V.max(axis=1)
    span = np.maximum(hi - lo, 1e-9)

    return float(np.mean((chosen - lo) / span))


if __name__ == "__main__":
    import sys
    s = pd.read_csv(sys.argv[1])
    a = pd.read_csv(sys.argv[2])
    print(f"{grade(s, a):.6f}")
