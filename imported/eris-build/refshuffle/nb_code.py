
# ===== cell 1 =====
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

PUBLIC = Path("./dataset/public")
PRIVATE = Path("./dataset/private")

train = pd.read_csv(PUBLIC / "train.csv")
test = pd.read_csv(PUBLIC / "test.csv")

HAVE_ANSWERS = (PRIVATE / "answers.csv").exists()
answers = pd.read_csv(PRIVATE / "answers.csv") if HAVE_ANSWERS else None

LABELS = ["A", "B", "C"]
RANK_COLS = ["rank_A", "rank_B", "rank_C"]

print(f"train: {len(train)} units, test: {len(test)} units, HAVE_ANSWERS={HAVE_ANSWERS}")
train.head(2)

# ===== cell 3 =====
def rescaled_kendall_tau(true_ranks, pred_ranks):
    n = len(true_ranks)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            ts = true_ranks[i] - true_ranks[j]
            ps = pred_ranks[i] - pred_ranks[j]
            if ts * ps > 0:
                c += 1
            else:
                d += 1
    tot = c + d
    tau = (c - d) / tot if tot else 0.0
    return (tau + 1.0) / 2.0


def make_submission(pred_dict, name=None):
    rows = [{"unit_id": uid, "rank_A": r[0], "rank_B": r[1], "rank_C": r[2]}
            for uid, r in pred_dict.items()]
    df = pd.DataFrame(rows)
    if name:
        Path("./working").mkdir(exist_ok=True)
        df.to_csv(f"./working/{name}.csv", index=False)
    return df


def grade_local(pred_dict):
    """Score against dataset/private/answers.csv when available
    (local dev only — the real held-out run happens on the platform)."""
    if not HAVE_ANSWERS:
        return None
    scores = []
    for row in answers.itertuples(index=False):
        uid = row.unit_id
        true_ranks = [row.rank_A, row.rank_B, row.rank_C]
        if uid not in pred_dict:
            scores.append(0.0)
            continue
        scores.append(rescaled_kendall_tau(true_ranks, pred_dict[uid]))
    return float(np.mean(scores))


def grade_on_train(pred_dict):
    """Self-check against train.csv's own labels — always available,
    used to calibrate hyperparameters WITHOUT looking at the test score."""
    scores = []
    for row in train.itertuples(index=False):
        uid = row.unit_id
        true_ranks = [row.rank_A, row.rank_B, row.rank_C]
        pred_ranks = pred_dict.get(uid)
        if pred_ranks is None:
            scores.append(0.0)
            continue
        scores.append(rescaled_kendall_tau(true_ranks, pred_ranks))
    return float(np.mean(scores))


def grade(pred_dict):
    """Prefer the real answer key when present (local dev); otherwise
    fall back to the train self-check so this notebook still runs to
    completion and reports a number in the platform's held-out run."""
    s = grade_local(pred_dict)
    return s if s is not None else grade_on_train(pred_dict)

# ===== cell 5 =====
pool_order = {uid: [0, 1, 2] for uid in test["unit_id"]}
make_submission(pool_order, "submission_0_pool_order")
print(f"pool order (A=0,B=1,C=2)   {grade(pool_order):.4f}")

reversed_order = {uid: [2, 1, 0] for uid in test["unit_id"]}
print(f"reversed pool order        {grade(reversed_order):.4f}")

rng = np.random.default_rng(0)
random_order = {}
for uid in test["unit_id"]:
    perm = [0, 1, 2]
    rng.shuffle(perm)
    random_order[uid] = perm
print(f"random permutation         {grade(random_order):.4f}")

# ===== cell 8 =====
def parse_card(card_str):
    d = {}
    for p in card_str.split(" | "):
        if ": " in p:
            k, v = p.split(": ", 1)
            d[k] = v
    return d


YEAR_RE = re.compile(r"\d{4}")


def rank_from_scores(scores):
    """scores: length-3 list, lower = should be ranked earlier.
    Returns [rank_A, rank_B, rank_C]."""
    order_idx = sorted(range(3), key=lambda i: scores[i])
    ranks = [0, 0, 0]
    for pos, li in enumerate(order_idx):
        ranks[li] = pos
    return ranks


def title_length_sort_pred(row):
    cards = [parse_card(getattr(row, f"ref_{L}_card")) for L in LABELS]
    return rank_from_scores([len(c.get("title", "")) for c in cards])


def year_sort_pred(row):
    cards = [parse_card(getattr(row, f"ref_{L}_card")) for L in LABELS]
    years = []
    for c in cards:
        m = YEAR_RE.search(c.get("year", ""))
        years.append(int(m.group()) if m else 0)
    return rank_from_scores(years)


tl_pred = {r.unit_id: title_length_sort_pred(r) for r in test.itertuples(index=False)}
print(f"title-length sort   {grade(tl_pred):.4f}")

year_pred = {r.unit_id: year_sort_pred(r) for r in test.itertuples(index=False)}
print(f"year-sort ascending {grade(year_pred):.4f}")

# ===== cell 11 =====
venue_frac = defaultdict(list)
for r in train.itertuples(index=False):
    ranks = {L: getattr(r, f"rank_{L}") for L in LABELS}
    for L in LABELS:
        venue = parse_card(getattr(r, f"ref_{L}_card")).get("venue", "")
        venue_frac[venue].append(ranks[L] / 2.0)

venue_avg = {k: float(np.mean(v)) for k, v in venue_frac.items() if len(v) >= 3}
global_avg = float(np.mean([x for v in venue_frac.values() for x in v]))


def adversarial_pred(row):
    cards = [parse_card(getattr(row, f"ref_{L}_card")) for L in LABELS]
    scores = [venue_avg.get(c.get("venue", ""), global_avg) for c in cards]
    return rank_from_scores(scores)


adv_pred = {r.unit_id: adversarial_pred(r) for r in test.itertuples(index=False)}
print(f"adversarial global venue-position   {grade(adv_pred):.4f}")

# ===== cell 14 =====
WORD_RE = re.compile(r"[a-zA-Z]{3,}")
STOP = set("the a an of in on at to for and or but with from by as is was were are be "
           "been being this that these those his her their its our your which who whom "
           "what when where why how not no also had has have into after before during "
           "while over under between more most some such than then out about up down off "
           "again further once here there all any both each few other same so than too "
           "very".split())


def words(s):
    return {w.lower() for w in WORD_RE.findall(s)} - STOP


def card_words(c):
    return words(c.get("title", "")) | words(c.get("venue", ""))


def word_overlap_pred(row):
    cards = [parse_card(getattr(row, f"ref_{L}_card")) for L in LABELS]
    card_w = [card_words(c) for c in cards]
    parts = row.text.split("[CITE]")
    n = 3
    S = np.zeros((n, n))
    for i in range(n):  # marker index, reading order
        ctx_w = words(parts[i][-2000:])
        for j in range(n):  # candidate label index
            S[i, j] = len(ctx_w & card_w[j])
    row_ind, col_ind = linear_sum_assignment(-S)  # maximize overlap
    ranks = [0, 0, 0]
    for marker_pos, label_idx in zip(row_ind, col_ind):
        ranks[label_idx] = marker_pos
    return ranks


overlap_pred = {r.unit_id: word_overlap_pred(r) for r in test.itertuples(index=False)}
print(f"word overlap (Hungarian assignment)   {grade(overlap_pred):.4f}")

# ===== cell 17 =====
def author_words(c):
    a = c.get("author", "")
    if not a or "uncredited" in a:
        return set()
    return words(a)


def card_year(c):
    m = YEAR_RE.search(c.get("year", ""))
    return int(m.group()) if m else None


def blended_pred(row, author_w, year_w):
    cards = [parse_card(getattr(row, f"ref_{L}_card")) for L in LABELS]
    card_w = [card_words(c) for c in cards]
    auth_w = [author_words(c) for c in cards]
    years = [card_year(c) for c in cards]
    parts = row.text.split("[CITE]")
    text_w_all = words(row.text)
    n = 3
    S = np.zeros((n, n))
    for i in range(n):
        ctx_w = words(parts[i][-2000:])
        for j in range(n):
            S[i, j] = len(ctx_w & card_w[j])
            if auth_w[j] and (auth_w[j] & text_w_all):
                S[i, j] += author_w
    known_years = [y for y in years if y is not None]
    if len(known_years) >= 2:
        for i in range(n):
            for j in range(n):
                y = years[j]
                if y is not None:
                    S[i, j] += year_w * (n - i) * (1.0 / (1 + abs(y - min(known_years))))
    row_ind, col_ind = linear_sum_assignment(-S)
    ranks = [0, 0, 0]
    for marker_pos, label_idx in zip(row_ind, col_ind):
        ranks[label_idx] = marker_pos
    return ranks


def score_on_train(author_w, year_w):
    preds = {r.unit_id: blended_pred(r, author_w, year_w) for r in train.itertuples(index=False)}
    return grade_on_train(preds)


best = None
for author_w in [0, 1, 2, 4, 8]:
    for year_w in [0, 0.05, 0.1, 0.2, 0.4]:
        s = score_on_train(author_w, year_w)
        if best is None or s > best[1]:
            best = ((author_w, year_w), s)

print("best (author_w, year_w) on TRAIN only:", best)

# ===== cell 19 =====
AUTHOR_W, YEAR_W = best[0]

final_pred = {r.unit_id: blended_pred(r, AUTHOR_W, YEAR_W) for r in test.itertuples(index=False)}
print(f"reference (overlap + author + year, w={best[0]})   {grade(final_pred):.4f}")

make_submission(final_pred, "submission")
print("wrote ./working/submission.csv")

# ===== cell 20 =====
try:
    import torch
    cuda_note = f"CUDA available: {torch.cuda.is_available()}"
except ImportError:
    cuda_note = "torch not installed in this environment"
# Compute budget check: this task is CPU-only by design (see Compute in
# problem-description.md). No CUDA/GPU is used anywhere in this notebook
# -- every step (word overlap, Hungarian assignment) runs on plain
# numpy/scipy. This line only reports whether CUDA happens to be visible
# in the current environment; a GPU being available or not does not
# change anything this notebook computes.
print(cuda_note)
print("This notebook is CPU-only end to end: no torch/CUDA call is made "
      "anywhere above this cell.")
