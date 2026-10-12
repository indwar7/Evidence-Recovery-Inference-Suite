"""
Grading contract
----------------
submission: columns [work_id, slot]
  - work_id    : the query work's id, matching test.csv's work_id column
  - slot       : an integer naming which of that row's candidate slots
                      (0-indexed, as listed in test.csv's candidate_* columns)
                      the solver believes shares the query's true research topic

answers: columns [work_id, slot]
  - work_id  : same join key as submission
  - slot     : the 0-indexed slot (among that row's own candidates) that is
               actually the query work's OpenAlex primary_topic

Both columns name the same object (a candidate slot index, 0..5) in the same
form (int). Score = accuracy over graded rows, chance-adjusted is NOT applied
here because every row has the same fixed candidate count (6), so raw
accuracy is already comparable row to row (chance = 1/6 for every row,
unlike variable-size candidate pools elsewhere in the portfolio).

Degrade, never reject:
  - a work_id in submission but absent from answers is ignored
  - a work_id in answers but absent from submission scores 0 for that row
  - a duplicated work_id in submission has ALL its rows discarded (scores 0)
  - a non-integer / out-of-range (not in 0..5) slot scores 0
  - only a missing required column raises InvalidSubmission
"""
import pandas as pd


class InvalidSubmission(ValueError):
    pass


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    required_sub_cols = {"work_id", "slot"}
    required_ans_cols = {"work_id", "slot"}
    if not required_sub_cols.issubset(submission.columns):
        raise InvalidSubmission(
            f"submission missing required columns: {required_sub_cols - set(submission.columns)}"
        )
    if not required_ans_cols.issubset(answers.columns):
        raise InvalidSubmission(
            f"answers missing required columns: {required_ans_cols - set(answers.columns)}"
        )

    sub = submission.copy()
    sub["work_id"] = sub["work_id"].astype(str)

    dup_mask = sub["work_id"].duplicated(keep=False)
    sub = sub[~dup_mask]

    sub = sub.drop_duplicates(subset="work_id", keep="first")
    sub_map = dict(zip(sub["work_id"], sub["slot"]))

    ans = answers.copy()
    ans["work_id"] = ans["work_id"].astype(str)

    n_total = len(ans)
    if n_total == 0:
        return 0.0

    n_correct = 0
    for work_id, true_slot in zip(ans["work_id"], ans["slot"]):
        pred = sub_map.get(work_id)
        if pred is None:
            continue
        try:
            pred_int = int(pred)
        except (ValueError, TypeError):
            continue
        if pred_int < 0 or pred_int > 5:
            continue
        try:
            true_int = int(true_slot)
        except (ValueError, TypeError):
            continue
        if pred_int == true_int:
            n_correct += 1

    return n_correct / n_total


if __name__ == "__main__":
    import pandas as pd

    answers = pd.DataFrame({"work_id": ["a", "b", "c", "d"], "slot": [0, 1, 2, 3]})

    exact = pd.DataFrame({"work_id": ["a", "b", "c", "d"], "slot": [0, 1, 2, 3]})
    assert grade(exact, answers) == 1.0, "exact match should score 1.0"

    one_wrong = pd.DataFrame({"work_id": ["a", "b", "c", "d"], "slot": [0, 1, 2, 0]})
    assert grade(one_wrong, answers) == 0.75, "one error out of four should score 0.75"

    none_match = pd.DataFrame({"work_id": ["a", "b", "c", "d"], "slot": [5, 5, 5, 5]})
    assert grade(none_match, answers) == 0.0, "all wrong should score 0.0"

    missing_row = pd.DataFrame({"work_id": ["a", "b", "c"], "slot": [0, 1, 2]})
    assert grade(missing_row, answers) == 0.75, "missing row should score 0 for that row, not raise"

    extra_row = pd.DataFrame(
        {"work_id": ["a", "b", "c", "d", "z"], "slot": [0, 1, 2, 3, 0]}
    )
    assert grade(extra_row, answers) == 1.0, "extra unknown id should be ignored"

    dup_id = pd.DataFrame(
        {"work_id": ["a", "a", "b", "c", "d"], "slot": [0, 1, 1, 2, 3]}
    )
    assert grade(dup_id, answers) == 0.75, "duplicated id should score 0 for that row, not use either value"

    bad_slot = pd.DataFrame({"work_id": ["a", "b", "c", "d"], "slot": [9, 1, 2, 3]})
    assert grade(bad_slot, answers) == 0.75, "out-of-range slot should score 0 for that row"

    non_numeric = pd.DataFrame({"work_id": ["a", "b", "c", "d"], "slot": ["x", 1, 2, 3]})
    assert grade(non_numeric, answers) == 0.75, "non-numeric slot should score 0 for that row, not raise"

    empty_sub = pd.DataFrame({"work_id": [], "slot": []})
    assert grade(empty_sub, answers) == 0.0, "empty submission should score 0.0"

    missing_col = pd.DataFrame({"work_id": ["a", "b", "c", "d"]})
    try:
        grade(missing_col, answers)
        assert False, "missing slot column should raise"
    except InvalidSubmission:
        pass

    print("all grade.py self-tests passed")
