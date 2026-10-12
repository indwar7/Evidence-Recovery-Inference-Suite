"""grade(submission, answers) -> float

Both arguments are already-loaded pandas DataFrames (or anything exposing the
same column access). `submission` has columns [id, names]; `answers` has
columns [id, names]. `names` is a '|'-joined sequence of predicted/true
identifier names, one per masked variable in that row's function, in the same
declaration order the dataset's `masked` column presents them (VAR0, VAR1, ...).

Score is per-row position accuracy — for each test row, the fraction of
VARi slots whose predicted name exactly matches the true name — averaged
across all rows in `answers`. A row scores 0 if: its id is missing from the
submission, its id is duplicated in the submission, its predicted name count
does not match the true count, or any predicted field is blank. Only a
missing required column raises InvalidSubmission.
"""


class InvalidSubmission(ValueError):
    pass


def grade(submission, answers) -> float:
    sub_cols = set(getattr(submission, "columns", []))
    ans_cols = set(getattr(answers, "columns", []))
    if "id" not in sub_cols or "names" not in sub_cols:
        raise InvalidSubmission("submission must have columns: id, names")
    if "id" not in ans_cols or "names" not in ans_cols:
        raise InvalidSubmission("answers must have columns: id, names")

    sub_records = submission.to_dict("records")
    id_counts = {}
    sub_by_id = {}
    for r in sub_records:
        rid = r.get("id")
        id_counts[rid] = id_counts.get(rid, 0) + 1
        sub_by_id[rid] = r.get("names")

    duplicated_ids = {rid for rid, c in id_counts.items() if c > 1}

    ans_records = answers.to_dict("records")
    if not ans_records:
        return 0.0

    total_score = 0.0
    for r in ans_records:
        rid = r.get("id")
        gold_str = r.get("names")
        gold_names = gold_str.split("|") if gold_str else []
        n = len(gold_names)
        if n == 0:
            continue

        if rid in duplicated_ids or rid not in sub_by_id:
            total_score += 0.0
            continue

        pred_str = sub_by_id[rid]
        if pred_str is None or (isinstance(pred_str, float)):
            total_score += 0.0
            continue

        pred_names = str(pred_str).split("|") if pred_str else []
        if len(pred_names) != n:
            total_score += 0.0
            continue

        correct = sum(1 for p, g in zip(pred_names, gold_names) if p.strip() == g.strip() and p.strip() != "")
        total_score += correct / n

    return total_score / len(ans_records)


def _make_df(rows):
    import pandas as pd

    return pd.DataFrame(rows)


def _run_tests():
    answers = _make_df(
        [
            {"id": "a", "names": "foo|bar"},
            {"id": "b", "names": "baz"},
            {"id": "c", "names": "x|y|z"},
        ]
    )

    # Case 1: exact match everywhere -> 1.0
    sub = _make_df(
        [
            {"id": "a", "names": "foo|bar"},
            {"id": "b", "names": "baz"},
            {"id": "c", "names": "x|y|z"},
        ]
    )
    assert grade(sub, answers) == 1.0, grade(sub, answers)

    # Case 2: one wrong slot in row c (1/3 correct there), rest exact
    sub = _make_df(
        [
            {"id": "a", "names": "foo|bar"},
            {"id": "b", "names": "baz"},
            {"id": "c", "names": "x|y|WRONG"},
        ]
    )
    expected = (1.0 + 1.0 + 2 / 3) / 3
    assert abs(grade(sub, answers) - expected) < 1e-9, grade(sub, answers)

    # Case 3: missing row -> scores 0 for that row, others unaffected
    sub = _make_df(
        [
            {"id": "a", "names": "foo|bar"},
            {"id": "c", "names": "x|y|z"},
        ]
    )
    expected = (1.0 + 0.0 + 1.0) / 3
    assert abs(grade(sub, answers) - expected) < 1e-9, grade(sub, answers)

    # Case 4: wrong-length prediction -> 0 for that row
    sub = _make_df(
        [
            {"id": "a", "names": "foo"},
            {"id": "b", "names": "baz"},
            {"id": "c", "names": "x|y|z"},
        ]
    )
    expected = (0.0 + 1.0 + 1.0) / 3
    assert abs(grade(sub, answers) - expected) < 1e-9, grade(sub, answers)

    # Case 5: duplicated id -> 0 for that row, does not score as correct
    sub = _make_df(
        [
            {"id": "a", "names": "foo|bar"},
            {"id": "a", "names": "foo|bar"},
            {"id": "b", "names": "baz"},
            {"id": "c", "names": "x|y|z"},
        ]
    )
    expected = (0.0 + 1.0 + 1.0) / 3
    assert abs(grade(sub, answers) - expected) < 1e-9, grade(sub, answers)

    # Case 6: blank prediction field -> that slot not counted correct
    sub = _make_df(
        [
            {"id": "a", "names": "|bar"},
            {"id": "b", "names": "baz"},
            {"id": "c", "names": "x|y|z"},
        ]
    )
    expected = (0.5 + 1.0 + 1.0) / 3
    assert abs(grade(sub, answers) - expected) < 1e-9, grade(sub, answers)

    # Case 7: malformed/unknown id in submission does not raise, is ignored
    sub = _make_df(
        [
            {"id": "a", "names": "foo|bar"},
            {"id": "ZZZ_UNKNOWN", "names": "???"},
            {"id": "b", "names": "baz"},
            {"id": "c", "names": "x|y|z"},
        ]
    )
    assert grade(sub, answers) == 1.0, grade(sub, answers)

    # Case 8: missing required column raises InvalidSubmission (ValueError subclass)
    bad_sub = _make_df([{"id": "a", "wrong_col": "foo|bar"}])
    try:
        grade(bad_sub, answers)
        assert False, "expected InvalidSubmission"
    except InvalidSubmission:
        pass
    assert issubclass(InvalidSubmission, ValueError)

    print("all grade.py tests passed")


if __name__ == "__main__":
    _run_tests()
