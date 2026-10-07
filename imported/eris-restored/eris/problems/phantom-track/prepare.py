"""Split raw generated data into public (released) and private (held-out) files.

public/  points_train.csv, matches_train.csv  -> supervised scenes with true links
         points_test.csv                       -> evaluation scenes, positions only
         sample_submission.csv                  -> identity baseline (legal, ~random)
private/ answers.csv                            -> hidden true links for the test scenes
"""
import os
import pandas as pd


def prepare(raw_dir, public_dir, private_dir):
    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(private_dir, exist_ok=True)

    pts_tr = pd.read_csv(os.path.join(raw_dir, "points_train.csv"))
    mt_tr = pd.read_csv(os.path.join(raw_dir, "matches_train.csv"))
    pts_te = pd.read_csv(os.path.join(raw_dir, "points_test.csv"))
    mt_te = pd.read_csv(os.path.join(raw_dir, "matches_test.csv"))

    pts_tr.to_csv(os.path.join(public_dir, "points_train.csv"), index=False)
    mt_tr.to_csv(os.path.join(public_dir, "matches_train.csv"), index=False)
    pts_te.to_csv(os.path.join(public_dir, "points_test.csv"), index=False)

    # identity baseline: predict b_id = a_id for every query (frame-B is shuffled -> ~random)
    sample = mt_te[["scene_id", "a_id"]].copy()
    sample["b_id"] = sample["a_id"]
    sample.to_csv(os.path.join(public_dir, "sample_submission.csv"), index=False)

    mt_te.to_csv(os.path.join(private_dir, "answers.csv"), index=False)
    return {
        "public/points_train.csv": len(pts_tr),
        "public/matches_train.csv": len(mt_tr),
        "public/points_test.csv": len(pts_te),
        "public/sample_submission.csv": len(sample),
        "private/answers.csv": len(mt_te),
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = prepare(os.path.join(here, "dataset", "raw"),
                  os.path.join(here, "prepared", "public"),
                  os.path.join(here, "prepared", "private"))
    print(out)
