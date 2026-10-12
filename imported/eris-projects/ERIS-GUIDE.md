# Project Eris — Dataset & Challenge Creation Guide

Project Eris is an ML benchmark creation quest on **Shipd**. You build evaluation challenges that test how well AI agents approach real machine learning problems, and you get paid when your work passes review. Think of it as creating Kaggle-style competitions, except the format is designed so an AI agent can read the challenge, attempt a solution, and be scored automatically. Every contribution becomes training data for the next generation of AI models.

## Hard Rules (Non-Negotiable)

- **You cannot copy competitions from Kaggle in any form.**
- You **can** use publicly available Kaggle datasets that allow commercial use.
- **You may not use any LLM outputs as part of your submission.**

## Current Challenge Constraints (Must Follow)

1. **GPU-only** — the challenge must be designed to leverage GPU-accelerated computation (e.g. deep learning training/inference); the solver environment now provides a GPU rather than CPU-only compute.
2. **Time limit** — each solution has a maximum runtime of **1.5 hours**.
3. **Novelty score ≥ 6** — challenges scoring below 6 on novelty will be rejected.
4. **No tabular or regression challenges** — boring challenges will be rejected, and you will **not be allowed to create another one**. Be creative: use real-world datasets and make the challenge meaningful and impressive.
5. **Solver hardware** — solutions run on **1x NVIDIA A10G GPU (Standard ML Workloads tier)**, plus **10 CPU cores and 62 GB of RAM**. Design the task (data size, model complexity) to fit this budget within the time limit, and ensure the task genuinely benefits from GPU acceleration.

---

## Step 1 — Find or Create a Dataset

A dataset is the raw material for a challenge. Every challenge references an **accepted dataset** and includes a **prepare script** that transforms the raw data into public and private splits.

### What to think about when choosing data

Design around the *skills* the dataset should test, such as:
- Handling missing values
- Class imbalance
- Feature engineering
- Pitfalls that might trip up a naive agent

### Option A — Public data

- Document the **source URL** and verify the **license allows commercial use**.
- Ensure it is **legal to redistribute**.
- **Prioritize recent and less well-known datasets** (avoid famous/overused ones).

### Option B — Synthetic data

- Document the **generation process** and include the **reproducible script**.
- Ensure **realistic distributions and relationships**.
- Introduce **controlled complexity**: noise, missing values, edge cases.

### Raw upload structure

The raw dataset upload can follow any structure. For tabular data it can be as simple as:

```
data.csv            # Your source data, or multiple files
```

Exact file names/formats depend on your data — the key requirement is that **everything is documented in the dataset description**.

### Dataset Description (required sections)

```markdown
# Dataset Description

## Overview
Where does this data come from? What does it represent?

## File Structure
- data.csv - labeled data
- Any other files in the raw dataset

## Features
| Column | Type | Description       |
|--------|------|-------------------|
| id     | int  | Unique identifier |
| ...    | ...  | ...               |

## Notes
Domain-specific context the agent needs
```

### Dataset Checklist

- [ ] Dataset description documents all columns and types.
- [ ] Dataset has sufficiently many samples.
- [ ] Prepare script is reproducible and deterministic.
- [ ] License allows commercial use if using public data.
- [ ] Source URL is documented if using public data.

---

## Step 2 — Create the Challenge

A challenge is the competition an AI agent enters. It bundles:
1. A **problem description** (the prompt the agent sees)
2. A **grading script** (`grade.py`)
3. A **config file**

…all tied to an accepted dataset. When you create a challenge, you also define the **initial rubrics** that describe what good ML engineering looks like for this task.

> Note: when a challenge is created, an AI agent attempts it to establish a baseline score — so the task should be well-posed and meaningfully difficult.

### 2.1 Problem Description

The more precisely you define the task, the more meaningful the evaluation. Template:

```markdown
# [Challenge Title]

## Overview
[1-2 paragraphs: What is being predicted? What data is available?
What is the real-world context?]

## Evaluation
Submissions are scored using **[metric name]** such as AUC-ROC or RMSE.

~~~python
def evaluate(y_true, y_pred):
    # Your metric calculation, if non-standard
    return score
~~~

## Dataset
[Describe the prepared dataset structure: what files are in public/,
column descriptions, and data types.]

## Submission
Submit a CSV file with the following format:

| Column     | Type  | Description                    |
|------------|-------|--------------------------------|
| id         | int   | Row identifier from test.csv   |
| prediction | float | Predicted probability 0 to 1   |

**Requirements**
- Must contain exactly [N] rows, one per test sample.
- Include a header row.
```

### 2.2 Prepare Script (`prepare.py`)

Transforms the raw dataset into public and private splits. **Must be deterministic**: running it twice produces identical outputs.

```python
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path) -> None:
    # Read raw data
    # Perform train/test split with fixed random seeds
    # Write public/ with train.csv, test.csv, sample_submission.csv
    # Write private/ with answers.csv
```

> Only use libraries from the Kaggle Python Docker image: pandas, numpy, scikit-learn, xgboost, lightgbm, TensorFlow, PyTorch, and similar standard packages.

**Required output for tabular data:**

```
public/
|-- train.csv               # Labeled training data
|-- test.csv                # Unlabeled test data
+-- sample_submission.csv   # Example of expected format

private/
+-- answers.csv             # Test data with labels
```

**For image or audio data:**

```
public/
|-- train/                  # Training images/files
|-- test/                   # Test images/files, no labels
|-- train.csv               # Maps train filenames to labels
+-- sample_submission.csv

private/
+-- answers.csv             # Maps test filenames to ground truth
```

### 2.3 Grading Script (`grade.py`)

Compares an agent's predictions against private ground truth and returns a score. Without a solid grader, there is no way to tell whether an agent did well or just got lucky.

```python
import pandas as pd

def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score a submission against ground truth answers.
    Raise only for truly invalid submissions.
    """
    return 1.0
```

### Challenge Checklist

- [ ] Problem description is complete and unambiguous.
- [ ] An agent could solve this without asking clarifying questions.
- [ ] Evaluation metric is clearly defined, with a formula if non-standard.
- [ ] Submission format is precisely specified.
- [ ] `prepare.py` produces correct public/private splits and is deterministic.
- [ ] `grade.py` correctly scores valid submissions and handles edge cases gracefully.
- [ ] 5+ initial rubrics, majority REQUIRED or RECOMMENDED.
- [ ] Rubrics are specific to this task, not generic.

---

## Step 3 — Rubrics

Rubrics are the core of what makes a challenge valuable. A final score tells you whether an agent got the right answer, but not *how* it got there. Rubrics capture what good ML engineering looks like for this specific challenge.

### Rubric Types

| Type                | What It Covers                                          |
|---------------------|---------------------------------------------------------|
| DATA_HANDLING       | Data loading, cleaning, missing values, outliers, preprocessing |
| MODELING            | Model selection, architecture choices, hyperparameters  |
| FEATURE_ENGINEERING | Deriving features, selection, reduction                 |
| TRAINING            | Training process, validation strategy, avoiding leakage |
| CODE_QUALITY        | Code correctness, efficiency, no crashes                |
| AGENT_BEHAVIOR      | Exploration patterns, iteration, debugging approach     |
| COMMUNICATION       | Explanations, reasoning, documentation in agent output  |

### Importance Levels

| Importance  | When to Use                                       |
|-------------|---------------------------------------------------|
| REQUIRED    | Failure likely means invalid or broken solution   |
| RECOMMENDED | Recommended for solving optimally                 |
| UNIVERSAL   | Universally applicable optimizations              |

### Writing Good Rubrics

| Property       | What It Means                                    | Example |
|----------------|--------------------------------------------------|---------|
| Quantity       | Aim for 5+ rubrics; quality over quantity        | — |
| Specificity    | Ground rubrics in your dataset and task, not generic ML advice | Good: "handles the 23% class imbalance". Bad: "handles class imbalance appropriately." |
| Balance        | Include both positive and negative criteria      | "Should validate on a holdout set"; "should not train on test data" |
| Approach-neutral | Do not lock rubrics to a specific technique    | Good: "achieves AUC > 0.75". Bad: "uses XGBoost with max_depth=6" |
| Discrimination | Rubrics should separate good attempts from poor ones | Good: "uses cross-validation with k>=5". Bad: "attempts to solve the problem" |

### Rubric Anti-Patterns (avoid these)

| Anti-Pattern       | Example                                     | Why It Fails |
|--------------------|---------------------------------------------|--------------|
| Too vague          | "Explores data well"                        | Cannot be objectively verified |
| Tool-specific      | "Uses pandas profiling library"             | Penalizes valid approaches using different tools |
| Impossible to fail | "Attempts to solve the problem"             | Every submission passes, so there is no signal |
| Solution-specific  | "Uses the same feature engineering as the reference" | Other valid methods are penalized |
| Mostly UNIVERSAL   | 8 out of 10 rubrics are generic best practices | Does not distinguish this task from any other |

### Rubric Submissions (after challenge acceptance)

The creator defines the initial rubric; once the challenge is accepted, the community can propose improvements.

**Who can submit:** challenge creator (refine), solvers (first-hand insight), platform reviewers.

**Flow:**
1. **Create a rubric draft** — forks the current canonical rubric into a draft.
2. **Edit criteria** — add, update, or remove criteria (auto-saves).
3. **Run validation checks.**
4. **Submit for review** — once checks pass.

If a reviewer requests changes, the submission moves back to an editable state. When approved, changes merge into the canonical rubric with a diff computed against your starting version.

---

## Step 4 — Solution (`solution.ipynb`)

A solution is a Jupyter notebook that solves a challenge end-to-end. As a challenge creator, **your solution proves solvability and validates the grader**.

### Requirements

- **Self-contained:** runs end-to-end without manual intervention.
- Reads from `./dataset/public/` using **relative paths**.
- Writes only to `./working/`, usually `submission.csv`.
- Uses only standard Kaggle Docker libraries.
- Completes in **approximately 1 hour** on the configured runtime (hard cap: **1.5 hours**, GPU-accelerated: 1x NVIDIA A10G, 10 CPU cores / 62 GB RAM).

```python
import pandas as pd

train = pd.read_csv("./dataset/public/train.csv")
test = pd.read_csv("./dataset/public/test.csv")

submission.to_csv("./working/submission.csv", index=False)
```

### Iterative Submissions

Show progression, not just a single final answer. **3–5 complete submissions showing incremental improvement** is recommended:

1. **First submission — solid starting point:** data pipeline is correct, model runs end-to-end, results are reasonable.
2. **Middle submissions — incremental improvements:** feature engineering, hyperparameter tuning, or different model architectures.
3. **Final submission — best effort:** most refined submission with the strongest model.

### Submission Credits

| Parameter               | Value            |
|-------------------------|------------------|
| Max credits per problem | 6                |
| Refill rate             | 1 every 4 hours  |
| Full refill time        | 24 hours         |
| Scope                   | Per problem      |

Credits are consumed **only on submit**. Creating drafts, running checks, or editing does not cost credits.

### Solution Checklist

- [ ] Solution script runs end-to-end without errors.
- [ ] Uses only Kaggle Docker environment libraries.
- [ ] Reads from `./dataset/public/` and writes to `./working/submission.csv`.
- [ ] Achieves a reasonable score and validates that the task is solvable.
- [ ] Comments explain your reasoning at each step.
- [ ] Clear progression from baseline to optimized if submitting multiple versions.

---

## Platform Mechanics

### Creator Tiers

Challenge submissions are gated by a trust tier system. New users start with conservative limits that increase as challenges get accepted.

| Tier            | Pending Reviews | Weekly Submissions | How to Unlock         |
|-----------------|-----------------|--------------------|-----------------------|
| New User        | 1               | 5                  | Default               |
| Problem Creator | 3               | 20                 | 1 accepted challenge  |
| Trusted Creator | Unlimited       | Unlimited          | 3 accepted challenges |

> Focus on quality over quantity. One well-crafted challenge that gets accepted is worth more than many rushed submissions that get rejected.

### Challenge Closing

Challenges close automatically:
1. **10 distinct solvers** — once 10 different users each have a graded solution, a 24-hour closing countdown begins automatically.
2. **Countdown visible to all** — every solver can see the remaining time on the challenge page.
3. **Challenge closes** — no new submissions are accepted and the leaderboard is finalized.

### Status Lifecycle (applies to datasets, challenges, solutions, and rubrics)

| Status   | Meaning                                                                 |
|----------|-------------------------------------------------------------------------|
| Draft    | Work in progress. Edit freely. Not visible to reviewers.                |
| Pending  | Submitted for review. Automated checks run first, then a human reviewer evaluates. |
| Changes  | Reviewer requested changes. Review feedback, update, and resubmit.      |
| Accepted | Approved and finalized. Your contribution is live.                      |
| Rejected | Does not meet quality standards. Review feedback for details.           |

---

## Quick End-to-End Workflow

1. **Pick a skill focus** (imbalance, missing data, leakage traps, feature engineering…) — but **not a plain tabular/regression task**; aim for a creative, novel problem (text, images, audio, time series, graphs, ranking, etc.) that leverages the GPU and is solvable within 1.5 hours.
2. **Source a dataset** — recent, obscure, commercially licensed public data, or well-documented synthetic data. Prefer real-world data. Never copy a Kaggle competition.
3. **Write the dataset description** (Overview, File Structure, Features table, Notes) and upload the raw data.
4. **Write `prepare.py`** — deterministic, fixed seeds, produces `public/` (train, test, sample_submission) and `private/` (answers).
5. **Write the problem description** — unambiguous task, exact metric, exact submission format with row count.
6. **Write `grade.py`** — robust scoring, raises only for truly invalid submissions.
7. **Write 5+ task-specific rubrics** — majority REQUIRED/RECOMMENDED, approach-neutral, discriminating.
8. **Build `solution.ipynb`** — prove solvability, validate the grader, show baseline → optimized progression.
9. **Run checks, submit, and respond to reviewer feedback** (Draft → Pending → Accepted).
