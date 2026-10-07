# Challenge configuration — Compiling Amended Regulations
name: recompile
title: "Compiling Amended Regulations"
domain: natural-language-processing
dataset:
  name: recompile-dataset
  title: "eCFR Point-in-Time Section Pairs With the Amendments That Produced Them"
  short_title: "eCFR Point-in-Time Amendment Pairs"
  description: dataset/dataset-description.md
  generator: dataset/generator/build_raw.py    # deterministic scan, oldest-first
  source_data: "REAL government text end to end. The amendatory instructions are
    the agency's own words as published in the Federal Register (<AMDPAR>
    elements of the bulk XML). The before and after texts are the Office of the
    Federal Register's own point-in-time compilation, served by the eCFR
    versioner API. Ground truth is what the Code of Federal Regulations actually
    said on the two mornings either side of the change. No synthetic text, no
    annotation, no LLM output anywhere in the pipeline."
  source_url: "https://www.ecfr.gov/api/versioner/v1"
  source_url_secondary: "https://www.govinfo.gov/bulkdata/FR"
  distinctness: "The labelled content comes from the eCFR point-in-time API, not
    from the Federal Register; the Federal Register supplies only the <AMDPAR>
    instruction strings. The unit of data is a versioned PAIR -- one CFR section
    on two dates either side of one rule -- which no corpus of Federal Register
    documents contains. This is not citation extraction, not linking, and not
    classification or summarization over regulatory text."
  citation: "United States Government. Federal Register (govinfo bulk XML) and
    Code of Federal Regulations (eCFR point-in-time API), final rules published
    @@DATE_RANGE@@."
  license: "Public domain. Works of the United States Government are not subject
    to copyright in the United States (17 U.S.C. 105). Commercial use and
    redistribution are unrestricted. attribution.csv records the Federal Register
    document number and citation behind every record."
  raw_files:
    - dataset/raw/items.jsonl         # instructions + before/after text per (rule, section)
    - dataset/raw/attribution.csv     # the FR rule behind every record
    - dataset/raw/meta.json           # source URLs, date range, parameters, counts
problem_description: problem-description.md
rubrics: rubrics.md
prepare:
  script: prepare.py                  # exposes prepare(raw, public, private); seed 20260810
  public:
    - dataset/public/items/<item_id>.json       # group_id + before_text + instruction blocks
    - dataset/public/train.csv                  # item_id, after_text
    - dataset/public/test.csv                   # item_id
    - dataset/public/sample_submission.csv      # section unchanged (valid, scores 0.000)
    - dataset/public/validate_submission.py     # the grader's structural checks, offline
    - dataset/public/dataset_stats.json
  private:
    - dataset/private/answers.csv               # item_id, before_text, after_text
  split_note: "Split is DISJOINT BY CFR PART, not by item. Agencies re-run the
    same amendment shape year after year -- civil-penalty inflation adjustments
    most obviously -- so an item-level split would put one year's version in
    train and the next year's in test. At most one item per (title, section).
    prepare.py also drops sections the agency reprinted in full, for which the
    answer is the printed block copied out."
grade:
  script: grade.py                    # exposes grade(submission, answers)
  usage: "python grade.py submission.csv dataset/private/answers.csv"
  submission_file: submission.csv
  submission_columns: [item_id, after_text]
  metric: weighted_edit_operation_f1
  metric_formula: "Token-level edit scripts (difflib opcodes) are computed from
    the shared before text to the official after text and to the submitted text.
    Score per item is F1 over those operations, each weighted by the tokens it
    moves (tokens removed plus tokens added). Gold and predicted operations are
    aligned one-to-one rather than intersected exactly: a pair earns credit only
    if aimed at the same place (spans overlap, or start within max(3, 0.25*size)
    tokens), and then earns 0.5*span_IoU + 0.5*token_F1(replacement), so the
    right amendment with the wrong extent earns partial credit. Averaged over
    test items. Returning the input unchanged scores 0.000; returning the
    official text scores 1.000."
  direction: maximize
  min_score: 0.0
  max_score: 1.0
  # ---- anchors measured on the SHIPPED dataset by work/measure.py ---- #
  copy_input: @@copy@@                       # return the section unchanged
  printed_only: @@printed_only@@             # return only the text the agency printed
  append_printed: @@append_printed@@         # append the printed text to the section
  regex_splice: @@regex_splice@@             # regex address, no hierarchy
  paragraph_splice: @@paragraph_splice@@     # hierarchy splice, no `* * *` handling
  reference_solution_score: @@reference@@    # GRADED ON THE PRIVATE TEST SET
  oracle: 1.000
  agent_gate: "@@GATE_ONELINE@@"
  note_band: "Difficulty is EMPIRICAL. The metric was corrected twice after
    measurement. First: with unweighted operation counts, returning only the
    printed block scored 0.499, because one catastrophic deletion costs the same
    as one small edit; weighting operations by tokens moved dropped that baseline
    to 0.229. Second: with exact operation intersection the per-item score was
    near-binary -- the reference scored exactly 1.000 on 13 items and exactly
    0.000 on 8 of 30 -- so soft alignment replaced it, giving 26 distinct per-item
    values across 47 items. The grader was also made to degrade rather than raise
    on structural problems, because a rejected submission produces no score and a
    run with no score measures nothing. Four answer leaks were found by inspecting
    real items and closed before the freeze -- see DESIGN.md section 4."
  band_evidence: "Five faithful partial implementations span 0.400 / 0.419 /
    0.445 / 0.494 / 0.522 -- distinct and strictly ordered. Dropping the paragraph
    hierarchy entirely falls to 0.192, which is the intended penalty for skipping
    the central difficulty. Two measured agent runs on the previous split scored
    1.08x and 1.32x that split's reference, projecting to 0.56-0.69 here, inside
    the intended 0.4-0.7 band at both ends."
  split_stratification: "Test is steered to 65 percent compositional items (a
    `* * *` partial replacement, a run of more than two instructions, a
    redesignation, or a quoted removal). The corpus is 58 percent compositional
    and the first random part split drew only 45 percent, which inflated every
    score measured on it -- the reference scores 0.820 on simple items against
    0.439 on compositional ones. The 0.65 target was chosen by sweeping 0.45-0.80
    and measuring the ladder at each point: past ~0.70 every solver collapses to
    the same score and one ablation overtakes the reference. Train is steered too
    and holds 29 compositional items of 56, so the hard cases stay developable."
submission:
  filename: submission.csv
  columns: [item_id, after_text]
  rows: one per item_id in test.csv
reference_solution: solution.ipynb
environment: kaggle-python-docker-image   # CPU-only, <1.5h
