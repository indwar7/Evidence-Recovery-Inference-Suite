# Matching Body-Worn Motion Sensors to Their Wearers

## Overview

Several people each wore five motion sensors at once: one on the chest, one on each wrist, one on the right thigh and one on the left ankle. Each test bag puts three, four or five of these people together, all recorded during the same quiet activity: standing, sitting, or sitting and talking. For every person in a bag you get one chest recording, and that recording stands for the person. You also get every wrist, thigh and ankle recording of everyone in the bag, shuffled, with no indication of who wore it. For every limb recording, say whose chest recording it belongs with.

```
bag teb64d59e48   activity: sit_and_talk
candidates (chest): seb9853fd9a s42e0d3596b s299bfbe794 s420a781ef5 s5a8f6cf9f8
left_wrist   sd9d22e7836   ->  ?
left_wrist   s462262ffe6   ->  ?
left_wrist   s91abcc7c2b   ->  ?
left_wrist   s143df173a9   ->  ?
left_wrist   sd6407d8b5a   ->  ?
...          (the same for right_wrist, right_thigh, left_ankle)
```

Every recording is 4.0 seconds (205 samples) of a 3-axis accelerometer and a 3-axis gyroscope at 51.2 Hz. The answer is a fact the original archive already records: which participant's session each sensor file belongs to. The archive stores one file per participant per sensor, and that file structure is the label. Nobody annotated anything for this task, and no part of the label was inferred or generated.

## Task

Each row of `test.csv` is one limb recording. It gives:

- `stream_id`: the recording's id;
- `bag_id`: the bag it belongs to;
- `activity`: the bag's activity (`stand`, `sit` or `sit_and_talk`);
- `site`: where the sensor was worn (`left_wrist`, `right_wrist`, `right_thigh` or `left_ankle`);
- `candidates`: the chest recordings of everyone in the bag, as space-separated stream ids.

The samples of every recording in the test bags, chest and limb, are in `test_signals.csv`.

Produce `wearer`: the stream id of the chest recording of the person who wore that limb sensor. It is always one of that row's candidates.

The bag structure is fixed and known. In a bag of K people, every person contributes exactly one recording from each of the four limb sites, so there are 4K limb rows, and at each site every candidate is the right answer for exactly one row.

## What makes this hard

**Everyone in a bag is doing the same quiet thing.** The people in a bag were recorded at different times, but all during the same activity, and none of them is walking, climbing stairs or changing posture. Activity and posture tell them apart no better than chance. A sitting or standing body still moves a little, and that small residual motion is almost all the information these recordings carry.

**The clocks do not agree.** Each limb recording starts at its own offset of up to 2 s (102 samples) before or after its wearer's chest recording, drawn independently for every limb. That reproduces how the original devices' clocks drifted. A limb and its true chest recording overlap in time for between about half and all of the window, and nothing in the data says by how much. A direct sample-by-sample comparison is therefore never available.

**Orientation is gone.** Each channel's mean over its 4-second window has been removed from every recording in every bag. The way a sensor sat on the body and its constant offset are not available. Those identify a recording session rather than a body, and leaving them in would let a solution match sessions instead of bodies.

**The test people are new.** Training uses six people. The five test people appear nowhere in the training data. Nothing a solution learns about a particular person carries over to the test set. It has to learn how motion relates across different places on one body.

**Bags are groups, not pairs.** A bag holds 3, 4 or 5 people, and every limb has to be placed among all of them. Bag sizes vary, so a guess has a different chance of being right in each, and the score corrects for that (see Evaluation).

## What is new here

- **The label is who wore the sensor.** It comes from the source archive's own file structure, one file per participant per device, with each device's clock aligned to the chest device of the same session. No one labelled anything.
- **Every person in a bag is still.** Test bags hold 24 standing, 18 sitting and 16 sitting-and-talking groups. There is no walking or climbing in any bag, so there are no steps to line up and no gait rhythm to compare.
- **The clocks are offset.** Every limb recording carries its own offset of up to 2 s from its chest recording.
- **The test people are unseen.** Six people for training, five different people for test.
- **Several people at once.** Test bags hold 3, 4 or 5 people (19, 27 and 12 bags), and the output is a full assignment of 4K limb recordings to K people, not a yes-or-no decision about one pair.

## How it relates to prior work

Deciding whether two sensors are on the same body has been studied before, as a way to tell when several devices are carried by one person (Lester, Hannaford and Borriello, 2004), and later for wrist- and ankle-worn devices (Cornelius and Kotz, 2011). That work relies on the wearer walking. Recognising *where* on the body a sensor is worn is a related but different problem (Kunze, Lukowicz, Junker and Tröster, 2005). Here the site is given.

This task differs from that line of work in four ways:

- every person in a bag is sitting or standing, which removes the walking that earlier work depends on;
- the sensors' clocks disagree by up to 2 s, independently per limb;
- the output is a one-to-one assignment of many recordings within a group, not a decision about one pair;
- the score is corrected for chance, so a constant or random answer scores about zero at every group size.

The recordings come from an activity-recognition dataset. This task does not ask which activity anyone was doing; that is given in every row.

## Dataset

Discover the data directory from the environment rather than assuming a path. It holds:

- `train_recordings.csv`: whole recordings of the six training people, every activity, all five sites. Columns `recording` (one id per person), `site`, `i` (sample index on that person's shared clock), `activity`, and `ax`, `ay`, `az` (m/s²), `gx`, `gy`, `gz` (rad/s). Unlike the bags, these are aligned in time across sites and keep each channel's mean. Samples a sensor dropped are absent rows, so `i` has gaps.
- `train.csv`: 70 bags cut from the training people exactly as the test bags are cut from the test people, 1,096 limb rows. Columns `stream_id`, `bag_id`, `activity`, `site`, `candidates`, `wearer`.
- `train_signals.csv`: the samples of every recording in `train.csv`, chest and limb. Columns `stream_id`, `i` (0 to 204), `ax`, `ay`, `az`, `gx`, `gy`, `gz`.
- `test.csv`: 58 bags, 900 limb rows. Columns `stream_id`, `bag_id`, `activity`, `site`, `candidates`.
- `test_signals.csv`: the samples of every test recording, in the same layout as `train_signals.csv`.
- `sample_submission.csv`: `stream_id`, `wearer`, with every limb assigned to its bag's first candidate. It is a valid submission and shows the required layout; by construction it scores at the floor.
- `dataset_stats.json`: split sizes, bags by size and by activity, the window length, the sample rate and the maximum clock offset.
- `validate_submission.py`: an offline structural check of a submission file. It needs no answers.

Stream ids, bag ids and row order come from a salted hash and carry no information. The order of `candidates` within a row is shuffled and carries no information either.

## Rules

**Use only the data directory.** The recordings come from a public research dataset. Matching test recordings against any copy of the original recordings, or against any other source of recordings of these people, is not solving the task. A submission built that way is invalid regardless of its score, and so is any submission that depends on data not shipped in the data directory. Nothing in the data can make such a comparison impossible, so this rule is what closes that route. The solver environment is expected to have no network access, and your solution must run to completion from the data directory alone.

**Treat every test bag on its own.** The same five people appear in many test bags. Grouping test recordings across bags, fitting anything on the test recordings as a collection, or using one bag's answers to decide another's is not allowed. Each bag's predictions must come from that bag's own recordings plus whatever was learned from the training data.

- Learn from the provided data only. No pretrained models, no LLMs and no external data of any kind.
- Use only libraries available in the Kaggle Python Docker image.
- No LLM-generated outputs may be used anywhere in your solution.
- Predictions must come from a model fitted to the training data. A hand-set rule or formula on its own is not a valid solution. Features designed by hand may be fed to a fitted model.
- Your notebook must run end to end, top to bottom: load the data, fit, then write `submission.csv`.
- Seed everything; your run should be reproducible.

**What not to use:** pretrained models or embeddings of any kind, task-specific libraries, external recordings or datasets, any copy of the source recordings, network access, and anything fitted on the test recordings or carried from one test bag to another.

## Evaluation

**Chance-corrected wearer accuracy**, averaged over all limb rows of the test set. Range 0 to 1, higher is better.

Each row is either right (the submitted `wearer` is the chest recording of the person who wore the limb sensor) or wrong. A right answer is worth 1. A wrong answer costs 1/(K − 1), where K is the number of people in that row's bag: 0.5 in a bag of three, 0.333 in a bag of four and 0.25 in a bag of five. A uniform guess among K candidates is right with probability 1/K, so its expected value is 1/K − (1 − 1/K)/(K − 1) = 0 whatever the bag size. The score is the mean over all rows, and a negative mean is reported as 0.

**Why chance-corrected accuracy.** Bags have three, four or five people, so a guess is right a third, a quarter or a fifth of the time. Two obvious alternatives were considered and rejected:

- *Plain accuracy.* The floor would depend on bag size and sit well above zero. A constant or random answer would earn roughly a quarter of the maximum, and a submission could gain points from the mix of bag sizes rather than from the recordings.
- *Whole-bag exact match* (a bag counts only if every one of its limbs is right). Most bags end up partly right, and this scoring gives a partly right bag nothing, discarding most of the evidence. It would also make big bags far harder than small ones for reasons that have nothing to do with the recordings.

Under the chosen metric, a constant answer (every limb assigned to the same candidate, as in `sample_submission.csv`) scores 0, and a random answer, one-to-one or not, scores about 0 in expectation at every bag size. Any score clearly above 0 has to come from the recordings.

## Submission

A CSV named `submission.csv` with exactly the columns `stream_id,wearer`, and one row per `stream_id` in `test.csv`, in any order.

```
stream_id,wearer
s3f1c9a2b7e,s0a7d41c95b
s91be20d4c3,s7c2e8f1a06
```

A missing row, a blank `wearer`, a `wearer` that is not one of the row's candidates, or a `stream_id` given more than once counts as wrong for that row; it does not void the submission. Unknown `stream_id`s are ignored. Only a file without the `stream_id` or `wearer` column is rejected. Run `python validate_submission.py submission.csv` to check the structure offline.

## Compute

CPU only: 10 cores, 62 GB RAM, 1.5 hours. No GPU.

The training data is small (six people, 70 bags, plus their full recordings), so this is a task about what is fitted, not about scale. Any model fitted to the training data that runs on CPU within the budget is acceptable. The whole test set is 900 limb rows over 58 bags, so prediction cost is negligible next to fitting.
