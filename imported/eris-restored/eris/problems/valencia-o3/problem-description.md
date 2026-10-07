# Low-Cost Air Quality Sensor Reliability Detection

## Overview

A regulatory air quality station ran for 238 days with two low-cost IoT sensor
nodes bolted to its enclosure, sampling the same air. The station carries
laboratory-grade analysers. The nodes carry a multi-sensor module costing a few
tens of euros. Both report ozone.

Most of the time the cheap node is roughly right. Sometimes it is badly wrong -
not broken, not obviously stuck, just reporting an ozone concentration that has
little to do with the air around it. This happens because a cheap
electrochemical cell is cross-sensitive: its response mixes ozone with nitrogen
dioxide, temperature and humidity in proportions the manufacturer never
published. Under some conditions that mixture stays close to the truth. Under
others it does not.

Nobody deploying these nodes at scale can afford a reference analyser beside
each one. So the practical question is not "what is the true value" but
**"should I believe this reading right now?"**

Your task: for every hour, score how likely it is that the node's ozone reading
is untrustworthy - using only what the nodes themselves report.

## Task

An hour is **anomalous** when the node's divergence from the co-located
regulatory analyser falls in the worst quarter of its own 14-day
neighbourhood. The comparison is local, so exactly 25 percent of hours are
anomalous in every part of the record - the label marks when this sensor was
unusually wrong *for the conditions it was in*, not simply when the air was
cold.

You are given the label for every training hour. You are not given it for the
test hours, and you never see the analyser at all.

Output one continuous score per test hour, higher meaning more suspect. Only
the ranking matters - the scale is irrelevant.

## What makes this hard

**The channel under suspicion tells you nothing about itself.** Scoring hours
by the node's own ozone reading gives 0.257, against a no-skill floor of 0.258
- literally no better than a constant. Disagreement between the two nodes is
no better (0.246). Whatever predicts unreliability, it is not the reading whose
reliability is in question.

**Train and test are consecutive in time.** Every test hour falls after the
last training hour, and the test period is colder and darker than anything in
training. The cross-sensitivities that made the cell misread in warm air are
not the ones that make it misread in cold air, so anything you fit is being
extrapolated across that boundary.

Measured on this data with the official grader:

| Approach | Average precision |
|---|---|
| Constant or random score | 0.24 - 0.26 |
| The node's own ozone channel as the score | 0.257 |
| Disagreement between the two nodes | 0.246 |
| Local outlier factor | 0.349 |
| Logistic regression | 0.494 |
| Isolation forest | 0.537 |
| Random forest | 0.549 |
| Gradient boosting | 0.552 |

The gap between the first three rows and the rest is the whole task. Getting
from 0.26 to 0.49 means finding that the signal lives in humidity, temperature
and the co-pollutant channels. Getting past 0.49 means handling the season
boundary without overfitting to it.

## Data files

All public files are in ./dataset/public/

**train.csv** - 3,651 hourly rows covering the warm season.

| Column | Type | Description |
|---|---|---|
| timestamp | datetime | The hour of the observation. |
| n1_* | float | Node 1's eleven channels for that hour. |
| n2_* | float | Node 2's eleven channels for that hour. |
| is_anomalous | int | Label. 1 if that hour's divergence was in the worst quarter of its 14-day neighbourhood, else 0. Exactly 25 percent of rows. |

Each node reports these eleven channels, prefixed `n1_` or `n2_`:

| Channel | Unit | Description |
|---|---|---|
| Temp | deg C | Air temperature at the node. |
| HR | percent | Relative humidity. |
| PM1 | ug/m3 | Particulate matter under 1 um. |
| PM2_5 | ug/m3 | Particulate matter under 2.5 um. |
| PM10 | ug/m3 | Particulate matter under 10 um. |
| VOC | index | Volatile organic compound index. |
| CH2O | mg/m3 | Formaldehyde. |
| CO2 | ppm | Carbon dioxide. |
| CO | ppm | Carbon monoxide. |
| O3 | node units | Ozone as reported by the node. This is the channel whose reliability you are judging. |
| NO2 | node units | Nitrogen dioxide as reported by the node. |

**test.csv** - 1,629 hourly rows immediately following the training period.
Same columns as train.csv without `is_anomalous`.

**sample_submission.csv** - the required output format, with the same score for
every hour. It scores 0.258.

The two nodes are separate physical units and disagree with each other: on
average they differ by 1.06 deg C on temperature and 8.52 ug/m3 on their ozone
channels. That spread is real hardware variation, not injected noise.

## Metric

**Average precision** - the area under the precision-recall curve for
identifying the anomalous hours, computed over all 1,629 test hours.

Higher is better. A constant or random score gives roughly the positive rate,
about 0.25. Only the ranking of your scores matters; their scale does not.

Range: 0.0 to 1.0.

## Submission format

Write a CSV named `submission.csv` to `./working/` with exactly two columns:

```
timestamp,score
2024-11-10 00:00:00,0.13
2024-11-10 01:00:00,0.82
...
```

- One row for each timestamp in test.csv, 1,629 rows plus a header, in any
  order.
- `score` is your suspicion score for that hour, as a number. Higher means more
  likely anomalous. Any range is fine.
- A timestamp you omit is ranked last rather than voiding the submission;
  duplicated timestamps keep their first row and unknown timestamps are
  ignored. Only an unreadable file or a missing `timestamp` or `score` column
  is unscorable.

## Rules & environment

- Learn from the provided data only. No pretrained models, no LLMs, and no
  external data of any kind.
- Use only libraries available in the Kaggle Python Docker image (pandas,
  numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch and similar).
- No LLM-generated outputs may be used anywhere in your solution.
- Your notebook must run end to end, top to bottom: load the public data, fit,
  then write `./working/submission.csv`
- Seed everything; your run should be reproducible.
