# Dataset Description

## Overview

238 consecutive days of hourly air quality measurements from Moli del Sol, a
regulatory air quality monitoring station in Valencia, Spain, together with the
output of two low-cost IoT sensor nodes that were physically co-located with
it for the whole period. Both the reference analysers and the cheap nodes were
sampling the same air, hour by hour, from 31 May 2024 to 23 January 2025.

The reference station runs laboratory-grade analysers of the kind used for
regulatory compliance reporting. The nodes carry a ZPHS01B multi-sensor module
costing a few tens of euros. The point of the deployment was to find out how
far the cheap hardware can be corrected toward the expensive hardware.

The dataset is used here to study a specific failure of cheap gas sensing: an
inexpensive electrochemical ozone cell is cross-sensitive, responding to
nitrogen dioxide, temperature and humidity as well as to ozone, in proportions
the manufacturer does not publish and that vary with conditions. Under some
conditions the node tracks the analyser closely; under others it diverges
badly. The task built on this dataset is to detect the divergent hours from
the node's own output alone - an anomaly detection problem, since a field
deployment has no reference analyser to compare against.

Data source: https://zenodo.org/records/17044034
DOI: 10.5281/zenodo.17044034
License: CC-BY 4.0 (commercial use permitted with attribution)
Collected by the dataset authors at the Moli del Sol AQMS, Valencian
Community, Spain. This is measured operational data, not simulated.

The files here are the hourly aggregation from that record, unmodified.

## File Structure

This dataset upload contains exactly three files at the top level:

- gva_60.csv - the regulatory station's hourly measurements, 5,679 rows.
- n1_60.csv - low-cost node 1, hourly, co-located, 5,678 rows.
- n2_60.csv - low-cost node 2, hourly, co-located, 5,679 rows.

All three are joined on the rounded_datetime column. The prepare.py script
submitted with the task builds the chronological train/test split from them,
publishes the node channels to solvers, and routes the reference ozone column
for the test period to the private grading key.

## Features

### gva_60.csv - regulatory station, 5,679 hourly rows

| Column | Type | Description |
|---|---|---|
| O3 | float | Ozone, ug/m3. Used to define the anomaly label, never published to solvers. |
| NO2 | float | Nitrogen dioxide, ug/m3. |
| NOx | float | Nitrogen oxides, ug/m3. |
| NO | float | Nitric oxide, ug/m3. |
| SO2 | float | Sulphur dioxide, ug/m3. |
| CO | float | Carbon monoxide, mg/m3. |
| PM10 | float | Particulate matter under 10 um, ug/m3. |
| PM2,5 | float | Particulate matter under 2.5 um, ug/m3. |
| PM1 | float | Particulate matter under 1 um, ug/m3. |
| PM10_S/C, PM2,5_S/C, PM1_S/C | float | Particulate channels, alternate processing. |
| SPL | float | Sound pressure level, dB. |
| Temp_Int | float | Internal enclosure temperature, deg C. |
| HR_Int | float | Internal enclosure relative humidity, percent. |
| HR | float | Relative humidity, percent. |
| Vac_Min | float | Instrument diagnostic. Empty for the entire record. |
| rounded_datetime | int64 | Hour of the observation, nanoseconds since epoch. |

Only the O3 column is used by this task, and only to construct the label; the
rest are documented because they are present in the file. The reference analysers have gaps for calibration and
maintenance: O3 is missing for about 6.8 percent of hours. Those hours cannot
be scored and are dropped by prepare.py rather than imputed.

### n1_60.csv and n2_60.csv - low-cost nodes, 5,678 / 5,679 hourly rows

Both files carry the same eleven channels. The two nodes are separate physical
units and disagree with each other: mean absolute difference between them is
1.06 deg C on temperature, 3.76 points on humidity, 5.76 ug/m3 on PM2.5 and
8.52 ug/m3 on the ozone channel. That unit-to-unit spread is real hardware
variation, not noise added for the task.

| Column | Type | Description |
|---|---|---|
| Temp | float | Temperature, deg C. |
| HR | float | Relative humidity, percent. |
| PM1 | float | Particulate matter under 1 um, ug/m3. |
| PM2_5 | float | Particulate matter under 2.5 um, ug/m3. |
| PM10 | float | Particulate matter under 10 um, ug/m3. |
| VOC | float | Volatile organic compounds index. |
| CH2O | float | Formaldehyde, mg/m3. |
| CO2 | float | Carbon dioxide, ppm. |
| CO | float | Carbon monoxide, ppm. |
| O3 | float | Ozone as reported by the node. Cross-sensitive, see Overview. |
| NO2 | float | Nitrogen dioxide as reported by the node. |
| rounded_datetime | int64 | Hour of the observation, nanoseconds since epoch. |

The node files have no missing values.

## Split Design

prepare.py splits chronologically, not randomly. Every hour up to and
including 9 November 2024 becomes training; 10 November 2024 to 23 January
2025 becomes test. This is deliberate and is the main source of difficulty:

- Training covers late spring, summer and autumn. Test is entirely winter.
- The anomaly label ranks each hour's divergence within its own 14-day
  neighbourhood and marks the worst quarter. Each window is scored using only
  hours inside it, so no test information reaches a training label.
- Ranking locally rather than against a fixed global cut is deliberate. The
  sensor drifts further from truth in cold air, so an absolute threshold would
  mark 10 percent of summer hours and 23 percent of winter hours, and the base
  rate would carry the season rather than the sensor's behaviour. Local ranking
  holds it at exactly 25.0 percent in both splits, which keeps a solver's own
  chronological hold-out informative about the test period.

A random split would leak neighbouring hours across the boundary and make the
task substantially easier, so it is not used.

## Measured Difficulty

All figures are average precision on the held-out winter period, computed with
the task's own grader.

| Approach | Average precision |
|---|---|
| Constant score (no skill) | 0.258 |
| Random score | 0.240 |
| The node's own ozone channel as the score | 0.257 |
| Disagreement between the two nodes | 0.246 |
| Local outlier factor | 0.349 |
| Logistic regression | 0.494 |
| Isolation forest | 0.537 |
| Random forest | 0.549 |
| Gradient boosting (reference solution) | 0.552 |

The channel whose reliability is being judged carries no information about its
own reliability - 0.257 against a no-skill floor of 0.258 - so a solver has to
discover that the signal lives in the humidity, temperature and co-pollutant
channels instead.

The design was checked for solvability the other way round as well: a
chronological hold-out taken from the training period ranks the tree ensembles
first, and they are also what wins on the real test period. A solver following
its own validation lands at 0.549, against the reference solution's 0.552, so
the reference is reachable from the public data rather than only in hindsight.

The ceiling is not artificial. A sensor of this cost does not carry enough
information to determine when it is wrong, so a substantial part of the
residual is irreducible.

## Sample-Size Summary

| Piece | Count |
|---|---|
| Hours with complete reference and both nodes | 5,281 |
| Training hours (31 May to 9 Nov 2024) | 3,651 |
| Test hours (10 Nov 2024 to 23 Jan 2025) | 1,629 |
| Node channels per hour | 22 (11 per node) |
| Hours dropped for missing reference ozone | 385 |
