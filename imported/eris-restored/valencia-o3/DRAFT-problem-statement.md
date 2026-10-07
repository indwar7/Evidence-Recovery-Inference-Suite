# The Channel That Lies: Reference-Grade Ozone from a Sensor That Cannot Measure It

## Overview

A regulatory air quality station in Valencia, Spain ran for 238 days with two
cheap IoT sensor nodes bolted to its enclosure, sampling the same air. The
station carries laboratory-grade analysers. The nodes carry a 30-euro
multi-sensor module. Both report ozone.

The node's ozone channel is nearly worthless. Fed straight through, it scores
worse than predicting the mean. Modelled on its own, it still scores below
zero. The reason is that cheap electrochemical cells are cross-sensitive: what
that channel actually reports is a blend of ozone, nitrogen dioxide,
temperature and humidity, in proportions the manufacturer never published and
that drift with the weather.

The information needed to reconstruct the true ozone concentration is present
in the node - just not in the channel labelled ozone. It is distributed across
the other channels, including ones with no chemical relationship to ozone at
all.

Your task: given the two nodes' full raw output, predict what the regulatory
analyser recorded.

## What makes this hard

The train and test splits are **consecutive in time, not shuffled**. You train
on 31 May to 9 November 2024 and predict 10 November to 23 January 2025. The
test period is colder, darker and lower in ozone than anything in training.
Every cross-sensitivity you fit on summer air is being extrapolated into
winter air.

This punishes the obvious moves. Measured on this data:

- Passing the node's ozone channel straight through: **-0.216**
- Modelling from that channel alone: **-0.065**
- Linear calibration on all channels: **0.386**
- Gradient boosting on all channels: **0.558**
- Adding hour-of-day: **0.590**
- Adding lagged and rolling-window features: **0.467**

That last line is the trap. Temporal features look like free signal and cost
you a fifth of your score, because the relationship they encode does not
survive the season change.

There is also a ceiling here that no method reaches. A 30-euro sensor does not
contain enough information to determine a laboratory measurement. Part of the
residual is irreducible, and it is not obvious from the data where the
learnable part ends.

## Task

Predict the reference station's hourly ozone concentration for every hour in
the test period, from the two nodes' readings for that hour.

## Data

Moli del Sol air quality station, Valencia, Spain. Source:
https://zenodo.org/records/17044034 (DOI 10.5281/zenodo.17044034),
CC-BY 4.0, commercial use permitted.

- `train.csv` - hourly rows, 31 May to 9 Nov 2024. Both nodes' 11 channels
  each (temperature, humidity, PM1, PM2.5, PM10, VOC, formaldehyde, CO2, CO,
  ozone, NO2), plus the reference station's ozone for that hour.
- `test.csv` - hourly rows, 10 Nov 2024 to 23 Jan 2025. Node channels only.
- Roughly 3,600 training hours and 1,500 test hours.

## Metric

Coefficient of determination (R squared) between predicted and reference
ozone over the test hours, floored at 0.

## Reference points

- Predict the training mean: 0.00
- Pass the node ozone channel through unchanged: 0.00 (floored from -0.216)
- Linear calibration: 0.39
- Reference solution: about 0.59

## Rules

- Learn from the provided data only. No pretrained models, no LLMs, no
  external data, no fetching the source dataset for the withheld period.
- Kaggle Python Docker image libraries only.
- The notebook must run end to end and write ./working/submission.csv.
- Seed everything.
