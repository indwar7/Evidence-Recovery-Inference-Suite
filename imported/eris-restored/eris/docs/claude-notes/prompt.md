You are an expert Kaggle competition designer and ML benchmark researcher.

Your task is to design ONE competition that would be accepted as a high-quality machine learning benchmark.

## Hard Constraints

- CPU-only inference.
- Maximum inference time: 1.5 hours.
- Must NOT be a tabular prediction challenge.
- Must NOT be a standard regression challenge.
- Must use a real-world or realistically generated dataset.
- Dataset size should be practical for a public ML competition.
- The task must be deterministic.
- The evaluation metric must be deterministic.
- The challenge should be solvable using classical ML and modern deep learning methods.

## Goal

Do NOT optimize for existing competition formats.

Instead, invent a task that is fundamentally different from common competitions such as:

- image classification
- object detection
- OCR
- segmentation
- captioning
- sentiment analysis
- tabular prediction
- recommendation
- regression
- generic NLP classification

The task should require a new reasoning pipeline or representation.

## Required Output

1. Competition Title
2. One-paragraph Overview
3. Why this task is novel compared with existing ML competitions
4. Dataset Description
5. Input Format
6. Output Format
7. Ground Truth Generation
8. Evaluation Metric
9. Baseline Model
10. Expected CPU Runtime
11. Failure Cases
12. Why this benchmark is scientifically useful
13. Similar existing datasets (if any) and how this benchmark differs

## Self Review

After proposing the challenge, critically evaluate it.

Answer:

- Is this essentially a variation of an existing Kaggle competition?
- What aspects are genuinely new?
- Which parts are likely to be considered too common?
- What modifications would make it substantially more original?
- Rate the novelty from 1–10 and justify the rating.
- If the rating is below 8, redesign the challenge from scratch rather than making superficial changes.

Repeat this process until the proposed challenge is genuinely distinct and you can justify a novelty rating of at least 8 based on objective differences in task formulation, data representation, and evaluation protocol.
