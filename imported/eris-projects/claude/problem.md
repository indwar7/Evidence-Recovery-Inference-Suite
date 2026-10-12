You are a senior ML benchmark designer and research scientist.

Your objective is to invent an ORIGINAL machine learning competition in the RAG domain.

Do not imitate existing Kaggle competitions or previously accepted challenges.

## Constraints

- Domain: Retrieval-Augmented Generation (RAG)
- CPU-only
- Maximum inference time: 1.5 hours
- No external knowledge required
- Dataset must be fully self-contained
- Must have deterministic evaluation
- Must not be a simple retrieval, reranking, QA, classification, regression, or embedding similarity task.

## Originality Requirements

Before proposing anything, think about the space of existing RAG tasks and explicitly avoid:

- document retrieval
- next-page prediction
- chunk ranking
- passage reranking
- semantic search
- duplicate detection
- QA retrieval
- citation prediction
- OCR reconstruction
- text ordering
- entity linking

Invent a task with a fundamentally different objective, representation, or supervision signal.

The novelty should arise from the task itself, not from changing terminology.

## Deliver

1. Competition title
2. Overview
3. Motivation
4. Dataset generation pipeline
5. Input format
6. Output format
7. Training data
8. Test data
9. Evaluation metric
10. Baseline
11. Why existing retrieval methods struggle
12. Failure modes
13. Scientific contribution

## Critical Review

After generating the idea, critique it as an independent reviewer.

Answer:

- Which existing papers or competitions are most similar?
- What could reviewers consider derivative?
- Which components are actually new?
- How would you redesign it to be substantially different?

If the proposal is too close to existing work, discard it entirely and generate a completely different challenge instead of making superficial edits.

Repeat until the proposal is genuinely distinct in task formulation and motivation.
