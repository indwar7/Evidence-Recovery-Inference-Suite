# Matching Abstracts by Shared Research Topic, Not by Name

A set-selection NLP challenge built from OpenAlex (CC0). Given a query
paper's abstract and six candidate groups — each shown only as one other
paper's abstract, never as a topic name — pick which group shares the
query's real OpenAlex-assigned research topic. Distractor groups are the
true topic's own nearest neighbors by embedding similarity, not random or
same-domain picks (see DESIGN.md for the measured difficulty-tuning
history).

## Files

```
config.yaml                       metadata, measured baseline ladder, prepare/grade pointers
problem-description.md            task shown to the solving agent
dataset-description.md            source, fetch method, file structure, license
rubrics.md                        task-specific grading rubrics
DESIGN.md                         measured design record, including the difficulty-tuning history
prepare.py                        prepare(raw_dir, public_dir, private_dir) -> None; topic grouping embedded inside
grade.py                          grade(submission, answers) -> float
measure_ladder.py                 local tool: measures chance/adversary/reference/oracle on the shipped split
solution.ipynb                    reference solution notebook
dataset/
  raw/works.jsonl                 fetched works (work_id, title, abstract, year) — no topic field
  generator/build_raw.py          the fetch script
  generator/build_hard_distractors.py  computes build/hard_neighbors.json
  generator/build_prepare_groups.py    embeds build/*.json into prepare.py
build/                            build-only: topic_map.json, hard_neighbors.json (never shipped)
  public/                         train.csv, test.csv, sample_submission.csv (generated)
  private/answers.csv             answer key (generated, never shipped in the raw zip)
```

## Reproduce

```bash
cd dataset/generator
python3 build_raw.py --out ../raw/works.jsonl --n-topics 1600 --per-topic 14 --topic-map-out ../../build/topic_map.json
python3 build_hard_distractors.py --raw ../raw/works.jsonl --topic-map ../../build/topic_map.json --out ../../build/hard_neighbors.json
python3 build_prepare_groups.py --topic-map ../../build/topic_map.json --neighbors ../../build/hard_neighbors.json --out ../../prepare.py --inline
cd ../..
python3 prepare.py dataset/raw dataset/public dataset/private
python3 grade.py   # runs grader self-tests
python3 measure_ladder.py   # reports chance/adversary/reference/oracle on the shipped split
```

## Provenance

- Source: OpenAlex (https://openalex.org), CC0 1.0 Universal.
- Fetched via the public OpenAlex REST API, one worker, seeded sampling.
- Abstracts reconstructed from OpenAlex's documented inverted-index format.
- Topic assignments are OpenAlex's own structured `primary_topic` field,
  not an annotation added by this build.
