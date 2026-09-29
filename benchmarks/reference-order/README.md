# The Reference Shuffle

**Domain:** natural-language-processing · **Tags:** text
**Metric:** mean rescaled Kendall's tau (chance is exactly 0.5) · **Runtime:** CPU only
**Licence:** CC BY-SA 4.0

## The task

You are given one real published paragraph with its inline citation markers
replaced by a neutral token, and a shuffled pool of that paragraph's own three
reference cards — title, author, venue, year. Recover the order in which the
references were actually cited.

Every card in the pool is a genuine citation of the paragraph. Nothing is a
distractor. The only thing withheld is order.

**A random ordering scores exactly 0.5**, verified by enumerating all six
permutations. A row that is missing or malformed scores 0.0, not 0.5, so
skipping a row is never better than attempting it.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Challenge configuration and 9 measured anchors |
| `problem-description.md` | The prompt an agent sees |
| `dataset-description.md` | Source, fetch method, schema, intended use, limitations |
| `rubrics.md` | 7 rubrics — 4 REQUIRED, 3 RECOMMENDED |
| `DESIGN.md` | The measured design record |
| `prepare.py` | Deterministic raw → public/private split (seed 20260925) |
| `grade.py` | `grade(submission, answers) -> float` |
| `solution.ipynb` | Reference solution: word overlap, author and year signal, Hungarian assignment |
| `dataset/` | `raw/`, `public/`, `private/`, `generator/` |

## Reproduce

```bash
python prepare.py                                        # raw -> public/private
python grade.py dataset/public/sample_submission.csv \
                dataset/private/answers.csv              # -> 0.502 (format example, chance)
```

The reference notebook imports `torch`, which is outside
`requirements.txt`, so it is not part of the offline `tools/verify.py` run.

## Data

Paragraphs from English Wikipedia Good and Featured articles. The label is
the order in which the article's own editors placed its citations — no
annotation created for this dataset, no model-generated content.

492 units, each with exactly three references, kept from 1,196 extracted
across 510 articles. The split is by **article**, so no two paragraphs from
the same page cross it: 383 train units from 226 articles, 109 test units
from 70 articles.

**CC BY-SA 4.0** — confirmed from Wikipedia's own MediaWiki `siteinfo` rights
endpoint. Commercial use permitted with attribution; derivatives of the data
must carry the same licence. See `dataset/raw/ATTRIBUTION.txt`.
