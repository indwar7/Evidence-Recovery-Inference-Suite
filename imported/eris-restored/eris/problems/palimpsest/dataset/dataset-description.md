# Dataset Description — Wikivoyage Edit-Lineage Windows

## Overview

Real revision histories from **English Wikivoyage**, the collaboratively written
travel guide. Every snapshot in this dataset is genuine wikitext exactly as it
stood at one moment in one article's history — no synthetic text, no
paraphrasing, and no LLM output anywhere in the pipeline. The only manufactured
thing is the *arrangement*: real revisions from several articles are shuffled
together into a bag, and the challenge is to recover the lineage they came from.

The ground truth is not a human annotation and not a model's opinion. It is the
wiki's own revision graph: revision *r* has parent *r−1* because that is
literally the version it was edited from, recorded by MediaWiki at the moment of
the edit. This is about as close to uncontestable ground truth as text data gets.

Wikivoyage was chosen over Wikipedia deliberately. Its articles are maintained
rather than merely grown — listings get re-priced, venues open and close,
sections get reorganised — so revision windows exist in which an article churns
substantially while ending up about the same size. Those are exactly the windows
where byte count stops being a proxy for chronology, and where the ordering
signal has to come from the language of the edits instead.

## Source

| | |
|---|---|
| Source | English Wikivoyage full-history XML dump |
| URL | `https://dumps.wikimedia.org/enwikivoyage/latest/enwikivoyage-latest-pages-meta-history.xml.7z` |
| Dump date | 2026-08-05 |
| License | **CC BY-SA 4.0** — commercial use and redistribution permitted, with attribution and share-alike |
| Attribution | Text © Wikivoyage contributors. `attribution.csv` lists every article and revision id used, satisfying the CC BY-SA attribution requirement. Derived data in this dataset is likewise released under CC BY-SA 4.0. |

`attribution.csv` is part of the **raw** dataset. It is deliberately **not**
carried into the public split: article titles and revision ids are precisely the
keys that would let a solver look the true chronology up instead of inferring
it.

## File Structure

Three files at the top level of the upload:

```
windows.jsonl      one line per article: the selected revision window, with text
attribution.csv    article title + revision ids used (CC BY-SA attribution)
meta.json          dump date, selection parameters, and counts
```

| File | Rows | Description |
|---|---|---|
| `windows.jsonl` | 5,600 | One JSON object per article — the selected 14-revision churn window, including the full wikitext of every revision. This is the entire corpus; 78,400 revisions in total. |
| `attribution.csv` | 78,400 | `title, revid, url` for every revision used, satisfying the CC BY-SA attribution requirement. |
| `meta.json` | — | Dump URL and date, selection parameters (window length, size bounds, churn threshold), and counts. |

`prepare.py` consumes `windows.jsonl` and derives the public/private splits from
it (shuffled snapshot bags, parent-pointer labels, and the hidden answer key).
Those derived files are products of the prepare step and are not part of this
raw upload.

## Features

### `windows.jsonl` — one JSON object per article

| Field | Type | Description |
|---|---|---|
| `title` | str | Wikivoyage article title |
| `churn_score` | float | Selection score for the chosen window (see Notes) |
| `window_start` | int | Index of the window's first revision within the article's full history |
| `n_total_revs` | int | Total revisions the article has |
| `revs` | list | The selected consecutive revisions, **oldest first** |

Each entry of `revs`:

| Field | Type | Description |
|---|---|---|
| `revid` | str | MediaWiki revision id |
| `ts` | str | ISO-8601 revision timestamp |
| `size` | int | Revision size in bytes, as recorded in the dump |
| `comment` | str | Edit summary, truncated to 200 characters |
| `text` | str | Full wikitext of the revision |

`ts`, `revid` and `comment` are the ground-truth and provenance fields. They are
consumed by `prepare.py` to derive the parent pointers and are never released to
solvers.

### `attribution.csv`

| Column | Type | Description |
|---|---|---|
| `title` | str | Article title |
| `revid` | str | Revision id used |
| `url` | str | Permanent link to that revision's history entry |

## Notes

**How a window is selected.** An article's history is scanned for the
consecutive run of 14 revisions maximising

```
churn_score = (total absolute byte movement across the run) / (1 + |net byte change|)
```

subject to at least 200 bytes of total movement. This favours periods of active
revision over periods of steady accretion. The criterion looks only at byte
sizes recorded in the dump — never at the text, never at the labels, and never
at any baseline's performance — so it cannot be tuning difficulty to a
particular solution. Its measured effect is to reduce how well byte size orders
a lineage from ~0.76 pairwise accuracy on arbitrary windows to ~0.60 on selected
ones.

**Eligibility.** Main-namespace articles only, no redirects, at least 20
revisions of history, and a median window size between 2,500 and 40,000 bytes.
The upper bound keeps items inside the compute budget; the lower bound excludes
stubs whose revisions are too short to carry an edit signature.

**Known shortcut, removed downstream.** Wikivoyage `{{listing}}` templates carry
a `lastedit=YYYY-MM-DD` parameter recording when an editor last verified the
listing. It appears in about 45% of revisions (93,107 occurrences here), and
the maximum `lastedit` date in a revision orders a lineage at 0.64 pairwise
accuracy — a pure metadata shortcut. Editors also misspell the parameter
(`lastedited`, `lastdit`, `lasted it`, `lastedot`, `lastedir`) and citation
`accessdate` carries the same kind of signal, so rather than chase spellings
`prepare.py` drops **any template parameter whose value is a bare ISO date** —
a rule that is exhaustive by construction and cannot be defeated by a typo.
After stripping, no date-valued parameter survives anywhere in the released
text; the 1.85% of snapshots that still contain an ISO date carry it in prose,
and those order lineages at 0.497 — chance. The parameters are left intact in
the raw data so the measurement stays reproducible.

**One article, several lineages.** `prepare.py` usually takes two or three
lineages from a single article's 14-revision window, drawn from adjacent
stretches with one revision dropped between them. The dropped revision is the
next lineage's true parent, so each lineage head is genuinely a `ROOT` and the
item stays well-posed — but the lineages are the same page days apart, which is
what makes grouping them a real judgement rather than a clustering exercise.

**Wikitext, not prose.** Snapshots are raw wikitext: templates, listings, link
markup, and section headings included. This is what an edit actually looks like,
and much of the directional signal (a listing gaining a `price`, a link losing a
dead URL) lives in the markup rather than the prose.
