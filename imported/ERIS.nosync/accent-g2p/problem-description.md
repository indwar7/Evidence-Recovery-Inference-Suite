# The Accent Translator

**Predict the target-accent pronunciation of a word from its spelling and
the two accents involved.**

You are given the word *bath*, told that the source accent is RP and the
target is GA, and asked for the GA transcription: `bæθ`. The RP
transcription `/bɑːθ/` is **not** provided — only the spelling is.

What makes this more than grapheme-to-phoneme is where the score comes from.
Your prediction is compared against the target only at the segments where the
two accents actually differ, so the task is really: **which sounds does this
accent rewrite, and into what?**

## Task

Each row gives you:

- `word` — `bath`
- `src_accent` — `RP`
- `tgt_accent` — `GA`

The source transcription is **not** given. Output the phoneme string of the
word in the target accent (`bæθ`), scored on the segments that differ from
what the *source* accent has — the accent's own rewrites.

- Accents: `RP` (Received Pronunciation), `GA` (General American), `AU`
  (Australian), `NZ` (New Zealand). All twelve directions occur.
- **~88% of rows require a real change.** The rest are identical across the
  two accents — and knowing *not* to rewrite is scored too.
- Test words are unseen in training, one row per word. You learn the rules,
  not the lexicon.

## What you are actually being asked

Predict the target-accent pronunciation from the word's spelling, and in
particular where that accent's pronunciation departs from the source
accent's. Post-vocalic `/ɹ/` appears in GA but
not RP. RP's `/ɑː/` in *bath* becomes GA `/æ/`, but the same vowel in
*father* does not. NZ centralises `/ɪ/`; AU fronts the vowel of *goat*. Each
is a context-sensitive rule you must recover from examples and apply to words
you have never seen.

## Dataset

The data directory is mounted by the solver environment. Locate it from the
environment you are running in rather than hard-coding a path.

- `train.csv` — `row_id, word, src_accent, tgt_accent, phonemes`
- `test.csv` — the same without `phonemes`
- `sample_submission.csv` — `row_id, phonemes`, word spelling copied through (≈0.09)
- `dataset_stats.json` — counts, direction distribution, share of rows needing change

Word-disjoint split. `row_id` is opaque.

## Rules

**Use only the data directory.** The transcriptions are real Wiktionary
content and the same headwords are searchable there. Recovering a test word's
pronunciation by looking it up — in Wiktionary, in any dump or mirror of it,
in a pronunciation dictionary such as CMUdict, or through a
grapheme-to-phoneme tool or model (`g2p_en`, `eng_to_ipa`, `espeak`,
`phonemizer` and equivalents) — is not solving the task; a submission built
that way is invalid regardless of its score, and so is any submission that
depends on data not shipped in the data directory. The solver environment is
expected to have no network access; your solution must run to completion from
the data directory alone.

**The source transcription is withheld on purpose.** `test.csv` gives the
spelling and the two accent labels, never the source-accent phoneme string.
Reconstructing that string from the spelling with an outside pronunciation
source sidesteps the task: what is being measured is the transfer function
between two accents, learned from `train.csv`, not access to an English
pronunciation lexicon.

- Learn from the provided data only. No pretrained models, no LLMs, and no
  external data of any kind — `train.csv` is the entire supervision signal.
- Use only libraries available in the Kaggle Python Docker image (pandas,
  numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch and similar).
- No LLM-generated outputs may be used anywhere in your solution.
- Your notebook must run end to end, top to bottom: load the public data, fit
  whatever you fit, then write `submission.csv`.
- Seed everything; your run should be reproducible.

## Evaluation

**Change-segment accuracy.** For each row, the true target is aligned to the
source and the segments that differ are identified — those are the accent's
rewrites. Your prediction is scored only on those positions:

```
per-row = (changed segments you got right) / (changed segments)
```

On rows where the two accents pronounce the word identically there are no
changed segments to score. For these rows, **the whole predicted phoneme
string must equal the whole target phoneme string exactly** — character for
character, after normalisation — to score 1; any difference scores 0, even
outside what would otherwise be a "changed segment." These are 11.7% of the
shipped test set. Mean over rows.

**Why not edit similarity.** About 80% of any transcription survives an accent
change. Under edit similarity, copying the source and doing nothing scores
≈0.80, leaving a 0.2 band to distinguish every real model. Scoring only the
changed segments sends naive copying to ≈0.1
and spends the full range on the one thing that matters: do you know the
rewrite.

Stress, length and syllable marks are stripped from all strings before
alignment; emitting them neither helps nor hurts.

## Submission

- `row_id` — from `test.csv`
- `phonemes` — predicted target-accent phoneme string

One row per `row_id`, header, UTF-8. Missing or blank rows score 0; only
missing columns reject.

## Compute

An encoder–decoder over `<src_accent> <tgt_accent> letters`, trained
to emit the target string, is the natural model, and the change-segment
metric rewards exactly the context-sensitive rewrites it can learn that rule
tables cannot. 1x NVIDIA A10G, 10 CPU cores, 62 GB RAM, 1.5 h.

