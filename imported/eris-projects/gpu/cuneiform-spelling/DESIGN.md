# DESIGN — Spelling Unseen Akkadian Words in Cuneiform Signs

Measured design record. Every number here was produced by a script in `work/`
against the files named; numbers marked (sample) come from the first seven
SAAo volumes used for feasibility and are superseded by the full-corpus ladder
further down.

## Task in one line

Given a normalized Akkadian word (dictionary-style transcription such as
`ušettaqūni`) with its lemma, gloss, part of speech, the normalized words
around it on the same line, and the text's period / genre / provenience,
produce the sign-by-sign transliteration the scribe actually wrote
(`u₂-še-ta-qu-u₂-ni`): which signs, syllabic or logographic, which homophone
sign variant, which determinative, whether a vowel is written plene.

## Why this shape

- The archive's own record is the label. ORACC editors record both the
  transliteration (`form`) and the normalization (`norm`) for every lemmatised
  word; no annotation is added here.
- The inverse direction (normalization -> spelling) is one-to-many with a
  measured distribution, so it ranks solutions instead of converging on an
  exact algorithm. The forward direction (spelling -> normalization) is the
  classical lemmatisation task and is not what is asked.
- Candidate ideas killed before building: USPTO trademark pseudo-marks (bulk
  download now behind an API key); Cantus chant text -> melody (CC BY-NC-SA);
  LDraw part geometry -> part name (reads as 3D captioning).

## Why the test set is restricted to unseen normalized forms

Random text-level 80/20 split, 7 SAAo volumes, 77,131 scorable words (sample):

- norm -> most-frequent-form lookup table: exact 0.614, sign-sim 0.715, coverage 0.850
- table with lemma fallback and naive syllabifier: exact 0.632, sign-sim 0.787
- per-norm majority ceiling measured on the test set itself (transductive oracle): 0.831

A lookup table already takes 74% of the attainable exact-match ceiling, so a
trained model could only land inside the table's noise. The split was
therefore changed so the table has zero coverage by construction: a test row is
a word in a held-out text whose normalized form never occurs in any training
text, whose lemma does occur in training, that is common vocabulary (no
proper names), and that has no missing (restored) signs.

On that test definition (sample, 1,441 rows):

- naive syllabifier of the normalized form: exact 0.144, sign-sim 0.500
- lemma table (most common spelling of the lemma): exact 0.026, sign-sim 0.312
- stem+suffix heuristic (closest training spelling of the same lemma by
  normalized prefix, syllabified remainder appended): exact 0.145, sign-sim 0.569
- char transformer, input = pos | lemma | gloss | norm (D=192, 2+2 layers,
  6 epochs, 35k capped rows): exact 0.173, sign-sim 0.528 — ABOVE the
  heuristic on exact match but BELOW it on the shipped metric (0.569). This is
  the register's "reference below a naive baseline" failure, caught before
  the full build. Cause: the heuristic sees the lemma's attested training
  spellings (it copies the nearest stem) while the model only saw the lemma
  string. Fix: the reference is retrieval-augmented — each row's input carries
  up to K=4 attested (normalized form = spelling) pairs of the same lemma from
  the training split, excluding pairs with the row's own normalized form (so
  training rows look like evaluation rows, where the form is unseen). The same
  attested pairs are what the heuristic consumes, so the ladder compares equals.
  Re-measured on the full corpus below.

Full corpus, feasibility split of `work/feas3.py` (same text-bucket rule as
prepare.py, 4,000 unseen-form rows, 251,624 capped training rows), D=256,
3+3 layers, 2 epochs on an M-series laptop:

- syllabifier 0.489 (exact 0.140); stem+suffix adversary 0.513 (exact 0.136)
- char transformer with attested spellings in the input: sign-sim 0.665,
  exact 0.367 — margin 0.15 over the adversary on the metric and 0.23 on exact
  match. The retrieval-augmented design is what makes the trained model win.

## Current build (v2, after the internal review) — ladder on the shipped split

Metric: 0.5 x exact sign-sequence match + 0.5 x sign-level similarity.
Produced by work/finalize.py (runs work/ladder.py, work/cpu_rules.py,
work/ceiling.py, and reads the executed solution.ipynb) into ladder.json.

- blank 0.0000; copy of the normalized form (sample submission) 0.0003
- lemma most-common training spelling 0.1304; within the text's period 0.1482
- naive syllabifier 0.3182 (exact 0.144); stem+suffix on the attested column
  0.3305 (exact 0.138); stem+suffix over all training forms of the lemma with
  the most frequent sign per syllable 0.3566; strongest rule stack found by the
  review (vowel-length-blind lookup within the lemma first) 0.3641
- reference = solution.ipynb as shipped, executed end to end: REFERENCE_SCORE
  (exact REFERENCE_EXACT; REFERENCE_MARGIN above the strongest rule stack;
  contiguous halves REFERENCE_HALVES; REFERENCE_WALL)
- practical ceiling estimate (train only; majority spelling of the same
  normalized form in other training texts, forms in 2-5 texts, 56,078 rows):
  0.7072 (exact 0.611); over all forms in 2+ texts 0.6289
- oracle 1.0000

## v1 ladder (superseded; pure sign-level similarity, first split)

- blank 0.0000; copy of the normalized form (sample submission) 0.0006
- lemma most-common training spelling 0.2452; conditioned on period 0.2773 (metadata probe)
- naive syllabifier of the normalized form 0.4935 (exact 0.146): content-based
  degenerate, no training data, the first rung a solver must clear
- stem+suffix heuristic (adversary; consumes exactly the attested column)
  0.5183 (exact 0.139)
- reference: 0.6798 (see "Reference" below)
- oracle 1.0000
- contiguous halves of answers.csv: syllabifier 0.4925 / 0.4945, adversary
  0.5110 / 0.5256

## Reference

v2: the published number is the shipped notebook's own final submission (fp32,
seed 0, configuration below, heuristic fallback for rows that decode to no
sign), executed with `jupyter nbconvert --execute` on Apple MPS. The text below
describes the v1 proxy run, which used the same code and configuration on the
v1 split.

`work/reference.py` (the same code is the transformer cell of solution.ipynb):
character-level encoder-decoder transformer trained from scratch on train.csv.
Input string `pos | lemma | gloss[:20] | norm | attested | period[:14] |
genre[:14]` truncated to 160 characters; output the spelling, at most 38
characters; greedy decoding with the encoder memory computed once per batch.
The character vocabulary comes from training rows only. A development split
holds out 3% of training docs and keeps their rows whose normalized form is
unseen in the remaining docs, mirroring the evaluation rule.

Published number: proxy configuration on Apple MPS, D=256, 3+3 layers, 4 heads,
2 epochs, batch 64, AdamW + one-cycle LR 4e-4, 250,000 shuffled training rows,
seed 0: 0.6798 on the shipped answers (exact 0.378; contiguous halves
0.6797 / 0.6799; development split 0.6600 on 1,149 rows; 2,684 s on MPS). The notebook's A10G
configuration (D=384, 4+4 layers, 6 epochs, batch 256, bf16, 65-minute
training budget) is larger.

An earlier reference build counted the evaluation inputs' characters into the
vocabulary; it was replaced before any number was published so that nothing
about the reference is fitted on evaluation rows.

## Metric

v2: 0.5 x [pred sign sequence == true sign sequence] + 0.5 x sign-level
similarity. Adopted after the review measured, on the v1 split, that under
sign similarity alone a naive syllabifier with no training data reached 0.49
and the review's estimate of GPU-agent scores was 0.60-0.72, above the
0.2-0.6 band; under the blend the same rungs were syllabifier 0.3196, rule
stack 0.3655, reference 0.5299. Pure exact match was the stricter alternative
(syllabifier 0.146, reference 0.380) but is bimodal per row.

v1 (superseded): mean over rows of normalized sign-level edit similarity
(1 - Levenshtein(pred_signs, true_signs) / max length). Alternatives measured
and rejected:

- exact match only: bimodal; a spelling with one wrong homophone sign out of
  six scores the same as an unrelated string.
- character-level similarity on the transliteration string: rewards the
  subscript digits and hyphens as much as the sign choice, which is what is
  being tested.

## Licence

Each ORACC corpus JSON file carries `"license": "This data is released under
the CC0 license"`; the project pages and the per-text credits say "released
under the Creative Commons Attribution Share-Alike license 3.0". The stricter
of the two, CC BY-SA 3.0, is declared, with attribution to ORACC and the
editing projects in every file.

## Build log

- 2026-10-11 feasibility on SAAo volumes 1-7 (`work/feas.py`, `work/feas2.py`);
  tiny model lost to the stem+suffix heuristic on sign-sim (0.528 vs 0.569).
- 2026-10-11 all 77 Akkadian ORACC volumes fetched (565 MB of JSON zips);
  retrieval-augmented reference measured with `work/feas3.py`.
- 2026-10-11 v1 shipped locally: reference 0.6798 (proxy run), checklist 65/65.

## Internal review round (2026-10-11) and what changed in v2

A five-reviewer workflow (committee review per claude/ai&novelty.md, a
rule-based attacker, a duplicate-text attacker, leak probes, a document audit),
each finding re-checked by an independent verifier. Confirmed findings and
the fixes:

- Novelty 5/10 (gate 6). "What is new here" claimed computational work only
  reads cuneiform; a reviewer finds restoration models (Fetaya et al. 2020,
  Lazar et al. 2021), EvaCun 2025 and 2026 English-to-Akkadian translation in
  one search, and Sahala et al. 2020 is the exact inverse mapping. Fixed by
  rewriting both sections: concede the string-transduction primitive
  (grapheme-to-phoneme, kana-kanji conversion, morphological reinflection),
  lead with the scribal-choice target and the unseen form of a known lemma,
  cite only verified works, and make no priority claim.
- Unseen-form guarantee was true only against training ROWS: 471 v1
  evaluation rows (394 norms) occurred in training texts on restored or
  untransliterated words. v2 checks every word of every training text.
- Duplicate manuscripts across the split (re-editions under new numbers,
  duplicate composites, parallel exemplars): the attack showed no exploitable
  lift (best lookup 0.5195 vs adversary 0.5183 in v1), but v2 drops held-out
  texts that largely duplicate one training text (53 texts).
- Row order tracked lemma frequency (greedy caps in the same salted order as
  the cut): v2 cuts and numbers rows in a second independent salted order.
- doc alias in test.csv invited cross-row pooling: v2 publishes text aliases
  only for training rows (train_docs.csv).
- Metric floor too high (see Metric); stronger CPU adversary existed (0.5535
  v1; 0.3641 v2) and is now a published rung; a train-derived ceiling estimate
  is published instead of the unsupported "ceiling below perfection" phrase.
- Reference was not from the notebook: v2 notebook uses exactly the published
  configuration in fp32 and is executed end to end; its own output is the
  reference.
- Documentation errors corrected: 69 contributing volumes (77 zips fetched;
  7 catalogue-only; RINAP 5/1 repeats RINAP 5), 11,770 of 13,231 texts contain
  Akkadian, 32% of tokens lack a normalization (47% cannot be rows), 49% of
  normalized tokens are Neo-Assyrian, complete period list, provenience
  variants, an invented example spelling (raṣ-pa-a-te) and a held-out-only
  compound sign replaced by training examples, rubric for every rule, flat
  dataset zip, titles unified, README commands.
