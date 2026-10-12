# Spelling Unseen Akkadian Words in Cuneiform Signs

A Neo-Assyrian scribe who had to write the verb form *ušettaqūni* (lemma *etēqu*, "to proceed") had no fixed spelling to copy. In one letter from Nimrud it is written `u₂-še-ta-qu-u₂-ni`, with the sign `u₂` rather than `u` or `u₃` and a plene vowel before the ending; in another letter, from Nineveh, the same form is written `u₂-še-taq-u-ni`; elsewhere the same lemma is written with the logogram `DIB`. Spelling was a choice, and the choice varied with the scribe, the archive and the period. This task asks a model to make those choices for words it has never seen spelled: given the normalized Akkadian word, what the editors know about it (lemma, gloss, part of speech), how the same lemma is spelled elsewhere in the training texts, the surrounding words of the line, and the period, genre and findspot of the text, produce the sign-by-sign transliteration the scribe actually wrote.

## Task

Each row is one word on one line of one cuneiform text, and the answer is that word's transliteration as the Oracc editors recorded it from the tablet. The input columns are:

- `norm`: the normalized (dictionary-style) form of the word, with vowel length and consonant doubling written out, e.g. `ušettaqūni`.
- `lemma`, `gloss`, `pos`: the editors' lemmatisation, e.g. `etēqu`, `proceed`, `V`.
- `attested`: up to four spellings of the same lemma taken from the training texts, written as `normalized=spelling` pairs separated by ` ; `, most frequent first, e.g. `etēqu=DIB ; ušettaq=u₂-še-taq ; etēqu=DIB-qa{MEŠ} ; ītiq=DIB`. A row's own normalized form is never among them.
- `context`: the normalized words of the whole line, with the target word in square brackets, words without a normalization (broken or unlemmatised) as `?`, and numerals as written.
- `period`, `genre`, `provenience`: catalogue metadata of the text, blank where the catalogue does not record it.
- `doc`: an opaque alias of the text, shared by rows from the same text.

The output is one string per row in the same convention as the training targets: signs joined by hyphens, logograms in upper case, syllabic values in lower case, homophone sign indices as subscript digits, determinatives in curly braces. For the Nimrud training row above the answer is `u₂-še-ta-qu-u₂-ni`.

Every evaluation row is a word whose normalized form occurs in no training text. Its lemma does occur, so the attested column is never empty, and the word is common vocabulary (nouns, verbs, adjectives, adverbs, prepositions and particles); proper names are excluded from evaluation, as are the few words whose spelling is identical to their normalized form. Evaluation texts are disjoint from training texts.

## What makes this hard

Akkadian cuneiform is a mixed script. A word can be written with a Sumerian logogram (`LUGAL` for *šarru*, "king"), syllable by syllable (`šar-ru`), or as a logogram followed by syllabic signs (*bēlīya*, "my lord", is both `EN-ia` and `be-li₂-ia` in the training texts); the same syllable can be written with several homophone signs (`u`, `u₂`, `u₃`; `ša`, `ša₂`); a long vowel may or may not be written plene (`lu` versus `lu-u`); a syllable boundary can fall `CV-VC` or `CVC` (`ra-aṣ-pa-a-te` versus `raṣ-pa-a-te`); and a determinative may precede the word (`{LU₂}` before professions, `{URU}` before cities). Which option a scribe took depends on convention: on the archive, the period, the genre, the word class, the particular lemma, and on habit. None of it is recoverable from the normalized form alone, and a word-to-spelling lookup is useless by construction because the evaluation words are unseen. The attested spellings of the lemma give the stem's conventional signs; the model has to decide how the inflected form continues from that stem, when a logographic stem switches to syllabic writing, and which signs carry the new syllables.

## What is new here

1. The direction. Computational work on cuneiform goes from signs to readings: sign recognition, transliteration to normalization and lemma, machine translation. This task goes from the reading back to the signs, which is the question a philologist faces when restoring a broken passage or judging whether a proposed reading fits the traces on the tablet.
2. The target is scribal choice, not a deterministic transform. Normalization to spelling is one-to-many; the answer key records what one scribe did, and the evaluation set is built so that no solver can have seen that choice. The task therefore ranks solutions by how well they model orthographic convention, with a measurable ceiling below perfection.
3. The unit is an unseen inflected form of a known lemma. Every row supplies the lemma's attested spellings, so the task isolates generalisation (continue a known stem into an unknown form; decide between logographic and syllabic writing; pick the homophone sign) from memorisation.
4. Real data at scale across nearly two millennia: 13,231 texts from 77 Oracc volumes, from the Old Assyrian and Old Babylonian periods to the Parthian, letters, royal inscriptions, legal documents, omens, astronomical diaries and commentaries, with the editors' own transliterations as the answer key.

## How it relates to prior work

Grapheme-to-phoneme conversion and its inverse, and the normalization of historical spellings to a modern standard, are the classical shapes nearest to this task; both map between a surface string and a canonical string of the same alphabet, and the inverse of normalization, generating a period-appropriate spelling, is rarely posed because the modern languages that have such corpora have one standard. Here the surface alphabet is a sign repertoire of several hundred values with logograms, determinatives and homophone sets, and the corpus comes with the lemmatisation that makes the one-to-many mapping learnable. Restoration of damaged ancient texts (predicting missing characters in Greek inscriptions from their context) is a different problem: there the word is unknown and the context must supply it; here the word is known and its written realisation is asked for. Akkadian normalization and lemmatisation tools go in the opposite direction of this task and are not usable inside it, since the solver environment is offline and external sign lists and dictionaries are not permitted.

## Dataset

Discover the data directory from the environment rather than assuming a path. It holds:

- `train.csv`, 340,876 rows: `row_id`, `doc`, `norm`, `lemma`, `gloss`, `pos`, `attested`, `context`, `period`, `genre`, `provenience`, `spelling`. One row per lemmatised Akkadian word with no restored or unreadable sign in the 10,539 training texts, capped at 40 rows per (normalized form, spelling) pair, with its transliteration as the target. The `attested` column of a training row is built the same way as for an evaluation row, from the training rows of the same lemma and excluding the row's own normalized form; it is empty for 14% of training rows, whose lemma has no other attested form, and never empty for an evaluation row.
- `test.csv`, 4,000 rows: the same columns without `spelling`. The rows come from 917 of the 2,692 held-out texts and cover 1,948 lemmas; no lemma contributes more than six rows and no normalized form more than two.
- `sample_submission.csv`: the required submission layout, filled with the normalized form as the prediction (a baseline that scores at the floor).
- `dataset_stats.json`: counts and the distribution of parts of speech and periods in each split.

Words with any restored (bracketed) or unreadable sign are not rows in either split, and numerals, unlemmatised words and Sumerian words appear only inside `context`. Row order carries no information: ids were assigned after a keyed shuffle.

## Rules

**Use only the data directory.** The editions behind these rows are public and searchable. Recovering a spelling by locating the edition of a text (through its normalized line, its metadata, or any other key) is not solving the task; a submission built that way is invalid regardless of its score, and so is any submission that depends on data not shipped in the data directory. The solver environment is expected to have no network access; your solution must run to completion from the data directory alone.

**No external Akkadian resources.** No sign lists, syllabaries, dictionaries, glossaries, lemmatisers, normalisers or corpora of Akkadian or Sumerian (ePSD, the Oracc gloss files, CDA, CAD, language models trained on Akkadian or Sumerian, or anything equivalent) may be used. The training table is the only source of sign values, logograms and spelling conventions.

**Score each row independently.** The prediction for a row may depend only on that row's own columns and on the training data. Fitting statistics on the evaluation inputs, or combining several evaluation rows to resolve one, is not permitted.

- Learn from the provided data only. No pretrained models, no LLMs, and no external data of any kind.
- Use only libraries available in the Kaggle Python Docker image.
- No LLM-generated outputs may be used anywhere in your solution.
- Your notebook must run end to end, top to bottom.
- Seed everything; your run should be reproducible.

## Evaluation

Each predicted spelling is split into sign tokens: each `{...}` determinative is one token, and the rest is split on hyphens, periods and colons. The same is done to the true spelling. A row's score is one minus the edit distance between the two token sequences divided by the longer sequence's length, so a prediction with every sign right scores one, a prediction that gets all but one of six signs right scores five sixths, and a prediction sharing no sign with the truth scores zero. Tokens compare as exact strings: `u` and `u₂` are different signs, as they are on the tablet, and a logogram is one token however many syllables it stands for. The submission score is the mean over all evaluation rows.

Worked example, for the true spelling `u₂-še-ta-qu-u₂-ni` (six signs):

- `u₂-še-ta-qu-u₂-ni` earns full credit.
- `u-še-ta-qu-u₂-ni` (wrong homophone for the first sign) earns five sixths.
- `u₂-še-taq-u-ni` (the other spelling of this very form in the training texts) earns one half: `taq` stands where `ta-qu` was written and the plene `u₂` is missing.
- `ušettaqūni` (the normalized form copied through) earns nothing, since it is one token that matches no sign.

Why this metric and not the alternatives considered: exact match collapses every near miss into zero, so a submission that gets the stem and the logogram right but writes one vowel plene would be scored like an unrelated string; character-level similarity on the transliteration string rewards hyphens and subscript digits as much as sign choice, and lets the copied normalized form collect partial credit for sharing letters; token-level similarity on signs measures exactly the decision the scribe made, one sign at a time, while still giving zero to a copied input and to a blank.

Measured on the shipped evaluation set:

- a blank submission scores 0.0000, and the sample submission (the normalized form copied through) 0.0006;
- the most frequent training spelling of the lemma scores 0.2452, or 0.2773 when chosen within the text's period;
- a naive syllabification of the normalized form, using no training data at all, scores 0.4935, because many syllabic spellings consist of the obvious consonant-vowel signs;
- a heuristic that copies the nearest attested spelling of the lemma (longest shared normalized prefix) and syllabifies the rest scores 0.5183; it uses exactly the information in the attested column and is the baseline a trained model has to beat;
- the reference, a character-level encoder-decoder trained from scratch on the training rows, scores 0.6798;
- the answer key scores 1.0000.

Exact-match rates are far lower (0.146 for the syllabifier, 0.139 for the heuristic, 0.378 for the reference), so the sign-level partial credit is what separates solutions.

## Submission

A CSV named `submission.csv` with columns `row_id`, `spelling`, one row per `row_id` in `test.csv`, in any order. `spelling` is the predicted transliteration string. A row that is missing, blank, duplicated, or whose id is unknown scores zero for that row; extra rows are ignored. Only a missing column rejects the file.

## Compute

One NVIDIA A10G, 10 CPU cores, 62 GB RAM, 1.5 hours wall clock for the whole notebook. The task is built for a character-level encoder-decoder trained from scratch on the training rows: the input is the concatenation of part of speech, lemma, gloss, normalized form, attested spellings and metadata (under 200 characters), the output is a spelling of at most forty characters, and 340,876 rows fit several epochs of a model with a few million parameters inside the budget with greedy decoding of 4,000 rows at the end. Table lookups and rule systems can be built on the CPU in minutes and are useful as fallbacks, but they cannot generalise to unseen forms, which is the whole of the evaluation set.
