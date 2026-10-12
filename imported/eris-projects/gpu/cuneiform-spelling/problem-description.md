# Spelling Unseen Akkadian Words in Cuneiform Signs

Given a normalized Akkadian word, its lemma, up to four attested spellings of that lemma, the line it stands in and the metadata of its text, predict the sign-by-sign transliteration the scribe actually wrote.

In the training texts the verb form *ušettaqūni* (lemma *etēqu*, "to proceed") is written `u₂-še-ta-qu-u₂-ni` in a letter from Nimrud and `u₂-še-taq-u-ni` in a letter from Nineveh, and other forms of the same verb are written with the logogram `DIB`. Spelling was a choice. This task scores how well a model makes that choice for words it has never seen spelled.

## Task

Each row is one word on one line of one cuneiform text. The answer is that word's transliteration as the Oracc editors recorded it from the tablet. The input columns are:

- `norm`: the normalized (dictionary-style) form of the word, with vowel length and consonant doubling written out, e.g. `ušettaqūni`.
- `lemma`, `gloss`, `pos`: the editors' lemmatisation, e.g. `etēqu`, `proceed`, `V`.
- `attested`: up to four spellings of the same lemma from the training texts, written as `normalized=spelling` pairs separated by ` ; `, most frequent first, e.g. `etēqu=DIB ; ušettaq=u₂-še-taq ; etēqu=DIB-qa{MEŠ} ; ītiq=DIB`. A row's own normalized form is never among them.
- `context`: the normalized words of the whole line, with the target word in square brackets, numerals as written, and words without a normalization (broken, unlemmatised or Sumerian) as `?`.
- `period`, `genre`, `provenience`: catalogue metadata of the text, blank where the catalogue does not record it.

The output is one string per row, in the same convention as the training targets: signs joined by hyphens, logograms in upper case, syllabic values in lower case, homophone sign indices as subscript digits, determinatives and phonetic complements in curly braces. For the Nimrud row above the answer is `u₂-še-ta-qu-u₂-ni`.

Every evaluation row is a word whose normalized form occurs on no word of any training text, damaged or not. Its lemma does occur in training, so the `attested` column is never empty. The word is common vocabulary: nouns, verbs, adjectives, adverbs, prepositions, pronouns, number words and particles. Proper names are excluded, as are the few words whose spelling is identical to their normalized form. Evaluation texts are disjoint from training texts, and a held-out text that largely duplicates a training text (a re-edition under a new number, a duplicate composite, a parallel exemplar) contributes no evaluation rows.

## What makes this hard

Akkadian cuneiform is a mixed script, and scribes had several valid ways to write the same word:

- **Logogram or syllables.** *šarru* ("king") is written `LUGAL`, `MAN`, or `LUGAL{+ru}` with a phonetic complement. *bēlīya* ("my lord") is both `EN-ia` and `be-li₂-ia`.
- **Which homophone sign.** The same syllable can be written with different signs, such as `u`, `u₂` and `u₃`, or `lib` and `lib₃`.
- **How syllables are cut.** A closed syllable can be one sign or two: *išpur* is `iš-pur` and `iš-pu-ur`, and *libbi* is `lib-bi`, `lib₃-bi`, `li-ib-bi` and `ŠA₃-bi`.
- **Plene writing.** A long vowel may or may not be written out: *lū* is `lu`, `lu-u` and `lu-u₂`.
- **Determinatives and numerals.** Classifiers such as `{LU₂}` and `{URU}` may precede a word, and number words are often written with numerals (*šitta* as `2-ta`).

Which option a scribe took depends on the archive, the period, the genre, the lemma and habit. None of it follows from the normalized form alone. An exact word lookup has zero coverage, because the evaluation words are unseen. The attested spellings give the conventions for the stem. The model has to carry them into a form it has never seen: whether a logographic stem continues in syllables, which signs write the new syllables, and how the ending is spelled.

## What is new here

1. **The target is one historical writer's choice among equally valid spellings.** No standard orthography exists to recover. Measured on training texts only, the majority spelling of the very same word in other texts reproduces a text's spelling exactly 61% of the time, for words attested in two to five texts. The answer key records what each scribe wrote, so the task ranks models by how well they model orthographic convention, not by how well they recover a norm.
2. **The unit is an unseen inflected form of a known lemma.** Each row gives up to four attested spellings of the lemma, and the form itself never occurs in a training text. The task isolates generalisation (carry a stem's sign conventions into a new form and decide how the ending is written) from memorisation.
3. **Conditioning on where and when the word was written.** Period, genre and findspot are given for every row. The same normalized form is spelled differently across archives, and a model that uses this metadata can learn regional and period conventions.
4. **The primitive is conceded, the capability is not.** Mechanically this is string transduction, the same primitive as grapheme-to-phoneme conversion, kana-kanji input conversion and morphological reinflection. Each of those targets a standard modern orthography. Here the target is a scribe's variable practice, generalised from a handful of attested spellings and scored sign by sign.
5. **Real data with a measured ladder.** There are 13,231 texts from 69 Oracc volumes, mostly first millennium BC but reaching back to a handful of Early Dynastic, Old Akkadian and Ur III texts. The editors' own transliterations are the answer key. Every rung below is measured, from a blank submission through rule systems to a trained model, with a ceiling estimate derived from the training data.

## How it relates to prior work

Computational work on Akkadian has mostly read cuneiform. One line converts transliteration into phonological transcription (Sahala et al., LREC 2020), the inverse of this task's mapping, and close to many-to-one. Another lemmatises and predicts missing tokens (the EvaCun 2025 shared task at ALP 2025). A third restores damaged passages (Fetaya et al., PNAS 2020; Lazar et al., EMNLP 2021). In restoration the identity of the missing word is unknown and the context must supply it. Here the word is given, and only its written form is asked for. Translation systems now also produce Akkadian from English, but they generate whole passages, and the choice among spellings of a given word is not what they are evaluated on. The nearest classical shapes are the transduction tasks named above: kana-kanji conversion, which also turns a phonetic reading into a mixed logographic and syllabic script with a homophone choice, and morphological reinflection, which also produces an unseen inflected form of a known lemma. Both have one correct answer per input; this task does not. The classical method for grapheme-to-phoneme conversion, the joint-sequence model, transfers directly, and it is the strongest CPU baseline measured below.

## Dataset

Discover the data directory from the environment rather than assuming a path. It holds:

- `train.csv`, 340,871 rows: `row_id`, `norm`, `lemma`, `gloss`, `pos`, `attested`, `context`, `period`, `genre`, `provenience`, `spelling`. There is one row per lemmatised Akkadian word with no restored or unreadable sign. The rows come from 8,768 of the 10,539 training texts and are capped at 40 rows per (normalized form, spelling) pair. Words written with the logogram `NA`, which common CSV readers turn into a missing value, are dropped. The `attested` column is built the same way as for evaluation rows: from the training texts, ranked by frequency before the cap, excluding the row's own normalized form. It is empty for 14% of training rows, whose lemma has no other attested form, and never empty for an evaluation row.
- `train_docs.csv`: `row_id`, `doc`, an opaque alias of the text each training row comes from, so that validation can hold out whole texts. Evaluation rows carry no text grouping.
- `test.csv`, 4,000 rows: the same columns as `train.csv` without `spelling`. The rows come from 885 of the 2,692 held-out texts and cover 1,974 lemmas. No lemma contributes more than six rows and no normalized form more than two.
- `sample_submission.csv`: the required submission layout, filled with the normalized form as the prediction, a baseline at the floor.
- `dataset_stats.json`: counts and the part-of-speech and period distributions of both splits, computed from the input columns.

Bare numerals (part of speech `n`), unlemmatised words and Sumerian words are never rows. Number words lemmatised as `NU` are rows and are often spelled with numerals. Row ids follow a salted shuffle, independent of text, line and lemma frequency.

## Rules

**Use only the data directory.** The editions behind these rows are public and searchable. Recovering a spelling by locating the edition of a text (through its normalized line, its metadata or any other key) is not solving the task. A submission built that way is invalid regardless of its score, and so is any submission that depends on data not shipped in the data directory. The solver environment is expected to have no network access, and your solution must run to completion from the data directory alone.

**No external Akkadian resources.** No sign lists, syllabaries, dictionaries, glossaries, lemmatisers, normalisers or corpora of Akkadian or Sumerian may be used. That includes ePSD, the Oracc gloss files, CDA, CAD, language models trained on Akkadian or Sumerian, and anything equivalent. The training table is the only source of sign values, logograms and spelling conventions.

**Score each row independently.** The prediction for a row may depend only on that row's own columns and on the training data. Fitting statistics on the evaluation inputs, or combining several evaluation rows to resolve one, is not permitted.

- Learn from the provided data only. No pretrained models, no LLMs, and no external data of any kind.
- Use only libraries available in the Kaggle Python Docker image.
- No LLM-generated outputs may be used anywhere in your solution.
- Your notebook must run end to end, top to bottom.
- Seed everything; your run should be reproducible.

## Evaluation

Both spellings are Unicode-NFC normalized and split into sign tokens. Each `{...}` determinative or complement is one token, and the rest is split on hyphens, periods, colons and whitespace. Tokens compare as exact strings: `u` and `u₂` are different signs, as they are on the tablet, and a logogram is one token however many syllables it stands for. A row scores

- 0.5 if the predicted sign sequence is exactly the scribe's, plus
- 0.5 times the sign-level similarity: one minus the edit distance between the two token sequences divided by the longer sequence's length.

The submission score is the mean over all evaluation rows.

Worked example, for the true spelling `u₂-še-ta-qu-u₂-ni` (six signs):

- `u₂-še-ta-qu-u₂-ni` earns 1.
- `u-še-ta-qu-u₂-ni` (wrong homophone for the first sign) earns 0.5 × 5/6 = 0.417.
- `u₂-še-taq-u-ni` (the other spelling of this very form in the training texts) earns 0.5 × 1/2 = 0.25. `taq` stands where `ta-qu` was written, and the plene `u₂` is missing.
- `ušettaqūni` (the normalized form copied through) earns 0, since it is one token that matches no sign.

**Why this metric and not the alternatives.**
- **Exact match alone** collapses every near miss into zero, so a spelling with the right stem and one homophone wrong scores like an unrelated string.
- **Sign similarity alone** gives too much to the obvious syllables. A naive syllabification of the normalized form, with no training data, would score 0.49 under it, and the gap between rule systems and trained models would shrink to a narrow band.
- **Character similarity** on the transliteration string rewards hyphens and subscript digits as much as sign choice, and lets a copied normalized form collect credit for shared letters.

The blend gives full credit only for the scribe's actual spelling, keeps sign-by-sign partial credit, and gives zero to a copied input and to a blank.

Measured on the shipped evaluation set:

- a blank submission scores 0.0000, and the sample submission (normalized form copied through) 0.0003;
- the most frequent training spelling of the lemma scores 0.1304, or 0.1482 when chosen within the text's period;
- a naive syllabification of the normalized form, using no training data, scores 0.3182;
- a heuristic that copies the nearest attested spelling of the lemma (longest shared normalized prefix) and syllabifies the rest scores 0.3305;
- the same heuristic over all training forms of the lemma, writing each syllable with its most frequent sign in training, scores 0.3566;
- the strongest hand-written rule system found, which adds a lookup of the lemma's training forms that ignores vowel length, scores 0.3641;
- a classical joint-sequence transducer of the kind used for grapheme-to-phoneme conversion, on the CPU only, scores CLASSICAL_SCORE. It aligns training spellings to the normalized forms, runs period- and genre-conditioned sign trigrams, copies candidate stems from the lemma's spellings, rescores right to left and with a sign language model, and picks a candidate with a gradient-boosted reranker trained on development folds;
- a character-level encoder-decoder of the reference's size that sees only the `attested` column, trained for two epochs with greedy decoding, scores 0.5200;
- the reference (the shipped notebook) scores REFERENCE_SCORE, with REFERENCE_EXACT of its spellings exactly right. It is the same encoder-decoder, but each row's input also carries spellings it retrieves from the training rows: the lemma's forms closest to the row's normalized form, and forms of other words with the same ending. It is trained for three epochs and decoded with beam search;
- the answer key scores 1.0000.

As a ceiling estimate from the training texts alone, predicting a word with the majority spelling of the very same normalized form in other training texts scores 0.707 (words attested in two to five texts). No method can do that on evaluation rows, whose forms are unseen.

## Submission

A CSV named `submission.csv` with columns `row_id` and `spelling`, one row per `row_id` in `test.csv`, in any order. `spelling` is the predicted transliteration string. A row that is missing, blank or duplicated, or whose id is unknown, scores zero; extra rows are ignored. Only a missing column rejects the file.

## Compute

One NVIDIA A10G, 10 CPU cores, 62 GB RAM, 1.5 hours wall clock for the whole notebook. The task is built for a character-level encoder-decoder trained from scratch. The input concatenates part of speech, lemma, gloss, normalized form, attested spellings and metadata: a median of 138 characters and at most 330. Training spellings are at most 51 characters, and all but four are 40 or fewer. The reference model has about 5.7 million parameters and reads inputs of up to 224 characters once the retrieved spellings are added. Three epochs over 250,000 rows took about REFERENCE_MIN minutes on a laptop GPU (Apple MPS). An A10G is several times faster, so larger models, more epochs and ensembles fit the budget. Exact table lookups have zero coverage on unseen forms. Hand-written rule systems reach 0.32 to 0.36, and the classical transducer reaches CLASSICAL_SCORE on the CPU; both are useful fallbacks and rescoring candidates.
