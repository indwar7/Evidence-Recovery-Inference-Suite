# Dataset Description

## Overview

This dataset holds the word-level editorial record of 13,231 cuneiform texts from the Akkadian corpora of Oracc, the Open Richly Annotated Cuneiform Corpus. 11,770 of the texts contain Akkadian words; the rest are Ugaritic or Sumerian texts from the same projects. For every lemmatised word the editors recorded two things. The transliteration lists sign by sign what the scribe wrote on the clay (`ARAD-MEŠ-šu₂`). The normalization is the dictionary-style reading of the same word (`urdānīšu`). Each word also carries its lemma, English gloss and part of speech. The dataset keeps both, line by line, for 328,054 lines and 1,830,469 word tokens from 69 Oracc project volumes, with each text's catalogue metadata (period, genre, provenience, dialect, script, object type, ruler).

The intended task, built by `prepare.py`, runs the other way from the editors' work: given the normalized word and what is known about the text, produce the spelling the scribe chose. No label is added here. The targets are the editors' own transliterations.

## Source

- Oracc, The Open Richly Annotated Cuneiform Corpus, JSON exports at `https://oracc.museum.upenn.edu/json/` (one zip per project volume).
- 77 project zips were fetched. 69 volumes contribute texts: SAAo (State Archives of Assyria online, 22 volumes), RINAP (5), RIAo (5), RIBo (8), CAMS (gkab), HBTIN, BLMS, CCPO, AEMW (amarna, ugarit, alalakh-idrimi), CKST, Suhu, OBTA, OBMC, CMAwRo (4), ADsD (5), babcity, borsippa, btto, caspo-akkpm, dccmt, glass, rimanum, asbp-ninmed and asbp-rlasb. Seven zips hold only catalogues and no editions (saao-aebp, saao-knpp, cams-akno, btmao, nimrud, oimea, qcat). RINAP 5 part 1 repeats RINAP 5's text numbers and adds no new text. The exact zip list is recorded in `meta.json`.
- Coverage by period (texts): Neo-Assyrian 6,898; Middle Babylonian 1,302; Hellenistic 815; Old Babylonian 748; Neo-Babylonian 450; Achaemenid 374; Middle Assyrian 279; Seleucid 181; Old Assyrian 129; First Millennium 129; Parthian 110; Uncertain 52; Old Akkadian 23; Ur III 8; Standard Babylonian 6; Early Dynastic IIIa and IIIb 5 each; unknown or unrecorded 1,717.
- Largest genres (texts): Royal Inscription 2,041; Administrative Letter 1,997; Legal 878; Legal Transaction 829; Astrological Report 566; Administrative Record 452; Astronomical Diary 401; Scholarly Letter 387.

## Fetch method

`generator/build_raw.py` downloads each project zip from the Oracc JSON root with a single worker. It sleeps one second between requests, makes three attempts per file with broad exception handling (a reset connection is not an HTTP error), and verifies every zip with `zipfile.testzip` before use. A zip that already verifies on disk is not fetched again, so an interrupted run resumes where it stopped. Extraction walks each text's `cdl` tree in document order, emits a record per line, and reads the catalogue metadata from each volume's `catalogue.json`. The run used for this release is dated in `meta.json`. It also counts the 88 corpus files that did not parse as JSON and the 265 repeated copies of a text number (138 of them RINAP 5 part 1), which are kept once, first volume wins.

## File Structure

The archive is flat: the four data files sit at its root, with the generator in one subfolder.

- `lines.jsonl` (311 MB): one JSON object per text line.
  - Line fields: `textid` (Oracc P, Q or X number), `project` (e.g. `saao/saa01`), `line_label` (the editors' label, such as `o 12` or `r 3'`), `line_idx` (0-based position of the line in the text), and `words` (a list, in line order).
  - Word fields:
    - `form`: the transliteration. Signs are joined by hyphens, determinatives and phonetic complements sit in braces, and sign indices are subscript digits.
    - `norm`: the normalization. It is null for unlemmatised, broken and Sumerian words.
    - `cf`, `gw`, `pos`: the citation form or lemma, the guide word (the English gloss), and the part of speech. Parts of speech include N, V, AJ, AV and PRP; the proper-noun tags PN, DN, GN, SN, RN and others; `n` for numerals; `u` for unlemmatised words.
    - `lang`: the Oracc language code, such as `akk-x-neoass`, `akk-x-stdbab` or `sux`.
    - Preservation fields: `n_signs` (signs in the word); `missing` (signs the editors restored inside square brackets); `damaged` (signs read from a damaged surface, marked with half brackets); `x` (true if any sign is unreadable).
- `texts.jsonl` (4.1 MB): one JSON object per text, with `textid`, `project`, `period`, `genre`, `subgenre`, `provenience`, `dialect`, `language`, `script`, `object_type`, `ruler`, `designation` (the volume siglum, e.g. `SAA 01 175`) and `n_lines`.
- `meta.json`: build date, the list of project zips fetched, record counts, the SHA-256 of the two JSONL files, and the licence statement.
- `ATTRIBUTION.txt`: the source, the licence and the editing projects, with the citation form Oracc asks for.
- `generator/build_raw.py`: the fetch and extraction script described above; it rebuilds the two JSONL files from the Oracc zips.

### What `prepare.py` derives (not shipped in this dataset)

The task's `prepare.py` reads the two JSONL files and writes the following into `public/`:
- a training table, with one row per lemmatised Akkadian word in a training text that has no restored or unreadable sign, and its spelling;
- a text-alias table for the training rows;
- an evaluation table, with one row per word in a held-out text whose normalized form occurs on no word of any training text;
- a sample submission;
- a summary statistics file.

It writes the answer key into `private/`. Those files exist only after `prepare.py` runs.

## Features

- **`form`** is the target of the derived task.
  - Upper case marks logographic writing with Sumerian sign names (`LUGAL`); lower case marks syllabic writing (`šar-ri`).
  - Subscript digits index homophone signs (`u`, `u₂`, `u₃`).
  - Braces mark determinatives (`{LU₂}`, `{URU}`, `{d}`) and phonetic complements (`{+ru}`).
  - Suffixes such as `@v` and `~v` mark sign variants. Compound signs are enclosed in vertical bars (for example KA×MI). All are kept exactly as the editors wrote them.
- **`norm`** is the input. Macrons and circumflexes mark vowel length and contraction, `ʾ` is the glottal stop, and doubled consonants are written out.
- **`cf`, `gw` and `pos`** carry the editors' lemmatisation. `gw` is one short English word or phrase per lemma; it is a gloss, not a translation of the word in context.
- **`missing`, `damaged` and `x`** make the preservation state explicit, so a consumer can decide which readings to trust. The derived task keeps words with damaged but read signs, and drops words with any restored or unreadable sign.
- **`period`, `genre` and `provenience`** come from each volume's catalogue. They are blank where the catalogue does not record them: period for 7% of texts and genre for 20%.

## Intended Use

- Primary: learning the mapping from normalized Akkadian to attested cuneiform spelling, that is, modelling scribal orthographic convention. This covers logogram versus syllabic writing, the choice of homophone sign, plene writing of long vowels, determinatives, and how these vary by period, genre and place.
- Also suitable for:
  - the forward direction (transliteration to normalization and lemma);
  - lemma-level studies of spelling variation;
  - sign-frequency and sign-inventory statistics by period;
  - a lemmatised Akkadian corpus for language modelling, with explicit preservation flags.
- Not suitable for:
  - translation, since no translations are included;
  - reading the tablets themselves, since there are no images or sign coordinates;
  - dating or provenance attribution as a classification target, since catalogue metadata is missing or coarse for a fifth of the texts and heavily skewed to Neo-Assyrian;
  - any claim about the original scribe's intent beyond what the editors transliterated.

## Limitations

- **Coverage is uneven.** 52% of texts and 49% of normalized word tokens are Neo-Assyrian, and 86% of the Neo-Assyrian texts come from Nineveh and Kalhu (letters and queries 42%, royal inscriptions 19%). Old Assyrian has 129 texts, Old Akkadian 23 and Ur III 8. Spelling conventions learned from this dataset are Neo-Assyrian conventions first.
- **Many tokens cannot be task rows.**
  - 32% of word tokens (589,098 of 1,830,469) have no normalization: numerals, unlemmatised words, Sumerian words, and words too broken to read.
  - 26% contain at least one restored sign, and 16% at least one unreadable sign.
  - In all, 47% of tokens cannot become rows of the derived task.
  - Preservation flags are per sign, as recorded by the editors. The dataset does not re-examine the tablets.
- **Editorial decisions differ between projects.** Normalization and lemmatisation differ in detail (vowel-length marking, Assyrian vowel harmony, loanwords), so a word spelled the same way can carry slightly different normalizations in two volumes.
- **Re-edited tablets appear twice.** A text number that appears in more than one volume is kept once (265 repeats). A tablet re-edited under a new number, such as SAA 1 175 and SAA 19 174, is kept under both numbers, and duplicate composites and parallel exemplars of royal inscriptions also occur. The derived task drops evaluation texts that largely duplicate a training text, but the raw files keep everything.
- **Some files and texts are omitted.** 88 corpus JSON files failed to parse, and 444 texts had no line structure (catalogue-only or empty editions).
- **Not every word is Akkadian.** 1,461 texts contain no Akkadian word (972 of them Ugaritic texts from AEMW), and Sumerian words inside Akkadian texts carry `lang: sux`. Consumers must filter on `lang` if they want Akkadian only.
- **Catalogue fields are free text.** Values are entered by each project and were not normalised here: `Kuyunjik (Nineveh)`, `Nineveh` and `Nineveh (mod. Kuyunjik)` all occur, as do `Nimrud (Kalhu)`, `Kalhu` and `Nimrud`.
- **Some editorial marks are rare and inconsistent.** Sign-variant markers (`TA@v`) and compound-sign notation follow slightly different conventions between projects.

## License

CC BY-SA 3.0. Oracc releases its editions under the Creative Commons Attribution Share-Alike license 3.0, as stated on the project pages and in the per-text credits. The JSON exports additionally state "This data is released under the CC0 license". The stricter CC BY-SA 3.0 is declared and observed: attribute Oracc and the editing projects named in `ATTRIBUTION.txt`, and distribute derivatives under the same licence. Licence URL: https://creativecommons.org/licenses/by-sa/3.0/
