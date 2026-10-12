# Dataset Description

## Overview

This dataset holds the word-level editorial record of 13,231 Akkadian cuneiform texts as published by Oracc, the Open Richly Annotated Cuneiform Corpus. For every lemmatised word the editors recorded two things: the transliteration, which lists sign by sign what the scribe wrote on the clay (`ARAD-MEŠ-šu₂`), and the normalization, the dictionary-style reading of the same word (`urdānīšu`), together with its lemma, English gloss and part of speech. The dataset keeps both, line by line, for 328,054 lines and 1,830,469 word tokens across 77 Oracc project volumes, with each text's catalogue metadata (period, genre, provenience, dialect, script, object type, ruler).

The intended task built on it, by `prepare.py`, is the inverse of what the editors did: given the normalized word and what is known about the text, produce the spelling the scribe chose. No label is added here; the targets are the editors' own transliterations.

## Source

- Oracc, The Open Richly Annotated Cuneiform Corpus, JSON exports at `https://oracc.museum.upenn.edu/json/` (one zip per project volume).
- Projects used: SAAo (State Archives of Assyria online, 24 volumes), RINAP (6), RIAo (5), RIBo (8), CAMS (gkab, akno), HBTIN, BLMS, CCPO, AEMW (amarna, ugarit, alalakh-idrimi), CKST, Suhu, OBTA, OBMC, CMAwRo (4), ADsD (5), babcity, borsippa, btto, btmao, caspo-akkpm, dccmt, glass, nimrud, oimea, qcat, rimanum, asbp-ninmed, asbp-rlasb. The exact zip list is recorded in `meta.json`.
- Coverage by period (texts): Neo-Assyrian 6,898; Middle Babylonian 1,302; Hellenistic 815; Old Babylonian 748; Neo-Babylonian 450; Achaemenid 374; Middle Assyrian 279; Seleucid 181; Old Assyrian 129; First Millennium 129; unknown or unrecorded 1,717.
- Largest genres (texts): Royal Inscription 2,041; Administrative Letter 1,997; Legal 878; Legal Transaction 829; Astrological Report 566; Administrative Record 452; Astronomical Diary 401; Scholarly Letter 387.

## Fetch method

`generator/build_raw.py` downloads each project zip from the Oracc JSON root with a single worker, sleeping one second between requests, three attempts per file with broad exception handling (a reset connection is not an HTTPError), and verifies every zip with `zipfile.testzip` before use. A zip that already verifies on disk is not fetched again, so an interrupted run resumes where it stopped. Extraction walks each text's `cdl` tree in document order, emits a record per line, and reads the catalogue metadata from each volume's `catalogue.json`. The run used for this release is dated in `meta.json`; 88 corpus files that did not parse as JSON and 265 texts that appear in more than one volume (kept once, first volume wins) are counted there.

## File Structure

- `lines.jsonl` (311 MB): one JSON object per text line. Fields: `textid` (Oracc P/Q/X number), `project` (e.g. `saao/saa01`), `line_label` (editors' line label such as `o 12` or `r 3'`), `line_idx` (0-based position of the line in the text), `words` (list, in line order). Each word has `form` (transliteration, signs joined by hyphens, determinatives in braces, sign indices as subscript digits), `norm` (normalization; null for unlemmatised or broken words), `cf` (citation form / lemma), `gw` (guide word, the English gloss), `pos` (part of speech: N, V, AJ, AV, PRP, ... and the proper-noun tags PN, DN, GN, SN, RN, ...; `n` for numerals, `u` for unlemmatised), `lang` (Oracc language code such as `akk-x-neoass`, `akk-x-stdbab`, `sux`), `n_signs` (signs in the word), `missing` (signs the editors restored inside square brackets), `damaged` (signs read from damaged surface, half brackets), `x` (true if any sign is unreadable).
- `texts.jsonl` (4.1 MB): one JSON object per text: `textid`, `project`, `period`, `genre`, `subgenre`, `provenience`, `dialect`, `language`, `script`, `object_type`, `ruler`, `designation` (the volume siglum, e.g. `SAA 01 175`), `n_lines`.
- `meta.json`: build date, the list of project zips used, record counts, SHA-256 of the two JSONL files, licence statement.
- `ATTRIBUTION.txt`: source, licence and the list of editing projects, with the citation form Oracc asks for.
- `generator/build_raw.py`: the fetch and extraction script described above; rebuilds the two JSONL files from the Oracc zips.

### What `prepare.py` derives (not shipped in this dataset)

The task's `prepare.py` reads the two JSONL files and writes a training table (one row per lemmatised Akkadian word with no restored or unreadable sign in a training text, with its spelling), an evaluation table (one row per word in a held-out text whose normalized form never occurs in the training texts), a sample submission and a summary statistics file into `public/`, and the answer key into `private/`. Those files exist only after `prepare.py` runs.

## Features

- `form` is the target of the derived task. Alphabetic case distinguishes logographic (upper-case, Sumerian sign names such as `LUGAL`) from syllabic (lower-case, `šar-ri`) writing; subscript digits index homophone signs (`u` vs `u₂` vs `u₃`); curly braces mark determinatives (`{LU₂}`, `{URU}`, `{d}`); `@v`, `~v` and similar suffixes are sign-variant markers and `|...|` encloses compound signs, all kept exactly as the editors wrote them.
- `norm` is the input. Macrons and circumflexes mark vowel length and contraction; `ʾ` is the glottal stop; doubled consonants are written out.
- `cf`, `gw`, `pos` carry the editors' lemmatisation. `gw` is one short English word or phrase per lemma and is a gloss, not a translation of the word in context.
- `missing`, `damaged`, `x` make the preservation state explicit so a consumer can decide which readings to trust. The derived task keeps words with damaged but read signs and drops words with any restored or unreadable sign.
- `period`, `genre`, `provenience` come from each volume's catalogue and are blank where the catalogue does not record them (period blank for 7% of texts, genre for 20%).

## Intended Use

- Primary: learning the mapping from normalized Akkadian to attested cuneiform spelling, i.e. modelling scribal orthographic convention: logogram versus syllabic writing, choice of homophone sign, plene writing of long vowels, determinatives, and how these vary by period, genre and place.
- Also suitable for: the forward direction (transliteration to normalization and lemma), lemma-level spelling-variation studies, sign-frequency and sign-inventory statistics by period, and as a lemmatised Akkadian corpus for language modelling with explicit preservation flags.
- Not suitable for: translation (no translations are included), reading the tablets themselves (no images, no sign coordinates), dating or provenance attribution as a classification target (catalogue metadata is missing or coarse for a fifth of the texts and is heavily skewed to Neo-Assyrian), or any claim about the original scribe's intent beyond what the editors transliterated.

## Limitations

- Coverage is uneven: 52% of texts and 58% of lemmatised words are Neo-Assyrian, mostly letters and queries from Nineveh and Kalhu; Old Akkadian, Old Assyrian and Ur III are represented by a few dozen texts. Spelling conventions learned from this dataset are Neo-Assyrian conventions first.
- 53% of word tokens have no normalization, because they are numerals, unlemmatised, or too broken to read. Preservation flags are per sign as recorded by the editors; the dataset does not re-examine the tablets.
- Normalization and lemmatisation are editorial decisions and differ between projects in detail (vowel-length marking, treatment of Assyrian vowel harmony, handling of loanwords). A word spelled the same way can carry slightly different normalizations in two volumes.
- 265 texts are edited in more than one project (for example a letter in both SAA 1 and SAA 19); only the first volume's edition is kept, so the other project's lemmatisation choices for that text are not represented.
- 88 corpus JSON files failed to parse and are omitted; 444 texts had no line structure (catalogue-only or empty editions) and are omitted.
- Sumerian words inside Akkadian texts carry `lang: sux` and are kept in `lines.jsonl` for context but are not Akkadian spellings; consumers must filter on `lang` if they want Akkadian only.
- Catalogue fields are free text as entered by each project (`Kuyunjik (Nineveh)`, `Nineveh`, `Kuyunjik` all occur) and were not normalised here.
- The transliteration keeps editorial marks that encode sign variants (`TA@v`) and compound signs (`|KA×KIB|`); these are rare and their conventions differ slightly between projects.

## License

CC BY-SA 3.0. Oracc releases its editions "under a Creative Commons Attribution Share-Alike license 3.0" (project pages and per-text credits); the JSON exports additionally state "This data is released under the CC0 license". The stricter CC BY-SA 3.0 is declared and observed: attribute Oracc and the editing projects named in `ATTRIBUTION.txt`, and distribute derivatives under the same licence. Licence URL: https://creativecommons.org/licenses/by-sa/3.0/
