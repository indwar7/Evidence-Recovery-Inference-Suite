# Dataset Description

## Overview

The complete data dump of the Research Organization Registry (ROR), release
v2.0 of 16 December 2025: 120,445 research organizations worldwide, each with
its identifier, canonical English display name, alternative names in other
languages, acronyms, aliases, location, type, founding year and relationships
to parent and child institutions.

ROR is a live community registry maintained by the ROR project and used by
publishers, funders and repositories to identify institutions unambiguously. It
is not a benchmark and was not assembled for machine learning. Every name in it
was contributed and curated by people who work with these institutions.

What makes it useful here is that the registry records, for the same
institution, both the canonical English display name and what the institution
calls itself in its own language. That is a large body of real paired name
data - 50,857 pairs across 134 languages - with no generation, translation or
augmentation involved on either side.

Data source: https://zenodo.org/records/17953395
DOI: 10.5281/zenodo.17953395
License: CC0 1.0 Universal (public domain dedication; commercial use permitted,
attribution not required)
Published by the Research Organization Registry project, 16 December 2025.
This is curated operational registry data, not simulated.

The file here is the JSON dump from that record, unmodified.

## File Structure

This dataset upload is a flat archive with four directories at the top level:

- raw/v2.0-2025-12-16-ror-data.json - the full registry dump, 120,445
  records, one JSON array. This is the only source of data.
- generator/prepare.py - the deterministic script that builds the split below
  from the raw file. Seed 20260801, pandas only.
- public/train.csv, public/test.csv, public/sample_submission.csv - the
  prepared public split, exactly as prepare.py writes it.
- private/answers.csv - the withheld English names for the test split.

The prepare.py script submitted with the task reads this file, extracts the
non-English label / English display name pairs, drops any label that occurs more than once in the registry
(compared case-insensitively with whitespace collapsed), splits the rest by
organization, and routes the English names for the test organizations to the
private grading key.

## Features

Each element of the JSON array is one organization. The fields the task uses
are marked; the rest are documented because they are present.

- id (str): ROR identifier, for example https://ror.org/04ttjf776. Used to key the organization-disjoint split.
- names (list): Used. Each entry is an object with value, types and lang.
- locations (list): Geonames record: country, subdivision, city, latitude, longitude.
- established (int): Founding year, where known.
- types (list): Education, healthcare, company, facility, government, nonprofit, archive, other.
- relationships (list): Parent, child, related and successor links to other ROR ids.
- external_ids (list): Cross-references to GRID, ISNI, Wikidata and Crossref Funder IDs.
- links (list): Official website and Wikipedia page.
- domains (list): Internet domains operated by the organization.
- status (str): active, inactive or withdrawn.
- admin (object): Record creation and last-modification metadata.


### The names list

- value (str): The name itself.
- types (list): One or more of ror_display, label, acronym, alias.
- lang (str): ISO 639 language code, or null.


Exactly one entry per organization carries ror_display; that is the canonical
English name and the task's target. Entries carrying label with a non-English
lang are the task's inputs. Across the registry there are 120,445
ror_display names, 157,487 label entries, 53,765 acronyms and 48,659
aliases.

## Task Construction

Pairs are formed by taking, for every organization that has an English display
name, each of its non-English labels:

- Organizations in the registry: 120,445
- Organizations contributing at least one pair: 46,869
- Pairs extracted from the registry: 51,690
- Pairs dropped because the label is not unique in the registry: 833
- Pairs kept: 50,857
- Pairs where source already equals target: 19,269 (37.9%)
- Distinct languages: 134 (99 of them present in the test split)
- Training pairs (80% of organizations): 40,698
- Test pairs (20% of organizations): 10,159


The largest source languages are French (7,605), Japanese (5,498), Chinese
(4,775), Spanish (4,721), German (4,626) and Portuguese (3,283), with a long
tail down to single-figure languages.

## Split Design

Before splitting, every label that occurs more than once in the registry -
compared case-insensitively with whitespace collapsed - is removed: 833
pairs. Such a label is either shared by two different organizations, which
gives one input two correct answers, or an exact repeat. Removing them guarantees that no feature row appears twice
anywhere, within a split or across the two.

The split is then by **organization**, not by pair, using a fixed seed. All of
an institution's labels land on the same side. This matters: several
organizations carry three or four labels in different languages, and a
row-level split would put the English display name in training and then ask
for the same string at test time. The task would collapse into a lookup.

With the organization-disjoint split there is nothing to look up. A solver has
to learn that universiteit, universite, universidad and universitaet
all become university, and apply it to institutions it has never seen.

## Measured Difficulty

Token-level F1 on the held-out organizations, computed with the task's own
grader.

- Empty output: 0.000
- Copy the source name through unchanged: 0.426
- Aligned word substitution table: 0.470
- Bag co-occurrence table: 0.486
- Per-language table with copy-through for rare tokens (reference): 0.489


Two properties are worth noting. The copy-through baseline is high because 38
percent of pairs are already identical and because proper nouns are shared
across both sides, so the headroom above it is what actually measures
translation skill. And a solver that translates aggressively scores worse than
one that knows when to leave a string alone.

The ceiling is well below 1.0 for any method that obeys the no-pretrained-model
rule. Recovering "University of Antwerp" from "Universiteit Antwerpen" requires
knowing that Antwerpen is Antwerp, and that fact is not derivable from the
training pairs unless that particular city happens to appear in them.
