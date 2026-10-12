#!/usr/bin/env python3
"""generate.py - deterministic generator for the "Budget-Constrained Index
Construction" challenge dataset.

The task
--------
A solver receives a corpus of documents (each a numbered list of sentences
with token counts), a query workload, and a per-document budget of index
windows. Each window is a contiguous span holding at most MAX_CHUNK_TOKENS
tokens; windows may NOT overlap and a document may carry at most budget_k of
them, so most of a document is left UN-indexed. The solver outputs the set of
index windows per document -- a capacity-constrained SELECTION, not a
partition.

A query is ANSWERED by an index iff some single window contains ALL of the
query's required sentences and NONE of its poison sentences (contradicting
statements that would corrupt a reader). The score is the mean over evaluation
queries -- a smooth, continuous quantity.

Where the difficulty lives
--------------------------
Every true fact sits beside a POISON TWIN: the same entity and attribute, the
same sentence template, a different value. There is no textual tell. Three
independent shortcuts are closed BY CONSTRUCTION, each an asserted invariant:

  1. Corpus-consensus voting.  Poison values come from a DONOR pool of real
     countries that never appear as entities, and every fact occurrence is
     matched by exactly one poison occurrence of the same (entity, attribute)
     pair.  Counting how often each value is stated therefore yields an exact
     tie -- frequency carries zero information.  (assert_balance)
  2. Knowledge-base membership.  Both the true value and the poison value are
     real values of real countries present in real_facts.json, so "is this
     number in the KB at all?" is uninformative.  Only an ENTITY-RESOLVED
     lookup separates them.  (donors are real rows of the KB)
  3. Positional priors.  Poison is placed before or after its fact with equal
     probability everywhere -- adjacent twins, cluster blocks and off-path
     poison alike.  (assert_placement)

What remains is the intended skill, and it is graded continuously: resolve the
entity (directly, via its ISO code, via its capital, or via a two-hop chain of
both), read the value the sentence actually states -- which may be written as
plain digits, comma-grouped digits, or a rounded phrase like "about 438.3
thousand" -- normalise it, and compare it to that entity's true value in the
shipped knowledge base.  A solver that matches integers exactly silently
misreads every comma-grouped and rounded surface; the generator asserts that
each such surface is still DECIDABLE (it lies strictly closer to its own source
value than to the competing twin), so the skill is real and not a coin flip.

Then, with the evidence known, allocate a scarce budget of token-bounded
non-overlapping windows across each document.

Why the score band is provable
------------------------------
Indexes are per-document and independent, so the EXACT optimum is computable
by a budgeted dynamic program (at most budget_k non-overlapping token-bounded
windows per document). The generator constructs the workload so that

  * a tuned fraction of queries is structurally unanswerable (required
    sentences farther apart than the token cap, or a poison sentence pinned
    between two required sentences), and
  * answerable queries deliberately conflict (interleaved windows that no
    legal index can satisfy simultaneously),

then verifies with the budgeted DP that the optimum lands in the target band
and that the measured anchors -- budget-blind tilings, a no-knowledge-base
guesser, a naive one-window-per-query allocator, a greedy allocator and an
exact-integer-matching "obvious pipeline" -- are all separated by the required
margins.  No submission can ever exceed the verified optimum.

Query anatomy (all evidence is textual, inside the same document):
  direct     : "What is the <attr> of <entity>?"        -> 1 fact sentence
  aliased    : entity asked by its ISO code             -> 2 sentences
  relational : entity asked by its capital              -> 2 sentences
  chained    : capital -> ISO code -> entity            -> 3 sentences

Determinism: numpy.random.default_rng (PCG64), fixed seed; two runs produce
byte-identical CSVs. Text is generated from closed template/vocabulary sets
(no external data, no LLM output). Only numpy + pandas are used.

Outputs:
  dataset/raw/documents.csv     doc_id, sent_id, n_tokens, text, budget_k
  dataset/raw/queries.csv       query_id, split, doc_id, question,
                                required_sent_ids, poison_sent_ids, qtype, cls
  dataset/raw/meta.json         MAX_CHUNK_TOKENS, verified score anchors

Usage:
    python generate.py [--out-dir ../raw]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

GLOBAL_SEED = 20260715
MAX_CHUNK_TOKENS = 60

# Budget-constrained index construction: per document you may build at most
# budget_k index windows (contiguous spans of at most MAX_CHUNK_TOKENS tokens),
# leaving the rest of the document UN-indexed. budget_k = ceil(KFRAC * #queries)
# so the budget binds: you cannot index every answerable region and must choose.
KFRAC = 0.45
BUDGET_MIN = 2

N_DOCS = 220
N_TRAIN_Q = 600
N_TEST_Q = 800

# ---------------------------------------------------------------------------
# Workload composition knobs. The auto-tuning loop in main() moves
# FRAC_IMPOSSIBLE and FRAC_HARD_SURFACE; the rest are fixed.
# ---------------------------------------------------------------------------
FRAC_IMPOSSIBLE = 0.10   # structurally unanswerable queries
FRAC_CONFLICT = 0.10     # interleaved pairs: any segmentation satisfies <=1 of 2
FRAC_TIGHT = 0.14        # answerable but wide windows (need deliberate chunks)
FRAC_CLUSTER = 0.30      # several facts packed in one window: merging saves
                         # budget (an allocator that spends one window per query
                         # here starves the rest of the document)
# remainder: comfortably answerable

# Fraction of TRUE fact sentences rendered with a non-plain numeric surface
# (comma-grouped digits or a rounded phrase). This is the primary difficulty
# knob: a solver that matches integers exactly, or that pulls the last \d+ run
# out of the sentence, misreads exactly these and must then guess between the
# twins. Auto-tuned so the "obvious pipeline" anchor lands inside AGENT_BAND.
FRAC_HARD_SURFACE = 0.55

# Target bands enforced by the auto-tuner (see gates() / main()).
OPT_BAND = (0.62, 0.85)        # verified optimum (the ceiling)
AGENT_BAND = (0.42, 0.52)      # obvious-pipeline anchor: exact-integer matching
                               # (kept inside the required 0.40-0.60 agent band,
                               # centred so there is headroom to the ceiling)
BLIND_MAX = 0.18               # budget-blind tilings must stay at or below this
MIN_ALLOC_GAP = 0.10           # optimum - naive-per-query allocator
MIN_KB_GAP = 0.15              # optimum - no-knowledge-base guesser
#
# NOTE on greedy: a greedy max-gain allocator is measured every attempt and
# reported, but it is NOT gated. Empirically it ties the budgeted DP exactly on
# these instances -- documents are small and each query's evidence is local, so
# the highest-gain window almost never blocks a better pair. The allocation
# skill this workload really grades is therefore budget-AWARENESS (merging
# clustered facts, conceding conflicts) rather than exact optimisation: the
# naive one-window-per-query allocator gives up MIN_ALLOC_GAP or more. The
# published anchors say exactly this rather than claiming a DP is required.

# ---------------------------------------------------------------------------
# Real-world reference facts (baked-in, self-contained)
# ---------------------------------------------------------------------------
# Planted facts are REAL: every entity is a real country and every fact value
# is that country's real attribute (population, land area, count of bordering
# countries, count of official languages), with its real ISO 3166-1 alpha-3
# code and real capital city. The snapshot `real_facts.json` ships with the
# dataset, so generation is self-contained (no live download, no source URL).
# Provenance: country attributes assembled from public reference data
# (ISO 3166 country codes, capitals, and openly published population / area /
# border / language figures).
REAL_FACTS = []    # all rows of the knowledge base
ENTITY_POOL = []   # rows usable as a question's subject
DONOR_POOL = []    # rows used ONLY to source poison values, never as subjects
NAME_IDX = {}      # name -> index in REAL_FACTS

# Every third row is reserved as a donor. Donors stay in the shipped KB (so a
# "is this value in the KB?" test cannot separate a fact from its twin) but are
# never the subject of a fact, so a poison value can never be traced back to
# some other entity mentioned in the corpus.
DONOR_STRIDE = 3

# ---------------------------------------------------------------------------
# Attribute phrasing
#
# THE POINT OF THIS BLOCK. In earlier versions of this dataset an attribute had
# exactly one wording, and a question and its answering sentence used the SAME
# wording. Matching a question to its fact was therefore literal string
# containment: a finite regex covered the whole workload, every competent solver
# wrote the same regex, and every competent solver landed on the same score.
# A reviewer correctly called that an optimization puzzle that does not rank
# solutions.
#
# Here each attribute has MANY wordings, and a question and its answering
# sentence draw their wording INDEPENDENTLY. A question may ask "how many people
# live in X" while the sentence that answers it says "X: resident count ...".
# There is no literal overlap to key on. A solver has to decide that two
# different phrases denote the same attribute, and the quality of that decision
# is what the score now ranks.
#
# Wordings are additionally split: TEST_ONLY wordings never appear in the
# training split, so a lookup table harvested from the labelled data cannot
# cover the evaluation set. A solver that generalises across wordings -- by
# token overlap, edit distance, embeddings, or a learned mapping -- keeps
# scoring; a solver that enumerates what it saw does not.
#
# Wordings are disjoint across attributes (asserted at build time), so the
# mapping from wording to attribute is always well defined and the task stays
# exactly decidable.
# ---------------------------------------------------------------------------

ATTR_PHRASINGS = {
    "population": {
        "train": ["population", "resident population", "number of people",
                  "number of residents", "count of inhabitants",
                  "total number of people", "people living there",
                  "resident headcount"],
        "test_only": ["inhabitant count", "resident count",
                      "number of inhabitants", "headcount of residents"],
        "units": [" people", " residents", " inhabitants"],
    },
    "area": {
        "train": ["land area", "surface area", "area of land", "landmass size",
                  "size of the territory", "total land area",
                  "extent of its territory"],
        "test_only": ["territorial extent", "landmass area", "expanse of land",
                      "size of its landmass"],
        "units": [" square kilometers", " sq km", " square km"],
    },
    "num_borders": {
        "train": ["count of bordering countries", "number of land neighbours",
                  "number of bordering states", "tally of land borders",
                  "count of neighbouring countries", "number of shared borders"],
        "test_only": ["count of adjoining nations", "number of border countries",
                      "tally of neighbouring states"],
        "units": [" countries", " neighbours", " bordering states"],
    },
    "num_languages": {
        "train": ["count of official languages", "number of official tongues",
                  "official language count", "tally of recognised languages",
                  "number of state languages"],
        "test_only": ["count of recognised tongues", "official tongue tally",
                      "number of designated languages"],
        "units": [" languages", " official languages", " tongues"],
    },
}

ATTR_KEYS = ["population", "area", "num_borders", "num_languages"]

# attributes whose value range is wide enough to carry rounded surface forms
WIDE_ATTRS = {"population", "area"}


def attr_phrases(attr_key, split):
    """Wordings usable for this attribute in this split.

    The training split sees only its own wordings. The evaluation split sees
    those PLUS the held-out ones, so some evaluation questions are phrased in
    ways no training example ever shows.
    """
    spec = ATTR_PHRASINGS[attr_key]
    return spec["train"] if split == "train" else spec["train"] + spec["test_only"]


def assert_phrasings_disjoint():
    """No wording may be usable for two different attributes, and no wording may
    be a substring of a wording belonging to a different attribute -- otherwise
    the phrase-to-attribute mapping would be ambiguous and the task would not be
    decidable."""
    owner = {}
    for key, spec in ATTR_PHRASINGS.items():
        for p in spec["train"] + spec["test_only"]:
            assert p not in owner, f"wording {p!r} claimed by two attributes"
            owner[p] = key
    for a in owner:
        for b in owner:
            if owner[a] != owner[b]:
                assert a not in b, f"wording {a!r} is a substring of {b!r}"
    return len(owner)


def load_real_facts(here):
    """Load the baked real-world country reference snapshot and split it into
    the entity pool (question subjects) and the donor pool (poison values)."""
    global REAL_FACTS, ENTITY_POOL, DONOR_POOL, NAME_IDX
    path = os.path.join(here, "real_facts.json")
    with open(path, encoding="utf-8") as f:
        REAL_FACTS = json.load(f)
    assert len(REAL_FACTS) >= 150, f"too few real facts: {len(REAL_FACTS)}"
    REAL_FACTS.sort(key=lambda r: r["cca3"])          # stable, source-order free
    NAME_IDX = {r["name"]: i for i, r in enumerate(REAL_FACTS)}
    ENTITY_POOL = [r for i, r in enumerate(REAL_FACTS) if i % DONOR_STRIDE]
    DONOR_POOL = [r for i, r in enumerate(REAL_FACTS) if not i % DONOR_STRIDE]
    assert len(ENTITY_POOL) >= 100 and len(DONOR_POOL) >= 40


FACT_TEMPLATES = [
    "The {attr} of {ent} is {val}{unit}.",
    "{ent} reports a {attr} of {val}{unit}.",
    "According to reference data, the {attr} of {ent} is {val}{unit}.",
    "Records list the {attr} of {ent} at {val}{unit}.",
    "For {ent}, the {attr} stands at {val}{unit}.",
    "{ent} is recorded with a {attr} of {val}{unit}.",
    "The register gives {ent} a {attr} of {val}{unit}.",
    "Filed for {ent}: a {attr} of {val}{unit}.",
    "A {attr} of {val}{unit} is listed against {ent}.",
    "{ent} shows a {attr} of {val}{unit} in the reference tables.",
    "Under {ent} the {attr} is entered as {val}{unit}.",
    "The tabulated {attr} for {ent} comes to {val}{unit}.",
]

# Poison sentences reuse their own fact's template, wording and numeric surface
# form, and carry a DIFFERENT real country's real value for the same attribute.
# They contain NO textual tell (no "outdated", "wrongly", "disputed", ...): once
# the number is masked a poison sentence is character-for-character identical to
# the true fact it shadows. Telling them apart requires resolving the entity and
# cross-referencing its true value in the shipped knowledge base.
POISON_TEMPLATES = FACT_TEMPLATES

ALIAS_TEMPLATES = [
    "The country with ISO code {code} is {ent}.",
    "ISO 3166 code {code} is assigned to {ent}.",
    "Code {code} in ISO 3166 denotes {ent}.",
    "{ent} is registered under the code {code}.",
    "The alpha-3 code {code} belongs to {ent}.",
]

LINK_TEMPLATES = [
    "The country whose capital is {cap} is {ent}.",
    "{cap} is the capital of {ent}.",
    "{ent} governs from its capital, {cap}.",
    "The seat of government of {ent} is {cap}.",
    "{cap} functions as the capital city of {ent}.",
]

# two-hop: capital -> ISO code (the code is then resolved by an ALIAS sentence)
CHAIN_TEMPLATES = [
    "The capital of the country with ISO code {code} is {cap}.",
    "{cap} serves as the capital of ISO 3166 country {code}.",
    "The state registered as {code} has {cap} as its capital.",
    "Capital city of entry {code}: {cap}.",
]

# Carrier "filler" is synthetic naturalistic business/registry prose,
# composed deterministically from grammatical slot templates + vocabularies.
# It is self-contained (no external download, no source URL, so there is
# nothing for a duplicate detector to match) and irrelevant to the task -- it
# is only the surrounding text between planted facts. FILLER_POOL is built
# once at load time.
FILLER_POOL = []

_F_TIME = ["last quarter", "this cycle", "earlier in the week",
           "over the review period", "ahead of the audit", "during the recess",
           "before the annual meeting", "since the reorganization",
           "throughout the fiscal year", "at the start of the term",
           "following the site visit", "in the interim", "on short notice",
           "without further delay", "for the third consecutive period"]
_F_ACTOR = ["The regional office", "The oversight committee", "The duty roster",
            "The steering group", "The facilities team", "The archive desk",
            "The compliance unit", "The planning board", "The logistics wing",
            "The reception log", "The standards panel", "The visiting delegation",
            "The night shift", "The procurement office", "The records room",
            "The maintenance crew", "The advisory council", "The front desk"]
_F_ACTION = ["reviewed the standing agenda", "filed the routine correspondence",
             "updated the distribution list", "posted the revised schedule",
             "signed off on the walkthrough", "archived the meeting minutes",
             "circulated the internal memo", "confirmed the seating plan",
             "approved the housekeeping notes", "logged the visitor entries",
             "rescheduled the briefing", "tidied the notice board",
             "collated the weekly summary", "reset the booking calendar",
             "acknowledged the handover notes", "renewed the parking passes",
             "restocked the supply cabinet", "verified the attendance sheet"]
_F_TAIL = ["without further action", "as a matter of routine",
           "pending confirmation", "for the record", "and adjourned promptly",
           "with no items outstanding", "ahead of schedule",
           "to keep matters tidy", "and moved on", "in the usual manner",
           "before closing up", "with little discussion",
           "and noted the outcome", "as previously agreed", "for completeness"]
_F_STANDALONE = [
    "No notable events were reported for the period.",
    "Standard procedures were followed during the walkthrough.",
    "Weather conditions delayed the outbound shipment by a day.",
    "Interns completed the orientation module on time.",
    "The canteen menu for the season was posted on the board.",
    "A minor clerical revision was applied to the cover sheet.",
    "Visitors are reminded to sign the logbook on arrival.",
    "The stationery order arrived in two separate parcels.",
    "The elevator inspection passed without any remarks.",
    "Refreshments were arranged for the afternoon session.",
    "The lost-and-found box was cleared at the end of the day.",
    "Parking was limited owing to the ongoing resurfacing work.",
]


def load_filler(here):
    """Build a large pool of unique naturalistic filler sentences from closed
    slot vocabularies. Fully synthetic and deterministic; no external source.
    Sentences are 5-18 tokens, plain ASCII, no reserved characters."""
    global FILLER_POOL
    pool, seen = [], set()

    def push(s):
        if s in seen:
            return
        if not all(32 <= ord(c) < 127 for c in s):
            return
        if any(c in s for c in '"|[]()*_'):
            return
        nt = len(s.split())
        if 5 <= nt <= 18:
            seen.add(s)
            pool.append(s)

    for s in _F_STANDALONE:
        push(s)
    # deterministic nested composition (no RNG, no hashing): actor + action +
    # extra, thinned by a fixed stride over a running counter so the pool is
    # varied but byte-identical every run.
    extras = _F_TIME + _F_TAIL
    counter = 0
    for actor in _F_ACTOR:
        for action in _F_ACTION:
            for extra in extras:
                counter += 1
                if counter % 4 != 0:      # keep 1 in 4 -> ~1600 sentences
                    continue
                push(f"{actor} {action} {extra}.")
    FILLER_POOL = pool
    assert len(FILLER_POOL) >= 1400, f"filler pool too small: {len(pool)}"


# Question frames. Several per form, so the frame itself is not a key either --
# what identifies the form is which referring expression it uses (a name, an ISO
# code, a capital, or a capital that must be resolved through a code).
QUESTION_TEMPLATES = {
    "direct": [
        "What is the {attr} of {ent}?",
        "For {ent}, what {attr} is on record?",
        "Give the {attr} of {ent}.",
        "{ent} has what {attr}?",
        "Report the {attr} recorded for {ent}.",
        "Which {attr} is listed for {ent}?",
    ],
    "aliased": [
        "What is the {attr} of the country with ISO code {code}?",
        "For the country registered as {code}, what is the {attr}?",
        "Give the {attr} of ISO 3166 entry {code}.",
        "Which {attr} is recorded for the state coded {code}?",
    ],
    "relational": [
        "What is the {attr} of the country whose capital is {cap}?",
        "For the country capitalled at {cap}, what is the {attr}?",
        "Give the {attr} of the country governed from {cap}.",
        "Which {attr} belongs to the country whose seat of government is {cap}?",
    ],
    "chained": [
        "What is the {attr} of the country whose capital city is {cap}?",
        "Taking the country whose capital city is {cap}, what is its {attr}?",
        "Give the {attr} of the state whose capital city is {cap}.",
        "Which {attr} is on record for the country with capital city {cap}?",
    ],
}


def n_tokens(text):
    return len(text.split())


# ---------------------------------------------------------------------------
# Poison values: deterministic, donor-sourced, frequency-neutral
# ---------------------------------------------------------------------------
_WRONG_CACHE = {}
_DIR_BALANCE = 0   # running bigger/smaller balance for poison values


def wrong_value_for(fact, attr_key):
    """The poison value for one (entity, attribute) pair.

    Deterministic in the pair alone -- no RNG -- so every occurrence of that
    pair anywhere in the corpus states the SAME wrong value. Combined with the
    one-poison-per-fact rule this makes value frequencies tie exactly, which is
    what kills corpus-consensus voting.

    The value is a real attribute of a real country drawn from DONOR_POOL, so
    it is present in the shipped knowledge base (a membership test cannot
    separate it from the truth) but is never attributed to any entity that
    appears in the corpus (so it cannot be traced to a second subject).
    """
    key = attr_key
    ckey = (fact["name"], attr_key)
    if ckey in _WRONG_CACHE:
        return _WRONG_CACHE[ckey]
    v = int(fact[key])
    base = NAME_IDX[fact["name"]] * 7 + ATTR_KEYS.index(attr_key) * 29
    # MAGNITUDE SYMMETRY: the poison must be larger than the truth about half
    # the time and smaller the other half. Without this, "the bigger number is
    # the poison" (or "the one with more digits") is a free statistical tell
    # that needs no knowledge base at all.
    #
    # A fixed parity is not enough: a country with no bigger donor (the most
    # populous ones) or no smaller one (v = 0 borders) can only go one way, and
    # those forced cases skew the corpus. So the direction is steered by a
    # running balance -- deterministic, because pairs are resolved in a fixed
    # generation order -- which self-corrects whenever a pair is forced.
    # Digit length is the other visible statistic. A donor drawn from the same
    # decade as the truth makes the two numbers the same length on the page, so
    # "the longer number is the fact" carries no information either; the 25%
    # separation still fits comfortably inside one decade.
    global _DIR_BALANCE
    want_bigger = _DIR_BALANCE <= 0
    ndig = len(str(v))
    best = None
    #                same length?  honour direction?
    for same_len, require_dir in ((True, True), (True, False),
                                  (False, True), (False, False)):
        for step in range(len(DONOR_POOL)):
            w = int(DONOR_POOL[(base + step) % len(DONOR_POOL)][key])
            if w == v or w <= 0:
                continue
            if same_len and len(str(w)) != ndig:
                continue
            if require_dir and ((w > v) != want_bigger):
                continue
            # keep the twins far enough apart that a rounded surface of one can
            # never be mistaken for the other (see render_pair's decidability
            # assertion): >= 25% relative separation on the wide attributes.
            if key in WIDE_ATTRS and abs(w - v) < 0.25 * max(v, 1):
                continue
            best = w
            break
        if best is not None:
            break
    if best is None:
        best = v + 1 if v > 0 else 2
    _DIR_BALANCE += 1 if best > v else -1
    _WRONG_CACHE[ckey] = best
    return best


# ---------------------------------------------------------------------------
# Numeric surface forms
#
# The same number can be written several ways. Plain digits are trivially
# parseable; the other forms are not, and they are what separates a solver
# that genuinely reconciles a stated value against the knowledge base from one
# that matches integers exactly or greps the last run of digits.
# ---------------------------------------------------------------------------


SURFACE_STYLES = ["plain", "grouped", "nearly", "thousand", "million"]
HARD_STYLES = {"grouped", "nearly", "thousand", "million"}


def _surfaces(v, wide):
    """style -> (text, denoted_value) renderings of `v`.

    `denoted_value` is the number the text literally says, which for the
    rounded styles is NOT `v`. That gap is the whole point: a solver has to
    normalise the surface and reconcile it against the knowledge base with a
    tolerance, instead of testing integer equality.
    """
    out = {"plain": (f"{v}", v)}
    if not wide or v < 10000:
        return out
    out["grouped"] = (f"{v:,}", v)                              # 438,317
    r3 = int(round(v, -3))
    if r3 > 0:
        out["nearly"] = (f"nearly {r3:,}", r3)                  # nearly 438,000
    out["thousand"] = (f"about {v / 1000:.1f} thousand",
                       int(round(v / 1000, 1) * 1000))          # about 438.3 thousand
    if v >= 1_000_000:
        out["million"] = (f"roughly {v / 1e6:.2f} million",
                          int(round(v / 1e6, 2) * 1e6))         # roughly 1.23 million
    return out


def render_pair(rng, v, w, attr_key, want_hard):
    """Render the true value `v` and its poison twin `w` in the SAME surface
    style. Returns (fact_surface, poison_surface, is_hard).

    Matching the style is what removes the last textual tell: if the truth
    could be written "about 438.3 thousand" while its twin were always plain
    digits, the rendering style itself would leak which sentence is which. With
    the style shared, the two sentences are identical token for token apart
    from the number they carry.

    DECIDABILITY INVARIANT: for each of the two, the number its surface denotes
    must lie strictly closer to its own source value than to the other's, so a
    solver that normalises correctly is never forced to guess. Styles failing
    this for either side are discarded; "plain" denotes each value exactly and
    is always available as a fallback.
    """
    wide = attr_key in WIDE_ATTRS
    fs, ps = _surfaces(int(v), wide), _surfaces(int(w), wide)
    ok = []
    for style in SURFACE_STYLES:
        if style not in fs or style not in ps:
            continue
        (ftext, fden), (ptext, pden) = fs[style], ps[style]
        if abs(fden - v) >= abs(fden - w):        # fact would read as the twin
            continue
        if abs(pden - w) >= abs(pden - v):        # twin would read as the fact
            continue
        ok.append((style, ftext, ptext))
    assert ok, f"no decidable shared surface for {v} vs {w}"
    hard_ok = [c for c in ok if c[0] in HARD_STYLES]
    plain_ok = [c for c in ok if c[0] not in HARD_STYLES]
    pick = hard_ok if (want_hard and hard_ok) else (plain_ok or ok)
    # Comma grouping is the cheap half of the skill -- stripping separators is
    # the first thing anyone tries. The ROUNDED phrasings are the half that
    # needs an actual tolerant reconciliation against the knowledge base, so
    # they carry most of the weight; otherwise the difficulty collapses the
    # moment a solver calls .replace(",", "").
    rounded = [c for c in pick if c[0] in ("nearly", "thousand", "million")]
    if rounded and rng.random() < 0.75:
        pick = rounded
    style, ftext, ptext = pick[int(rng.integers(len(pick)))]
    return ftext, ptext, style in HARD_STYLES


def make_twin_sentences(rng, fact, attr_key, wrong, want_hard, split):
    """Build the true fact sentence and its poison twin together.

    The two share a template, an attribute wording, a unit and a numeric surface
    style, so once the number is masked they are character-for-character
    identical and there is nothing to key on but the value itself.

    The wording is drawn from the pool allowed for this example's split, and
    INDEPENDENTLY of the wording its question will use -- that independence is
    what stops a solver from matching a question to its answer by string
    overlap.
    """
    val = int(fact[attr_key])
    fsurf, psurf, hard = render_pair(rng, val, wrong, attr_key, want_hard)
    phrases = attr_phrases(attr_key, split)
    phrase = phrases[int(rng.integers(len(phrases)))]
    units = ATTR_PHRASINGS[attr_key]["units"]
    unit = units[int(rng.integers(len(units)))]
    t = FACT_TEMPLATES[int(rng.integers(len(FACT_TEMPLATES)))]
    return (t.format(attr=phrase, ent=fact["name"], val=fsurf, unit=unit),
            t.format(attr=phrase, ent=fact["name"], val=psurf, unit=unit),
            hard)


# ---------------------------------------------------------------------------
# Document assembly
#
# Each document is built as a list of "slots". We first lay out filler, then
# place query evidence with controlled gaps measured in TOKENS, because the
# chunk constraint is in tokens. Placement classes:
#   easy       : evidence sentences adjacent or 1 filler apart
#   tight      : evidence gap just UNDER the cap (needs a deliberate chunk)
#   impossible : evidence gap OVER the cap, or poison pinned between evidence
#   cluster    : several facts packed so ONE window answers all of them
#   conflict   : two queries interleaved so no legal index satisfies both
#
# INVARIANT: exactly one poison sentence is planted per true fact sentence,
# and its side (before / after the fact) is drawn with probability 1/2.
# ---------------------------------------------------------------------------


class DocBuilder:
    def __init__(self, rng, doc_id):
        self.rng = rng
        self.doc_id = doc_id
        self.sentences = []          # list of text
        self.queries = []            # dicts with required/poison slot indices

    def add(self, text):
        self.sentences.append(text)
        return len(self.sentences) - 1

    def add_filler(self, k):
        for _ in range(k):
            self.add(FILLER_POOL[int(self.rng.integers(len(FILLER_POOL)))])


_FILLER_AVG = None


def gap_filler_count(rng, target_tokens):
    """How many filler sentences approximate target_tokens of gap."""
    global _FILLER_AVG
    if _FILLER_AVG is None:
        _FILLER_AVG = float(np.mean([n_tokens(t) for t in FILLER_POOL]))
    return max(0, int(round(target_tokens / _FILLER_AVG)))


def _name_collides(name, used):
    """True if `name` is a substring of, or superstring of, any used name
    (real country names like Niger/Nigeria, Sudan/South Sudan would make text
    evidence ambiguous within one document)."""
    for u in used:
        if name in u or u in name:
            return True
    return False


def make_query_material(rng, qtype, used_names=None, want_hard=False,
                        split="train"):
    """Returns (evidence sentences, question text, meta), all grounded in a
    real country and its real attribute value. Picks a country whose name does
    not collide (substring either way) with names already used in this
    document. The FACT sentence is always evidence[-1].

    The question's wording for the attribute is drawn INDEPENDENTLY of the
    fact sentence's wording for the same attribute. A question may ask for the
    "number of residents" while the sentence that answers it records a
    "resident headcount": there is no lexical overlap to match on, and a solver
    has to decide the two denote the same attribute.

    meta = (fact_row, attr_key, poison_sentence_text, used_hard_surface)
    """
    used_names = used_names if used_names is not None else set()
    fact = ENTITY_POOL[int(rng.integers(len(ENTITY_POOL)))]
    for _ in range(30):
        if not _name_collides(fact["name"], used_names):
            break
        fact = ENTITY_POOL[int(rng.integers(len(ENTITY_POOL)))]
    attr_key = ATTR_KEYS[int(rng.integers(len(ATTR_KEYS)))]
    wrong = wrong_value_for(fact, attr_key)
    fs, poison_text, hard = make_twin_sentences(rng, fact, attr_key, wrong,
                                                want_hard, split)
    # the question's own wording -- a separate draw from the same split pool
    qphrases = attr_phrases(attr_key, split)
    qphrase = qphrases[int(rng.integers(len(qphrases)))]

    def frame(form):
        opts = QUESTION_TEMPLATES[form]
        return opts[int(rng.integers(len(opts)))]

    if qtype == "direct":
        evidence = [fs]
        question = frame("direct").format(attr=qphrase, ent=fact["name"])
    elif qtype == "aliased":
        alias = ALIAS_TEMPLATES[int(rng.integers(len(ALIAS_TEMPLATES)))]
        evidence = [alias.format(ent=fact["name"], code=fact["cca3"]), fs]
        question = frame("aliased").format(attr=qphrase, code=fact["cca3"])
    elif qtype == "chained":
        # capital -> ISO code -> country -> fact : three evidence sentences
        chain = CHAIN_TEMPLATES[int(rng.integers(len(CHAIN_TEMPLATES)))]
        alias = ALIAS_TEMPLATES[int(rng.integers(len(ALIAS_TEMPLATES)))]
        evidence = [chain.format(cap=fact["capital"], code=fact["cca3"]),
                    alias.format(ent=fact["name"], code=fact["cca3"]), fs]
        question = frame("chained").format(attr=qphrase, cap=fact["capital"])
    else:  # relational (via real capital -> country)
        link = LINK_TEMPLATES[int(rng.integers(len(LINK_TEMPLATES)))]
        evidence = [link.format(cap=fact["capital"], ent=fact["name"]), fs]
        question = frame("relational").format(attr=qphrase, cap=fact["capital"])
    return evidence, question, (fact, attr_key, poison_text, hard)


def build_conflict_pair(b, rng, planA, planB, used_names, want_hard):
    # each plan carries its own split; wording pools follow from it
    """Interleave two 2-span queries A1 B1 A2 B2 so that each fits the cap on
    its own but no legal segmentation can satisfy both (verified in tokens).
    The two poison twins are placed as a block on a randomly chosen side of the
    interleaved region, so the token arithmetic of the conflict is untouched.
    Returns True on success."""
    for _ in range(12):
        evA, qA, mA = make_query_material(rng, planA["qtype"], used_names,
                                          want_hard, planA["split"])
        evB, qB, mB = make_query_material(
            rng, planB["qtype"], used_names | {mA[0]["name"]}, want_hard,
            planB["split"])
        tA1, tA2 = n_tokens(evA[0]), n_tokens(evA[1])
        tB1, tB2 = n_tokens(evB[0]), n_tokens(evB[1])

        # filler slots: g1 between A1,B1 (widens A's window); g3 between
        # A2,B2 (widens B's window); both widen the joint window. Search for
        # counts that keep each window legal but push the joint one over cap.
        fillers = [FILLER_POOL[int(rng.integers(len(FILLER_POOL)))]
                   for _ in range(6)]
        ftok = [n_tokens(s) for s in fillers]
        found = None
        for g1 in range(4):
            for g3 in range(4):
                add1 = sum(ftok[:g1])
                add3 = sum(ftok[3:3 + g3])
                w_A = tA1 + tB1 + tA2 + add1
                w_B = tB1 + tA2 + tB2 + add3
                w_AB = tA1 + tB1 + tA2 + tB2 + add1 + add3
                if (w_A <= MAX_CHUNK_TOKENS and w_B <= MAX_CHUNK_TOKENS
                        and w_AB > MAX_CHUNK_TOKENS):
                    found = (g1, g3)
                    break
            if found:
                break
        if found is None:
            continue
        g1, g3 = found

        poiA, poiB = mA[2], mB[2]
        before = rng.random() < 0.5
        pid = {}
        if before:                     # poison block ahead of the interleave
            b.add_filler(int(rng.integers(0, 2)))
            pid["A"], pid["B"] = b.add(poiA), b.add(poiB)
            b.add_filler(int(rng.integers(1, 3)))

        reqA = [b.add(evA[0])]
        for s in fillers[:g1]:
            b.add(s)
        reqB = [b.add(evB[0])]
        reqA.append(b.add(evA[1]))
        for s in fillers[3:3 + g3]:
            b.add(s)
        reqB.append(b.add(evB[1]))

        if not before:                 # poison block after the interleave
            b.add_filler(int(rng.integers(0, 2)))
            pid["A"], pid["B"] = b.add(poiA), b.add(poiB)

        b.queries.append({"qtype": planA["qtype"], "cls": "conflict",
                          "question": qA, "required": reqA,
                          "poison": [pid["A"]], "hard": mA[3],
                          "ent": mA[0]["name"], "attr_key": mA[1],
                          "split": planA["split"]})
        b.queries.append({"qtype": planB["qtype"], "cls": "conflict",
                          "question": qB, "required": reqB,
                          "poison": [pid["B"]], "hard": mB[3],
                          "ent": mB[0]["name"], "attr_key": mB[1],
                          "split": planB["split"]})
        b.add_filler(int(rng.integers(1, 4)))
        used_names.update({mA[0]["name"], mB[0]["name"]})
        return True
    return False


def build_cluster_group(b, rng, plans, used_names, want_hard):
    """Pack several DIRECT facts consecutively so ONE window (<= cap tokens)
    answers all of them at once. The budget then rewards MERGING: an optimal
    allocator spends a single window here and saves the rest for elsewhere,
    while a per-query allocator burns one window each.

    Every member's poison twin goes into a block on a randomly chosen side of
    the fact run, so a window that greedily swallows past the facts loses the
    whole cluster. Returns #queries added."""
    mats = []
    for pl in plans:
        ev, q, (fact, ai, poison_text, hard) = make_query_material(
            rng, "direct", used_names, want_hard, pl["split"])
        used_names.add(fact["name"])
        mats.append((ev[0], q, fact, ai, poison_text, hard))
    # keep as many as fit one window (leave a little headroom under the cap)
    C = len(mats)
    while C >= 1 and sum(n_tokens(m[0]) for m in mats[:C]) > MAX_CHUNK_TOKENS - 4:
        C -= 1
    if C < 2:
        return 0                       # not a usable cluster
    mats = mats[:C]

    poisons = [m[4] for m in mats]
    before = rng.random() < 0.5

    b.add_filler(int(rng.integers(1, 3)))
    pids = []
    if before:
        pids = [b.add(p) for p in poisons]
        b.add_filler(int(rng.integers(0, 2)))
    ids = [b.add(m[0]) for m in mats]                  # consecutive facts
    if not before:
        b.add_filler(int(rng.integers(0, 2)))
        pids = [b.add(p) for p in poisons]

    for idx, m in enumerate(mats):
        b.queries.append({"qtype": "direct", "cls": "cluster", "question": m[1],
                          "required": [ids[idx]], "poison": [pids[idx]],
                          "hard": m[5], "ent": m[2]["name"], "attr_key": m[3],
                          "split": plans[idx]["split"]})
    b.add_filler(int(rng.integers(1, 4)))
    return C


def build_doc(rng, doc_id, plans, frac_hard):
    """plans: list of query plans to embed. Returns sentences + queries with
    resolved sentence ids."""
    b = DocBuilder(rng, doc_id)
    b.add_filler(int(rng.integers(2, 5)))
    used_names = set()

    def want_hard():
        return bool(rng.random() < frac_hard)

    conflicts = [p for p in plans if p["cls"] == "conflict"]
    clusters = [p for p in plans if p["cls"] == "cluster"]
    others = [p for p in plans if p["cls"] not in ("conflict", "cluster")]

    # cluster plans: group into bunches of 2-4 and pack each
    while len(clusters) >= 2:
        take = min(len(clusters), 2 + int(rng.integers(0, 3)))   # 2..4
        grp = [clusters.pop() for _ in range(take)]
        if build_cluster_group(b, rng, grp, used_names, want_hard()) < 2:
            others += [{"cls": "easy", "qtype": "direct",
                        "split": g["split"]} for g in grp]
    for p in clusters:                       # leftover single -> easy
        others.append({"cls": "easy", "qtype": p["qtype"],
                       "split": p["split"]})
    while len(conflicts) >= 2:
        pA, pB = conflicts.pop(), conflicts.pop()
        if not build_conflict_pair(b, rng, pA, pB, used_names, want_hard()):
            others += [{"cls": "tight", "qtype": pA["qtype"],
                        "split": pA["split"]},
                       {"cls": "tight", "qtype": pB["qtype"],
                        "split": pB["split"]}]
    for p in conflicts:
        others.append({"cls": "tight", "qtype": p["qtype"],
                       "split": p["split"]})

    for plan in others:
        qtype, cls = plan["qtype"], plan["cls"]
        evidence, question, (fact, attr_key, twin, hard) = \
            make_query_material(rng, qtype, used_names, want_hard(),
                                plan["split"])
        used_names.add(fact["name"])
        required, poison = [], []
        ev_tokens = sum(n_tokens(s) for s in evidence)

        if cls == "easy":
            gap = int(rng.integers(0, 2))
        elif cls == "tight":
            budget = MAX_CHUNK_TOKENS - ev_tokens - int(rng.integers(4, 12))
            gap = gap_filler_count(rng, max(0, budget))
        else:  # impossible via distance
            over = MAX_CHUNK_TOKENS - ev_tokens + int(rng.integers(15, 45))
            gap = gap_filler_count(rng, max(0, over)) + 1

        if len(evidence) == 1:
            # DIRECT query -> the poison twin sits IMMEDIATELY ADJACENT in
            # RANDOMIZED order. The twin is word-for-word identical to the true
            # fact except the value it states, so position carries no signal --
            # a solver must resolve the entity and reconcile the stated value
            # against real_facts.json to know which of the two single-sentence
            # windows is the fact. Because the two sit adjacent, excluding the
            # twin forces a 1-sentence window on the fact; a solver that cannot
            # tell them apart must hedge (spend a second window, straining the
            # budget) or guess (miss half). This is the skill the score ranks.
            if rng.random() < 0.5:
                poison.append(b.add(twin))
                required.append(b.add(evidence[0]))
            else:
                required.append(b.add(evidence[0]))
                poison.append(b.add(twin))
        else:
            # multi-span: link sentence(s) first, fact last
            if cls == "impossible" and rng.random() < 0.5:
                # poison pinned between evidence: any chunk covering the whole
                # evidence span must include it -> structurally unanswerable
                for s in evidence[:-1]:
                    required.append(b.add(s))
                mid = max(1, gap // 2)
                b.add_filler(mid)
                poison.append(b.add(twin))
                b.add_filler(max(0, gap - mid))
                required.append(b.add(evidence[-1]))
            else:
                before = rng.random() < 0.5
                if before:
                    poison.append(b.add(twin))
                    b.add_filler(int(rng.integers(0, 2)))
                for s in evidence[:-1]:
                    required.append(b.add(s))
                b.add_filler(gap)
                required.append(b.add(evidence[-1]))
                if not before:
                    b.add_filler(int(rng.integers(0, 2)))
                    poison.append(b.add(twin))

        b.add_filler(int(rng.integers(1, 4)))
        b.queries.append({"qtype": qtype, "cls": cls, "question": question,
                          "required": required, "poison": poison,
                          "hard": hard, "ent": fact["name"],
                          "attr_key": attr_key,
                          "split": plan["split"]})

    return b


# ---------------------------------------------------------------------------
# Allocation: exact optimum and the weaker allocators used as score anchors
# ---------------------------------------------------------------------------


def _gain_fn(queries):
    qinfo = [(min(req), max(req), set(poi)) for req, poi in queries]

    def gain(j, i):
        g = 0
        for lo, hi, poi in qinfo:
            if j <= lo and hi < i and not any(j <= p < i for p in poi):
                g += 1
        return g
    return gain, qinfo


def doc_optimum_budget(tok, queries, cap, K):
    """Exact max #queries answerable using AT MOST K non-overlapping index
    windows (each a contiguous span of <= cap tokens); gaps allowed.

    A window [j, i) answers every query whose required span fits in [j, i)
    with no poison inside it. Windows are disjoint, so each query is counted by
    at most one window. DP over (sentences consumed, windows used):
        g[k][i] = best coverage using <= k windows within the prefix [0, i).
    Transition: skip sentence i (g[k][i-1]), or end a k-th window at i that
    starts at some j with pref[i]-pref[j] <= cap (g[k-1][j] + gain(j, i)).
    O(n^2 * K * Q); documents are small. This is the provable ceiling."""
    n = len(tok)
    pref = np.concatenate([[0], np.cumsum(tok)])
    gain, _ = _gain_fn(queries)

    g = [[0] * (n + 1) for _ in range(K + 1)]
    for k in range(1, K + 1):
        for i in range(1, n + 1):
            best = max(g[k][i - 1], g[k - 1][i])
            j = i - 1
            while j >= 0 and pref[i] - pref[j] <= cap:
                v = g[k - 1][j] + gain(j, i)
                if v > best:
                    best = v
                j -= 1
            g[k][i] = best
    return g[K][n]


def doc_greedy_budget(tok, queries, cap, K):
    """Greedy allocator: repeatedly take the single highest-gain legal window
    that does not overlap what is already chosen. A strong heuristic, and the
    thing a budgeted DP has to beat."""
    n = len(tok)
    pref = np.concatenate([[0], np.cumsum(tok)])
    _, qinfo = _gain_fn(queries)
    spans, remaining = [], list(range(len(qinfo)))
    while len(spans) < K and remaining:
        best = (0, None)
        for a in range(n):
            for bnd in range(a + 1, n + 1):
                if pref[bnd] - pref[a] > cap:
                    break
                if any(not (bnd <= x or a >= y) for x, y in spans):
                    continue
                g = 0
                for qi in remaining:
                    lo, hi, poi = qinfo[qi]
                    if a <= lo and hi < bnd and not any(a <= p < bnd for p in poi):
                        g += 1
                if g > best[0]:
                    best = (g, (a, bnd))
        if best[1] is None:
            break
        a, bnd = best[1]
        spans.append((a, bnd))
        remaining = [qi for qi in remaining
                     if not (a <= qinfo[qi][0] and qinfo[qi][1] < bnd
                             and not any(a <= p < bnd for p in qinfo[qi][2]))]
    return len(qinfo) - len(remaining)


def doc_naive_budget(tok, queries, cap, K):
    """Naive allocator: one minimal window per query, in query order, until the
    budget runs out. No merging, no conflict resolution -- the allocation a
    solver produces when it treats the budget as an afterthought."""
    pref = np.concatenate([[0], np.cumsum(tok)])
    _, qinfo = _gain_fn(queries)
    spans = []
    for lo, hi, poi in qinfo:
        if len(spans) >= K:
            break
        if pref[hi + 1] - pref[lo] > cap or any(lo <= p <= hi for p in poi):
            continue
        if any(not (hi + 1 <= x or lo >= y) for x, y in spans):
            continue
        spans.append((lo, hi + 1))
    sat = 0
    for lo, hi, poi in qinfo:
        for a, bnd in spans:
            if a <= lo and hi < bnd and not any(a <= p < bnd for p in poi):
                sat += 1
                break
    return sat


def naive_budget_score(tok_by_doc, queries_by_doc, budget_by_doc, cap, size):
    """Budget-blind baseline: cut the document into fixed `size`-sentence
    windows (respecting the token cap) and KEEP the first budget_k of them.
    A solver who does not read the queries would do about this well."""
    total, sat = 0, 0
    for d, tok in tok_by_doc.items():
        bounds = [0]
        cur_tok, cur_cnt = 0, 0
        for i, t in enumerate(tok):
            if cur_cnt >= size or cur_tok + t > cap:
                bounds.append(i)
                cur_tok, cur_cnt = 0, 0
            cur_tok += t
            cur_cnt += 1
        bounds.append(len(tok))
        parts = list(zip(bounds[:-1], bounds[1:]))[
            :budget_by_doc.get(d, BUDGET_MIN)]
        for req, poi in queries_by_doc.get(d, []):
            total += 1
            lo, hi = min(req), max(req)
            for j, i in parts:
                if j <= lo and hi < i and not any(j <= p < i for p in poi):
                    sat += 1
                    break
    return sat / max(1, total)


def budget_k_for(n_queries):
    """Per-document index budget: how many windows you may build."""
    return max(BUDGET_MIN, int(round(KFRAC * n_queries)))


# ---------------------------------------------------------------------------
# Shortcut invariants -- asserted, not hoped for
# ---------------------------------------------------------------------------


def assert_no_shortcuts(doc_rows, all_queries, sent_text):
    """Prove the three closed shortcuts hold on the generated corpus."""
    import collections
    import re

    # (1) frequency balance: every (entity, attribute) pair must be stated as a
    #     fact exactly as often as it is stated as poison, so a corpus-consensus
    #     vote between the two candidate values is an exact tie.
    fact_ct, poison_ct = collections.Counter(), collections.Counter()
    for q in all_queries:
        did = q["doc_id"]
        key = (q["ent"], q["attr_key"])
        fact_ct[key] += 1
        poison_ct[key] += len(q["poison"])
    bad = [k for k in fact_ct if fact_ct[k] != poison_ct[k]]
    assert not bad, f"fact/poison frequency imbalance on {len(bad)} pairs"

    # (2) donor isolation: no poison value is ever attributable to an entity
    #     that appears in the corpus.
    entity_names = {q["ent"] for q in all_queries}
    donor_names = {r["name"] for r in DONOR_POOL}
    assert not (entity_names & donor_names), "a donor country was used as an entity"

    # (3) placement symmetry: poison lands before its fact about half the time.
    before = sum(1 for q in all_queries
                 for p in q["poison"] if p < max(q["required"]))
    total_p = sum(len(q["poison"]) for q in all_queries)
    frac = before / max(1, total_p)
    assert 0.40 <= frac <= 0.60, f"positional prior in poison placement: {frac:.3f}"

    # (4) no textual tell: with the number masked out, a fact sentence and its
    #     own poison twin must be character-for-character identical -- same
    #     template, same numeric surface style, same unit.
    num = re.compile(r"[\d][\d.,]*")
    mismatched = 0
    for q in all_queries:
        fshape = num.sub("#", sent_text[(q["doc_id"], max(q["required"]))])
        for p in q["poison"]:
            if num.sub("#", sent_text[(q["doc_id"], p)]) != fshape:
                mismatched += 1
    assert not mismatched, (
        f"{mismatched} poison sentences differ from their twin beyond the number")

    # (5) magnitude symmetry: "the bigger number is the poison" and "the number
    #     with more digits is the poison" must both be coin flips, or a solver
    #     gets the discrimination for free with no knowledge base.
    digits = re.compile(r"[\d][\d,]*")

    def _val(s):
        m = digits.search(s)
        return float(m.group(0).replace(",", "")) if m else 0.0

    bigger = f_longer = p_longer = npairs = 0
    for q in all_queries:
        fs = sent_text[(q["doc_id"], max(q["required"]))]
        for p in q["poison"]:
            ps = sent_text[(q["doc_id"], p)]
            fv, pv = _val(fs), _val(ps)
            npairs += 1
            bigger += fv > pv
            # digit length of the surface as written, which is what a solver
            # actually sees. Ties are common and carry no signal, so the test
            # is whether the two DIRECTIONS are balanced against each other --
            # not whether either reaches one half.
            lf, lp = len(str(int(fv))), len(str(int(pv)))
            f_longer += lf > lp
            p_longer += lp > lf
    fb = bigger / max(1, npairs)
    fd = f_longer / max(1, npairs)
    pd_ = p_longer / max(1, npairs)
    assert 0.40 <= fb <= 0.60, f"magnitude prior: fact larger {fb:.3f} of the time"
    assert abs(fd - pd_) <= 0.08, (
        f"digit-length prior: fact longer {fd:.3f} vs poison longer {pd_:.3f}")

    return {"pairs": len(fact_ct), "poison_before_frac": round(frac, 4),
            "fact_larger_frac": round(fb, 4),
            "fact_longer_frac": round(fd, 4), "poison_longer_frac": round(pd_, 4),
            "entities": len(entity_names), "donors": len(donor_names)}


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def build_corpus(seed, frac_impossible, frac_hard):
    """Build one full corpus candidate; returns everything needed to measure
    the score anchors."""
    global _WRONG_CACHE, _DIR_BALANCE
    _WRONG_CACHE = {}
    _DIR_BALANCE = 0
    rng = np.random.default_rng(seed)

    n_q_total = N_TRAIN_Q + N_TEST_Q
    per_doc = n_q_total / N_DOCS

    n_conf = int(round(FRAC_CONFLICT * n_q_total / 2)) * 2  # even
    classes = (["impossible"] * int(round(frac_impossible * n_q_total))
               + ["conflict"] * n_conf
               + ["cluster"] * int(round(FRAC_CLUSTER * n_q_total))
               + ["tight"] * int(round(FRAC_TIGHT * n_q_total)))
    classes += ["easy"] * max(0, n_q_total - len(classes))
    classes = classes[:n_q_total]
    multi_span = ["aliased", "relational", "chained"]
    plans = []
    for c in classes:
        if c in ("impossible", "conflict"):
            # conflict interleaving is defined for 2-span queries
            qt = (multi_span[int(rng.integers(3))] if c == "impossible"
                  else multi_span[int(rng.integers(2))])
        elif c == "cluster":
            qt = "direct"                          # cluster packs direct facts
        else:
            # bias easy/tight toward DIRECT so most answerable queries carry an
            # adjacent poison twin -- the value-discrimination skill the score
            # ranks -- and make the two-hop `chained` form half of what is left,
            # so a solver that only handles single-hop resolution gives up a
            # measurable slice rather than a rounding error.
            if rng.random() < 0.62:
                qt = "direct"
            else:
                qt = "chained" if rng.random() < 0.5 else multi_span[
                    int(rng.integers(2))]
        plans.append({"cls": c, "qtype": qt})
    rng.shuffle(plans)

    # SPLIT IS DECIDED HERE, BEFORE ANY TEXT EXISTS. Attribute wordings are
    # drawn from a pool that depends on the split, so held-out wordings can
    # appear only in evaluation examples. Assigning the split after rendering
    # (as earlier versions did) would make that impossible.
    order = rng.permutation(len(plans))
    test_plan = set(order[:N_TEST_Q].tolist())
    for i, pl in enumerate(plans):
        pl["split"] = "test" if i in test_plan else "train"

    doc_rows, query_rows, all_queries = [], [], []
    tok_by_doc, queries_by_doc = {}, {}
    sent_text = {}
    qid = 0
    idx = 0
    d = 0
    while idx < len(plans):
        k = int(np.clip(rng.poisson(per_doc), 3, 11))
        my_plans = plans[idx: idx + k]
        idx += k
        if not my_plans:
            continue
        d += 1
        doc_id = f"doc_{d:04d}"
        b = build_doc(rng, doc_id, my_plans, frac_hard)

        tok = [n_tokens(s) for s in b.sentences]
        tok_by_doc[doc_id] = tok
        queries_by_doc[doc_id] = []
        for s_i, text in enumerate(b.sentences):
            doc_rows.append({"doc_id": doc_id, "sent_id": s_i,
                             "n_tokens": tok[s_i], "text": text})
            sent_text[(doc_id, s_i)] = text
        for q in b.queries:
            qid += 1
            query_rows.append({
                "query_id": f"q_{qid:05d}",
                "doc_id": doc_id,
                "question": q["question"],
                "required_sent_ids": "|".join(map(str, q["required"])),
                "poison_sent_ids": "|".join(map(str, q["poison"])),
                "qtype": q["qtype"], "cls": q["cls"],
                "split": q["split"],
            })
            queries_by_doc[doc_id].append((q["required"], q["poison"]))
            all_queries.append({"doc_id": doc_id, "required": q["required"],
                                "poison": q["poison"], "hard": q["hard"],
                                "qtype": q["qtype"], "cls": q["cls"],
                                "ent": q["ent"], "attr_key": q["attr_key"],
                                "split": q["split"]})

    qdf = pd.DataFrame(query_rows)
    qdf["split"] = [q["split"] for q in all_queries]

    return doc_rows, qdf, all_queries, tok_by_doc, sent_text


def measure(doc_rows, qdf, all_queries, tok_by_doc):
    """Measure every published score anchor on the TEST workload."""
    test_by_doc, hard_by_doc = {}, {}
    for q in all_queries:
        if q["split"] != "test":
            continue
        test_by_doc.setdefault(q["doc_id"], []).append((q["required"],
                                                        q["poison"]))
        hard_by_doc.setdefault(q["doc_id"], []).append(q["hard"])

    budget_by_doc = {d: budget_k_for(len(qs)) for d, qs in test_by_doc.items()}
    for row in doc_rows:
        row["budget_k"] = budget_by_doc.get(row["doc_id"], BUDGET_MIN)

    n_test = sum(len(v) for v in test_by_doc.values())
    cap = MAX_CHUNK_TOKENS
    opt = greedy = naive = obvious = guess = 0
    for dd, qs in test_by_doc.items():
        tok, K = tok_by_doc[dd], budget_by_doc[dd]
        opt += doc_optimum_budget(tok, qs, cap, K)
        greedy += doc_greedy_budget(tok, qs, cap, K)
        naive += doc_naive_budget(tok, qs, cap, K)
        # "obvious pipeline": exact-integer knowledge-base matching. Every fact
        # written with a comma-grouped or rounded surface is misread, so the
        # solver cannot tell that query's twins apart and drops the query.
        keep = [q for q, h in zip(qs, hard_by_doc[dd]) if not h]
        obvious += _score_subset(tok, qs, keep, cap, K)
        # "no knowledge base": the entity is resolved and every sentence about
        # its (entity, attribute) pair is found, but with nothing to compare
        # values against, one of those candidates is picked at random as the
        # fact and the rest are treated as poison. This is what a solver that
        # skips the knowledge-base lookup entirely can do -- the candidates are
        # the true fact plus all of its twins, whatever their distance.
        rng = np.random.default_rng(int(dd.split("_")[1]) * 977 + 11)
        flipped = []
        for req, poi in qs:
            fact = max(req)
            cands = sorted([fact] + list(poi))
            pick = cands[int(rng.integers(len(cands)))]
            link = [s for s in req if s != fact]
            flipped.append((sorted(link + [pick]),
                            [c for c in cands if c != pick]))
        guess += _score_subset(tok, qs, flipped, cap, K)

    fixed = {s: naive_budget_score(tok_by_doc, test_by_doc, budget_by_doc,
                                   cap, s) for s in (1, 3, 5, 8, 12)}
    return {"optimum": opt / n_test, "greedy": greedy / n_test,
            "naive_alloc": naive / n_test, "obvious": obvious / n_test,
            "no_kb_guess": guess / n_test, "fixed": fixed,
            "budget_by_doc": budget_by_doc, "n_test": n_test,
            "test_by_doc": test_by_doc}


def _score_subset(tok, all_qs, believed, cap, K):
    """Allocate with the DP against `believed` queries, then score against the
    TRUE queries `all_qs`. This is how a solver with an imperfect view of the
    workload is actually graded."""
    if not believed:
        return 0
    spans = _dp_spans(tok, believed, cap, K)
    sat = 0
    for req, poi in all_qs:
        lo, hi = min(req), max(req)
        for a, b in spans:
            if a <= lo and hi < b and not any(a <= p < b for p in poi):
                sat += 1
                break
    return sat


def _dp_spans(tok, queries, cap, K):
    """Budgeted DP with backtracking -- returns the chosen windows."""
    n = len(tok)
    pref = np.concatenate([[0], np.cumsum(tok)])
    gain, _ = _gain_fn(queries)
    g = [[0] * (n + 1) for _ in range(K + 1)]
    bk = [[None] * (n + 1) for _ in range(K + 1)]
    for k in range(1, K + 1):
        for i in range(1, n + 1):
            best, arg = g[k][i - 1], ("skip", i - 1)
            if g[k - 1][i] > best:
                best, arg = g[k - 1][i], ("drop", i)
            j = i - 1
            while j >= 0 and pref[i] - pref[j] <= cap:
                v = g[k - 1][j] + gain(j, i)
                if v > best:
                    best, arg = v, ("win", j)
                j -= 1
            g[k][i], bk[k][i] = best, arg
    spans, k, i = [], K, n
    while k > 0 and i > 0:
        kind, j = bk[k][i]
        if kind == "win":
            spans.append((j, i)); k -= 1; i = j
        elif kind == "drop":
            k -= 1
        else:
            i = j
    return sorted(spans)


def gates(m):
    """Every band the accepted corpus must satisfy, as (name, ok) pairs."""
    best_fixed = max(m["fixed"].values())
    return [
        ("optimum in band", OPT_BAND[0] <= m["optimum"] <= OPT_BAND[1]),
        ("agent band", AGENT_BAND[0] <= m["obvious"] <= AGENT_BAND[1]),
        ("blind baselines low", best_fixed <= BLIND_MAX),
        ("allocation gap", m["optimum"] - m["naive_alloc"] >= MIN_ALLOC_GAP),
        ("kb gap", m["optimum"] - m["no_kb_guess"] >= MIN_KB_GAP),
    ]


def main(out_dir):
    """Deterministic AUTO-TUNING loop: candidate corpora are generated with a
    fixed seed schedule, every score anchor is measured exactly, and the first
    candidate that satisfies ALL gates is accepted. Two proportional
    controllers run between attempts -- one on the unanswerable fraction (moves
    the ceiling) and one on the hard-surface fraction (moves the obvious-
    pipeline anchor) -- so the loop itself is deterministic end to end."""
    os.makedirs(out_dir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    load_real_facts(here)
    load_filler(here)
    n_word = assert_phrasings_disjoint()
    n_heldout = sum(len(v["test_only"]) for v in ATTR_PHRASINGS.values())
    print(f"attribute wordings: {n_word} total, {n_heldout} held out of training")
    print(f"knowledge base: {len(REAL_FACTS)} countries "
          f"({len(ENTITY_POOL)} entities / {len(DONOR_POOL)} poison donors)")
    print(f"filler pool   : {len(FILLER_POOL)} synthetic carrier sentences")

    frac_imp, frac_hard = FRAC_IMPOSSIBLE, FRAC_HARD_SURFACE
    accepted = None
    for attempt in range(40):
        seed = GLOBAL_SEED * 100 + attempt
        doc_rows, qdf, all_q, tok_by_doc, sent_text = build_corpus(
            seed, frac_imp, frac_hard)
        m = measure(doc_rows, qdf, all_q, tok_by_doc)
        gg = gates(m)
        ok = all(v for _, v in gg)
        print(f"  attempt {attempt:2d}: imp={frac_imp:.3f} hard={frac_hard:.3f} "
              f"opt={m['optimum']:.4f} obvious={m['obvious']:.4f} "
              f"greedy={m['greedy']:.4f} naive={m['naive_alloc']:.4f} "
              f"noKB={m['no_kb_guess']:.4f} blind={max(m['fixed'].values()):.4f}"
              f"{'  ACCEPT' if ok else '  [' + ','.join(n for n, v in gg if not v) + ']'}")
        if ok:
            accepted = (seed, frac_imp, frac_hard, doc_rows, qdf, all_q,
                        tok_by_doc, sent_text, m)
            break
        # controller 1: the unanswerable fraction moves the ceiling
        mid_opt = sum(OPT_BAND) / 2
        frac_imp = float(np.clip(frac_imp + (m["optimum"] - mid_opt) * 0.55,
                                 0.02, 0.45))
        # controller 2: the hard-surface fraction moves the obvious-pipeline
        # anchor (more hard surfaces -> lower obvious score)
        mid_agent = sum(AGENT_BAND) / 2
        frac_hard = float(np.clip(frac_hard + (m["obvious"] - mid_agent) * 0.9,
                                  0.05, 0.95))
    if accepted is None:
        raise RuntimeError("auto-tuning failed to satisfy every gate")

    (seed, frac_imp, frac_hard, doc_rows, qdf, all_q, tok_by_doc,
     sent_text, m) = accepted

    inv = assert_no_shortcuts(doc_rows, all_q, sent_text)
    print(f"shortcut invariants PROVEN: {inv}")

    meta = {"max_chunk_tokens": MAX_CHUNK_TOKENS,
            "budget_kfrac": KFRAC,
            "budget_min": BUDGET_MIN,
            "accepted_seed": seed,
            "frac_impossible": round(frac_imp, 4),
            "frac_hard_surface": round(frac_hard, 4),
            "n_docs": int(qdf["doc_id"].nunique()),
            "n_sentences": len(doc_rows),
            "n_train_q": int((qdf.split == "train").sum()),
            "n_test_q": int((qdf.split == "test").sum()),
            "verified_optimum_test": round(m["optimum"], 4),
            "greedy_alloc_test": round(m["greedy"], 4),
            "naive_alloc_test": round(m["naive_alloc"], 4),
            "obvious_pipeline_test": round(m["obvious"], 4),
            "no_kb_guess_test": round(m["no_kb_guess"], 4),
            "fixed_size_baselines_test": {str(k): round(v, 4)
                                          for k, v in m["fixed"].items()},
            "shortcut_invariants": inv}

    pd.DataFrame(doc_rows).to_csv(os.path.join(out_dir, "documents.csv"),
                                  index=False)
    qdf.to_csv(os.path.join(out_dir, "queries.csv"), index=False)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    print(f"docs: {meta['n_docs']}, sentences: {len(doc_rows):,}, "
          f"queries: {len(qdf)} (train {meta['n_train_q']} / "
          f"test {meta['n_test_q']})")
    print(f"VERIFIED OPTIMUM (test): {m['optimum']:.4f}  <- hard ceiling")
    print(f"  greedy allocator     : {m['greedy']:.4f}")
    print(f"  naive allocator      : {m['naive_alloc']:.4f}")
    print(f"  obvious pipeline     : {m['obvious']:.4f}  <- agent band")
    print(f"  no knowledge base    : {m['no_kb_guess']:.4f}")
    print(f"  budget-blind tilings : {meta['fixed_size_baselines_test']}")
    print("class mix (test):",
          qdf[qdf.split == 'test']['cls'].value_counts().to_dict())
    print("qtype mix (test):",
          qdf[qdf.split == 'test']['qtype'].value_counts().to_dict())


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(here, "..", "raw"))
    args = parser.parse_args()
    main(args.out_dir)
