#!/usr/bin/env python3
"""components.py - the prompt-component catalog.

Sixteen components, each a small, ordinary edit to a classification prompt of
the kind a practitioner would actually try. Every component is a pure function
from (prompt_parts) to (prompt_parts), so any ordered pair composes cleanly and
the catalog stays closed under composition.

None of them is a jailbreak, a trick, or an instruction to game the label. They
are the moves people reach for when a prompt is not working: state a role, give
examples, ask for reasoning, constrain the output, reorder the options, warn
about a confusion, and so on.

The point of the task is that their effect is NOT uniform. A component that
helps on one article can hurt on another, because what it really does is move
one specific model's decision boundary in one specific place.
"""

# Each entry: (id, short name, human description)
CATALOG = [
    ("C01", "expert_role",      "Prefix a one-line role: 'You are a newsroom editor.'"),
    ("C02", "taxonomy_hint",    "State that the labels come from a standard news taxonomy."),
    ("C03", "one_shot",         "Prepend one worked example from a fixed held-out article."),
    ("C04", "three_shot",       "Prepend three worked examples from fixed held-out articles."),
    ("C05", "think_first",      "Ask the model to consider the main subject before answering."),
    ("C06", "single_letter",    "Demand the answer be a single letter and nothing else."),
    ("C07", "no_hedging",       "Forbid hedging or refusing; require a choice."),
    ("C08", "reverse_options",  "Reverse the order in which the options are listed."),
    ("C09", "alpha_options",    "Sort the option labels alphabetically instead of by taxonomy."),
    ("C10", "title_only",       "Show only the headline, dropping the body text."),
    ("C11", "body_first",       "Put the body before the headline instead of after."),
    ("C12", "truncate_short",   "Cut the article to its first 200 characters."),
    ("C13", "confusion_warn",   "Warn that two named categories are commonly confused."),
    ("C14", "exclude_hint",     "Say which single category is least likely, without naming the answer."),
    ("C15", "confidence_ask",   "Ask for the most probable label under uncertainty."),
    ("C16", "terse_frame",      "Strip all instructions down to 'Category:'"),
]

IDS = [c[0] for c in CATALOG]
N = len(IDS)

# Ordered pairs: apply the first component, then the second. A component may be
# paired with itself, which simply means applying it once (the functions are
# idempotent by construction), so the catalog gives N*N legal moves.
PAIRS = [(a, b) for a in IDS for b in IDS]

assert len(PAIRS) == N * N, "pair enumeration must be complete"
