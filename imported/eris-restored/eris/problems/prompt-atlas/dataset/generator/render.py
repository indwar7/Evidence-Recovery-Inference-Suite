#!/usr/bin/env python3
"""render.py - turn (article, component pair) into the exact prompt string.

A prompt is built from a mutable `Parts` dict. Each component is a pure
function Parts -> Parts, so an ordered pair is just function composition. Every
component is idempotent, which is what makes pairing a component with itself
well defined.

The option letters are assigned AFTER all components have run, because two of
the components (reverse_options, alpha_options) change the order in which the
categories are listed and therefore which letter each category receives. The
caller gets the letter->category map back so measured probabilities can be
folded back onto categories rather than letters.
"""

import string

# The 17 MN-DS level-1 categories, in the taxonomy's own order.
CATEGORIES = [
    "arts, culture, entertainment and media",
    "conflict, war and peace",
    "crime, law and justice",
    "disaster, accident and emergency incident",
    "economy, business and finance",
    "education",
    "environment",
    "health",
    "human interest",
    "labour",
    "lifestyle and leisure",
    "politics",
    "religion and belief",
    "science and technology",
    "society",
    "sport",
    "weather",
]
LETTERS = list(string.ascii_uppercase[: len(CATEGORIES)])   # A .. Q

# Short display names used inside the prompt. The taxonomy's own labels are
# long enough that repeating all seventeen would dominate the prompt; these are
# the same categories under their conventional short forms.
DISPLAY = {
    "arts, culture, entertainment and media": "arts and media",
    "conflict, war and peace": "conflict and war",
    "crime, law and justice": "crime and justice",
    "disaster, accident and emergency incident": "disaster and accident",
    "economy, business and finance": "business and finance",
    "education": "education",
    "environment": "environment",
    "health": "health",
    "human interest": "human interest",
    "labour": "labour",
    "lifestyle and leisure": "lifestyle",
    "politics": "politics",
    "religion and belief": "religion",
    "science and technology": "science and technology",
    "society": "society",
    "sport": "sport",
    "weather": "weather",
}

BODY_CHARS = 260          # default body budget
TRUNC_CHARS = 110         # component C12

# Three fixed worked examples, written once and reused for every prompt. They
# are short paraphrases, not rows of the corpus, so no test article can leak in.
EXAMPLES = [
    ("Central bank holds rates steady for a third quarter",
     "economy, business and finance"),
    ("Wildfire forces evacuation of two coastal villages",
     "disaster, accident and emergency incident"),
    ("National side reaches the semi-final after extra time",
     "sport"),
]

CONFUSABLE = ("society", "human interest")


def base_parts(title, body):
    return {
        "role": None,
        "taxonomy": False,
        "examples": 0,
        "think": False,
        "single_letter": False,
        "no_hedge": False,
        "option_order": "taxonomy",
        "title_only": False,
        "body_first": False,
        "body_chars": BODY_CHARS,
        "confusion": False,
        "exclude": False,
        "confidence": False,
        "terse": False,
        "title": title,
        "body": body,
    }


# ---- the sixteen components ---------------------------------------------- #

def c01(p): p["role"] = "You are a newsroom editor."; return p
def c02(p): p["taxonomy"] = True; return p
def c03(p): p["examples"] = max(p["examples"], 1); return p
def c04(p): p["examples"] = 3; return p
def c05(p): p["think"] = True; return p
def c06(p): p["single_letter"] = True; return p
def c07(p): p["no_hedge"] = True; return p
def c08(p): p["option_order"] = "reverse"; return p
def c09(p): p["option_order"] = "alpha"; return p
def c10(p): p["title_only"] = True; return p
def c11(p): p["body_first"] = True; return p
def c12(p): p["body_chars"] = min(p["body_chars"], TRUNC_CHARS); return p
def c13(p): p["confusion"] = True; return p
def c14(p): p["exclude"] = True; return p
def c15(p): p["confidence"] = True; return p
def c16(p): p["terse"] = True; return p


COMPONENTS = {
    "C01": c01, "C02": c02, "C03": c03, "C04": c04,
    "C05": c05, "C06": c06, "C07": c07, "C08": c08,
    "C09": c09, "C10": c10, "C11": c11, "C12": c12,
    "C13": c13, "C14": c14, "C15": c15, "C16": c16,
}


def _ordered_categories(order):
    if order == "reverse":
        return list(reversed(CATEGORIES))
    if order == "alpha":
        return sorted(CATEGORIES)
    return list(CATEGORIES)


def render(title, body, comp_a, comp_b, least_likely=None):
    """Apply comp_a then comp_b. Returns (prompt_text, letter_to_category)."""
    p = base_parts(title, body)
    p = COMPONENTS[comp_a](p)
    p = COMPONENTS[comp_b](p)

    cats = _ordered_categories(p["option_order"])
    letter_to_cat = dict(zip(LETTERS, cats))
    option_block = "\n".join(f"{L}. {DISPLAY[c]}" for L, c in zip(LETTERS, cats))

    head = []
    if p["role"]:
        head.append(p["role"])
    if not p["terse"]:
        head.append("Classify the news article below into exactly one category.")
    if p["taxonomy"]:
        head.append("The categories are the top level of a standard news taxonomy.")
    if p["confusion"]:
        head.append(f"Note that '{DISPLAY[CONFUSABLE[0]]}' and "
                    f"'{DISPLAY[CONFUSABLE[1]]}' are commonly confused.")
    if p["exclude"] and least_likely:
        head.append(f"The category '{DISPLAY[least_likely]}' is very "
                    f"unlikely to be correct.")
    if p["confidence"]:
        head.append("If you are unsure, give the single most probable category.")
    if p["no_hedge"]:
        head.append("Do not hedge and do not refuse. You must choose one option.")
    if p["think"]:
        head.append("Consider what the article is mainly about before answering.")

    ex = ""
    if p["examples"]:
        lines = []
        for t, c in EXAMPLES[: p["examples"]]:
            letter = LETTERS[cats.index(c)]
            lines.append(f"Headline: {t}\nAnswer: {letter}")
        ex = "Examples:\n" + "\n\n".join(lines) + "\n\n"

    body_text = "" if p["title_only"] else body[: p["body_chars"]].strip()
    if p["title_only"]:
        article = f"Headline: {title}"
    elif p["body_first"]:
        article = f"Article: {body_text}\nHeadline: {title}"
    else:
        article = f"Headline: {title}\nArticle: {body_text}"

    tail = []
    if p["single_letter"]:
        tail.append("Answer with a single letter and nothing else.")
    tail.append("Answer:")

    parts = []
    if head:
        parts.append("\n".join(head))
    parts.append("Options:\n" + option_block)
    if ex:
        parts.append(ex.rstrip())
    parts.append(article)
    parts.append("\n".join(tail))

    return "\n\n".join(parts), letter_to_cat
