"""Redacts citation signal from opinion text so the citation-graph task
cannot be solved by string-matching / regex extraction instead of
recognizing legal/topical dependency between opinions.

Two independent redaction passes, both required (either alone is
insufficient):

1. FORMAL REPORTER CITATIONS -- patterns like "410 U.S. 113", "347 F.3d
   572", "123 F. Supp. 2d 456", "2015 WL 9592538". These are stripped
   regardless of which opinion they point to (a solver should not be
   able to look up "citation 410 U.S. 113 appears in this bag's opinion
   X" against any external or internal index).

2. CASE-NAME PROSE REFERENCES -- e.g. "Roe v. Wade" appearing in running
   text without a formal citation attached. Every OTHER opinion's
   case_name / case_name_short in the whole raw corpus (not just the
   current bag) is checked as a literal substring and masked -- a
   citing opinion very often names the case it cites in prose even when
   the pinpoint citation is elsewhere in the sentence, and this is
   exactly the shortcut a solver would exploit if left in.
"""
import re

_REPORTER_ABBREVS = (
    r"U\.\s?S\.|F\.\s?(?:2d|3d|4th)?|F\.\s?Supp\.\s?(?:2d|3d)?|F\.\s?App'?x|"
    r"S\.\s?Ct\.|L\.\s?Ed\.\s?(?:2d)?|Cal\.\s?(?:App\.\s?)?(?:2d|3d|4th|5th)?|"
    r"N\.\s?E\.\s?(?:2d|3d)?|N\.\s?W\.\s?(?:2d)?|S\.\s?E\.\s?(?:2d)?|"
    r"S\.\s?W\.\s?(?:2d|3d)?|A\.\s?(?:2d|3d)?|P\.\s?(?:2d|3d)?|So\.\s?(?:2d|3d)?|"
    r"B\.\s?R\.|F\.\s?R\.\s?D\.|WL"
)
CITATION_RE = re.compile(
    rf"\b\d{{1,4}}\s+(?:{_REPORTER_ABBREVS})\s+\d+\b",
    re.IGNORECASE,
)

# a bare "v." / "vs." case-name shape not already caught by an exact
# case-name substring match -- catches paraphrased or partially-named
# references the exact-match pass below would miss, at some cost of
# over-redacting ordinary prose that happens to contain "X v. Y"
# (acceptable: the task is about noise-free discussion of holdings, not
# about names, so over-redacting a stray "v." is a safe direction to
# err in)
GENERIC_VS_RE = re.compile(
    r"\b[A-Z][A-Za-z.'&-]*(?:\s+[A-Z][A-Za-z.'&-]*){0,4}\s+v\.?\s+"
    r"[A-Z][A-Za-z.'&-]*(?:\s+[A-Z][A-Za-z.'&-]*){0,4}\b"
)

CITATION_TOKEN = "[CITATION]"
CASE_NAME_TOKEN = "[CASE]"

# Real OCR-extracted opinion text contains page-break form-feeds (\x0c)
# and old-Mac-style lone carriage returns (\r not followed by \n) --
# both are valid UTF-8 but broke a platform CSV validator's strict
# tabular-data parser (measured: shipping them raised "could not be
# read as UTF-8 delimited data" against a file that decodes as UTF-8
# cleanly in Python). Normalized to a single space; this is whitespace
# collapsing, not signal removal -- no citation or case-name content is
# affected, only mid-field line-break/control-character artifacts.
_CONTROL_CHAR_RE = re.compile(r"\r\n|\r|\n|[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize_whitespace(text: str) -> str:
    return _CONTROL_CHAR_RE.sub(" ", text)


def build_case_name_matcher(case_names: list) -> re.Pattern | None:
    """case_names: every case_name / case_name_short string in the raw
    corpus (across ALL clusters, not just the current bag -- a citing
    opinion may name a case outside its own bag's contents, and that is
    still signal that must not leak). Returns a compiled regex that
    matches any of them as a literal, case-sensitive substring, longest
    names first so overlapping matches prefer the more specific one.
    Names shorter than 6 characters are dropped (too many false-positive
    substring hits in ordinary prose, e.g. "In Re")."""
    names = sorted({n.strip() for n in case_names if n and len(n.strip()) >= 6},
                    key=len, reverse=True)
    if not names:
        return None
    return re.compile("|".join(re.escape(n) for n in names))


def redact(text: str, case_name_re: re.Pattern | None) -> str:
    if not text:
        return text
    text = normalize_whitespace(text)
    out = CITATION_RE.sub(CITATION_TOKEN, text)
    if case_name_re is not None:
        out = case_name_re.sub(CASE_NAME_TOKEN, out)
    out = GENERIC_VS_RE.sub(CASE_NAME_TOKEN, out)
    return out


def scan_leaks(text: str) -> dict:
    """Post-redaction check: count any surviving formal citation or
    generic 'X v. Y' shape. Should be ~0 across a shipped corpus."""
    return {
        "surviving_citations": len(CITATION_RE.findall(text)),
        "surviving_vs_patterns": len(GENERIC_VS_RE.findall(text)),
    }
