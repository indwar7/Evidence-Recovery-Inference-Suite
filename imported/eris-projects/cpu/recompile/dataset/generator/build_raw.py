"""Build the raw corpus: real CFR sections before/after a real Federal Register rule.

For every final rule published in the Federal Register we have

  * the amendatory instructions the agency wrote (FR XML <AMDPAR>), together with
    whatever replacement text the agency printed underneath each one, and
  * the official text of the affected CFR section on the day before the rule took
    effect and on the day it took effect (eCFR point-in-time XML).

The pair (before, after) is the Office of the Federal Register's own compilation
of that rule.  Nothing here is annotated, paraphrased or model-generated.

Sources, both US Government works in the public domain:
  govinfo bulk Federal Register XML   https://www.govinfo.gov/bulkdata/FR
  eCFR point-in-time API              https://www.ecfr.gov/api/versioner/v1

Run:  python build_raw.py <out_dir> [n_rules]

Streams results to disk as it goes (a long scan must survive a restart).
"""
from pathlib import Path
import json, os, re, sys, time, urllib.request, urllib.error, datetime, hashlib, html

UA = {"User-Agent": "eris-benchmark-build/1.0 (research; contact via Shipd)"}
FR_API = "https://www.federalregister.gov/api/v1/documents.json"
GOVINFO = "https://www.govinfo.gov/bulkdata/FR/{y}/{m}/FR-{d}.xml"
ECFR = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml?part={part}"

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
MAX_PART_BYTES = 12_000_000       # skip only the true monsters
REQ_PAUSE = 0.34                  # be a good citizen with the public APIs

# Well-posedness: the share of tokens the amendment introduces that must appear in
# what the agency printed.  Below this the gold text cannot be produced from the
# inputs, which in practice means the change belongs to a different rule that
# landed in the same window.  Measured, the distribution is sharply bimodal --
# 85 items at exactly 1.00, 69 below 0.50, and a near-empty valley between -- so
# this threshold separates genuine amendments from mis-attributed ones rather
# than tuning difficulty.  The few between 0.90 and 1.00 are cases where eCFR
# renders a token slightly differently from the printed rule.
WELL_POSED = 0.90


# --------------------------------------------------------------------------- io

def fetch(url, timeout=300, retries=6):
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return fh.read()
        except Exception as e:                       # noqa: BLE001 - network flake
            last = e
            code = getattr(e, "code", None)
            if code in (404, 400):
                raise
            # eCFR answers 429 when several workers push at once; backing off
            # hard is the difference between finishing and losing the part.
            time.sleep(10 * (a + 1) if code == 429 else 2 ** a)
    raise last if last else RuntimeError(url)


class Cache:
    """Disk cache so a restart never re-downloads."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key):
        return self.root / (hashlib.sha1(key.encode()).hexdigest()[:24] + ".bin")

    def has(self, key):
        return self.path(key).exists()

    def get(self, key, url):
        p = self.path(key)
        if p.exists():
            return p.read_bytes()
        if os.environ.get("ERIS_SKIP_PREFETCH") == "1":
            return b""            # cache-only: never reach for the network
        try:
            blob = fetch(url)
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):
                p.write_bytes(b"")
                return b""
            raise
        time.sleep(REQ_PAUSE)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(blob)
        tmp.rename(p)                 # atomic: a killed run never leaves a truncated file
        return blob

    def prefetch(self, jobs, workers=4, label=""):
        """Fetch many URLs concurrently. Downloads dominate this build, and doing
        them one at a time takes hours; the public APIs tolerate a handful of
        parallel readers comfortably."""
        from concurrent.futures import ThreadPoolExecutor
        seen, todo = set(), []
        for k, u in jobs:                    # many rules share an issue or a part
            if k in seen or self.has(k):
                continue
            seen.add(k)
            todo.append((k, u))
        if not todo:
            return
        done = [0]

        def one(job):
            k, u = job
            try:
                self.get(k, u)
            except Exception as e:                   # noqa: BLE001
                print(f"    prefetch fail {k}: {e}", flush=True)
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"    {label} {done[0]}/{len(todo)}", flush=True)

        print(f"  prefetching {len(todo)} {label} with {workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, todo))


# ------------------------------------------------------------------ normalising

# Inline emphasis markup sits *inside* words and paragraph designators -- turning
# <E T="03">a</E> into " a " would rewrite "(a)" as "( a )".  Drop those without
# leaving a space; every other tag becomes whitespace.
INLINE_TAG = re.compile(r"</?(?:E|SU|SUB|I|B)\b[^>]*>", re.I)
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")
# eCFR inserts an editorial pointer at the top of any section that has a pending
# amendment.  It names the very rule we are asking about -- a direct leak -- and
# it ends in a date, so it must be consumed through the year, not through the
# first full stop ("... at 88 FR 123, Jan. 3, 2023.").
# The pointer ends "... at 88 FR 3502, Jan. 19, 2023." -- stopping at the first
# full stop leaves "19, 2023." glued to the front of the section, so run to the
# year.  A bounded lazy match keeps it from swallowing the section on a miss.
ECFR_POINTER = re.compile(
    r"Link to an amendment published at.{0,200}?\d{4}\s*\.\s*", re.I)
ECFR_POINTER2 = re.compile(
    r"Link to (?:an?|the) [a-z ]{0,40}published at.{0,200}?\d{4}\s*\.\s*", re.I)
EFFECT_NOTE = re.compile(
    r"(?:Effective date note|Editorial note|Note:\s*At \d+ FR)[^.]*\.", re.I)
# Any Federal Register citation identifies the rule being applied, so both sides
# get the same placeholder rather than a half-hearted strip of bracketed credits.
FR_CITE = re.compile(r"\b\d{1,3}\s*FR\s*\d{2,6}\b", re.I)
# Source credits carry the FR citation of every rule that ever touched the
# section -- including, in the AFTER text, the rule being applied.
SOURCE_CREDIT = re.compile(r"\[\s*\d+\s*FR\s*\d+[^\]]*\]")
CFR_CITE = re.compile(
    r"(?:§+\s*\d+[A-Za-z]?\.\d+[a-z0-9\-]*)|(?:\b\d+\s+CFR\s+(?:part\s+|§+\s*)?\d+[.\d]*)",
    re.I)

UNI = {"‘": "'", "’": "'", "“": '"', "”": '"',
       "–": "-", "—": "-", "−": "-", " ": " ",
       "…": "...", "¶": " "}


def detag(x):
    x = html.unescape(x)                 # &#xA7; must become § before anything else
    x = INLINE_TAG.sub("", x)
    return WS.sub(" ", TAG.sub(" ", x)).strip()


def canon(text):
    """One canonical single-line form used for input, gold and grading."""
    text = html.unescape(text)
    for k, v in UNI.items():
        text = text.replace(k, v)
    text = ECFR_POINTER.sub(" ", text)
    text = ECFR_POINTER2.sub(" ", text)
    text = EFFECT_NOTE.sub(" ", text)
    text = SOURCE_CREDIT.sub(" ", text)
    text = CFR_CITE.sub("<CITE>", text)
    text = FR_CITE.sub("<FRCITE>", text)
    return WS.sub(" ", text).strip()


def toks(text):
    return text.split()


# ------------------------------------------------------------- federal register

def list_rules(n_rules):
    """Final rules with an effective date and a single CFR title."""
    out, cursor = [], None
    fields = ["document_number", "publication_date", "effective_on",
              "cfr_references", "title", "agencies", "citation"]
    while len(out) < n_rules * 4:
        q = (f"{FR_API}?per_page=1000&order=oldest"
             f"&conditions[type][]=RULE"
             f"&conditions[publication_date][gte]={START_DATE}"
             f"&conditions[publication_date][lte]={END_DATE}"
             + "".join(f"&fields[]={f}" for f in fields))
        if cursor:
            q += f"&search_after_cursor={cursor}"
        d = json.loads(fetch(q))
        res = d.get("results") or []
        if not res:
            break
        for r in res:
            refs = r.get("cfr_references") or []
            titles = {x.get("title") for x in refs if x.get("title")}
            parts = {str(x.get("part")) for x in refs if x.get("part")}
            if not r.get("effective_on") or len(titles) != 1 or not 1 <= len(parts) <= 3:
                continue
            r["_title"] = titles.pop()
            r["_parts"] = sorted(parts)
            out.append(r)
        nxt = d.get("next_page_url")
        if not nxt:
            break
        m = re.search(r"search_after_cursor=([^&]+)", nxt)
        if not m:
            break
        cursor = m.group(1)
        time.sleep(REQ_PAUSE)
    return out


def rule_regtext(cache, pub_date, doc_number):
    """Return [(instruction, printed_text, section_no)] for one FR document."""
    y, m, _ = pub_date.split("-")
    url = GOVINFO.format(y=y, m=m, d=pub_date)
    blob = cache.get("fr:" + pub_date, url)
    if not blob:
        return []
    xml = blob.decode("utf8", "replace")
    for chunk in re.findall(r"<RULE>(.*?)</RULE>", xml, re.S):
        if doc_number not in chunk:
            continue
        return parse_regtext(chunk)
    return []


SECTNO = re.compile(
    r"<SECTNO>\s*(?:&#167;|&sect;|§)?\s*([0-9A-Za-z.\-]+)\s*</SECTNO>")


SEC_IN_TEXT = re.compile(r"§+\s*(\d+[A-Za-z]?\.\d+[a-z0-9\-]*)")


def parse_regtext(chunk):
    """Walk the regulatory text, grouping instructions with the section they hit.

    Two things make this less trivial than pairing each AMDPAR with the next
    block.  A single section is often amended by a run of consecutive AMDPARs
    ("Amend § 15.407 by:" / "a. Removing the headings ..." / "b. Revising ..."),
    with only whitespace between them -- those must accumulate rather than reset.
    And amendments that only delete text print nothing at all, so the section has
    to be read out of the instruction prose instead of a <SECTNO> element.
    """
    out = []

    def flush(pending, printed, secno):
        if not pending:
            return
        if secno is None:
            for t in pending:
                m = SEC_IN_TEXT.search(t)
                if m:
                    secno = m.group(1)
                    break
        if secno is not None:
            out.append((list(pending), printed, secno))

    for reg in re.findall(r"<REGTEXT(.*?)</REGTEXT>", chunk, re.S):
        pieces = re.split(r"(<AMDPAR>.*?</AMDPAR>)", reg, flags=re.S)
        pending = []
        for p in pieces:
            if p.startswith("<AMDPAR>"):
                txt = detag(p)
                named = SEC_IN_TEXT.search(txt)
                # A new instruction naming a different section starts a new group.
                if named and pending:
                    prev = next((SEC_IN_TEXT.search(t) for t in pending
                                 if SEC_IN_TEXT.search(t)), None)
                    if prev and prev.group(1) != named.group(1):
                        flush(pending, "", None)
                        pending = []
                pending.append(txt)
                continue
            if not pending:
                continue
            body = re.search(r"<SECTION>(.*?)</SECTION>", p, re.S)
            if body is None:
                if not detag(p):
                    continue            # whitespace between consecutive instructions
                continue                # prose we do not need (notes, tables of contents)
            sec = SECTNO.search(p)
            inner = re.sub(r"<SECTNO>.*?</SECTNO>", " ", body.group(1), flags=re.S)
            inner = re.sub(r"<SUBJECT>.*?</SUBJECT>", " ", inner, flags=re.S)
            flush(pending, detag(inner), sec.group(1) if sec else None)
            pending = []
        flush(pending, "", None)
    return out


# ------------------------------------------------------------------------ eCFR

DIV8 = re.compile(r'<DIV8[^>]*\bN="([^"]+)"[^>]*\bTYPE="SECTION"[^>]*>(.*?)</DIV8>', re.S)


def ecfr_key(date, title, part, section):
    return f"ecfrsec:{date}:{title}:{part}:{section}"


def ecfr_url(date, title, part, section):
    return ECFR.format(date=date, title=title, part=part) + f"&section={section}"


def ecfr_section(cache, date, title, part, section):
    """One section as it stood on `date`, canonicalised.

    Fetching the section rather than the whole part: a CFR part runs to several
    megabytes and eCFR serves at a few hundred KB/s, so a part costs ~20s while a
    section costs ~1s and about 500 bytes.  Measured on this corpus: whole-part
    fetching ran at 3 files/min, section fetching at over 100 requests/min.  It
    also lets the largest parts stay in the corpus instead of being skipped.
    """
    blob = cache.get(ecfr_key(date, title, part, section),
                     ecfr_url(date, title, part, section))
    if not blob:
        return None
    xml = blob.decode("utf8", "replace")
    m = DIV8.search(xml)
    if not m:
        return None
    body = re.sub(r"<HEAD>.*?</HEAD>", " ", m.group(2), count=1, flags=re.S)
    return canon(detag(body))


def prev_day(d):
    y, m, dd = (int(x) for x in d.split("-"))
    return (datetime.date(y, m, dd) - datetime.timedelta(days=1)).isoformat()


def _d(s):
    y, m, dd = (int(x) for x in s.split("-"))
    return datetime.date(y, m, dd)


VERSIONS = "https://www.ecfr.gov/api/versioner/v1/versions/title-{title}.json?part={part}"


def version_dates(cache, title, part):
    """{section -> sorted list of dates on which eCFR recorded a change}.

    The Federal Register's stated effective date and the date eCFR actually
    applies the change differ by a few days often enough that pairing on
    effective_on alone loses most of the corpus.  eCFR's own record is the
    authority for when its text changed, so we pair on that instead.
    """
    blob = cache.get(f"ver:{title}:{part}", VERSIONS.format(title=title, part=part))
    if not blob:
        return {}
    try:
        d = json.loads(blob)
    except Exception:                                # noqa: BLE001
        return {}
    out = {}
    for c in d.get("content_versions", []):
        if c.get("substantive") and c.get("date") and not c.get("removed"):
            out.setdefault(c["identifier"], set()).add(c["date"])
    return {k: sorted(v) for k, v in out.items()}


def pick_change_date(dates, pub_date, eff_date):
    """The recorded change date this rule is responsible for.

    An exact hit on the stated effective date is unambiguous.  Otherwise take the
    nearest recorded change within a month and let the well-posedness filter throw
    the item out if the resulting diff is not the one this rule printed -- busy
    parts change often enough that insisting on a unique date in a wide window
    discards two thirds of the corpus.
    """
    if eff_date in dates:
        return eff_date
    e = _d(eff_date)
    cands = [x for x in dates if abs((_d(x) - e).days) <= 30]
    if not cands:
        return None
    return min(cands, key=lambda x: abs((_d(x) - e).days))


# ----------------------------------------------------------------------- items

def diff_ops(before, after):
    """Token-level edit script turning `before` into `after`."""
    import difflib
    a, b = toks(before), toks(after)
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            ops.append((i1, i2, " ".join(b[j1:j2])))
    return ops


def build_meta(items_path: Path):
    """Provenance and selection parameters, plus the coverage actually achieved.

    The requested scan window and the window actually covered are recorded
    separately: the eCFR prefetch runs in date order and can be stopped early, so
    reporting the request as though it were the coverage would overstate the
    corpus.
    """
    pubs, parts, agencies, n = [], set(), set(), 0
    if items_path.exists():
        with open(items_path, encoding="utf8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                n += 1
                parts.add(f"{r.get('cfr_title')}-{r.get('cfr_part')}")
                agencies.add(r.get("agency"))
                if r.get("publication_date"):
                    pubs.append(r["publication_date"])
    return {
        "source_ecfr_primary": "https://www.ecfr.gov/api/versioner/v1",
        "source_fr_secondary": "https://www.govinfo.gov/bulkdata/FR",
        "source_rule_metadata": "https://www.federalregister.gov/api/v1/documents.json",
        "license": "public domain (17 U.S.C. 105)",
        "scan_window_requested": [START_DATE, END_DATE],
        "coverage_actual": [min(pubs), max(pubs)] if pubs else None,
        "n_records": n,
        "n_cfr_parts": len(parts),
        "n_agencies": len(agencies),
        "selection": {
            "well_posed_min_supplied_token_fraction": WELL_POSED,
            "min_before_tokens": 25,
            "max_before_tokens": 1500,
            "max_edit_operations": 40,
            "change_date_rule": "eCFR recorded change on the rule's effective date, "
                                "else the nearest recorded change within 30 days",
        },
        "built": "deterministic scan of Federal Register final rules, oldest first",
    }


def build(out_dir: Path, n_rules: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(out_dir / "_cache")
    done = set()
    items_path = out_dir / "items.jsonl"
    if items_path.exists():
        with open(items_path, encoding="utf8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["uid"])
                except Exception:                    # noqa: BLE001
                    pass
    print(f"resuming with {len(done)} items already on disk", flush=True)

    rules = list_rules(n_rules)
    print(f"candidate rules: {len(rules)}", flush=True)

    # --- phase 1: the daily Federal Register issues and the eCFR version records.
    cache.prefetch([("fr:" + r["publication_date"],
                     GOVINFO.format(y=r["publication_date"][:4],
                                    m=r["publication_date"][5:7],
                                    d=r["publication_date"]))
                    for r in rules], label="FR issues")
    cache.prefetch([(f"ver:{r['_title']}:{p}",
                     VERSIONS.format(title=r["_title"], part=p))
                    for r in rules for p in r["_parts"]], label="version records")  # noqa

    # --- phase 2: work out which point-in-time snapshots we will actually need,
    #     now that everything phase 1 answers is on disk, and fetch those too.
    need = set()
    for r in rules:
        try:
            reg = rule_regtext(cache, r["publication_date"], r["document_number"])
        except Exception:                                 # noqa: BLE001
            continue
        secs = {s for _, _, s in reg if s}
        if not secs:
            continue
        for part in r["_parts"]:
            try:
                vers = version_dates(cache, r["_title"], part)
            except Exception:                             # noqa: BLE001
                continue
            for sec in secs:
                d = pick_change_date(vers.get(sec, []), r["publication_date"],
                                     r["effective_on"])
                if d:
                    need.add((d, r["_title"], part, sec))
                    need.add((prev_day(d), r["_title"], part, sec))
    # eCFR sustains about 24 requests a minute whatever the concurrency, so the
    # prefetch is the whole cost of this build.  It runs in date order, so cutting
    # it short leaves a contiguous coverage window rather than a ragged sample --
    # set ERIS_SKIP_PREFETCH=1 to build from whatever is already cached.
    if os.environ.get("ERIS_SKIP_PREFETCH") != "1":
        cache.prefetch([(ecfr_key(d, t, p, sc), ecfr_url(d, t, p, sc))
                        for d, t, p, sc in sorted(need)], workers=8,
                       label="eCFR section snapshots")
    else:
        have = sum(cache.has(ecfr_key(d, t, p, sc)) for d, t, p, sc in need)
        print(f"  skipping prefetch: {have}/{len(need)} snapshots already cached",
              flush=True)

    attrib = open(out_dir / "attribution.csv", "a", encoding="utf8")
    if attrib.tell() == 0:
        attrib.write("document_number,citation,publication_date,effective_on,cfr_title,cfr_part\n")
    fh_items = open(items_path, "a", encoding="utf8")

    kept = 0
    for ri, r in enumerate(rules):
        if kept >= n_rules:
            break
        doc = r["document_number"]
        try:
            reg = rule_regtext(cache, r["publication_date"], doc)
        except Exception as e:                        # noqa: BLE001
            print("  fr fail", doc, e, flush=True)
            continue
        if not reg:
            continue

        # group instructions by the section they print
        by_sec = {}
        for instrs, printed, sec in reg:
            if sec is None:
                continue
            by_sec.setdefault(sec, []).append({"instructions": instrs, "printed": printed})
        if not by_sec:
            continue

        eff = r["effective_on"]
        title = r["_title"]
        got_any = False
        for part in r["_parts"]:
            try:
                vers = version_dates(cache, title, part)
            except Exception as e:                    # noqa: BLE001
                print("  ver fail", title, part, e, flush=True)
                continue
            if not vers:
                continue
            # group this rule's sections by the date eCFR applied the change
            wanted = {}
            for sec in by_sec:
                d = pick_change_date(vers.get(sec, []), r["publication_date"], eff)
                if d:
                    wanted.setdefault(d, []).append(sec)
            for change_date, secs in wanted.items():
                for sec in secs:
                    blocks = by_sec[sec]
                    uid = f"{doc}:{title}:{sec}"
                    if uid in done:
                        continue
                    try:
                        b = ecfr_section(cache, prev_day(change_date), title, part, sec)
                        a = ecfr_section(cache, change_date, title, part, sec)
                    except Exception as e:            # noqa: BLE001
                        print("  ecfr fail", title, part, sec, e, flush=True)
                        continue
                    if b is None or a is None:
                        continue
                    if b == a or not b or not a:
                        continue
                    if len(toks(b)) < 25 or len(toks(b)) > 1500:
                        continue
                    ops = diff_ops(b, a)
                    if not ops or len(ops) > 40:
                        continue

                    # WELL-POSEDNESS: every token the amendment introduces has to
                    # be present in what the agency printed, otherwise no solver --
                    # and no oracle -- could produce the gold text from the inputs
                    # given.  This is also what guards against mis-attributing a
                    # change to the wrong rule.
                    supply = canon(" ".join(
                        " ".join(x["instructions"]) + " " + x["printed"] for x in blocks))
                    supply_set = set(toks(supply))
                    added = [t for _, _, ins in ops for t in toks(ins)]
                    if added and sum(t in supply_set for t in added) / len(added) < WELL_POSED:
                        continue

                    rec = {
                        "uid": uid, "document_number": doc, "citation": r.get("citation"),
                        "publication_date": r["publication_date"], "effective_on": eff,
                        "change_date": change_date,
                        "cfr_title": title, "cfr_part": part, "section": sec,
                        "agency": (r.get("agencies") or [{}])[0].get("name"),
                        "rule_title": r.get("title"),
                        "before": b, "after": a,
                        # Instructions go through the same canonicalisation as the
                        # text.  They must: an uncanonicalised instruction still
                        # names the section ("Revise § 648.146 ..."), which is the
                        # lookup key for the answer, and still uses curly quotes,
                        # which would not match the quoted strings in the section.
                        "blocks": [{"instructions": [canon(i) for i in x["instructions"]],
                                    "printed": canon(x["printed"])} for x in blocks],
                        "n_ops": len(ops),
                    }
                    fh_items.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done.add(uid)
                    got_any = True
            attrib.write(f"{doc},{r.get('citation')},{r['publication_date']},"
                         f"{eff},{title},{part}\n")
        if got_any:
            kept += 1
            fh_items.flush()
            attrib.flush()
        if ri % 25 == 0:
            print(f"  [{ri}/{len(rules)}] rules kept={kept} items={len(done)}", flush=True)

    fh_items.close()
    attrib.close()
    (out_dir / "meta.json").write_text(json.dumps(build_meta(items_path), indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    build(Path(a[0] if a else "raw"), int(a[1]) if len(a) > 1 else 400)
