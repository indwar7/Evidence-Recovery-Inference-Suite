"""
prepare(raw, public, private) -> None

Builds the public/private split for the citation-candidate-selection
challenge from dataset/raw/{opinions,opinion_clusters,citation_edges}.csv.

TASK SHAPE
Each graded unit is ONE QUERY opinion presented with a fixed pool of
CANDIDATE opinions. Some candidates are opinions the query actually
cites (taken from the archive's own citation-extraction table); the rest
are real opinions the query does not cite. The solver reads the query's
redacted text and each candidate's redacted text and selects the subset
it believes the query cites.

Why this shape rather than "reconstruct all edges inside a bag": an
earlier build of this dataset scored every ordered pair inside a bag of
mutually-citing opinions. That version leaked -- a content-free probe
that predicted every (earlier-position -> later-position) pair scored
0.143 against a 0.098 reference, i.e. a solver reading no text at all
beat the reference. Measured cause: with every opinion in a bag both a
potential source and a potential target, predicting roughly half the
pair universe matched the true edge density well enough to score. The
query/candidate shape removes this: the query is fixed and separate,
only query->candidate membership is scored, and candidate order is
randomised per query, so position carries no usable signal.

REDACTION
Every shipped opinion text -- query and candidate alike -- is redacted
of formal reporter-citation strings ("410 U.S. 113") and of case-name
prose mentions, matched against EVERY case name in the raw corpus (not
just the ones in this unit, since a query may name a case outside its
own candidate pool). Without this the task is a regex exercise rather
than a reading task. The redaction logic is INLINED in this file rather
than imported from a sibling module: the platform runs this file in
isolation as /data/prepare.py with only raw/, public/ and private/
around it, so a sys.path.insert + separate-module import worked locally
but raised on the platform.

SPLIT
Query opinions are split into train and test by CONNECTED COMPONENT of
the citation subgraph, so no citation edge crosses between a test query
and any train query. Candidate distractors are drawn from the same side
of the split as their query.

GROUND TRUTH
private/answers.csv is ONE ROW PER QUERY:
    query_id, cited_positions, n_candidates
where `cited_positions` is a whitespace-separated list of 0-indexed
candidate positions the query really cites (may be several; never empty
by construction, never all of them).
"""
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

csv.field_size_limit(10**9)

SEED = 20260921

# Candidate pool size per query. Swept during construction against the
# reference model and the random-guess floor (see DESIGN.md); larger
# pools make the task harder because there are more distractors to
# reject per true citation.
N_CANDIDATES = 12

# Queries must cite at least 1 and at most 4 opinions that survived the
# raw build, so the positive rate stays well below 50% of the pool.
#
# MIN_TRUE_CITES was 2 in the first build, which capped the usable pool at
# 789 queries -- measured on the shipped raw corpus, 2,653 opinions cite at
# least one surviving opinion but only 789 cite two or more. A reviewer asked
# for 1,000+ train and several hundred test queries, which is unreachable at
# 2. Lowering to 1 opens the full 2,653-query pool. The metric is unaffected:
# per-query MCC is defined for any true-positive count from 1 to
# n_candidates-1, and the grader already skips only the empty and
# all-positive cases.
MIN_TRUE_CITES = 1
MAX_TRUE_CITES = 4

# Test-side target. The platform splits answers row-wise into a public
# and a private leaderboard slice, each of which needs enough rows, so
# this is kept comfortably above the documented minimum.
N_TEST_QUERIES_TARGET = 400

# Fraction of connected components (by opinion count) held out for test.
TEST_COMPONENT_FRACTION = 0.22

# Cap on shipped text length per opinion (characters, post-redaction).
# Bounds both the redaction pass and downstream solver runtime; legal
# opinions are front-loaded (holding, then reasoning) so a prefix keeps
# the load-bearing content.
MAX_TEXT_CHARS = 6000


# ---------------------------------------------------------------------
# Redaction (inlined -- see module docstring)
# ---------------------------------------------------------------------

_REPORTER_ABBREVS = (
    r"U\.\s*S\.|F\.\s*(?:2d|3d|4th)?|F\.\s*Supp\.\s*(?:2d|3d)?|F\.\s*App'?x|"
    r"S\.\s*Ct\.|L\.\s*Ed\.\s*(?:2d)?|Cal\.\s*(?:App\.\s*)?(?:2d|3d|4th|5th)?|"
    r"N\.\s*E\.\s*(?:2d|3d)?|N\.\s*W\.\s*(?:2d)?|S\.\s*E\.\s*(?:2d)?|"
    r"S\.\s*W\.\s*(?:2d|3d)?|A\.\s*(?:2d|3d)?|P\.\s*(?:2d|3d)?|So\.\s*(?:2d|3d)?|"
    r"B\.\s*R\.|F\.\s*R\.\s*D\.|WL"
)
# NOTE: \s+ (not \s?) between the volume, reporter and page. OCR'd opinion
# text runs page-break whitespace into the citation, e.g.
# "80 L. Ed.        2 d 674". A single-space pattern missed those and left
# 8 reporter fragments in an earlier build of the shipped split.
CITATION_RE = re.compile(
    rf"\b\d{{1,4}}\s+(?:{_REPORTER_ABBREVS})\s+\d+\b",
    re.IGNORECASE,
)

# A bare "X v. Y" case-name shape not already caught by an exact
# case-name substring match. Over-redacts the occasional ordinary phrase,
# which is the safe direction to err in: the task is about the substance
# of the legal discussion, not about names.
GENERIC_VS_RE = re.compile(
    r"\b[A-Z][A-Za-z.'&-]*(?:\s+[A-Z][A-Za-z.'&-]*){0,4}\s+v\.?\s+"
    r"[A-Z][A-Za-z.'&-]*(?:\s+[A-Z][A-Za-z.'&-]*){0,4}\b"
)

CITATION_TOKEN = "[CITATION]"
CASE_NAME_TOKEN = "[CASE]"

# OCR-extracted opinion text carries page-break form-feeds and old-Mac
# lone carriage returns. Both are valid UTF-8 but broke a platform CSV
# validator's strict tabular parser, so they are collapsed to spaces.
# This is whitespace normalisation, not signal removal.
_CONTROL_CHAR_RE = re.compile(r"\r\n|\r|\n|[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize_whitespace(text: str) -> str:
    return _CONTROL_CHAR_RE.sub(" ", text)


def build_case_name_matcher(case_names: list):
    """Compile one alternation over every case name in the raw corpus,
    longest first so overlapping matches prefer the more specific name.
    Names under 6 characters are dropped (too many false-positive
    substring hits in ordinary prose, e.g. "In Re")."""
    names = sorted({n.strip() for n in case_names if n and len(n.strip()) >= 6},
                   key=len, reverse=True)
    if not names:
        return None
    return re.compile("|".join(re.escape(n) for n in names))


def redact(text: str, case_name_re) -> str:
    if not text:
        return text
    text = normalize_whitespace(text)
    out = CITATION_RE.sub(CITATION_TOKEN, text)
    if case_name_re is not None:
        out = case_name_re.sub(CASE_NAME_TOKEN, out)
    return GENERIC_VS_RE.sub(CASE_NAME_TOKEN, out)


def scan_leaks(text: str) -> dict:
    """Post-redaction check: count any surviving formal citation or bare
    'X v. Y' shape. Should be 0 across a shipped corpus."""
    return {
        "surviving_citations": len(CITATION_RE.findall(text)),
        "surviving_vs_patterns": len(GENERIC_VS_RE.findall(text)),
    }


# ---------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------


def _resolve_raw(raw: Path) -> Path:
    """Find the directory that actually holds the raw CSVs.

    The platform passes a `raw_root` whose exact layout is not
    guaranteed: depending on how the uploaded archive is unpacked, the
    CSVs may sit directly in `raw_root`, or one level down in a nested
    `raw/` (or `dataset/raw/`) folder. Resolving this explicitly avoids
    a FileNotFoundError deep inside the first csv read, which is how
    this surfaced on the platform.
    """
    raw = Path(raw)
    marker = "opinions.csv"
    if (raw / marker).exists():
        return raw
    for sub in ("raw", "dataset/raw"):
        if (raw / sub / marker).exists():
            return raw / sub
    # last resort: search a couple of levels down for the marker
    for depth in ("*/", "*/*/"):
        hits = sorted(raw.glob(f"{depth}{marker}"))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(
        f"could not find {marker} under {raw!s}; "
        f"contents: {sorted(p.name for p in raw.iterdir())[:20]}")


def _load_csv(path: Path) -> list:
    if not Path(path).exists():
        raise FileNotFoundError(f"missing expected raw file: {path!s}")
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _connected_components(nodes: set, edges: list) -> list:
    """Union-find over the citation subgraph, undirected for
    connectivity purposes only (direction is preserved for grading)."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)

    comps = {}
    for n in sorted(nodes):
        comps.setdefault(find(n), set()).add(n)
    return [comps[k] for k in sorted(comps)]


def _split_components(components: list, rng: random.Random, test_fraction: float) -> tuple:
    """Assign whole components to test until the target opinion-count
    fraction is reached, largest first so the split does not depend on
    one giant component landing arbitrarily on one side."""
    ordered = sorted(components, key=len, reverse=True)
    rng.shuffle(ordered)  # break ties among same-size components
    ordered.sort(key=len, reverse=True)
    total = sum(len(c) for c in components)
    target_test = total * test_fraction

    test_comps, train_comps = [], []
    test_count = 0
    for comp in ordered:
        if test_count < target_test and len(comp) <= (target_test - test_count) * 1.8:
            test_comps.append(comp)
            test_count += len(comp)
        else:
            train_comps.append(comp)
    return test_comps, train_comps


def _build_units(queries: list, cites: dict, pool: list, rng: random.Random,
                 n_candidates: int) -> list:
    """One unit per query: its true cites plus distractors drawn from
    `pool`, shuffled. Returns [(query_id, [candidate_id...],
    {true positions})]."""
    units = []
    pool_set = set(pool)
    for q in queries:
        true_c = sorted(c for c in cites[q] if c in pool_set)
        if not (MIN_TRUE_CITES <= len(true_c) <= MAX_TRUE_CITES):
            continue
        if len(true_c) >= n_candidates:
            continue
        forbidden = cites[q] | {q}
        distractor_pool = [o for o in pool if o not in forbidden]
        n_need = n_candidates - len(true_c)
        if len(distractor_pool) < n_need:
            continue
        cands = true_c + rng.sample(distractor_pool, n_need)
        rng.shuffle(cands)
        truth = {i for i, c in enumerate(cands) if c in cites[q]}
        units.append((q, cands, truth))
    return units


def prepare(raw: Path, public: Path, private: Path) -> None:
    t0 = time.time()
    raw, public, private = Path(raw), Path(public), Path(private)
    raw = _resolve_raw(raw)
    print(f"[0.0s] reading raw files from {raw}", file=sys.stderr)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    opinions = {r["id"]: r for r in _load_csv(raw / "opinions.csv")}
    print(f"[{time.time()-t0:.1f}s] loaded {len(opinions)} opinions", file=sys.stderr)
    clusters = {r["id"]: r for r in _load_csv(raw / "opinion_clusters.csv")}
    print(f"[{time.time()-t0:.1f}s] loaded {len(clusters)} clusters", file=sys.stderr)
    edges_raw = _load_csv(raw / "citation_edges.csv")
    print(f"[{time.time()-t0:.1f}s] loaded {len(edges_raw)} edges", file=sys.stderr)

    all_case_names = [c["case_name"] for c in clusters.values() if c.get("case_name")] + \
                     [c["case_name_short"] for c in clusters.values() if c.get("case_name_short")]
    case_name_re = build_case_name_matcher(all_case_names)
    print(f"[{time.time()-t0:.1f}s] built case-name matcher over {len(all_case_names)} names",
          file=sys.stderr)

    edges = [(e["citing_opinion_id"], e["cited_opinion_id"]) for e in edges_raw
             if e["citing_opinion_id"] in opinions and e["cited_opinion_id"] in opinions]

    cites = {}
    for a, b in edges:
        cites.setdefault(a, set()).add(b)

    connected_nodes = set()
    for a, b in edges:
        connected_nodes.add(a)
        connected_nodes.add(b)

    components = _connected_components(connected_nodes, edges)
    test_comps, train_comps = _split_components(components, rng, TEST_COMPONENT_FRACTION)
    test_nodes = sorted({n for c in test_comps for n in c})
    train_nodes = sorted({n for c in train_comps for n in c})
    print(f"[{time.time()-t0:.1f}s] {len(components)} components -> "
          f"{len(test_nodes)} test nodes, {len(train_nodes)} train nodes", file=sys.stderr)

    test_queries = [q for q in test_nodes if q in cites]
    train_queries = [q for q in train_nodes if q in cites]
    rng.shuffle(test_queries)
    rng.shuffle(train_queries)

    test_units = _build_units(test_queries, cites, test_nodes, rng, N_CANDIDATES)
    test_units = test_units[:N_TEST_QUERIES_TARGET]
    train_units = _build_units(train_queries, cites, train_nodes, rng, N_CANDIDATES)
    print(f"[{time.time()-t0:.1f}s] built {len(train_units)} train units, "
          f"{len(test_units)} test units", file=sys.stderr)

    # Redact once per opinion actually shipped
    needed = {q for q, _, _ in train_units + test_units}
    needed |= {c for _, cs, _ in train_units + test_units for c in cs}
    redacted = {}
    for i, oid in enumerate(sorted(needed)):
        redacted[oid] = redact(opinions[oid]["plain_text"][:MAX_TEXT_CHARS], case_name_re)
        if i % 500 == 0:
            print(f"[{time.time()-t0:.1f}s] redacted {i}/{len(needed)}", file=sys.stderr)
    print(f"[{time.time()-t0:.1f}s] redacted all {len(redacted)} opinions", file=sys.stderr)

    def emit(units, prefix):
        rows = []
        for n, (q, cands, truth) in enumerate(units):
            qid = f"{prefix}_{n:05d}"
            row = {"query_id": qid, "query_text": redacted[q]}
            for k, c in enumerate(cands):
                row[f"candidate_{k:02d}_text"] = redacted[c]
            rows.append((row, qid, truth, len(cands)))
        return rows

    train_emitted = emit(train_units, "train")
    test_emitted = emit(test_units, "test")

    cand_cols = [f"candidate_{k:02d}_text" for k in range(N_CANDIDATES)]

    # train.csv carries the label column; test.csv carries the same
    # feature columns MINUS that label column.
    with open(public / "train.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "query_text"] + cand_cols + ["cited_positions"])
        w.writeheader()
        for row, qid, truth, _ in train_emitted:
            out = dict(row)
            out["cited_positions"] = " ".join(str(i) for i in sorted(truth))
            w.writerow(out)

    with open(public / "test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "query_text"] + cand_cols)
        w.writeheader()
        for row, _, _, _ in test_emitted:
            w.writerow(row)

    with open(public / "sample_submission.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "cited_positions"])
        w.writeheader()
        # Format example only, NOT a prediction. The values are rotated
        # deterministically through a small set of shapes so the column
        # is neither blank (a profiler rejects a mostly-empty column) nor
        # constant (a profiler rejects that too), and so the file shows
        # that a row may name a different number of positions.
        shapes = ["0 1", "2 5 9", "3 7", "1 4 8 11", "6 10"]
        for i, (_, qid, _, _) in enumerate(test_emitted):
            w.writerow({"query_id": qid, "cited_positions": shapes[i % len(shapes)]})

    with open(private / "answers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "cited_positions", "n_candidates"])
        w.writeheader()
        for _, qid, truth, ncand in test_emitted:
            w.writerow({"query_id": qid,
                        "cited_positions": " ".join(str(i) for i in sorted(truth)),
                        "n_candidates": ncand})

    # Scan each text field SEPARATELY. Joining them into one string first
    # lets the "X v. Y" pattern match across the seam between two
    # documents (end of one text + " v. " + start of the next), which
    # reported 7 phantom leaks that do not exist in any shipped field.
    leaks = {"surviving_citations": 0, "surviving_vs_patterns": 0}
    for row, _, _, _ in train_emitted + test_emitted:
        for k, v in row.items():
            if not k.endswith("_text"):
                continue
            hit = scan_leaks(v)
            leaks["surviving_citations"] += hit["surviving_citations"]
            leaks["surviving_vs_patterns"] += hit["surviving_vs_patterns"]

    stats = {
        "task_shape": "one query opinion + a fixed candidate pool; select the cited candidates",
        "n_train_queries": len(train_emitted),
        "n_test_queries": len(test_emitted),
        "n_candidates_per_query": N_CANDIDATES,
        "true_cites_per_query_min": MIN_TRUE_CITES,
        "true_cites_per_query_max": MAX_TRUE_CITES,
        "n_components_total": len(components),
        "n_components_test": len(test_comps),
        "n_components_train": len(train_comps),
        "leak_scan": leaks,
        "submission_shape": "one row per query: query_id, cited_positions "
                            "(whitespace-separated 0-indexed candidate positions)",
        "split_note": "queries split by CONNECTED COMPONENT of the citation "
                      "subgraph; candidates drawn from the same side of the "
                      "split as their query, so no citation edge crosses "
                      "between a test query and any train query",
    }
    with open(public / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    raw_dir = here / "dataset" / "raw"
    if not raw_dir.exists():
        raw_dir = Path("raw")
    prepare(raw_dir, here / "dataset" / "public", here / "dataset" / "private")
