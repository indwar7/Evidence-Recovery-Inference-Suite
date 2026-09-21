# Archive

Superseded build artifacts, kept for provenance rather than for use. Nothing here is
referenced by any benchmark, grader, or verification run, and nothing here needs to be
downloaded to use the suite.

| Path | What it is | Why it is here |
|---|---|---|
| `experimental-order-raw-v1-215papers/` | The first PLOS corpus build: 215 papers, seed 20260913 | Superseded by the shipped 259-paper build at `benchmarks/experimental-order/dataset/raw/`. Retained so the corpus-growth step is auditable. |
| `release-bundles/` | Zipped copies of three `dataset/raw/` directories | Packaged as GitHub release assets during construction. Byte-for-byte redundant with the committed `dataset/raw/` trees, which are the authoritative copies. |

If you are evaluating this repository, you can ignore this directory entirely. If you are
auditing how the corpora evolved, start with the `meta.json` in each archived build and
compare its `seed`, `fetch_date` and filter settings against the shipped one.
