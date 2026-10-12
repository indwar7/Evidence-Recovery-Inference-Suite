# Rubrics

**[REQUIRED] Produces exactly one predicted slot per test row, from the
row's own six candidates.** `slot` must be an integer in 0-5 for
every `work_id` in `test.csv`. The candidate pool is reshuffled per row and
drawn from over a thousand distinct research topics; a submission that always
emits the same slot index scores at chance (~0.167) by construction, since
the true slot's position carries no cross-row signal.

**[REQUIRED] Uses only the shipped exemplar abstracts to judge each
candidate — never a topic name, field name, or domain name.** No such label
string is present anywhere in `dataset/public/`. A submission that attempts
to look up the query's catalogued subject (via network access, a cached
catalogue snapshot, or a classifier fine-tuned on a scholarly subject taxonomy) is
invalid per the Rules section regardless of its score.

**[REQUIRED] Runs end to end from `dataset/public/` alone, with no network
access at inference time.** The solver environment has no internet access;
a solution that calls an external API or downloads a model at submission
time will fail to run, not merely score poorly.

**[RECOMMENDED] Distinguishes nearest-neighbor distractors, not only
obviously unrelated ones.** `prepare.py` draws every distractor topic from
the query's true topic's own nearest neighbors by embedding similarity
(rank 2-30 of over a thousand topics), confirmed by measurement to be
substantially harder than random or same-field distractors. A
solution tuned only to separate obviously unrelated fields will
underperform, because the test set contains no easy rows of that kind by
construction.

**[RECOMMENDED] Treats each candidate's single exemplar as a noisy,
one-shot description of its topic, not a reliable summary.** Each candidate
is defined by exactly one other paper's abstract, so a solution that can
recover from an atypical or narrowly-scoped exemplar (e.g. by also
leveraging distributional structure learned from `train.csv`'s many
topic-exemplar pairs, rather than scoring each test row's exemplar in
isolation) should score measurably better than one that treats every
exemplar as equally representative of its topic.

**[REQUIRED] Reports a reproducible run.** All randomness (data loading
order, any model initialization, any sampling at inference time) must be
seeded; two runs of the same submission notebook must produce the same
`slot` for every row.

**[RECOMMENDED] Uses `train.csv`'s `slot` labels for supervised tuning or
model selection, not only as a held-out sanity check.** `train.csv` is built
from an entirely disjoint set of topics from `test.csv`, carries the true
slot for every row, and shares the exact same six-candidate structure, so
it is a direct, representative, labelled proxy for test performance — a
solution that ignores it is discarding the one legitimate training signal
the dataset provides.
