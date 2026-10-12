# Approach

Note on notation: the dataset masks certain strings with placeholder tokens that are
written inside angle brackets. To keep this write-up safe to paste into a form, I refer to
them here without the brackets, as the CAPABILITY placeholder, the MODEL placeholder and
the NUM placeholder.

Before writing any model code I spent a while just reading rows. That turned out to be the
whole game here, because the data gives away its own structure if you look at it long
enough.

## What I noticed reading the data

The evidence fragment is literally the rest of the sentence. A profile card fragment that
stops at "...and adherence" pairs with an evidence fragment that starts "to security
standards which directly relate to security testing within the SQA phase." Someone took a
rationale paragraph, cut it in half, put the front half in the profile and the back half
in the evidence bank. So this is not really a semantic retrieval problem, it is a text
reassembly problem. That explains why my first TF-IDF similarity baseline was so bad, at
0.26 edge accuracy. The distractors are topically identical by construction, so topic
similarity tells you nothing. Where the two halves join tells you everything.

Every source triple appears twice. There are 1580 train cases but only 790 distinct profile
triples, and 316 test cases but only 158 triples. The brief mentions this, but the
consequence is bigger than it sounds. A profile's true evidence has to be present in both
views' evidence banks, so if you intersect the two seven-element banks you get a much
smaller candidate set. In practice the intersection is exactly three elements for 93% of
triples, and those three are exactly the three gold evidences. That collapses the evidence
route from a seven-way retrieval into a three-by-three permutation. The same trick on the
task side takes nine candidates down to about six.

The CAPABILITY placeholder marks the answer. In the profile text the gold capability is
masked out. I checked this properly: across 1800 gold rows the gold string appears
literally in the profile only 6 times, while other candidates from the same bank appear 443
times unmasked. So a candidate you can literally see in the profile is almost certainly
wrong, and the mask position is a fill-in-the-blank slot. Separately, the gold does leak
literally into the evidence text about 25% of the time, and when it does, usually no other
candidate does.

## What I built

Two scorers, decoded together.

For evidence, a pairwise ranker over the reduced candidate set. The main features are a
stupid-backoff word n-gram model and a character n-gram model, both scored at the junction
between the prefix and the candidate suffix, in both directions. Around that I added the
things that tell you whether two halves of a sentence belong together:

- whether the prefix ended mid-sentence and the suffix starts lowercase
- whether quotes and brackets balance out once you glue the halves together
- whether both halves refer to the same SDLC phase (SI, SQA, SD, SM, SR)
- whether the same placeholder tokens appear on both sides
- ordinary idf-weighted overlap

The language models are built over every card fragment and evidence text in train and test.
That is safe because they only ever see the texts, never a pairing.

For tasks, a 32-way TF-IDF classifier over profile plus evidence text, since there are only
32 distinct capability strings in the whole dataset. On top of that sits a pairwise ranker
that also receives the fill-in-the-blank score for each candidate substituted into the
CAPABILITY slot, the literal-mention flags described above, and label priors.

For both rankers I append a copy of every feature centred within its own candidate set,
which is a cheap way to make a pointwise model behave listwise.

Decoding is joint. For each triple I enumerate the evidence permutations, and for each one
I find the best one-to-one task assignment given that evidence hypothesis, then take the
best combination overall. Doing this jointly rather than in two independent passes took
evidence F1 from 0.83 to 0.91, because a confident task assignment rules out bad evidence
permutations and vice versa. Finally, both views of a triple emit the same underlying
assignment, re-rendered into each view's local tokens.

## Things that did not work

Plain TF-IDF retrieval, as described above.

Gradient boosting on the task ranker. It beat logistic regression on the evidence route but
badly overfit on tasks, 0.84 against 0.92, so the final version uses boosting for evidence
and logistic regression for tasks. Blending even 20% of the boosted score into the task
ranker cost about a point.

Adding reconstructed train pairs into the language model corpus. This looked mildly helpful
with a linear model but hurt the boosted one, because the training-fold junctions become
memorised and the model then over-trusts that feature.

A second profile-only classifier as an extra task feature, which came out flat to slightly
negative.

## Validation

The test partition holds out complete model owners, so a split that only keeps the two
views of a triple together is not strict enough. Owner identity is not recoverable from
anonymous fragments, and trying to recover it is prohibited anyway, so I use a proxy that
does not need it. Profiles whose card templates are near-duplicates of each other get
forced into the same fold. Concretely, I take TF-IDF cosine over the profile text, link any
pair above a threshold, run union-find, and lift the resulting profile clusters up to whole
triples. Over-merging is the safe direction, because merging two triples that actually have
different owners only makes validation stricter, never leakier.

My first two attempts at this failed and I nearly gave up on it. Linking on any shared rare
shingle chained transitively through the shared boilerplate, phrases like "the model is
designed for", and collapsed 787 of the 790 triples into a single cluster. Grouping by
technical tag signature did the same thing, because tags such as the apache license with
the NUM placeholder are everywhere. The fix was simply to stop linking on single shared
features and use a strict whole-document similarity threshold instead.

Owner-aware three-fold results, sweeping how aggressively the proxy merges:

- threshold 0.80, 636 clusters: score 0.8387
- threshold 0.70, 482 clusters, the default: score 0.8324
- threshold 0.60, 311 clusters: score 0.8094

For reference, splitting by triple alone gives 0.838 and 0.844 across two seeds, so the
owner-aware penalty at the default threshold is only about half a point. That was the main
thing I wanted to know, because it says the model is not living off publisher templates. It
makes sense given where the signal comes from, since the junction language models, the
quote and punctuation balance and the mask substitution are all structural rather than
publisher-specific.

The 0.60 row is a pessimistic bound rather than a fair estimate. At that threshold the
largest cluster is 55% of all triples, so one fold trains on well under half the data, and
part of that drop is simply less training data rather than owner leakage.

At the default threshold the component scores are task edge F1 0.878, evidence edge F1
0.910, and exact ledger accuracy 0.657. The validation runs inside solution.py and prints
its report before the final fit, but it is purely diagnostic and cannot change the
predictions. The whole script takes about two minutes on CPU, of which roughly 80 seconds
is the validation.

Even under this stricter scheme the classifier's regularisation stays where the sweep put
it, at C equals 4, rather than the looser setting that scored better on random splits.

## Practical notes

The script takes the dataset directory and the output CSV path as its two positional
arguments, and writes one row per required test id.

The primary path uses scikit-learn. There is a pure-numpy fallback so the script still runs
and emits a valid file without it, but that fallback is a safety net rather than an
equivalent, and it scores noticeably lower.

Setting the environment variable SKIP_LOCAL_CV to 1 turns off the local validation if
runtime matters. It does not affect the predictions.
