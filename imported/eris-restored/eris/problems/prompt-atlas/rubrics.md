# Initial Rubrics -- Prompt Component Pair Response Atlas

Eight rubrics. Six REQUIRED, one RECOMMENDED, one OPTIONAL.

---

**1. Uses the published response surface, not just the article text. [REQUIRED]**

train_atlas.csv is the only place the hidden classifier's behaviour is visible.
A solution that reads train.csv and never opens train_atlas.csv is guessing at
a surface it was handed. Evidence of use: the 256 measured probabilities per
training article are loaded, indexed by ordered pair, and turned into
supervision for at least the chosen_pair term.

**2. Normalises each item's surface before learning from it. [REQUIRED]**

Absolute target probabilities differ by more than an order of magnitude across
items, so a model fitted on raw probabilities is dominated by a handful of
high-probability items and learns nothing about the rest. Because the grader
scores each item against its own floor and ceiling, the supervision has to be
normalised the same way: per item, map the 256 values to [0, 1] using that
item's own minimum and maximum.

**3. Respects the article-disjoint split when validating. [REQUIRED]**

Every article contributes three items. A validation split that puts two of an
article's items in fit and the third in validation leaks that article's entire
response surface and reports a score far above what the test set will give.
Any held-out set must be split by article_id, matching how the real test set
was built.

**4. Treats the ordered pair as ordered. [REQUIRED]**

The pair (C04, C12) and the pair (C12, C04) are different prompts and generally
different measurements, which is exactly what the order_sensitive field is
about. A solution that collapses pairs to unordered sets, or that models only
the sixteen components and adds their effects, cannot express the interaction
the task is built on and should be marked down.

**5. Derives the surface-dependent fields from one predicted surface. [REQUIRED]**

chosen_pair, best_first_component, helpful_components and harmful_components
are four readings of the same 256-value surface. Predicting the surface once
and reading all four off it keeps them mutually consistent and shares
statistical strength; fitting four unrelated models does neither. A solution
that produces a helpful_components list not containing its own chosen_pair
components, without an explicit reason, is showing this problem.

**6. Handles the submission contract exactly. [REQUIRED]**

Output is a CSV with exactly the columns id and answer_json, one row per test
id, with answer_json a valid JSON object carrying all nine fields. Component
ids are C01 through C16 and category names are the exact strings from
packet_json. The helpful and harmful lists carry at most five ids. A row that
fails to parse scores zero, so the solution should validate its own output
before writing it.

**7. Establishes a real baseline before modelling. [RECOMMENDED]**

The obvious baselines are worth measuring: submit the pair that is best on
average across all training items, submit the training-set majority for each
categorical field. Knowing what those score locally tells you whether a fitted
model is adding anything, and the gap between them and a text-conditioned model
is the substance of the task.

**8. Exploits the structure of the pair space. [OPTIONAL]**

The 256 pairs are the outer product of a 16-element catalog, so the surface has
a natural low-rank structure: a first-position effect, a second-position effect
and an interaction term. Factorising it that way, rather than treating the 256
pairs as unrelated outputs, uses the training articles more efficiently. This
is one good approach among several and is not required.
