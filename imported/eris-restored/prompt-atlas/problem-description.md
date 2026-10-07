# Prompt Component Pair Response Atlas

## Overview

**You are predicting how a hidden news classifier reacts to different wordings of the same question.**

A small text classifier answers a seventeen-way news category question. Its prompt is not fixed. Each prompt is assembled from a catalog of sixteen prompt components applied as an ordered pair: the first component is applied, then the second. A component is an ordinary edit of the kind a practitioner makes when a prompt is not working, such as stating a role, adding worked examples, demanding a single-letter answer, reversing the option order, or cutting the article short. Sixteen components in an ordered pair give 16 times 16, or 256, distinct prompts for the same article.

Those 256 prompts do not all produce the same answer. Some push the classifier toward one category, some toward another, and which ones do what changes from article to article.

**What each row gives you:** one news article and one target category.

**What you predict:** nine facts about how the classifier behaves on that article across all 256 prompts. The main one, carrying 30 percent of the score, is which single ordered pair drives the target category's probability highest. The other eight describe the same behaviour from other angles: which component works best in first position, which five components help the target most and which five hurt it most, which category the classifier picks by default, which category competes hardest with the target, whether the target can be made the top answer at all, whether swapping the two components changes anything, and how high the target's probability can be pushed. All nine are listed with exact definitions in the Dataset section below.

**What you are given to learn from:** the complete measured behaviour of the classifier for every training article, all 256 prompts each, in train_atlas.csv. You are given no measurements at all for the test articles, and no article appears in both splits. So the task is to learn how the components behave as a function of what an article is about, and then apply that to articles you have never seen.

You are never asked to run the hidden classifier yourself. Its responses were measured in advance.

## Dataset

The files are:

train.csv: labeled items, three target categories per training article.
test.csv: unlabeled items, three target categories per test article.
train_atlas.csv: the complete measured response surface for every training article, 256 rows per article.
sample_submission.csv: a weak valid submission showing the required columns and JSON shape.

Columns in train.csv:

id (string): row identifier.
article_id (string): the article this row is about. Rows sharing an article_id differ only in target category. Use it to join to train_atlas.csv.
prompt (string): task instruction.
packet_json (JSON string): item context. It contains headline (string), article (string, the exact text the classifier was shown), true_category (string), target_category (string), categories (list of the seventeen category names), components (list of sixteen objects with component_id, name and description), n_pairs (integer, always 256), and max_set_size (integer, always 5).
answer_format_json (JSON string): output schema summary.
answer_json (JSON string): target response-atlas record.

Columns in test.csv are the same except answer_json is omitted.

Columns in train_atlas.csv:

article_id (string): joins to train.csv.
comp_a (string): the component applied first.
comp_b (string): the component applied second.
p_arts, p_conflict, p_crime, p_disaster, p_economy, p_education, p_environment, p_health, p_human, p_labour, p_lifestyle, p_politics, p_religion, p_science, p_society, p_sport, p_weather (float): the hidden classifier's probability on each of the seventeen categories under that ordered pair. The seventeen values sum to 1. Probabilities are recorded against categories, not against option letters, so the two components that reorder the options are already folded in correctly.

No response surface is published for any test article. The split is article-disjoint: no article appears on both sides.

The submitted answer_json must contain:

chosen_pair (object): the ordered pair you believe pushes the target category highest. It contains first (string component id) and second (string component id). A component may be paired with itself.
best_first_component (string): the component id that gives the highest mean target probability when it is the component applied first, averaged over all sixteen partners.
helpful_components (list of strings): up to five component ids with the highest mean target probability, counting every pair in which the component appears in either position. Unordered.
harmful_components (list of strings): up to five component ids with the lowest mean target probability, counted the same way. Unordered.
default_category (string): the category the classifier picks most often across the 256 pairs. One of the seventeen category names.
top_competitor (string): the category other than the target with the highest mean probability across the 256 pairs. One of the seventeen category names.
target_reachable (string): yes if the target category is the classifier's top choice under at least one of the 256 pairs, otherwise no.
order_sensitive (string): yes if swapping the two components changes the target probability appreciably, otherwise no. Formally: take every unordered pair of two DIFFERENT components, which is 16 choose 2, or 120 pairs. Self-pairs such as C03 with C03 are excluded from this count, because reversing a self-pair gives the same prompt back and its difference is always zero. For each of the 120, take the absolute difference between the target probability under (a, b) and under (b, a). The answer is yes when the mean of those 120 differences exceeds one tenth of that item's own range from its lowest to its highest target probability.
peak_probability_bin (string): the bin of the highest target probability reachable over the 256 pairs. very_low is below 0.05, low is 0.05 up to 0.15, medium is 0.15 up to 0.40, and high is 0.40 or above.

Component ids are C01 through C16. Category names are the seventeen strings listed in packet_json, written exactly as they appear there.

## Evaluation

Submissions are scored from 0 to 1. The final score is the mean row score over all hidden test rows.

For each row, row_score is the SUM of these nine terms. Every term is a
non-negative contribution; nothing is ever subtracted. The nine weights add up
to exactly 1.00, so row_score always lies between 0 and 1.

- 0.30 times steering_score
- 0.12 times helpful_component_f1
- 0.10 times harmful_component_f1
- 0.10 times default_category_exact
- 0.08 times best_first_component_exact
- 0.08 times top_competitor_exact
- 0.08 times order_sensitive_exact
- 0.07 times target_reachable_exact
- 0.07 times peak_probability_bin_exact

steering_score is measured, not judged. The grading key holds the hidden classifier's target probability for all 256 ordered pairs of that item, along with that item's lowest value p_min and highest value p_max. Let p_chosen be the recorded probability for the pair you submitted. Then steering_score = (p_chosen - p_min) / (p_max - p_min). Naming the best pair scores 1. Naming the worst pair scores 0. Naming a pair that does not exist, or omitting chosen_pair, is treated as the worst pair. If p_max equals p_min the term is 0 for every submission.

Normalising against the item's own floor and ceiling is deliberate. An item whose components barely move the classifier cannot outweigh one where they move it a great deal.

helpful_component_f1 and harmful_component_f1 are the standard set F1 against the true five-component sets. Standard precision: the number of submitted ids that appear in the true set, divided by the number of ids you submitted. Standard recall: the number of submitted ids that appear in the true set, divided by the number of ids in the true set. F1 is then 2 * precision * recall / (precision + recall), and is 0 when precision and recall are both 0. If both sides are empty the term is 1. Submitting more than five ids in either list keeps the first five.

The six exact terms score 1 for a case-insensitive exact string match and 0 otherwise.

A row whose answer_json is missing, unparseable, or not a JSON object scores zero for that row rather than voiding the submission. The CSV must still have exactly the required columns.

## Submission

Submit a CSV with exactly two columns:

id (string)
answer_json (JSON string)

Example:

id,answer_json
atl_000031_0,"{""chosen_pair"":{""first"":""C07"",""second"":""C12""},""best_first_component"":""C07"",""helpful_components"":[""C02"",""C04"",""C07"",""C12"",""C13""],""harmful_components"":[""C08"",""C09"",""C10"",""C11"",""C16""],""default_category"":""politics"",""top_competitor"":""society"",""target_reachable"":""yes"",""order_sensitive"":""no"",""peak_probability_bin"":""medium""}"
atl_000031_1,"{""chosen_pair"":{""first"":""C14"",""second"":""C06""},""best_first_component"":""C14"",""helpful_components"":[""C01"",""C05"",""C06"",""C14"",""C15""],""harmful_components"":[""C03"",""C09"",""C10"",""C12"",""C16""],""default_category"":""politics"",""top_competitor"":""health"",""target_reachable"":""no"",""order_sensitive"":""yes"",""peak_probability_bin"":""low""}"

One row for each id in test.csv, in any order. An id you omit scores zero for that row. Duplicate ids keep the first row. Unknown ids are ignored.

## What Not To Use

Do not query any hosted language model API during your run.
Do not download or load pretrained checkpoints of the hidden classifier or attempt to identify which model produced the measurements.
Do not use external news corpora or category annotation files.
Do not search for the original articles or their published category labels.
Do not use any LLM-generated content in your submitted solution.
Do not manually label the test items.
Do not exploit row order, hidden files, or package-generation details.

## Rules and environment

Learn from the provided data only.
Use only libraries available in the standard Kaggle Python Docker image, such as pandas, numpy, scikit-learn, xgboost, lightgbm, tensorflow and pytorch.
Your notebook must run end to end, top to bottom: load the public files, fit, then write submission.csv to ./working/
Seed everything. Your run should be reproducible.
