# Inside the Loop or After It

Python code has lost all of its indentation. Give every line back its block depth.

```
def mean(values):                      0
total = 0                              1
for v in values:                       1
total += v                             2
return total / len(values)             1   <- the loop ends here
```

In Python, indentation is the program. The line `return total / len(values)` runs once if it sits at depth 1, and on the first pass through the loop if it sits at depth 2. Both versions are valid Python, and nothing in the grammar says which one the author wrote. A block header (a line ending in a colon) tells you where a block opens. Nothing tells you where it closes. That is the part of the task that is scored.

## Task

Each test row is one real Python source file, flattened:

- `lines`: a JSON list of the file's logical lines, in order. Blank lines and comment-only lines are removed, trailing comments are cut, a statement continued over several physical lines (inside brackets, a multi-line string, or after a backslash) is joined into one line with single spaces, and every line's leading whitespace is removed.

Produce `depths`: one non-negative integer per line, the line's block depth. Module-level lines have depth 0, the body of a top-level `def` or `class` has depth 1, and so on.

The files are real, tested Python source files written by many people. Training and test files come from different directories of the source tree: no directory contributes files to both sets, and every file on either side has at least 3 scored block closures.

**What you are actually being asked.** Most of a file's depths follow from rules. The line after a header is one level deeper. `else`, `elif`, `except` and `finally` line up with the statement they continue. A line after `return` would never run if it stayed in the same block, so the block almost always closes there. Those lines are not scored. The scored lines are the ones the rules leave open: a line that follows an ordinary statement, where the block may continue or may close, and if it closes, may close one block or several.

This is not code formatting. A formatter for Java or JavaScript decides where whitespace looks best, and any choice gives the same program. Here every choice gives a different program, and the answer is the one the author wrote. It is also not code completion: nothing is generated, and the words of every line are given.

The evidence is in the code. A line that uses the loop variable is probably still in the loop. A line that uses a total accumulated inside the loop, and not the loop variable, has probably left it. A second `if` that tests the same thing as the first is probably its sibling, not its child. A learned model weighs these cues; a fixed rule cannot.

## What is new here

1. **Restoring meaning, not appearance.** In Python, the depth of a line decides what the program does, so restoring it means recovering the author's program from text that admits several.
2. **Only the closures the grammar leaves open are scored.** Every line whose depth follows from a header, from `else` or `except` alignment, from a `def` or `class`, or from a preceding `return`, `raise`, `break`, `continue` or `pass` is left out. What remains is the decision no parser can make.
3. **Real code with real ambiguity.** The labels are the indentation the code's authors wrote, in files that pass their project's tests.

## How it relates to prior work

The block structure comes from the off-side rule, introduced for the ISWIM family of languages (Landin, 1966): a block lasts as long as its lines are indented past the line that opened it. Learning code layout from a corpus has been studied for languages where layout is cosmetic. Allamanis, Barr, Bird and Sutton (2014) learned coding conventions, including formatting, from a project's own code. Parr and Vinju (2016) trained a formatter that predicts the whitespace and indentation of each token from examples. This task takes the same idea, learning layout from a corpus, into a language where layout is semantics. It scores only the decisions that change the program.

## Dataset

The data directory is `/data` in the solver environment, `dataset/public` when run locally, and `/kaggle/input/python-indentation/public` on Kaggle. Do not hardcode one of them: check each in turn and use the first that contains `test.csv`.

- `train.csv`: `file_id`, `lines`, `depths`. One row per training file, with its depths.
- `test.csv`: `file_id`, `lines`. One row per test file.
- `sample_submission.csv`: `file_id`, `depths`. The depths the block headers alone imply: one level deeper after every header, and no block ever closed. A valid submission that scores 0.000.
- `validate_submission.py`: an offline structural check of your submission. It reads `test.csv` and needs no answers.

Both sets were drawn from one pool, the files with at least 3 scored block closures, and that pool was split by directory, so training and test files never share a directory. The data directory holds nothing else: no file names or paths, no build statistics, and nothing derived from the test files' depths. `file_id` and row order come from a salted hash.

## Rules

- **Use only the data directory.** The inputs are real, public source files, so nothing in the data can make looking them up impossible; this rule is what closes that route. Recovering the indentation by finding a file elsewhere, through any code hosting site, code search service, package index, archive, website or local copy of source code, is not solving the task. A submission built that way is invalid regardless of its score, and so is any submission that depends on data not shipped in the data directory. The solver environment is expected to have no network access, and your solution must run to completion from the data directory alone.
- **Do not use pretrained models.** Pretrained code models and language models have read large amounts of public code, possibly including these files, so they can reproduce indentation from memory. Train every model from scratch on `train.csv`.
- **Learn from the provided data only.** No pretrained models, no LLMs, and no external data of any kind.
- Use only libraries available in the Kaggle Python Docker image.
- No LLM-generated outputs may be used anywhere in your solution.
- Predict each test file from its own lines plus the training data. Do not fit statistics on the test files as a corpus.
- Build any tokenizer, vocabulary or category encoding from the training data only, or use a fixed character set.
- **Predictions must come from parameters trained on the CPU from the training files.** Hand-coded rules, string heuristics, lookup of training lines and Python's parser on its own are not valid solutions, and a trained model may not be reduced to a fallback behind such a method. Heuristic features, such as whether a line uses a name bound in the current block, may be fed to a trained model as input, and the parser may be used to rule out invalid candidates.
- Your notebook must run end to end, top to bottom.
- Seed everything; your run should be reproducible.

**What not to use:** pretrained models of any kind (code models, language models, pretrained embeddings or tokenizers), source code from outside the data directory in any form, code search services, code hosting sites or package indexes, network access, code formatters or linters used to choose depths, and statistics or vocabularies fitted on the test files. Everything your solution uses must be built from `train.csv` and the test file it is predicting.

## Evaluation

**Block-closure accuracy**, averaged over test files, in [0, 1], higher is better.

Within each file, a line is *eligible* when three things hold. The line before it is not a block header (it does not end with a colon). The line before it does not begin with `return`, `raise`, `break`, `continue` or `pass`. The line itself does not begin with `elif`, `else`, `except`, `finally`, `case`, `def`, `class`, `async` or `@`. ("Begins with" means the line's first word, or its first character if that is not a letter or underscore.) An eligible line is *scored* when its true depth differs from the true depth of the line before it, or your depth differs from your depth of the line before it. In other words, it is scored when either you or the author closed a block there. A scored line is *correct* when your change of depth from the previous line equals the author's. A file's score is its correct scored lines divided by its scored lines. The final score is the mean over test files.

Only the change of depth is compared, so a mistake on one line does not count again on every line after it.

Take the example above. Line 5 (`return total / len(values)`) follows an ordinary statement, so it is eligible, and the author closed one block there (depth 2 to 1). Suppose a second scored line in the file is one where you closed a block and the author did not.

- You put line 5 at depth 1 and close nothing elsewhere: 1 of 1 scored lines correct, 1.000.
- You put line 5 at depth 2 (never closing the loop): the line is still scored because the author closed a block, and you are wrong, 0.000.
- You put line 5 at depth 0: you closed two blocks where the author closed one, 0.000.
- You get line 5 right but also close a block at the other line: 1 of 2 correct, 0.500.

**Why block-closure accuracy.** The alternatives were measured on a feasibility split of the same files before this metric was frozen:

- *Rejected: accuracy of the depth on every line.* Most lines simply stay in their block, so a rule that never closes a block scores well while getting every interesting decision wrong. Absolute depths also cascade: one missed closure puts every later line of the block at the wrong depth.
- *Rejected: accuracy of every change of depth.* Changes after a header, at `else` and `except`, at `def` and after `return` are settled by grammar or by a fixed rule. On a feasibility split that scored every change of depth, a hand-written rule with no learning reached 0.499 against 0.770 for a trained model. The rule got 0.88 of the `else`/`except` lines, 0.90 of the lines after `return` and similar statements, and 0.71 of the `def`/`class` lines right. The skill that separates the two is closing a block after an ordinary statement. There the rule scored 0.000 and the model 0.383.
- *Kept: both kinds of error count.* A scored line is one where the author closed a block or you did. Missing a closure and inventing one both cost you, so neither never closing nor always closing can score well. Both are measured below.

## Submission

A CSV named `submission.csv` with exactly the columns `file_id,depths`, and one row per `file_id` in `test.csv`. `depths` is a space-separated list of non-negative integers with exactly one entry per line of the file.

```
file_id,depths
te_00000,0 0 0 1 1 2 2 1
te_00001,0 1 1 2 2 1
```

Scoring degrades and never rejects. A missing file, a blank or unparsable value, a value with the wrong number of entries, a negative depth, or a `file_id` given more than once scores 0 for that file. Unknown `file_id`s and extra columns are ignored. The only thing rejected outright is a file missing the `file_id` or `depths` column. Run `python validate_submission.py submission.csv` to check the structure offline.

## Compute

- CPU only. 1.5 hours.
- The intended model class is a classifier trained from scratch on features of the code, applied line by line while keeping track of the open blocks. Walk each file top to bottom with a stack of open blocks. At every line that does not follow a header, list the possible depths (stay, close one block, close two, and so on), describe each by the blocks it would close and the block the line would join, and let the trained model choose. Gradient-boosted trees over such candidates train and predict well inside the budget on a CPU.
- A per-candidate classifier fits the metric because the metric judges each closure decision on its own. Sequence models that read the whole file, beam search over the open-block stack, and richer name and data-flow features all fit within the budget.
