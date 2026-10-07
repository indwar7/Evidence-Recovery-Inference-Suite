# EDRMI — solution

## Files
- `solution.py` — self-contained: trains the encoder, decodes test, writes submission.
- `submission.csv` — structurally valid (1500 rows, 31 cols, 0 rule violations).
  Currently from an untrained stub; **regenerate with the GPU run below** for a real score.
- `FINDINGS.md` — structural analysis of the dataset that drove the design.

## Run (A10G, ~60 min)
```bash
pip install torch transformers
python solution.py --train train.csv --test test.csv --out submission.csv \
    --max_steps 6000 --holdout 200
```
`--holdout 200` prints the official metric on held-out train episodes before test decoding.
Set `--holdout 0` to train on everything.

## Key knobs
| flag | default | note |
|---|---|---|
| `--max_steps` | 6000 | fixed step budget (deterministic); 0 = full epochs |
| `--bs` | 192 | lower to 128 if OOM |
| `--epochs` | 2 | time cap usually binds first |
| `--backbone` | all-MiniLM-L6-v2 | statements are short; small model is the right trade |

## Approach
The episode-local "hidden codebook" is presentation, not a learning problem — underneath
sit ~10 globally recurring semantic registers. So the split is:

1. **Neural (learned once).** Bi-encoder + InfoNCE placing same-register statements together.
   Supervision is exact, never guessed:
   - positives: `step_i` across an operator's C0 demos
   - hard negatives: any two slots of one register bank (guaranteed distinct registers,
     same dialogue, same topic — exactly the confusion the task tests)
   An auxiliary head predicts execution position, giving the C1 permutation a second
   source of evidence beyond the single C1 demo per operator.

2. **Symbolic (exact).** Induce base prototypes from C0 demos, recover the episode-local
   C1 permutation, jointly pick the intrusion probe and a bijective slot trace, then solve
   the 6×6 operator assignment with Hungarian matching so each operator is used exactly once.

The bank's one-statement-per-register guarantee is what makes step 2 exact: it turns trace
prediction into a bijective assignment, so the constraint repairs individually uncertain calls.
