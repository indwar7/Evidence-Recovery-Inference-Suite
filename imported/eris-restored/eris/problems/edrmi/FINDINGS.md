# EDRMI structural findings

- train 9000 episodes / 44 cols; test 1500 / 14 cols. Balanced D1-D4 (2250 each), C0/C1 ~50/50.
- Every demo has exactly 3 steps. 6 ops x 3 demos = 18 demos per episode.
- **Dialogues repeat across episodes** (10907 unique; 9558 appear >=3x). Each occurrence
  draws a *different answer variant* per latent dim -> per-dialogue pool ~30 statements
  = ~10 dims x ~3 variants. Verbatim overlap across occurrences exists but is sparse (~25%).
- Register bank = exactly one statement per latent dim => **bijective 10x10 assignment**.
- Operator program = fixed ordered dim triple; C0 demos of an op share the base order
  (verified: identical surface-signature sequence in ~50% of ops even with a crude 4-way
  signature, i.e. consistency is real but needs a semantic classifier).
- C1 applies an episode-local non-identity permutation of the same 3 dims.
- Statements have stable positional identity: 34110 have exactly one C0 position vs 816 with two.
- Unsupervised TF-IDF clustering fails: it clusters by TOPIC not by semantic ROLE.

## Consequence for design
The episode-local "codebook" is presentation, not a learning problem. Solve as:
1. (neural, learned once) statement -> latent-dim scorer
2. (symbolic, exact) per episode: induce each op's program + control permutation from demos,
   Hungarian-match probe->bank, then solve the 6x6 operator assignment globally.
