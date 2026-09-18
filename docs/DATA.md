# Data contract (Phase 5)

Seed generator: `train/generate_seed.py`. Reference output: `data/seed.jsonl`
(gitignored). Everything is deterministic from one seed; a run pins its exact
input via `data_sha256`.

## JSONL row schema
```jsonc
{
  "id": "<family>:<state>",              // unique
  "state_id": "<state>",                  // groups questions of one state
  "family_id": "<family>",                // the generating source group
  "split": "train | dev | calibration | test | ood",
  "state": "<string | object | array>",   // what the questions are about (JSONL-safe)
  "questions": { "<qid>": { "type": "boolean|choice|score", "instructions": "…", "criteria": … } },
  "gold_probs": { "<qid>": { "<candidate>": <p> } },      // THE soft target distribution
  "gold_probs_kind": "programmatic_conditional_distribution",
  "gold_label": { "<qid>": <0|1> },        // observed outcome, sampled once from exact q (seeded)
  "gold_label_kind": "observed_outcome_sample",
  "metadata": { "source_group_id": "<family>", "n_words": <int> }
}
```

### The event and the exact distribution q
For the bag-of-tokens task, each state is a multiset of words drawn from a known
category prior. The **event** is "which category does the next sampled word come
from?"; the exact distribution is the additively-smoothed empirical share
`q_c = (count_c + alpha) / (N + K*alpha)`, `alpha=0.1`. Every question's gold is
a function of this q:

| Question | type | gold |
|---|---|---|
| `blob`  | choice (K categories) | `gold[c] = q_c` |
| `islead`| boolean (target = most-loaded category) | `gold P(yes) = q_target`; `gold_label` sampled ~q |
| `band`  | score (2 ordered bands) | `gold[band] = sum of q_c over the band` |

`gold_probs` is the cold truth distribution — this is what makes it a
parallel-decision objective (softmax target, not one-hot).

## Split hygiene (non-negotiable)
- Splits are GROUP-SEPARATED: one `source_group_id` never appears in more than
  one split. The generator mints a fresh family per split, so no leakage.
- The five splits: `train`, `dev` (checkpoint selection), `calibration` (for
  any post-hoc temperature/ECE work), `test` (reported), `ood` (held-out family
  priors, deliberately shifted → calibration must degrade there).
- A run that violates family-split-separation is invalid; the generator enforces it.

## Adding domain data (the moat)
Append rows in the SAME schema. To add real domain intent rows:
1. Give each provenance a `source_group_id`.
2. Provide `gold_probs` (a full distribution, not just a label) — this IS the
   parallel-decision product. If you only have a label, provide the observed
   outcome in `gold_label` and mark `gold_probs_kind` accordingly, but the
   calibration story needs a distribution target.
3. Keep `family_id` distinct per provenance and split rows by GROUP, never by
   individual row, so calibration/test stay honest.
4. Re-run `data_sha256` over the whole file and freeze it into the run config.

## Reproducibility
```bash
python train/generate_seed.py --out data/seed.jsonl --seed 17
sha256sum data/seed.jsonl          # freeze into the training run config
```
A training run dir records: checkpoint, config.json (incl. `data_sha256`),
train log, and per-split predictability. Calibration metrics are computed from
those saved predictions, never by re-running inference (see train/ece.py).