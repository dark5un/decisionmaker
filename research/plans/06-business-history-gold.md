# 06 — Example gold-loader: business decisions from historical outcome data

> STATUS: this is an **illustrative example** of the pluggable gold-loader
> surface in the Gold Compiler (plan 05), not a hard build requirement. It
> shows one way a non-skill domain can source exact gold. Build it only if
> time permits / the user asks; the general mechanism is the deliverable.

The highest-value *illustration* of the Decision-Maker generalization. Where the
skill corpus needed hand-compiled (Plan A), hand-programmed (Plan B), or
sparse-probed (Plan C) gold, **business decisions have gold fall out of historical
logs**: the empirical outcome distribution of a candidate given a decision-state is
*measured, real, and reproducible*. This is Plan C at scale — the most defensible
trust source, with no teacher LLM in the path.

This plan defines the gold-source loader and its contract so it plugs into the
*unchanged* training harness and serve path.

## Mapping (the whole trick)

| Decision-Maker concept | Business source |
|------------------------|-----------------|
| `state`                | decision-time feature vector (segment, market, deal size, tenure…) rendered to prose or structured text |
| candidate `leaf`       | the discrete choice (approve / decline, renew / lapse, invest tier A/B/C…) |
| `gold_probs`           | **empirical outcome distribution** for that state bucket from the historical log |
| question type          | `boolean` (2 candidates) or `choice` (K candidates, K≥2) — score not used for outcome gold |
| human-gold / referee   | the same real log, held out by **time** (train on past, evaluate on future) — the strongest generalization test available |

## Non-goals / standing decisions

- **No teacher LLM labels the gold.** History *is* the ground truth.
- **Calibration ≠ correctness of policy.** If the historical close-rate was driven
  by a flawed policy, the head replicates that policy with high calibration.
  This is reported, never hidden — same honesty caveat as "Plan A compiles
  whatever the skill says, even if it is wrong."
- **No causal claim from raw history.** Outcome gold is a *conditional outcome
  distribution*, not an estimate of the effect of a past decision. Bucketing must
  control for treatment/confounders before gold is trusted for decision-causal use.
- **No changes to `train.py`, the schema, or the serve path.** The loader emits
  the existing `seed.jsonl` row schema so training is byte-identical in behavior.

## Deliverable: `scripts/history_to_gold.py`

A small, deterministic, torch-free loader. Input is a tabular log (CSV or SQL
query result / parquet) with at least these columns:

```
<state features...>, <decision>, <candidate>, <outcome>
```

Contract:

- **Bucket:** collapse rows by (state features → `state`), `candidate`.
- **Empirical gold:** `gold_probs[candidate] = count(outcome=1 for that bucket)
  / count(rows in bucket)`, smoothed with `alpha` additive smoothing (reuse the
  exact `smoothed()` shape from `train/generate_seed.py`).
- **Exact-by-construction:** two runs over identical (log, seed) are byte-identical;
  emit `data_sha256` (reuse the same hashing). Determinism is CI-checkable.
- **Row schema:** emit the identical shape as `generate_seed.py` / `data/seed.jsonl`
  — `state` (prose or canonical-JSON of features), `questions{qid}`, `gold_probs{qid}`,
  group-separated `split` by source family, `metadata`.
- **Splits:** train/dev/cal by feature family (or time block for held-out),
  **test = a future time block** the head never saw → the OOD-style honest
  generalization signal.
- **Confounder guard (optional `--treat` flag):** when a treatment column exists,
  bucket gold *within* treatment arm so outcome gold is not a mix of
  decision-correctness and luck-of-action.
- **Refuse path:** a bucket with too few observations (`--min-count`, default ~20)
  is dropped or marked `gold: insufficient` — never padded with invented rows.

### CLI

```
scripts/history_to_gold.py --input deals.csv --state 'segment,campaign,size'
                          --decision-col decision --outcome-col closed
                          --out research/data/biz_train.jsonl
                          [--alpha 0.1] [--min-count 20] [--treat arm] [--seed 17]
```

## Acceptance gates

1. **Reproducibility:** double-run on a fixed seed → identical bytes + `data_sha256`.
2. **Unchanged harness:** `python -m train.train --data biz_train.jsonl` runs with
   the existing recipe (loss=ce, target=soft, epochs=30) and writes a checkpoint —
   no edits to `train.py`.
3. **Honest ceiling surfaced:** print mean gold-peak per candidate (the data ceiling
   the calibration reference requires — recompute before diagnosing flatness).
4. **Time hold-out:** the head's ECE on the held-out future block is reported
   (this is the real business generalization metric, not a synthetic floor).
5. **Refusal:** sub-min-count buckets are omitted or flagged, never fabricated.

## Risks & honesty

- **Concept drift** is the OOD canary: a head calibrated on last year is not
  calibrated today. The existing OOD ECE metric *is* the drift signal; surface it
  in any deployment report (the bake-off measured the same effect, 0.17 → 0.48,
  when priors shifted).
- **Data ceiling on genuinely ambiguous decisions:** where history is near
  coin-flip, calibrated peak p is low *by construction*. Low p there is correct,
  not a bug — communicate it as a gate ("escalate at p<0.6"), not as failure.
- **Policy lock-in is a product question, not a bug:** high calibration on a bad
  historical policy is the expected, honest output. The deployment decision is a
  human call (as everywhere in this system).

## Relation to the build package

`history_to_gold.py` is one gold-loader in the generalized pipeline (plan 05).
It shares the schema, the row shape, the harness, the ECE/OOD gates, and the
Calibration-first philosophy with the skill compiler — only the *source of gold*
differs (historical log instead of compiled rules). An agent building it reuses
`train/generate_seed.py`'s smoothing+split+hashing, `decision.schema.json`, and
`scripts/eval_head.py` verbatim.
