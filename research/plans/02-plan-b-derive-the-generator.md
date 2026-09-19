# 02 — Plan B: Derive the Generator (executable synthesis)

Gold source: **the output of a program the LLM writes.** Don't extract rules as
inert data; have the LLM write a small generating function that COMPUTES gold
from synthetic states — the `generate_seed.py` exact-gold pattern, parameterized
per skill.

## Premise
`generate_seed.py` already proves the strongest reproducibility layer in this
project: when gold is *computed* by a deterministic program over a known prior,
two runs over identical (program, seed) are byte-identical. Plan B generalizes
that: per skill, the "seed generator" becomes a small hand-checked Python
function. Gold is executed, not extracted. Failure mode it guards against: a
generator that is subtly wrong but satisfies its own generated tests by
construction — closed with a human-once review of the *code*, and cross-check of
its branches against the skill's own text.

## Dependencies
- Step 0 complete (triage, schema, holdout, human gold, eval_head.py).
- Case study: re-read `train/generate_seed.py` before writing any generator; it
  is the template (deterministic, torch-free, exact q, group-separated splits,
  alpha smoothing knob, data_sha256).

## Steps an agent will follow

### B1 — Select skills
The SAME `decision-dense` subset as Plan A (that is a cross-plan consistency
rule so the bake-off is apples-to-apples). Start narrow (3-4 skills).

### B2 — Have the LLM write a generator per skill
For each skill, prompt the LLM to produce `research/generators/<skill>.py` — a
self-contained Python module mirroring `generate_seed.py`:
```python
# make_row(sid, fid, rng, pi, alpha) -> row  # exact-gold pattern
# q = computable function of state features   # e.g. smoothed counts
# gold_probs = {qid: exact-distribution}      # no guesses
# splits: group-separated (train/dev/cal/test/ood)
# deterministic: everything seeded, data_sha256 emitted
```
The gold is whatever the program computes — byte-reproducible from (program +
seed). Require it to reproduce the plan's fixed schema (same `question` types:
boolean / choice / score) so downstream training is uniform.

### B3 — Judge and review the PROGRAM, not the data (human-once)
This is the single most important step and it is NOT automatable away:
- LLM-as-judge checks the generator against the skill's source spans: does the
  program's branch logic faithfully encode what the skill states?
- A human reviews the generator code ONCE per skill and signs off (the authority
  on truth is the skill author, not a second model).
- Detection harness: for every decision_id in the skill, run the generator with
  a fresh seed and confirm its produced gold for triggered branches matches the
  skill's prescribed behavior. Any mismatch → fix the generator, not the data.

### B4 — Synthesis + held-out check (the checkable part)
- Run the generator over multiple seeds to produce `planB_train.jsonl`
  (train/dev/calibration/test + an OOD family that deliberately shifts priors —
  the seed pattern's defining move).
- Confirm OOD calibration DEGRADES (that is correct and expected; it proves the
  OOD machinery works), and held-out branches (skill statements not hand-seeded)
  are recovered in-family.
- Two runs over the same (program, seed) must be byte-identical (canonical
  hash); assert this in CI (`data_sha256`).

### B5 — Emit + train
- `research/data/planB_train.jsonl` (+ `planB_generators.json` recording program
  hashes + seeds).
- Train with the proven recipe: `python -m train.train --loss ce --target soft
  --epochs 30 --run-dir runs/planB`, then `scripts/fit_temperature.py`
  (expect T≈1; gate on ECE, never deploy a temperature that breaks it).
- Evaluate with `eval_head.py` on the SHARED holdout set.

## Deliverables
- `research/generators/<skill>.py` (hand-reviewed generators)
- `research/data/planB_train.jsonl` (+ program/seed manifest)
- `runs/planB/checkpoint.pt` + predictions + ECE report
- a human sign-off note per reviewed generator

## Risks & honesty
- The failure mode is a generator whose logic is internally consistent but
  wrong relative to the skill. Mitigated by B3's human-once program review and
  branch cross-check — do not skip it, because the data will be "well-calibrated"
  on a wrong prior.
- Highest reproducibility of all three by construction, but only as good as the
  program/seed manifest captures (hash + seed must be committed with the data).
- Gold is "what the program computes," anchored to the skill's text by review —
  still one abstraction removed from measured runtime truth (Plan C).