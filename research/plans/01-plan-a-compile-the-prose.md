# 01 — Plan A: Compile the Prose (constrained extraction)

Gold source: **what the skill's text says.** Trust the words, compile them
faithfully into decision tables.

## Premise
Your skills are already semi-structured (bullets, `if/when/never/prefer`
markers, code fences, inline commands). Plan A extracts decision points from
that prose into the fixed schema using schema-enforced extraction, then
verifies each rule is faithful to its source span. Failure mode it guards
against: the LLM emitting *schema-valid but wrong* rules — schema conformance is
NOT correctness. Everything below is aimed at closing that gap with evidence.

## Dependencies
- Step 0 complete: skill triage, `decision.schema.json`,
  `holdout_questions.jsonl`, `human_gold.jsonl`, `scripts/eval_head.py`.

## Steps an agent will follow

### A1 — Select skill subset
Use the `decision-dense` bucket from Step 0 (`skill_triage.json`). Start with
**3-4 skills** (recommended: podman-quadlet-deploy, systematic-debugging, the
NVIDIA pinning guidance, one Ryoku/Hyprland ops skill) before scaling to the
rest — proving the mechanism on a handful beats a broad half-faithful sweep.

### A2 — Extraction (schema-enforced, thought-free)
For each SKILL.md, in bounded chunks (one Skill section per call), prompt the
extractor to produce decision-table JSON matching `decision.schema.json` using
**Structured Outputs** (or self-hosted vLLM+Outlines/XGrammar constrained
decoding).

- temperature = 0.
- Give the model an UNCONSTRAINED reasoning field first, and the CONSTRAINED
  rule object last (constrain the output, not the thought). Pattern:
  `reasoning: "...", rules: [ ...conforming JSON ]`.
- Include the source text; require every `source` span to be a verbatim slice
  of it.

### A3 — Intermediate grounding pass
Before final extraction, burn one pass rewriting the prose as pseudo-code
condition primitives (`if cdi_yaml_absent then regenerate_cdi`), then extract
the table from the pseudo-code (the "executable grounding" finding of
BREX/ExIde: it beats plain prompting on complex/nested conditions).

### A4 — Judge + repair (bounded)
Run an LLM-as-judge scoring each emitted rule on: fidelity-to-source,
non-hallucination, rule-type correctness, condition completeness, actionability.
Any field below threshold (θ=0.9) → regenerate that rule only (≤3 attempts,
keep best). Upstream context (definitions) verified before rules, so repair
runs against trusted context.

### A5 — Evidence binding (the anti-hallucination guard)
For every extracted condition/value/outcome, verify it is grounded in a real
source span:
- condition mentions a field that actually appears in the input_schema;
- outcome labels are within the question's criteria set;
- the rule, when run, yields the branch the skill text prescribes for a state
  that triggers it.
Ungrounded → flag to a human at end of run; never silently accept. (Expect
~majority+ reduction in false extractions vs prompt-only, per the
extraction-pipeline literature.)

### A6 — Verify against gold (the checkable part)
Feed the extracted rules into the existing state-synthesizer (reuse the
`generate_seed.py` pattern). This proves:
- held-out branches (rules the skill states but you never hand-seeded) recover
  as the skill prescribed → the rules generalize, not memorize;
- states that trigger no rule fall through to `default_outcome` → honest
  low-confidence behavior.
Where a branch can't be synthesized-and-recovered, mark the rule
`verification: incomplete` for human review.

### A7 — Emit + train
- Aggregate into `research/data/planA_rules.jsonl` (+ `generate_seed`-style
  synthetic states → `planA_train.jsonl`).
- Train with the proven recipe: `python -m train.train --loss ce --target soft
  --epochs 30 --run-dir runs/planA` (matches the winning `seed_soft30`), then
  `scripts/fit_temperature.py` (expect T≈1; do not deploy a temperature that
  breaks ECE).
- ALSO synthesize held-out-branch questions and confirm they're recovered.

## Deliverables
- `research/data/planA_rules.jsonl` (the compiled decision tables, evidence-bound)
- `research/data/planA_train.jsonl` (synthetic states + gold)
- `runs/planA/checkpoint.pt` + predictions + ECE report (existing harness outputs)
- per-rule `verification:` label + the human-review surface (ungrounded flags)

## Risks & honesty
- Gold is "what the skill said," not "what is true." If the skill is wrong,
  Plan A faithfully compiles a wrong rule. Gold here is fidelity, not validity.
- Structured Outputs gives schema conformance, not semantic correctness — the
  judge+repair+evidence loop is not optional.
- Reproducibility is bounded: extraction content is model-dependent (temperature
  0 + same prompt reduces variance but does not prove determinism). The schema
  and the train data are fully reproducible once the rules exist.