# Decision-Maker ← Hermes Skills: Plan Hub

Goal: produce calibrated, trainable decision data for Decision-Maker from the
Hermes skill corpus (~76 `SKILL.md` files), in three competing ways, then trade
them off in a controlled bake-off.

Why three ways: Decision-Maker's entire worth is CALIBRATION, which rests on
gold being exact and checkable, not guessed. The three plans differ by WHERE
gold comes from, because that is the one thing that decides calibration:

| Plan | Gold source          | Trust       | Coverage |
|------|----------------------|-------------|----------|
| A    | the prose (compiled) | words       | highest  |
| B    | a derived program    | executable  | medium   |
| C    | measured behavior    | runtime     | lowest   |

Trust ascends A < B < C; coverage descends A > B > C. Expect a trade-off
curve, not a clean winner.

All four plans are written for an agent to execute directly. Follow the plan
hub shared-prereqs (`00-shared-prerequisites.md`) first — every plan depends on
Step 0.

## Reading order
1. `00-shared-prerequisites.md` — Step 0 (skill triage, fixed schema, held-out eval set). RUN FIRST, once.
2. `01-plan-a-compile-the-prose.md` — constrained extraction.
3. `02-plan-b-derive-the-generator.md` — executable synthesis.
4. `03-plan-c-measure-behavior.md` — empirical bootstrap.
5. `04-plan-bake-off.md` — train 3 heads with the proven harness, compare.

## Non-goals / standing decisions (do not re-litigate)
- Do NOT build a broad skill-router/retriever head. This is about decisions
  INSIDE skills, not which skill to load.
- Do NOT get "correct answer" labels from a teacher LLM judging gold. That
  distills teacher bias and silently destroys calibration (proven). Gold must be
  exact-by-construction (B), compiled-from-text (A, evidence-bound), or
  measured (C).
- The trained head is served from `DECISIONMAKER_HEAD_CHECKPOINT` at T=1.0, the
  `seed_soft30` recipe (soft targets, ~30 epochs = hundreds of steps). The flat
  `seed_ce` (36 steps) and hard-target arms are disqualified.
- Skills whose decisions are pure judgment prose (e.g. `humanizer`) are NOT
  trainable by any of the three gold sources. They are triaged out in Step 0.

## Cross-plan consistency rules
- SAME schema everywhere: Rulebook-style decision table JSON (`input_schema` +
  ordered rules + `default_outcome` + canonical hash).
- SAME skill subset feeds all three plans (decided in Step 0).
- SAME held-out evaluation set + human-judged gold sample (decided in Step 0).
- SAME training harness + hyperparameters (the `seed_soft30` recipe).
- Dataset size NORMALIZED to the smallest plan's coverage, or "more data won"
  will trivially be the result and the bake-off is meaningless.

## Definition of project success
A comparison report showing, per method: internal ECE, pred-peak vs gold
ceiling, performance on the SHARED held-out set, a cross-generalization matrix
(A-states→model B, B-states→model C, etc.), and the human-arbitrated verdict.
The bake-off answers: "is decision data for Hermes something you compile, you
program, or you measure?"