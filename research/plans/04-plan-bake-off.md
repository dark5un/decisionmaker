# 04 — The Bake-Off: train 3 heads, compare on the same gates

Precondition: Plans A, B, C each produced a train set + a trained head
(`runs/planA`, `runs/planB`, `runs/planC`). This plan is the CONTROLLED
comparison. Its purpose is not to crown a winner by the number that each plan
can trivially game — it is to answer: "is decision data for Hermes something you
compile, you program, or you measure?"

## Non-negotiable controls (make the comparison meaningful)
- SAME harness + hyperparameters for all three: the proven `seed_soft30` recipe
  (`--loss ce --target soft --epochs 30`); T=1.0 unless a temperature demonstrably
  improves ECE without breaking it (per Phase-A finding, it usually won't).
- SAME schema, SAME skill subset, SAME held-out evaluation set + human gold
  (Step 0.3). Nobody gets a private eval.
- **Normalize dataset size.** Cap/subsample all three to the SMALLEST plan's
  coverage (Plan C is expected smallest). If you don't, "Plan A won because it
  had 10× the data" will be the result and the comparison is meaningless. Record
  the effective trained-N per model.
- Same `scripts/eval_head.py`. No ad-hoc metrics per plan.

## Evaluation protocol (in order; all models on all steps)
1. **Internal check** (each model on its OWN split):
   - boolean ECE on the plan's own test split (+ OOD split if the plan produced
     one),
   - pred-peak vs gold ceiling per question type (the honest-margin measure).
   This tells you each method's self-consistency. Expected: all three pass their
   own gate easily (they were trained on their own data). Do NOT stop here.

2. **Shared held-out gate** (the real test):
   - `eval_head.py` on `holdout_questions.jsonl` (fully held-out skills).
   - Report ECE, mean confidence, and per-type margin for each model on the SAME
     questions. This is where compile-vs-program-vs-measure separates.

3. **Cross-generalization matrix** (the scientifically interesting part):
   - Feed Plan A's generated states → model-B and model-C; Plan B's → model-A
     and model-C; Plan C's → model-A and model-B.
   - Output an N×N matrix of per-model ECE / margin on the OTHER plans' states.
   - Reading: cells that agree = converging manifolds; cells that diverge =
     methodologies disagree on what "correct" means. That divergence is the
     finding — it tells you whether the three gold sources are interchangeable
     or contradictory.

4. **Human-arbitrated verdict:**
   - Score all three models against `human_gold.jsonl` (the only truth beyond
     the three self-consistent truths).
   - This arbitrates the bake-off when internal and cross-matrix signals
     disagree.

## Deliverable: `research/plans/report/BAKEOFF.md`
A single report with:
- control table (N per model, hyperparameters identical, held-out set identical);
- per-model: internal ECE, pred-peak vs ceiling;
- shared held-out table (ECE / confidence / margin per question type);
- the cross-generalization matrix;
- the human-arbitrated table + verdict;
- a conclusion that names the winning method AND why, plus whether
  compile/program/measure are complementary rather than exclusive (e.g.
  "measure is the gold, compile is the coverage, generate is the volume" — if
  the matrix shows they agree, the honest recommendation is to merge: use Plan C
  truth to validate a Plan A/B dataset, not to choose one).

## Decision rules
- If all three are well-calibrated on the shared held-out set and the matrix is
  mostly-agree: the methods are interchangeable for training; pick by cost
  (A cheapest) and use C's measurements as the ongoing gold std.
- If only one calibrates on the shared set: that gold source is the real one;
  the others overfit their own synthetic splits.
- If the matrix shows disagreement where human gold sides with one: human gold
  wins; the others' training truths are wrong-but-self-consistent.
- If Plan C had too few samples to train, report C as a gold-reference rather
  than a model, and judge A/B against C's measured states instead.

## Edge cases / honesty
- A plan may be too small or too poor to "train a clean head." Say so rather
  than force it (mirrors the honest "insufficient sample" rule in Plan C).
- Do not tune three different epoch counts to make them all look good — the
  comparable regime is a fixed recipe, and that is the point.
- The human gold set is small (~10-20). Treat its verdict as directional, not a
  precision measurement; combine it with the cross-matrix, which is large.

## Exit criteria
`BAKEOFF.md` written, tables filled with real runs (no placeholders), a named
winner with the reason, and a recommendation on whether to merge methods. Then
the follow-on decision (which dataset feeds the served head) is a human call.