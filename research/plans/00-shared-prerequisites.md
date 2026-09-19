# 00 — Shared Prerequisites (Step 0) — RUN FIRST, once

Everything else depends on this. Run once, commit the outputs. All three plans
and the bake-off consume the artifacts produced here.

## Inputs
- Hermes skill corpus: `/home/px/.hermes/skills/**/SKILL.md` (76 files).
- Proven training harness: `~/workspace/decisionmaker/train/train.py` +
  `scripts/fit_temperature.py`.
- Prior validated findings (do not re-run): `runs/seed_soft30` is the winning
  serve head; T=1.0; hard-target and 36-step arms disqualified.

## 0.1 — Skill triage (bucket all skills)
Read every `SKILL.md`. Assign exactly one bucket:

- **`decision-dense`** — decisions are expressed as explicit conditional rules
  or stated preferences with a correct answer (e.g. podman-quadlet, GO setup).
  These proceed to all three plans.
- **`judgment-prose`** — decisions are soft judgment ("make it sound natural"),
  no pinnable correct answer (e.g. humanizer, press). NOT trainable by any of
  the three gold sources. Triaged out, but LISTED as a known non-goal.
- **`chore`** — no decisions, just invocation steps (e.g. "run this command").
  Lists how-to only. Triaged out.

Deliverable: `research/plans/skill_triage.json`
```json
{
  "decision-dense": ["podman-quadlet-deploy", "..."],
  "judgment-prose": ["humanizer", "..."],
  "chore": ["git-...", "..."],
  "metrics": { "total": 76, "decision_dense": 25 }
}
```

IMPORTANT: prefer skills that are BOTH decision-dense AND partially observable
(rules whose outcome can be measured by running a command/check). One candidate
shortlist to start narrower: podman-quadlet-deploy, systematic-debugging, the
NVIDIA GPU pinning guidance, and one Ryoku/Hyprland ops skill. Starting narrow
keeps Plan C non-empty and the bake-off honest.

## 0.2 — Fixed schema (decided; do not redesign)
One decision-table JSON per decision point:

```jsonc
{
  "skill": "podman-quadlet-deploy",
  "decision_id": "gpu-cdi-presence",
  "source": "SKILL.md#L77",
  "input_schema": { "required": ["cdi_yaml_absent", "gpu_vanilla"],
                    "properties": { "cdi_yaml_absent": {"type":"boolean"},
                                    "gpu_vanilla": {"type":"boolean"} } },
  "question": { "type": "choice",
                "criteria": { "regenerate_cdi": "no-op",
                              "regen_and_restart": "restart",
                              "normal": "nothing needed" } },
  "rules": [ { "id":"r1", "priority":100,
               "condition": {"all":[{"field":"cdi_yaml_absent","operator":"eq","value":true}]},
               "outcome":"regenerate_cdi" },
             { "id":"r0", "priority":0, "condition": null,
               "outcome":"normal", "is_default": true } ],
  "canonical_hash": "sha256:<computed on commit>"
}
```

Include a JSON Schema validator file (`research/plans/decision.schema.json`).
Rules without an explicit condition become `is_default:true` (the "no rule
matched → review/escalate" case → honest low confidence).

Deliverable: `research/plans/decision.schema.json`.

## 0.3 — Fixed held-out evaluation set (THE gate)
- Pick a small set of FULLY-HELD-OUT skills whose decisions no training set will
  see (sample from `decision-dense`). Their decision points become the
  evaluation questions.
- Sample a human-judged gold set: ~10-20 decision questions labeled by a human
  as ground truth. THIS is the only "truth" beyond the three self-consistent
  synthetic truths. It arbitrates the bake-off.

Deliverable: `research/plans/holdout_questions.jsonl` +
`research/plans/human_gold.jsonl`.

## 0.4 — Shared evaluator (one tool, all plans)
Write `scripts/eval_head.py <checkpoint> --questions holdout_questions.jsonl`
that loads a head (same path as `run_server.py`), runs the held-out questions,
and prints:
- per-question predicted distribution + confidence
- boolean ECE on any held-out boolean questions (vs human gold)
- per-question type pred-peak (max prob) and the resulting margin
- mean confidence per type

All three models are measured with THIS tool against THIS set. No other metric
is accepted for cross-plan comparison.

## Exit criteria
All of 0.1-0.4 exist, validated, and committed. Then proceed to Plans A, B, C
in any order (they are independent), then to the bake-off.