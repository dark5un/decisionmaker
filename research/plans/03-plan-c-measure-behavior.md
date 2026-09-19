# 03 — Plan C: Measure Behavior (empirical bootstrap)

Gold source: **what the system actually does at runtime.** Don't trust the words
(Plan A) or a derived program (Plan B); run the skill's own prescribed
checks/commands on synthetic states and record the outcome as empirical gold.

## Premise
This is the empirical-verification discipline you already live by (verify the
engine, verify the GPU, re-read /proc, never trust a cached map) applied to
*data generation* instead of *diagnosis*. Gold = observed outcome of an actual
execution on a real state. The most defensible ground truth of the three — but
only reachable for decisions whose outcome is OBSERVABLE as a command/check, so
coverage is lowest. Failure mode it guards against is the opposite of the other
two: no LLM in the gold path, so no bias can be distilled. The correct framing:
fewer, higher-trust samples over volume.

## Dependencies
- Step 0 complete (triage must be enriched for observability — see C1).
- A working probe environment: the target system (this Ryoku/Arch box) plus the
  cluster of tools the skills describe (podman, systemd, nvidia-smi, git, etc.).

## Steps an agent will follow

### C1 — Observability triage (adds a dimension to Step 0's triage)
For each `decision-dense` skill, classify each decision point as:
- **observable** — its outcome is reachable by running a command/check on a
  constructed state (e.g. "newuidmap lost setuid → everything fails": set the
  state, run `ls -l /usr/bin/newuidmap` + `podman info`, read the outcome).
- **not observable** — outcome is judgment or unmeasurable (e.g. "how natural
  is this phrasing?"). Excluded from Plan C, or human-gold-only.
Keep only skills with a meaningful count of observable decisions. If the narrow
start set has <2 observable decisions, pick a different skill — the plan must
not be empty.

### C2 — Build a probe harness
`scripts/probe_decisions.py <skill> <decision_id> <state>` that:
- materializes the state (create/remove a file, set an env, start/stop a unit);
- runs the exact check/tool the skill prescribes (the SKILL's commands, verbatim);
- captures the observable outcome (exit code, presence of a file, a log line,
  a device in `nvidia-smi`, etc.) as the gold;
- reverts the state (never leaves the box altered between probes).
Log every probe with: state, command(s) run, raw output, parsed outcome, and
the exact source span of the check in SKILL.md (traceability).

### C3 — Measure, don't assert
Enumerate decision points × meaningful state variations (the axes the skill's
conditions describe). Run the harness. Record outcomes empirically. Where the
observed outcome disagrees with what the skill's text predicts, that is data:
it marks (a) a skill bug, (b) a state the skill didn't cover (→
`default_outcome`), or (c) a measurement error. Do NOT silently reconcile —
flag the discrepancy to the human; it is exactly the kind of "verify, don't
trust" signal this plan exists to surface.

### C4 — Sanity the heuristic before scaling
Before generating many samples, confirm the first ~measured outcomes match
first-principles expectation (a reviewer checks a handful by hand). If the
harness misreads common outcomes, fix measurement BEFORE bulk generation —
the same discipline as validating a generator before a big run.

### C5 — Emit + train
- `research/data/planC_train.jsonl` (state + empirical gold), each row carrying
  probe provenance (state, command, raw output, parsed outcome, source span).
- Train with the proven recipe: `python -m train.train --loss ce --target soft
  --epochs 30 --run-dir runs/planC`, then `scripts/fit_temperature.py`
  (gate on ECE). Truthfully: N may be small (plan C is the sparse one) — if it
  is too small to train a clean head, say so and treat Plan C as the
  highest-trust calibration reference rather than a train set.
- Evaluate with `eval_head.py` on the SHARED holdout set.

## Deliverables
- `scripts/probe_decisions.py` (+ per-probe provenance log)
- `research/data/planC_train.jsonl` with full probe provenance
- `runs/planC/checkpoint.pt` + predictions + ECE report (or an honest
  "insufficient sample to train cleanly" note)
- discrepancy report (state where measured ≠ skill-text prediction)

## Risks & honesty
- Lowest coverage by design. Do not pad it with synthetic labels — padding Plan
  C with LLM-judged outcomes would make it Plan A by another name and defeat
  its purpose.
- Probes are slow, stateful, and can alter the box if the revert is wrong. The
  harness must be run where it is safe and must ALWAYS revert state.
- Highest-trust gold, smallest dataset. Its role in the bake-off is honorable:
  if Plan C calibrates reasonably on a fraction of the data, that is a strong
  statement about the other two plans' efficiency (they train on words/programs,
  C trains on truth).