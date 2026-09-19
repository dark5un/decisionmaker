# BUILD BRIEF — hand this file (and the plans it references) to a build agent

Read `research/plans/05-gold-compiler-plan.md` and
`research/plans/06-business-history-gold.md` first. This brief is the
authoritative contract: what to build, to what standard, and what must NOT be
changed. The agent needs this file + the two plans + the repo's existing harness.

## Mission

Build a **Gold Compiler**: a CLI that turns raw prose (a SKILL.md or any
decision-dense document) into **exact-gold, calibrated Decision-Maker training
data**, WITHOUT a teacher LLM as the source of truth. The core is document →
*verified rules* → *computed gold*. A separate **pluggable gold-loader** surface
lets OTHER domains source gold from something besides compiled rules (a historical
outcome log is ONE such example; your own custom loader can also drop in). Business
/history is an *illustrative* use, not a hard build requirement — build the general
mechanism; the specific loader is optional scaffolding for it.

Gold must be exact-by-construction: extract rules from text, *verify* them
constructively (not by a second model judging), then **compute** the gold
distribution from those rules. Two runs over the same (input, seed) are
byte-identical.

## Hard constraints (do not violate)

1. **No teacher-LLM judging gold.** No "second model scores whether this is
   correct." Verifier = deterministic interpreters + verbatim span matching.
2. **No changes to** `train/train.py`, `server/src/*`, `decision.schema.json`,
   `scripts/eval_head.py`, or the `seed.jsonl` row schema. The compiler's output
   must load in `train.py` unchanged.
3. **Refuse judgment-prose.** Input with no pinnable answer → exit 2, tag
   `no_gold`. Never emit invented gold for uncalibratable text.
4. **Read-only probes** (Plan C referee): never mutate the machine; revert any
   crafted state.
5. **Determinism is a test, not a hope:** the synthesizer double-run must be
   byte-identical (`data_sha256`), CI-checkable.
6. **Do not fabricate results.** If a probe/measurement can't be obtained, say so
   and mark the rule `verification: incomplete` for a human.

## Environment (facts, not assumptions)

- Repo here on disk: `/home/px/workspace/decisionmaker` (already has the harness,
  `runs/seed_soft30` winning head, `scripts/eval_head.py`, `scripts/probe_decisions.py`,
  `research/planA/rules.py`, `research/planA/synthesize_planA.py`,
  `research/generators/planB_generator.py`, `data/seed.jsonl`).
- Training runs in a container (host python has no torch):
  `podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v /home/px/workspace/decisionmaker:/app:Z \
  -v /home/px/.cache/huggingface:/root/.cache/huggingface:Z --workdir /app \
  localhost/decisionmaker-serve:local python -m train.train ...`
- Training recipe (FIXED, matches winning arm): `--loss ce --target soft
  --epochs 30`. Temperature: keep T=1.0 unless a fit demonstrably improves ECE
  without breaking it (calibration reference: `References/calibration…` in the
  decisionmaker skill). GPU: `--gpu GPU-34466eba-2642-c31a-2e44-6451b6a6b425`
  (the free 5090).
- Data dir is gitignored; force-add `research/data/*` if it must be committed
  (`git add -f`).
- User standing rules (bind): never run state-changing commands unilaterally
  (no chmod/install/start-units without handing the user the exact command);
  verify empirically, re-read live state, don't trust cached maps.

## Deliverables to build (in dependency order)

Command is ONE tool: `scripts/gold_compiler.py` plus the history loader.

**Phase 0 — contract lock**
- `scripts/gold_compiler.py` CLI: `--input <SKILL.md|text> --out <dir>` with exit
  taxonomy: 0 = emitted, 2 = refused no-gold, 3 = emitted with human-review flags.
- Add `validate_table()` (jsonschema over `decision.schema.json`); refuse path for
  known `judgment-prose` returns exit 2 with evidence.

**Phase 1 — extract**
- `gold_compiler.py extract <text>` → schema-valid candidate `rules` + free-form
  `reasoning`, temperature 0, structured constrained output. Chunk long text one
  section per call. Every `source` is a verbatim slice of the input.
- Acceptance: 100% schema-valid on the 3 shortlist + 3 held-out skills; two runs
  with same seed → identical rule JSON.

**Phase 2 — verify (the bulk; the anti-hallucination wall)**
- `gold_compiler.py verify <table>`:
  (a) **evidence binding (static):** every condition `field` in `input_schema`,
      every outcome in the criteria set, `source` is verbatim;
  (b) **behavioral oracle:** the rule interpreter
      (`research/planA/rules.py` `evaluate_condition`/`run_rules` already exist)
      runs each rule on representative trigger states and asserts the branch a
      hand-marked anchor says it should take. Anchors = the only human-in-loop
      step (a few states per rule).
- Verdicts `verified | incomplete | contradicted`; contradicted → regenerate that
  rule only (≤3 attempts, keep best). Human review manifest at end.
- Acceptance: re-derive the 10 shortlist tables → all `verified`; inject a
  known-bad table (bad field, outcome not in criteria, behavior ≠ anchor) → caught.

**Phase 3 — synthesize**
- `gold_compiler.py synthesize <verified_table>` → generalize
  `research/planA/synthesize_planA.py`: enumerate state space from `input_schema`,
  render states to prose (pluggable templating), run interpreter to compute gold,
  emit `train.jsonl` (group-separated splits, soft gold, data_sha256). `--k-phrasings`.
- Acceptance: byte-identical double-run; `train.py` loads it; no family leak.

**Phase 4 — probe referee (optional, reuse)**
- `gold_compiler.py probe <table>` reuses `scripts/probe_decisions.py` to compare
  measured vs rule-predicted outcomes on observable decisions. Discrepancy → flag
  a rule for regen, never silently reconcile. Read-only.

**Phase 5 — CI + optional extractor model**
- Makefile/pre-commit: `gold-compiler-lint`, `gold-compiler-verify-corpus`
  (re-verify 10 shortlist tables), `gold-compiler-determinism` (sha).
- Optional: once N verified prose→table pairs exist, fine-tune a small
  schema-decoder emitting rules directly into `decision.schema.json` grammar —
  SAFE ONLY because the verifier catches its errors. Metric = verifier pass-rate +
  measured agreement, NOT JSON-parse success.

**Optional/illustrative gold-loader (shows the pluggable surface) — plan 06**
Plan 06 (`06-business-history-gold.md`) describes ONE example of the pluggable
gold-loader interface: sourcing gold from a historical outcome log instead of
compiled rules. Build it **only if time permits or the user asks** — its purpose
is to demonstrate that the gold source is swappable. The general mechanism
(compiler + loader interface) is the required deliverable; this specific loader is
example scaffolding. If built, it must satisfy: bucket rows → empirical gold
distribution per state, additive-smoothed (reuse `generate_seed.py` smoothing),
same seed.jsonl schema, train/dev/cal + a held-out test block, `--min-count`
refusal, optional confounder guard. And it must be reproducible and train unchanged.

## Definition of done
All five compiler phases built; the pluggable gold-loader interface exists (the
business/history loader in plan 06 is optional scaffolding — done only if time
permits); acceptance gates pass on real runs (no placeholders);
`gold-compiler-verify-corpus` green on the 10 shortlist tables; every emitted rule
evidence-bound or human-flagged; deterministic double-runs; and
`python -m train.train --data <emitted>` trains a checkpoint on the unchanged
harness. Commit the source + plans; force-add `research/data/*` outputs.

## If stuck / blocked
Say so directly and hand over the exact command or blocker to the user — do not
invent output. Prefer the existing harness and Plan-A modules over new abstractions.
