# 05 — Gold Compiler: raw text → calibrated Decision-Maker training data

Design goal: a pipeline that takes **raw prose** (a SKILL.md, a doc, any decision-
dense text) and emits **exact-gold training rows** for Decision-Maker — without a
teacher LLM as the source of truth. Gold is exact-by-construction: a program
extracts *rules* from text, a verifier proves they're faithful, and a synthesizer
**computes** the gold distribution from those rules.

This is the generalized Plan-A compile loop (`rules.py` + `synthesize_planA.py`)
plus the Plan-C referee (`probe_decisions.py`), turned into one tool.

---

## Non-goals / standing decisions (do not re-litigate)

- **No teacher-LLM judging gold.** A second model scoring "is this correct?" is
  banned by the plan hub — it distills bias and silently destroys calibration.
  Gold comes from extracted-and-verified rules, then is *computed*.
- **No uncalibratable input.** `judgment-prose` (humanizer, p5js, meeting-action-
  items, ~23/78 skills) has no pinnable answer. The tool must **refuse** (tag
  `no_gold`) rather than emit noise.
- **No change to the serve path / head selection.** This tool generates training
  data only; which head feeds the service stays a human call.
- **No new architecture.** `seed_soft30` recipe, `decision.schema.json`,
  `seed.jsonl` row schema all stay. The compiler's output plugs into `train.py`
  unchanged.

---

## System overview

```
 raw prose ──▶ EXTRACT ──▶ VERIFY ──▶ SYNTHESIZE ──▶ decision_table.json
   (SKILL.md)    |           |           |                 + train.jsonl
                 └───────────┴───────────┴──▶ PROBE (referee, Plan C, optional)
                                             (measured ≠ predicted ⇒ regen rule)
```

One tool: `scripts/gold_compiler.py --input <SKILL.md> [--out DIR] [--probe]`

---

## Phase 0 — Contract & schema lock (smallest, everything depends on it)

**Deliverable**
- `scripts/gold_compiler.py` CLI skeleton with the wire contract frozen:
  `--input <path|text> --out <dir> --schema decision.schema.json [--probe] [--refuse-if-judgment]`
  plus a golden-handled exit taxonomy (0 = emitted, 2 = refused-no-gold,
  3 = emitted-with-human-review flags).

**Acceptance gates**
- Schema: `decision.schema.json` is the single importable validator; add a
  `validate_table()` in `gold_compiler` (jsonschema-backed) that a table must
  pass before any row is emitted.
- Refuse path: feeding a known `judgment-prose` snippet returns exit 2 with
  `{"reason":"judgment-prose","evidence":[...]}` from Step 0's triage categories.

**Why first** — every later phase consumes the schema/contract; locking it now
prevents rework. Reuses what we already validated.

---

## Phase 1 — Extractor (bounded, one section at a time)

**Deliverable**
- `gold_compiler.py extract <text>` → emits `reasoning: "..."` + a
  schema-conforming `rules` list, **constraining the output, not the thought**
  (temperature 0, structured output / Outlines-constrained decoding; reasoning
  field free-form).
- Chunking: for long text, one Skill/decision section per call, not the whole doc
  (matches Plan A2).

**Acceptance gates**
- 100% schema-valid output over a canned corpus of 3 shortlist skills + 3 held-out
  skills (re-derive the exact tables we hand-built in `rules.py`).
- Determinism check: same chunk + same seed ⇒ identical rule JSON (two runs).
- Every emitted `source` is a **verbatim slice** of the input (string-matched).

**Note** — the extractor is an *LLM writing candidate rules*. Its output is NOT
trusted; Phase 2 is the authority. We optimize extraction for *recall of correct
candidate rules*, not for being right (the verifier decides right).

---

## Phase 2 — Verifier (the anti-complacency wall; the real innovation)

**Deliverable**
- `gold_compiler.py verify <table>` runs two *constructive* checks, no second LLM:
  1. **Evidence binding (static):** every condition `field` ∈ `input_schema`;
     every outcome label ∈ the question's criteria set; every rule's `source`
     is a verbatim span; rule references resolve.
  2. **Behavioral check (the rule-"oracle"):** a tiny rule interpreter
     (`research/planA/rules.evaluate_condition` already exists) runs each rule
     against representative trigger states and asserts the branch a *hand-marked
     anchor* says it should take. Anchor marking is the ONLY human-in-loop step
     (a few states per rule, cheap).
- Per-rule verdict: `verified | incomplete | contradicted`, with the violating
  span/state. `contradicted` ⇒ regenerate that rule only (≤3 attempts, keep best).
- Output surface for humans: a review manifest listing every ungrounded /
  contradicted rule flagged at end of run — nothing silently accepted.

**Acceptance gates**
- Re-derive the 10 shortlist tables; **all 10 pass `verified`** (they encode the
  same semantics as `rules.py`).
- Inject a known-bad table (a rule referencing a non-schema field, an outcome not
  in criteria, a behavior that contradicts its anchor) ⇒ caught and flagged, never
  passed.
- Fidelity metric reported: % of emitted rules that pass evidence binding
  (target ≥ 0.9 on real skills; the literature-"majority+ reduction in false
  extractions" the plan hub promised).

**Why this isn't "an LLM judging"** — the verifier is a deterministic interpreter +
span equality, not a graded model answer. No judge bias can leak.

---

## Phase 3 — Synthesizer (exact-gold rows; reuse, don't rewrite)

**Deliverable**
- `gold_compiler.py synthesize <verified_table>` → generalizes `synthesize_planA.py`:
  enumerate the state space from `input_schema`, render states to prose (a small
  per-skill templating layer, pluggable), run the interpreter to **compute** the
  gold distribution, emit `train.jsonl` in `seed.jsonl` schema (group-separated
  splits, soft-target gold, `data_sha256`).
- `--k-phrasings` knob controls prose re-rendering per state (Plan A's data-width).

**Acceptance gates**
- Byte-identical double-run (sha check) — the determinism test harness.
- Output loads in `train.py` unchanged (`python -m train.train --data <out>` runs).
- Group separation: every source family appears in exactly one split.

---

## Phase 4 — Probe referee (Plan C integration; the ongoing truth source)

**Deliverable**
- `gold_compiler.py probe <table|train.jsonl>` → for each **partially-observable**
  decision, reuse `probe_decisions.py` to run the skill's actual commands on a
  sample of states and compare measured outcome vs rule-predicted outcome.
  Discrepancy ⇒ flag a specific rule for regeneration (bounded) — never silently
  reconcile.
- Read-only guarantee: probes never mutate the box (matches the user standing rule
  and Plan C's revert discipline).

**Acceptance gates**
- On the 5 existing shortlist probe states, measured == predicted where the skills
  were truthful; disagreement is surfaced as a `discrepancy` report (even when the
  *skill text* is what's wrong — that's data, not a bug to hide).
- A `--probe` run never leaves the box altered (diff-check before/after).

**Note** — Plan C stays the *gold-reference*, not a train set, at this scale
(measured outcomes too sparse to train). Its job here is referee, exactly as the
bake-off recommended.

---

## Phase 5 — CLI polish, CI, and the extractor-model graduation (optional)

**Deliverables**
- Makefile entries + CI: `gold-compiler-lint`, `gold-compiler-verify-corpus`
  (re-verify the 10 shortlist tables), `gold-compiler-determinism` (sha), all of
  which must go green on `pre-commit`.
- **Extractor model (graduation, only if demand justifies):** once we hold N
  hand-verified SKILL.md→Rulebook pairs, fine-tune a small **schema-decoder**
  (emit rules directly into `decision.schema.json` grammar) on those pairs. It's
  safe *only because* the verifier makes its errors recoverable. Metric optimized:
  **verifier pass-rate + measured-state agreement**, NOT JSON-parse success.

**Acceptance gates**
- CI green on all three checks from a clean checkout.
- Extractor model, if built, beats the baseline extractor on verifier pass-rate
  (not on "does it parse").

---

## Honest limits (state them in the README, don't hide them)

1. **Judgment-prose refuses** — no extractor conjures calibration for it. The
   tool emits `no_gold`, matching Step 0's triage.
2. **Coverage caps usefulness:** a head stays calibrated only within the schema it
   trained on; this compiler widens coverage per-skill but never makes a head
   general across schemas. That's the service's limit, not the compiler's.
3. **Partial observability:** where a decision's outcome can't be measured (probe
   unavailable), `verification` caps at `incomplete` for human review — no
   fabricated "verified."
4. **Extractor content is model-dependent** (temp 0 + seed reduces, doesn't
   eliminate, variance). The trainer rows are fully reproducible once the table
   exists; the table's *candidate rules* are not guaranteed byte-deterministic
   across extractor versions — but the verifier normalizes that.

---

## Deliverable summary (what "done" means)

| # | Artifact | Gate |
|---|----------|------|
| 0 | `scripts/gold_compiler.py` CLI + schema validator + refuse path | contract frozen, exit taxonomy green |
| 1 | `extract` — schema-valid candidate rules, verbatim sources | 100% valid on 6-skill corpus, deterministic |
| 2 | `verify` — constructive evidence + behavioral oracle | 10/10 shortlist verified; injections caught |
| 3 | `synthesize` — exact-gold `train.jsonl` | byte-deterministic, trains unchanged, no family leak |
| 4 | `probe` — Plan C referee | measured==predicted where truthful; discrepancies surfaced; non-mutating |
| 5 | CI (lint/verify-corpus/determinism) + optional extractor model | green on clean checkout |

**End of Phase 5 = a repeatable "SKILL.md in → calibrated train.jsonl out" tool**,
with the judge out of the loop, every rule evidence-bound or human-flagged, and
Plan C as the ongoing referee. The human call on which head serves remains
upstream of all of it.

---

## Suggested build order & effort
Phase 0 (0.5 day) → Phase 1 (1 day) → Phase 2 (1–1.5 day, the bulk) →
Phase 3 (0.5 day, reuse) → Phase 4 (0.5 day, reuse `probe_decisions.py`) →
Phase 5 (0.5–1 day + optional model spike).
Worth doing **all of 0–4** before any extractor-model ambitions; the model is only
an efficiency gain on top of a verifier that already makes it safe.

---

## Scope / generalization — it's schema-bound, not skill-bound

Everything above is written about `SKILL.md`, but **the compiler and the service
are domain-agnostic**. The unit the whole pipeline operates on is the **Rulebook
decision table** (schema + verified rules + computed gold), and that object is not
specific to skills. `raw text → calibrated train.jsonl` generalizes to *any*
decision-dense prose whose outcomes are pinnable and (ideally) observable.

Where it transfers cleanly (each is the same three-phase machinery, new corpus only):

- **Docs → decision router:** converting policy / SOP / decision manuals
  ("if condition C then step S") into a calibrated, auditable router with p-gates.
- **Config-generator decisions:** a generator script that must pick a *tier /
  quant / profile / branch* from discrete conditions — gold is exact-by-
  construction (this is exactly podman-quadlet's "pick the tuning tier" pattern
  we already compiled).
- **Runtime ops gates:** "retry / rollback / escalate" on a health observation —
  highly *observable*, so Plan C's probe referee is at its strongest here, better
  than skills (real commands, real outcomes).
- **Preference / spec resolution:** turning a written spec into the *resolved*
  setting with a confidence attached (not the value alone).

Per-domain cost when generalizing is small and fixed: (a) a verifier anchor pass
(a human marks a handful of trigger states per rule), and (b) if you want the
Plan-C referee, the outcome must be *observable* by a command/check.

Where it does NOT transfer (the refusal is by design, not laziness):

- **Judgment-prose / creative** ("make it sound natural") — no pinnable answer;
  same `no_gold` refusal as skills.
- **Open-ended generation / numeric regression** — the architecture scores
  *candidate leaves*, so it needs a discrete candidate set and a per-candidate
  Gold distribution to regress against; it is not a text generator.
- **General reasoning** — it answers "how much should I trust candidate X for
  this frozen decision," not "*what* is the answer to this open question."

The honest framing: **Decision-Maker + Gold Compiler is a calibrated-confidence
and auditability layer for any discrete, rule-governed decision — skills were the
first corpus, not the only one.** The thing to scope per new domain is gold, not
the machinery. The live question is which domain has prose worth compiling AND an
observable (or exactly-constructible) outcome — that intersection is where it
pays for itself.

