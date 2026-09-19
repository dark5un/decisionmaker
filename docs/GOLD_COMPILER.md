# Gold Compiler — raw prose → calibrated Decision-Maker training data

Implements plans 05 (`research/plans/05-gold-compiler-plan.md`) and 06
(`research/plans/06-business-history-gold.md`). One tool turns a decision-dense
document (a SKILL.md, a SOP, a policy manual) into **exact-gold, calibrated**
training rows for Decision-Maker, **without a teacher LLM as the source of
truth**: extract → verify (constructive) → synthesize (compute gold).

## The core idea: gold is exact-by-construction

```
 raw prose ──▶ EXTRACT ──▶ VERIFY ──▶ SYNTHESIZE ──▶ train.jsonl
 (SKILL.md)     |           |           |
                └───────────┴───────────┴──▶ PROBE (Plan C referee, read-only)
```

- **EXTRACT** emits *candidate* rules (recall-first — not trusted).
- **VERIFY** is the authority and the anti-complacency wall. Two *constructive*
  checks, **no second model judging**:
  1. *Static evidence binding* — every condition `field` resolves in
     `input_schema`; every outcome is a member of the question's criteria set;
     sources that claim verbatim spans must be substrings of the input.
  2. *Behavioral oracle* — the compiled rule interpreter runs each rule over
     hand-marked **anchor states** and asserts the fired branch matches the
     anchor. Anchors are the only human-in-loop step (a few states per rule).
- **SYNTHESIZE** enumerates the state space from `input_schema`, renders each
  state to prose (pluggable templates, `--k-phrasings`), runs the interpreter to
  **compute** the soft gold distribution, and emits `train.jsonl` — byte-
  deterministic, group-separated, `data_sha256` stamped.
- **PROBE** is the ongoing referee: on partially-observable decisions it runs the
  skill's own commands (read-only) and reports measured vs predicted — a
  discrepancy flags a rule for regen, never silently reconciles.

## CLI

```
scripts/gold_compiler.py --input <SKILL.md|text|corpus> --out <dir>   # full pipeline
scripts/gold_compiler.py extract    <input> [--out dir]
scripts/gold_compiler.py verify     <table> [--anchors] [--source-text X]
scripts/gold_compiler.py synthesize <verified-table> [--out train.jsonl]
scripts/gold_compiler.py probe      <table>
```

Exit taxonomy (the frozen contract):
`0` emitted · `2` refused (judgment-prose → `no_gold`, or no extractor) ·
`3` emitted with human-review flags (some rule incomplete/contradicted, or a
probe discrepancy).

`--refuse-if-judgment` makes the tool refuse (exit 2) on text with no pinnable
answer (tone/vibe/aesthetic/etc. → `{"reason":"judgment-prose","evidence":[...]}`).

Extractor surface: the deterministic path re-derives the corpus rulebook. A
pluggable LLM-backed extractor can be registered (`register_llm_extractor`); the
tool refuses (`no_llm_extractor`, exit 2) when none is present rather than
fabricate.

## Files

- `scripts/gold_compiler.py` — the tool (CLI + pipeline).
- `research/compiler/core.py` — deterministic machinery (validate/verify/
  synthesize/probe).
- `research/compiler/corpus.py` — 10 shortlist tables (from `research/planA/rules.py`)
  + 8 held-out tables authored from real SKILL.md files.
- `research/compiler/anchors.py` — the hand-marked anchors driving the oracle.
- `research/compiler/renderers.py` — pluggable prose renderers.
- `scripts/history_to_gold.py` — **plan 06** example gold-loader: empirical gold
  from a historical outcome log (a different source of gold, plugged into the
  same unchanged harness). `scripts/gen_business_history.py` makes its fixture.
- `scripts/gold_compiler_ci.py` — the three CI gates (`make gold-compiler*`).
- `research/data/gold_compiler/` — committed outputs (force-added).

## Acceptance status (verified on this machine)

| Gate | Status |
|------|--------|
| 10 shortlist + 8 held-out tables schema-valid | PASS (18/18) |
| verify-corpus → all tables `verified` | PASS (18/18) |
| Injected bad field / bad outcome / bad behavior → caught | PASS |
| Synthesize double-run byte-identical | PASS |
| `train.jsonl` loads in unchanged `train.py`, gold sums to 1, no family leak | PASS |
| Probe referee read-only, measured==predicted on observable decisions | PASS |
| plan 06: reproducibility (identical bytes), harness-loadable, `--min-count` refusal, `--treat` confounder guard, time-hold-out test | PASS |
| pytest `tests/test_gold_compiler.py` | PASS (12/12) |
| `make gold-compiler` (lint + verify-corpus + determinism) | PASS |

Training a checkpoint on the emitted rows runs in the serve container, exactly
as any other run:

```
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v /home/px/workspace/decisionmaker:/app:Z \
  -v /home/px/.cache/huggingface:/root/.cache/huggingface:Z --workdir /app \
  localhost/decisionmaker-serve:local python -m train.train \
  --data research/data/gold_compiler/train.jsonl --loss ce --target soft \
  --epochs 30 --gpu GPU-34466eba-2642-c31a-2e44-6451b6a6b425
```

## Honest limits

1. **Judgment-prose refuses** — no extractor conjures calibration for it
   (`no_gold`, exit 2).
2. **Partial observability** — a rule no anchor exercises caps at `incomplete`
   (human review); a degenerate default rule (no condition, e.g. `rule-of-three`
   r0, provably unreachable) is verified by construction when static binding
   passes and no anchor contradicts it. Nothing is silently "verified."
3. **Extractor content is model-dependent** — temp-0 + seed reduce, not remove,
   variance; the verifier normalizes it. Trainer rows are fully reproducible
   once a table exists.
4. **Coverage caps usefulness** — a head stays calibrated only within the schema
   it trained on.
5. **History gold is policy + luck, not causality** — plan 06's empirical gold is
   a conditional outcome *distribution*; on a flawed historical policy the head
   reproduces that policy with high calibration (reported, never hidden). The
   deployment call stays human.