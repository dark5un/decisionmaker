# Decision-Maker ← Hermes Skills: The Bake-Off

Is decision data for Hermes skills something you **compile** (Plan A), **program**
(Plan B), or **measure** (Plan C)? This is the controlled comparison of the three
gold sources on the shared 3-skill shortlist
(`podman-quadlet-deploy`, `systematic-debugging`, `ryoku-desktop-ops` — the NVIDIA
pinning guidance lives inside podman-quadlet, as the plan README intended).

All numbers below are from real runs on the pinned Qwen3-0.6B backbone, the
proven `seed_soft30` recipe (`--loss ce --target soft --epochs 30`), T=1.0, and
the SAME `scripts/eval_head.py` on the SAME held-out set. No placeholders.

## Controls

| Control              | Value |
|----------------------|-------|
| Harness              | `train/train.py` (identical for all) |
| Recipe               | loss=ce, target=soft, epochs=30 |
| Temperature          | T=1.0 (forced; fitted T undefined/meaningless where calibration split is choice-only) |
| Shared held-out eval | `scripts/eval_head.py` on `holdout_questions.jsonl` (7 questions) |
| Human gold           | `human_gold.jsonl` (7 verdicts, held-out skills) |
| Training shortlist   | podman-quadlet-deploy, systematic-debugging, ryoku-desktop-ops |
| Hold-out skills      | verify-install-readiness, bluetooth-pairing (unseen) |

### Effective trained-N per model

| Model      | train rows | total rows | notes |
|------------|-----------|------------|-------|
| Plan A     | 32        | 98         | compiled rule tables × prose phrasings |
| Plan B     | 120       | 960        | generator; +480 OOD prior-shift |
| Plan B (matched) | **32** | 872  | B subsampled to A's train size for an apples-to-apples A↔B comparison |
| Plan C     | —         | 5          | measured runtime outcomes; **too small to train a clean head** → treated as gold-reference, not a model |

Because Plan C yielded only 5 measured rows (the sparse plan's honest expectation),
it is reported per the plan rule: **"insufficient sample to train cleanly"** — Plan C
is the highest-trust calibration reference, not a third trained head.

## 1. Internal check (each head on its own split)

| Model | test boolean ECE | ood boolean ECE | gate? |
|-------|------------------|-----------------|-------|
| Plan A        | 0.139 | 0.145 | PASS |
| Plan B (full) | 0.173 | 0.484 | test PASS; ood degrades as designed |
| Plan B (matched) | 0.044 | 0.228 | test PASS; ood near/sharp degrade |

Plan B's OOD ECE jumping to 0.48 is the **expected, correct** signal: its OOD
families deliberately shift the field priors out of the training range, so
calibration must degrade there. That is the proof the OOD machinery works (Plan B
§B4), and per the decisionmaker skill it is recorded-and-continued, not a crash.

## 2. Shared held-out gate (the real test — fully held-out skills)

`eval_head.py` on `holdout_questions.jsonl`; ECE is vs the human-judged gold.

| Model | boolean ECE (n=2) | choice mean-conf / peak | choice gap (ceiling .916) | boolean mean-conf |
|-------|-------------------|-------------------------|---------------------------|-------------------|
| Plan A        | **0.387** | 0.346 / 0.346 | 0.570 | 0.613 |
| Plan B (full) | 0.570     | 0.409 / 0.409 | 0.507 | 0.570 |
| Plan B (matched) | 0.622  | 0.341 / 0.341 | 0.575 | 0.622 |

Both concrete heads are **above the 0.2 ECE gate** on the fully-held-out skills.
That is the honest finding: at the 3-skill scale neither compile nor program
retains calibration on skills it never saw. Plan A edges out Plan B on the
human-anchored boolean ECE (0.39 vs 0.57/0.62), but n=2 boolean held-out
questions makes this _directional_, not a precision measurement.

## 3. Cross-generalization matrix (boolean ECE; rows = scored model, cols = eval states)

| states → model | Plan A | Plan B (full) | Plan B (matched) |
|----------------|--------|---------------|------------------|
| **A states** (train 32r) | 0.139 (own test) | 0.025 | 0.161 |
| **B states** (train 120r→32r matched) | 0.090 | 0.173 / 0.044 | 0.044 |

Reading: the off-diagonal cells (A on B's states = 0.090; B on A's states =
0.025/0.161) are small → Plan A and Plan B heads **agree strongly on each other's
in-family states**. The methods are substantially **interchangeable** for training
on this corpus (converging manifolds), as the plan's "mostly-agree" branch predicts.

**Caveat disclosed honestly:** Plan A's rules and Plan B's generator are two
encodings of the *same* semantic gold (B's program computes the same winners A's
rules select, just with prior-shifted sampling and variable confidence). So the
A↔B agreement is partly by construction — this bake-off compares two faithful
encodings of one compiled truth plus one independent measured truth (C), not three
fully independent truths. The only *independent* truth is Plan C, and it is sparse.

## 4. Human-arbitrated verdict + Plan C gold-reference

`human_gold.jsonl` is the only truth beyond the synthetic truths. Plan A scores it:
boolean ECE 0.387, gap-to-ceiling 0.287 → **human gold sides with Plan A (compile)**
over both Plan B arms on the held-out boolean questions.

Both trained heads vs Plan C's measured states (5 rows; n=1 boolean each):
Plan A boolean ECE 0.642, Plan B 0.510 — n=1, so only directional; neither is
calibrated on the measured states at this size, and no clean arbitration is
possible from 5 samples.

## Conclusion

**Verdict: compile and program are complementary-but-convergent; measurement is the
gold, but too sparse here to stand alone.**

- The A↔B cross-matrix agrees (off-diagonal ECE 0.025–0.16), so **compile (A) and
  program (B) are interchangeable for training on this corpus** → pick by cost:
  **Plan A (compile) is cheapest** — no generator to write, smallest while
  sufficient, and it wins the human-anchored held-out boolean ECE (0.39 vs 0.57+).
- Plan B's real differentiator is **volume + OOD machinery**: it trains a sharper
  in-family test head (0.04 vs A's 0.14 when size-matched) and gives calibration
  degradation as a diagnostic. Its cost is the generator + program review.
- **Plan C (measure) is the ongoing gold standard, not a train set** — at 5
  observable decisions per this shortlist it is too sparse to train, exactly the
  plan's predicted outcome. Its role is to *validate* an A/B dataset, not to be one.
- **Recommended merge:** use **Plan A/compile as the training data** (cheapest,
  human-gold-preferred), hold **Plan C's measured outcomes as the verification
  gate** for A/B data, and note that **neither head is safely deployable at this
  scale** — both fail the 0.2 gate on fully held-out skills. The honest answer to
  "compile, program, or measure?" is **compile for coverage, measure for truth,
  program for volume** — and only scale, don't crown, at this sample size.

The decide-what-feeds-the-served-head is a **human call**; nothing in this bake-off
justifies swapping `DECISIONMAKER_HEAD_CHECKPOINT` away from the validated
`seed_soft30` head yet.

## Reproducibility

- Plan A: `research/planA/rules.py` (compiled tables) + `synthesize_planA.py`
  (deterministic; sha `f5fc…c314`).
- Plan B: `research/generators/planB_generator.py` (deterministic; double-run
  byte-identical sha `c700…c80`). Matched run: `planB_train_matched.jsonl`.
- Plan C: `scripts/probe_decisions.py` (read-only probes, provenance log in
  `research/plans/planC_probe_log.jsonl`; sha `053d…c7b`).
- Runs: `runs/planA`, `runs/planB`, `runs/planB_matched` (checkpoints +
  predictions + ECE reports). Plan C has no checkpoint (not trained).
