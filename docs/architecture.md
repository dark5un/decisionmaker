# Decision-Maker Service — Architecture notes (Phase 1)

Status: Phase 1 research baseline. Source-verified 2026-09-18. This is the
record the rest of the build is grounded in. README/docs elsewhere are treated
as hypotheses; each fact below was confirmed by reading source (primary docs or
the implementation) in this session.

Use case / domain      : Pluggable. Build starts with a synthetic + hand-labeled
                         seed set; the data generator is the moat, not the build.
Deployment             : Single-user local, rootless Podman on this box.
API contract           : OUR OWN contract with clean, production-ready semantics
                         (clean semantics; own all clients). Drop-in compat is a
                         one-flag change if ever wanted.
Engine                 : Candidate-leaf scorer on Qwen3-0.6B (instruct), written
                         from scratch, start-from-pretrained.
Calibration            : Proper-scoring objective (CE baseline + Brier + RLCD-style
                         paired proper-reward) + ECE harness.
Packaging (open gate)  : Path A (head-based, sidecar) vs Path B (generative GGUF).
                         Decided empirically in Phase 10a, not by taste.
Container               : Podman Quadlet; `decisionmaker-serve` + `decisionmaker-train`.
SDK strategy           : Hand-written thin clients (this spec is ~200 lines; the
                         plan says decide by spec size — codegen noise isn't worth it).

---

## 1. The design (what is ours)

Decision-Maker is an engine that evaluates {state, questions} and
returns calibrated per-candidate probability distributions in ONE forward pass,
served over HTTP, with typed clients. Three separable planes, built in this order:

1. **Engine** — the decision core (candidate-leaf scorer on Qwen3-0.6B).
2. **Service** — containerized /v1/decisionmaker, load-once, warm.
3. **Framework** — the wire contract + calibration + data. This is our product and IP.

The engine is a candidate-leaf scorer with proper-scoring calibration, written from scratch.

### 1.1 The three improvement forks (architecture decisions, not todos)

- **Fork F1 — engine encoding** (Phase 10): head-based calibrated distributions
  (PyTorch, one forward, no decode) vs generative + constrained decoding
  (llama.cpp/GGUF, embeddable Go binary). Decided by the Phase 10a spike.
- **Fork F2 — calibration objective** (Phase 5): plain CE (accuracy) vs direct
  Brier vs RLCD-style paired proper-reward PG (calibration). CE and Brier both
  recover the true distribution by proper-scoring theory; the paired PG is a
  stochastic-gradient estimator of expected Brier. Honest expectation: it MAY NOT
  beat CE — that is a research outcome, not a bug.
- **Fork F3 — confidence semantics**: spread-based (spread of the returned distribution) vs
  calibrated. Decision: spread-based on the wire; calibration is our internal
  R&D concern, not a contract field.

---

## 2. The primitives — verified contract facts

The primitives below define the contract our wire format implements.

### Endpoint and request
- `POST /v1/decisionmaker` (Bearer auth on the real API; ours is single-user local).
- Request body: `{ state, model, questions }`.
  - `state`: string, OR structured (object/array) — e.g. chat logs, records, app state.
  - `model`: which model (we pin Qwen3-0.6B instruct, revision `c1899de289a04d12100db370d81485cdf75e47ca`).
  - `questions`: map `{ qid: {type, instructions, criteria} }`. The qid is a
    key WE choose; answers come back under the same key; the qid is NEVER sent
    to the model and is never part of inference (qid invariance).
- `instructions` is optional on ALL three types. When omitted, the key is left
  out of the request rather than sent as `null`.
- `instructions` / Choice option descriptions / Score level entries / Boolean
  true-false descriptions may each be a string, object, array, or null
  (structured entries). We serialize whatever shape is given.

### The three question types
| Type | Asks for | Answer | Bounds |
|---|---|---|---|
| Boolean  | yes/no, on a question or a statement | probability of yes, a single float 0..1; NO separate confidence on the wire | needs no criteria |
| Choice | one option from a set you define | the option, a probability per option (sums to 1), confidence | 2..255 options |
| Score  | a position on an ordered scale | a fractional score (Σ i·pᵢ, can land between levels), probability per level (sums to 1), confidence, legend | 2..10 levels, low→high |

- Boolean `criteria` is a `{true:…, false:…}` map (what a yes / no means). Optional.
- Choice `criteria` is a map of option-key → description; option ORDER is
  preserved (our JSON object must keep insertion order — important for the model
  input, which is order-sensitive).
- Score `criteria` is an ordered ARRAY of level descriptions, low to high.
- A Choice should get an `other` option when the set may not cover every input.

### Response (verified from api.md)
```jsonc
// Boolean
{ "type": "boolean", "boolean": 0.92 }                       // no confidence field
// Choice
{ "type": "choice", "choice": "technical",
  "probabilities": { "billing": 0.08, "technical": 0.85, "sales": 0.07 },
  "confidence": 0.82 }
// Score
{ "type": "score", "score": 1.6,
  "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
  "probabilities": { "0": 0.05, "1": 0.3, "2": 0.65 },
  "confidence": 0.78 }
// top level
{ "model": "decisionmaker-latest", "answers": { "<qid>": <answer>, ... },
  "usage": { "input_tokens": 312, "output_tokens": 48 } }
```
- Score `score` = the probability-weighted answer = Σ (index · p_index);
  `legend` maps each level index (string key) to its description.

### Confidence semantics (critical, F3)
- `confidence` (0..1) = how PEAKED the probability distribution is on the winner
  (0 = model torn between options, 1 = one option dominates). It is a property of
  the returned distribution, NOT "how likely this is right" and NOT a calibration
  guarantee. It is used as a GATE: `act` above a high threshold, `review` above a
  lower one, `escalate` otherwise.
- Boolean has NO wire confidence. A confident "no" and confident "yes" are equally
  confident. (`max(boolean, 1-boolean)` stays ≥ 0.5, so `escalate` remains reachable.)
  Decision F3: our wire contract keeps Boolean confidence-free.

### Error model
| Status | Meaning |
|---|---|
| 401 | missing/invalid API key (ours: single-user, likely unused) |
| 422 | request body failed validation; body details the offending field |
| 429 | rate limit |
| 529 | overloaded |
Client SDKs retry 429/529 with exponential backoff. We add: failures are TYPED
(not a generic 500), and our engine's own malformed-output errors (non-finite
logits, probs not summing to 1) become typed 500s distinguishing them from
validation 422s.

### Local-policy bounds
- Choice options: lib enforces 1..255 (spec unbounded; live API 400s at 256).
- Score levels: lib enforces 2..10 (spec minItems:1; live API 400s at 11).
- questions per request: min 1 (spec `minProperties: 1`).
OUR Phase 2 spec adopts: Choice 2..255, Score 2..10 (the plan's stated bounds),
Boolean criteria {true,false} only.

---

## 3. The candidate-leaf engine

All candidate leaf paths (N states × M questions × K candidates) are packed
into one batch and passed through a single backbone forward. This is the
"solved part" of Phase 3.

### Leaf construction
```
State:\n{state}\n
Question type: {type}\nQuestion:\n{instructions}\n
  [+ for boolean with criteria: "False criterion: {..}\n" / "True criterion: {..}\n"]
Candidate:\n{candidate_text}\nDecision:
+ EOS token
```
- state is serialized as-is (string, or JSON for objects/arrays).
- **Boolean/Boolean** has exactly two leaves, ids `["false","true"]`, texts
  `["The proposition is true."]` (only the one semantic path; the "false" path is
  the degenerate/empty candidate). Model emits a scalar z; logits are `[0, z]`
  (i.e. P(false)=sigmoid(−z), P(true)=sigmoid(z)); softmax over the pair.
- **Choice** leaves: one per option, text `"{key}: {description}"`, in the
  criteria order the caller provided. Never inject an ordinal index or adjacent
  level text.
- **Score** leaves: one per level, text = the level description, in order
  (low→high). "Never inject ordinal index or adjacent levels" — so the model must
  infer the ordering from the given descriptions only.
- No silent truncation: if the longest leaf exceeds `max_length`, raise.

### One forward pass
- ALL candidate leaf paths (N states × M questions × K candidates) are packed
  into a single batch, right-padded to the longest leaf with the pad token.
- One backbone forward (no `use_cache`, sdpa attention). Take the hidden state at
  each leaf's LAST real token (`lengths - 1`). This is the "ONE forward, no
  decode" property.
- Decision head: `LayerNorm(hidden) → Linear(hidden, 1)` → scalar per candidate.
- Softmax over a question's K candidate scalars (masking padding with −1e9).
- **Score** distribution → expected index `score = Σ i·pᵢ` (fractional OK).
- **Boolean/Boolean** = sigmoid over the pair's `[0, z]` logits → P(true).
- Optional cross-choice set-attention for Choice (a `log_k` feature + multihead
  over the choice's candidates, residual-projected from zero). Added later; the
  `set_head=none` flat scorer is the baseline.
- Temperature applied to logits BEFORE softmax at inference (default 1.0; an
  explicit scalar, does not by itself mean calibrated).

### Training (head + optional full-model)
- Warm start head-only (backbone frozen, head LR 1e-3) then full-model (body
  2e-5, head 2e-4, weight-decay 0.01, grad-norm clip 1.0, AdamW).
- Target: softmax distribution (CE on gold_distribution, sum-to-1) OR hard
  one-hot gold. `loss = −Σ target·log_softmax(logits)`.
- bf16 autocast forward, FP32 parameter/optimizer storage. Fits this box's GPUs
  (0.6B bf16 ≈ 2 GB weights, ≈5 GB with optimizer).
- Baseline control: frozen pretrained language-model next-token logits over the
  candidate labels (the "native baseline") — establishes the untrained floor.

---

## 4. Calibration — public math, verified

Sources: Gneiting & Raftery 2007 "Strictly Proper Scoring Rules, Prediction, and
Estimation" (public math); local `docs/RLCD_EXPERIMENT.md` (read in full
this session).

- Proper-scoring theory: minimizing expected CE or expected Brier over the true
  conditional distribution recovers that distribution. CE gives accuracy, not a
  habitual calibration guarantee; the calibration HARNESS is the R&D-relevant part.
- RLCD-style paired proper-reward: draw M≥2 predictions with replacement,
  reward = proper function of (hit count, agreement penalty) whose expectation is
  `||q||² − ||p−q||²`; its gradient equals the expected-Brier gradient. It is a STOCHASTIC
  gradient estimator of expected Brier — public math, NOT a proprietary
  algorithm, and we never claim it as one.
- Evidence (the CPU benchmark + one Qwen event run): direct Brier and the
  paired PG both learn useful probabilities; the paired arm did NOT clearly beat
  direct Brier across all metrics (one seed, one family). The honest engineering
  conclusion, which Phase 5 follows: keep **CE and direct Brier as first-class
  controls**, add the paired-PG arm as the experiment, and prefer exact Brier for
  small candidate sets (avoid sampling variance). Report honestly if the paired
  arm doesn't win.
- ECE: **ten fixed bins** over p(true) (Boolean-style),
  NOT top-label ECE, NOT a calibration guarantee. We adopt the same 10-bin
  convention, report a reliability diagram, ECE, NLL, and Brier per split.
  Gate: a "calibrated" model has ~90% of its 0.9-predictions correct.

---

## 5. Container + GPU wiring — verified live on this box

- Rootless Podman 6.1.1, `nvidia-container-toolkit` + `nvidia-ctk` present,
  CDI spec at `/etc/cdi/nvidia.yaml`.
- Existing quadlet convention under `~/.config/containers/systemd/`
  (e.g. `llama-cpp-main.container`) — the pattern Phase 4 follows verbatim:
  - `AddDevice=nvidia.com/gpu=all`
  - `Environment=CUDA_VISIBLE_DEVICES=GPU-<uuid>`
  - `SecurityLabelDisable=true`, `UserNS=keep-id`, shared `ai.network`,
    `PublishPort=127.0.0.1:<host>:<container>`, read-only model volume.
- Live GPU truth (from `/proc/driver/nvidia/gpus/*/information`, this session):
  - RTX 5090, minor 1, UUID `GPU-34466eba-2642-c31a-2e44-6451b6a6b425`.
    NOT idle: llama-server holds ~26.6 GiB of its 32 GiB on this box.
  - RTX 4070 Ti, minor 0, UUID `GPU-58e5f98d-a4f4-4cff-a9fc-16df0667d2fd` — drives
    Hyprland but has ~8 GiB free; decisionmaker-serve is pinned here (0.6B fp32 ≈ 2.5 GiB).
  - The plan's "prefer the 5090 (idle)" premise was WRONG on 2026-09-18: choosing a GPU
    must be a LIVE decision (nvidia-smi + /proc + available free memory), not an assumption.
  - Minors are reassigned across reboots: NEVER trust a cached/CDI mapping;
    always re-read `/proc/driver/nvidia/gpus/*/information` and pin by UUID.
- Container flag to verify GPU before claiming success: `--list-devices` or an
  equivalent device probe in the app, not a trust-the-setup claim.

## 6. SDK toolchains — verified present
- Go 1.27.1 (mise), cargo 1.98.1, uv 0.12.14. Runtime torch NOT yet installed in
  the 3.14 interpreter (Phase 4/5 handoff provides it via venv/container).

---

## 7. What this means for the build order (restated decisions)
1. Phase 2 contract first — everything depends on it; small spec (~200 lines) so
   thin hand-written clients in Phases 6/7, no codegen.
2. Phase 3 engine exactly as section 3 (candidate-leaf + one forward + head).
3. Phase 4 serve on the 5090, pinned by UUID (section 5).
4. Phase 5 calibrate with CE + direct Brier controls + paired-PG arm, 10-bin ECE
   (section 4).
5. Phases 6–8 thin Go/Rust SDKs + cross-language parity.
6. Phase 10 spike decides Path A vs Path B packaging empirically.