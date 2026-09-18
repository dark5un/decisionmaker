# Decision-Maker Service — HTTP API

Own contract, not wire-compatible by default (clean semantics; own all clients).
Versioned at `/v1/decisionmaker`; do not break wire compat without a new version.
The OpenAPI document at `spec/openapi.yaml` is the source of truth — this file
is the human reference and the worked examples in `tests/fixtures/`.

Default local endpoint: `http://127.0.0.1:8090` (rootless Podman; see `deploy/`).

## Endpoints

| Method | Path             | Purpose |
|---|---|---|
| POST   | `/v1/decisionmaker`  | Evaluate a state against typed questions |
| GET    | `/health`        | Liveness/readiness: 200 only when the model is loaded once |

## POST /v1/decisionmaker

Request body (`application/json`):

```jsonc
{
  "state": "string | object | array (non-empty)",
  "model": "string, optional (pinned default if omitted)",
  "questions": {
    "<qid>": {
      "type": "boolean | choice | score",
      "instructions": "string | object | array | null, optional",
      "criteria": "type-dependent, see below"
    }
  }
}
```

Validation rules (structural bounds in the spec; runtime rules here):

- `state` must be a non-empty string, or a non-empty object/array.
- `questions` must have at least one entry.
- Each `qid` is a non-empty string. The qid is NEVER sent to the model and is
  never part of inference (qid-invariance — see Phase 3).
- `instructions` is optional on all three types.
- `type` one of: `boolean`, `choice`, `score`.

### The three question types

**Boolean** — yes/no probability. `criteria` optional `{true, false}` descriptions.
```json
{ "type": "boolean", "instructions": "Does this convey urgency?",
  "criteria": { "true": "Explicitly time-sensitive", "false": "No urgency expressed" } }
```

**Choice** — one option from 2..255. `criteria` is a map of option-key →
description (or `null` when the key speaks for itself). Order is preserved and
passed to the model unchanged.
```json
{ "type": "choice", "instructions": "Which team should handle this?",
  "criteria": { "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts" } }
```

**Score** — position on an ordered scale, 2..10 levels LOW to HIGH. `criteria`
is an ordered array. The model infers the ordering from the descriptions only.
```json
{ "type": "score", "instructions": "How frustrated is the customer?",
  "criteria": ["Calm", "Frustrated", "Very angry"] }
```

Structured entries: `instructions`, Choice option descriptions, Score level
entries, and Boolean true/false descriptions may each be a string, object, or array
(label the fields; the model sees JSON rather than a flattened template).

### Response (200)

```jsonc
{
  "model": "decisionmaker-latest",
  "answers": {
    "<qid>": { ... answer, keyed by the same qid ... }
  },
  "usage": { "input_tokens": 312, "output_tokens": 48 }
}
```

Answer shapes (the `type` matches the question):

```jsonc
// Boolean — NO confidence field (a confident no and a confident yes are equally confident)
{ "type": "boolean", "boolean": 0.92 }

// Choice
{ "type": "choice", "choice": "technical",
  "probabilities": { "billing": 0.08, "technical": 0.85, "sales": 0.07 },
  "confidence": 0.82 }

// Score
{ "type": "score", "score": 1.6,
  "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
  "probabilities": { "0": 0.05, "1": 0.3, "2": 0.65 },
  "confidence": 0.78 }
```

- Choice `probabilities`: every option-key → probability, values sum to 1.
- Score `score` = Σ (index × p_index) — fractional, can land between levels.
  Uniform over K levels → score = (K−1)/2.
- Score `legend`: level index (string key) → description.
- Runtime invariants (enforced by the engine, not just the schema):
  probabilities are all finite in [0,1] and sum to 1; `score` for a score answer.

### Confidence — READ THIS
`confidence` (0..1) is how PEAKED the returned distribution is on its winner:
0 = model torn between options, 1 = one option dominates. It is a property of
the returned distribution, NOT "how likely this is right" and NOT a calibration
guarantee. Use it as a GATE: `act` above a high threshold (e.g. 0.8), `review`
above a lower one, `escalate` otherwise. Boolean carries NO confidence.

### Errors (typed, never a generic 500)

```jsonc
{ "error": { "type": "validation | internal | rate_limit | overloaded | not_ready",
             "message": "...", "details": { ...field-level pointers on 422... } } }
```

| Status | type | Meaning |
|---|---|---|
| 400 | validation | Unparseable body (malformed JSON). |
| 422 | validation | Request failed contract validation; `details` names the field. |
| 429 | rate_limit | Rate limited; retry with exponential backoff. |
| 500 | internal | Engine produced an invalid result (non-finite logits, probs not summing to 1). Distinguish from a client-side 422. |
| 503 | not_ready | Model not loaded yet; re-check `/health`. |

## GET /health

200 when the model is loaded once and warm:
```jsonc
{ "status": "ready", "model": "qwen3-0.6b...", "uptime_seconds": 120 }
```
503 (`not_ready`) before the model finishes loading.

## Worked example

The `tests/fixtures/domain_shape.request.json` payload (a request in the exact
shape the public docs show) is accepted by this contract and its counterpart
`tests/fixtures/domain_shape.response.json` validates against the 200 schema.