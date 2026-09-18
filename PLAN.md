# Decision-Maker Service & Framework — Remaining work

This repo is fully built and proven (see `docs/AGENTS.md` for the verified status of
every phase). Only the items below are unimplemented. Each one was left as an explicit
open gate by the build; nothing here is speculative.

## Open gates (in order)

1. **Phase 9 tail — resume-anywhere.** Confirm a fresh agent can pick this repo up from
   `docs/AGENTS.md` alone: `make test` exit 0 plus a working train handoff (start
   `decisionmaker-train` and the ECE gate reports 0 pass / 2 fail).

2. **Phase 10 — packaging fork (the single Go binary).** Decide between:
   - **Path A** — head-based calibrated distributions, Go CLI + model SIDECAR (current
     engine). One forward, no token decode.
   - **Path B** — generative + constrained decoding (llama.cpp/GGUF embedded in a Go
     binary via `go:embed`).
   Deciding spike: build a throwaway llama.cpp + Go bundle with constrained decoding
   against a small instruct model on a handful of hand-labeled domain intents; measure
   its ECE/accuracy on the SAME eval set as the head-based engine. If constrained
   decoding is within a small epsilon → Path B wins; if the head is materially better
   (expected) → Path A wins. Both share `spec/`, the SDKs, and `/v1/decisionmaker`.

3. **Phase 10 — CLI UX** (whichever fork wins): `decisionmaker eval` (read
   question-blocks from stdin/file as JSON or YAML, print structured outcome,
   `--format json|yaml`), `decisionmaker serve [--host --port --model]` (daemon),
   `decisionmaker list-models`, `decisionmaker version`. Typed errors matching the spec.

4. **Optional, low priority** — soak test (long-run stability) and per-request
   memory / GPU-util measurement.

## Note
`docs/AGENTS.md` is the authoritative resume-anywhere handoff and the place to track
new status.