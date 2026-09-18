# AGENTS.md — resume-anywhere handoff (written 2026-09-18)

Read PLAN.md too. This file is the authoritative "where we are / what's next"
for a fresh agent. Everything here was VERIFIED by running it (not assumed).

## Quick start (5 min)
```bash
cd "$HOME/workspace/decisionmaker"
.venv/bin/python -m pytest tests/ -q        # expect ~52 passed, 1 torch-gated skip
curl -s http://127.0.0.1:8090/health         # expect {"status":"ready",...} (service is UP)
```
The dev venv is `.venv` (managed by uv; `uv.lock` pins it). Source layout:
`server/src/engine.py` (pure core + DecisionEngine), `server/src/model.py` (torch
DecisionHead/Model), `server/src/app.py` (FastAPI), `spec/openapi.yaml`,
`train/`, `sdk/go/`, `deploy/`, `tests/`, `scripts/`, `docs/`.

## THE MODEL + GPU (verified; do NOT assume the plan's words)
- Model: `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`
  (verified via HF: that sha IS the repo's main commit; `-Instruct` is GATED/401,
  `-Base` lacks the rev). Loaded with `dtype=torch.float32` (transformers 5.17
  renamed `torch_dtype`→`dtype`), `attn_implementation="sdpa"`, bf16 autocast forward.
- GPU: decisionmaker is pinned to the RTX 4070 Ti `GPU-58e5f98d-a4f4-4cff-a9fc-16df0667d2fd`.
  The RTX 5090 is NOT free — llama-server holds ~26.6 GiB of its 32 GiB (98%).
  Re-verify live with `scripts/verify_gpu.sh --serve` before any GPU work.
  fp32 0.6B ≈ 2.4 GiB; the serve container uses ~2.5-2.9 GiB on the 4070 Ti.
- Deploy (already RUNNING): image `localhost/decisionmaker-serve:local` (built from
  `deploy/Containerfile.serve`, base `docker.io/pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime`
  — required for sm_120/Blackwell), quadlet installed at
  `~/.config/containers/systemd/decisionmaker-serve.container` (with `decisionmaker.network`).
  Manage: `systemctl --user restart|stop decisionmaker-serve`; logs `journalctl --user -u decisionmaker-serve -f`.
  The repo is mounted at `/app:ro`, so code edits apply after a `systemctl --user restart`.
- Quadlet gotchas already fixed: DO NOT use `UserNS=keep-id` with a writable HF
  cache (it broke a PermissionError writing the cache token file); tensors must go
  on the BACKBONE device and the head must be `.to(cuda)` (device mismatch); the
  first GPU request incurs ~14s CUDA warmup, steady-state ~25ms.

## Phase-by-phase status (see PLAN.md checklist)
- **Phase 1 research** — DONE. `docs/architecture.md` (primitives, calibration theory,
  container GPU, SDK decision).
- **Phase 2 contract** — DONE. `spec/openapi.yaml` (3.1), `docs/API.md`,
  `tests/fixtures/` (valid/invalid/response), `scripts/validate_spec.py` → 13/13 green.
- **Phase 3 engine** — DONE. Pure torch-free core + torch head; qid-invariance test.
  Structured entries (instructions/score levels/choice descs/boolean criteria) are
  serialized to compact JSON in leaves; engine validation == spec.
- **Phase 4 service** — DONE (LIVE). /health 200; every fixture sums-to-1; 7 invalid→422;
  malformed→400; warm ~24ms. Service is UP at 127.0.0.1:8090.
- **Phase 5 train/calibration** — DONE (ran live on the RTX 5090). Head-only fine-tune
  (backbone frozen + bf16; fp32 full-graph OOM'd ~8 GiB — fix in `train/train.py`). Three
  arms on `data/seed.jsonl`: CE (0.0261/0.0424), Brier (0.0002/0.0147), paired (0.0436/0.0247),
  ALL ECE-gate PASS. Brier strongest; paired does NOT beat CE on test (honest). See
  `docs/CALIBRATION.md`. Real run artifacts in `runs/seed_{ce,brier,paired}/` (gitignored).
  `train/train.py` gets `--loss {ce|brier|paired}` and `--run-dir`; rerun via the quadlet.
- **Phase 6 Go SDK** — DONE. `sdk/go/{decisionmaker,client,decision}.go` real typed client
  (Choice/Boolean/Score request/response, probabilistic maps, typed spec errors with
  `errors.Is`, context transport, `Answer.Decide(threshold)` gate). `probe.go` removed.
  `cmd/cli/main.go` example CLI (parity producer). `go test` green 9/9 incl. LIVE
  Go==curl parity + malformed->422. gofmt clean.
- **Phase 7 Rust SDK** — DONE. `sdk/rust/src/{lib,blocking}.rs` thin typed client,
  `serde` tagged enums, async reqwest+tokio (default) / `blocking` feature-gated,
  typed `ApiError`, `Answer::decide` gate. `examples/cli.rs`. `cargo test --all-features`
  11/11 incl. live parity; binary demonstrates Go/Rust `HashMap`-sorted-key encoding ==
  canonical wire (see below). fmt/clippy clean.
- **Phase 8 cross-language parity** — DONE. `Makefile` (unit/contract/parity/go/rust/
  lint/bench); `scripts/parity_golden.py` feeds the SAME canonicalized fixture bytes to
  curl/Go/Rust and asserts identical normalized distributions + sums-to-1; 3/3 green.
  `make test` exit 0 (spec 13/13, pytest 51+1skip, Go, Rust, parity).
  KEY WIRE INVARIANT: SDK encoders sort object keys (Go/Rust std maps); curl sends raw
  bytes — so parity canonicalizes fixtures to sorted-key form. Do NOT add a fixture with
  unsorted structured-object state and expect SDK==curl byte-for-byte.
- **Benchmark** — DONE (real). `scripts/benchmark.py` + `make bench`: warm p50=23.6ms
  p95=25.1ms mean=23.9ms; ~43 req/s @8-concurrency; correctness gate OK (live 4070 Ti).
- **Phase 9 docs** — DONE (arch/API/DATA/DEPLOY/AGENTS). Plan checklist updated.
- **Phase 10 packaging fork** — not started. Spike: constrained-decode llama.cpp+Go vs
  head-based ECE on the same eval set → decide Path A (sidecar) vs Path B (embedded GGUF).

## Next-agent TODO (in order)
1. (Phase 10) The packaging spike — constrained-decode llama.cpp+Go vs head-based ECE on the
   same eval set; decide Path A (sidecar) vs Path B (embedded GGUF).
2. (Phase 9 tail) resume-anywhere run: `make test`; the calibration/pattern skill update
   (offer to user).

## Commands that matter
- Tests: `.venv/bin/python -m pytest tests/ -q`
- Spec gate: `.venv/bin/python scripts/validate_spec.py`
- Regenerate seed: `.venv/bin/python train/generate_seed.py --out data/seed.jsonl`
- Oracle ECE baseline: `PYTHONPATH=. .venv/bin/python -m train.verify_seed --input data/seed.jsonl`
- Rebuild image (only if Containerfile changed): `podman build -f deploy/Containerfile.serve -t decisionmaker-serve:local .`
- GPU truth: `scripts/verify_gpu.sh --serve`
- Live check: `curl -s -X POST http://127.0.0.1:8090/v1/decisionmaker -H 'Content-Type: application/json' -d @tests/fixtures/domain_shape.request.json`