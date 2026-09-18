# Decision-Maker: single-command build/test/verify entrypoints.
# Everything runs read-only; start the service with `make serve-up` first.

SHELL := /bin/bash
PY    := .venv/bin/python
POD   := podman

.PHONY: help test unit contract parity go rust lint bench serve-up serve-down serve-restart \
        seed verify-seed build-image clean

help:
	@echo "targets:"
	@echo "  make test            full gate: unit + contract + parity + go + rust"
	@echo "  make unit            pytest (engine/service/train) + spec validation"
	@echo "  make contract        spec-gate + fixtures"
	@echo "  make parity          cross-language golden parity (needs server up)"
	@echo "  make go              go test (sdk/go)"
	@echo "  make rust            cargo test (sdk/rust, all features)"
	@echo "  make lint            gofmt + go vet + cargo fmt --check + cargo clippy"
	@echo "  make serve-up        start the service container (quadlet)"
	@echo "  make serve-down      stop it"
	@echo "  make seed            regenerate synthetic seed data"
	@echo "  make verify-seed     oracle ECE baseline on the seed"

test: unit parity go rust

unit: contract
	$(PY) -m pytest tests/ -q

contract:
	$(PY) scripts/validate_spec.py

parity: 
	./scripts/parity_golden.py

go:
	cd sdk/go && go test ./...

rust:
	cd sdk/rust && cargo test --all-features

lint:
	cd sdk/go && gofmt -l . && go vet ./...
	cd sdk/rust && cargo fmt --check && cargo clippy --all-features

bench:
	$(PY) scripts/benchmark.py

serve-up:
	$(POD) build -f deploy/Containerfile.serve -t decisionmaker-serve:local . && \
	systemctl --user restart decisionmaker-serve && scripts/poll_health.sh

serve-down:
	systemctl --user stop decisionmaker-serve

serve-restart:
	systemctl --user restart decisionmaker-serve

seed:
	$(PY) train/generate_seed.py --out data/seed.jsonl

verify-seed:
	$(PY) train/verify_seed.py --input data/seed.jsonl

build-image:
	$(POD) build -f deploy/Containerfile.serve -t decisionmaker-serve:local .

clean: serve-down
	cd sdk/rust && cargo clean