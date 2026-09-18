#!/usr/bin/env bash
# Podman quadlet healthcheck for the Decision-Maker serve container: ready only when
# /health returns 200 (model loaded once). Must be executable in the container.
set -euo pipefail
curl -fsS -o /dev/null http://127.0.0.1:8090/health