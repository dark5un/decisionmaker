#!/usr/bin/env bash
# Podman quadlet healthcheck for the decisionmaker serve container: ready only when
# /health returns 200 (model loaded once). Uses python3 because curl/wget are
# NOT in the pytorch/ubuntu image (curl-free image -> curl-based check was always
# exit 127, so podman marked the healthy container "unhealthy").
set -euo pipefail
/opt/conda/bin/python3 -c \
  "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/health',timeout=5).status==200 else 1)"