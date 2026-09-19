#!/usr/bin/env bash
# Phase B retrains on the 5090 (serve keeps the 4070 Ti). Sequential: both share the 5090.
# NOTE: train.py returns EXIT 2 for a failed ECE gate (a legitimate "arm didn't meet
# calibration" signal, NOT a crash). So the pipeline must record-and-continue on code 2,
# and only abort on a real error (e.g. code 1/126/137). Using bare `set -e` or aborting
# on non-zero would silently skip the remaining arms after a gate-fail.
set -uo pipefail
cd /home/px/workspace/decisionmaker
GPU=GPU-34466eba-2642-c31a-2e44-6451b6a6b425
IMG=localhost/decisionmaker-serve:local
RUN=(podman run --rm --device nvidia.com/gpu=all --security-opt label=disable
     -v /home/px/workspace/decisionmaker:/app:Z
     -v /home/px/.cache/huggingface:/root/.cache/huggingface:Z --workdir /app)

run_arm() {  # $1=label $2...=args
  local label="$1"; shift
  echo "=== [$label] $* ==="
  "${RUN[@]}" "$IMG" python -m train.train "$@" --gpu "$GPU"
  local rc=$?
  if [ "$rc" = "0" ]; then echo "=== [$label] OK (gate PASS) ==="
  elif [ "$rc" = "2" ]; then echo "=== [$label] ECE gate FAIL (recorded; continuing) ==="
  else echo "=== [$label] ERROR rc=$rc (aborting sweep) ==="; exit "$rc"; fi
}

run_arm B1-hard     --data data/seed.jsonl --run-dir runs/seed_hard   --loss ce --target hard --epochs 3
run_arm B2-soft30   --data data/seed.jsonl --run-dir runs/seed_soft30 --loss ce --target soft --epochs 30
echo "=== ALL PHASE B ARMS DONE ==="