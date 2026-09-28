#!/usr/bin/env python3
"""Load the MLX model once, then serve /v1/decisionmaker -- mirrors
scripts/run_server.py's role for the torch/CUDA path, but native on Apple
Silicon: no CUDA_VISIBLE_DEVICES, no GPU-UUID pinning, no container. MLX talks
to the one Metal GPU on this machine directly (mx.default_device()).

Run from mlx/'s own venv (mlx/pyproject.toml -- no torch anywhere):
    cd mlx && uv run python run_server.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import uvicorn
from mlx_lm import load

MLX_DIR = Path(__file__).resolve().parent
REPO_ROOT = MLX_DIR.parent
for p in (MLX_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from server.src.app import create_app
from server.src.engine import EngineConfig

from model import DecisionHead
from engine import MlxDecisionEngine

DEFAULT_MLX_MODEL_PATH = MLX_DIR / "converted" / "qwen3-0.6b-c1899de"


def build_engine() -> tuple[MlxDecisionEngine, EngineConfig]:
    # Temperature precedence: explicit DECISIONMAKER_TEMPERATURE override >
    # fitted T from temperature.json beside the head checkpoint > 1.0 (no
    # sharpening) -- same precedence as scripts/run_server.py.
    temp_env = os.environ.get("DECISIONMAKER_TEMPERATURE")
    config = EngineConfig(
        max_length=int(os.environ.get("DECISIONMAKER_MAX_LENGTH", "512")),
        temperature=float(temp_env) if temp_env else 1.0,
        model=os.environ.get("DECISIONMAKER_MODEL", "Qwen/Qwen3-0.6B"),
        revision=os.environ.get("DECISIONMAKER_REVISION", "c1899de289a04d12100db370d81485cdf75e47ca"),
    )

    mlx_model_path = os.environ.get("DECISIONMAKER_MLX_MODEL_PATH", str(DEFAULT_MLX_MODEL_PATH))
    if not Path(mlx_model_path).is_dir():
        raise RuntimeError(
            f"no MLX model at {mlx_model_path}; convert it first, e.g.:\n"
            f"  uv run mlx_lm.convert --hf-path Qwen/Qwen3-0.6B "
            f"--mlx-path {mlx_model_path} --dtype bfloat16\n"
            f"(pin the same revision as config.revision -- see mlx/spike_forward.py)"
        )

    full_model, tokenizer = load(mlx_model_path)
    backbone = full_model.model  # inner transformer only, no LM head (matches torch's AutoModel)

    head = DecisionHead(int(backbone.args.hidden_size))

    # Load the fine-tuned head if available. Without this the head is
    # random-init and every /v1/decisionmaker answer collapses to ~uniform
    # probabilities -- same silent-failure shape as the torch path
    # (docs/DEPLOY.md "Troubleshooting GPU confusion"), just MLX weights
    # (.safetensors via nn.Module.save_weights/load_weights) instead of a
    # torch state_dict. Checkpoint format is defined by mlx/train.py (not yet
    # built) -- this path is forward-looking until that exists.
    head_checkpoint = os.environ.get("DECISIONMAKER_HEAD_CHECKPOINT")
    if head_checkpoint and Path(head_checkpoint).is_file():
        head.load_weights(head_checkpoint)
        print(f"head loaded from {head_checkpoint}", flush=True)
        temp_json = Path(head_checkpoint).with_name("temperature.json")
        if not temp_env and temp_json.is_file():
            fitted = float(json.loads(temp_json.read_text())["T"])
            config.temperature = fitted
            print(f"fitted temperature loaded from {temp_json}: T={fitted}", flush=True)
    else:
        print(
            "WARNING: no DECISIONMAKER_HEAD_CHECKPOINT (or file missing); "
            "serving a random-init head (responses will be ~uniform)",
            flush=True,
        )

    return MlxDecisionEngine(tokenizer, config, backbone=backbone, head=head), config


def main() -> None:
    print(f"loading Qwen3-0.6B (MLX, device={mx.default_device()}) ...", flush=True)
    engine, config = build_engine()
    print(f"engine ready (model={config.model}, revision={config.revision}); serving", flush=True)
    app = create_app(engine)
    # Default host is 127.0.0.1 (not 0.0.0.0): this runs as a bare process on
    # the dev machine, not inside a container's isolated network namespace,
    # so there's no reason to default to listening on every interface.
    uvicorn.run(app, host=os.environ.get("DECISIONMAKER_HOST", "127.0.0.1"),
                port=int(os.environ.get("DECISIONMAKER_PORT", "8090")), log_level="info")


if __name__ == "__main__":
    main()
