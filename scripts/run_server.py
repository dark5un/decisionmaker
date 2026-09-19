#!/usr/bin/env python3
"""Load the model once, then serve /v1/decisionmaker.

The GPU is selected by CUDA_VISIBLE_DEVICES=GPU-<uuid> injected by the quadlet
(deploy/decisionmaker-serve.container). This script verifies a real CUDA device is
visible before serving and never fabricates GPU success.

Run inside the container from the repo root: PYTHONPATH=/app (set in the
Containerfile) makes `server.src.*` importable, so all imports sit at module top.
"""
from __future__ import annotations

import os
import json

# Map the role GPU variable (deploy/gpu.env -> DECISIONMAKER_SERVE_GPU_DEVICES) to
# CUDA_VISIBLE_DEVICES BEFORE torch is imported (torch caches CUDA_VISIBLE_DEVICES
# at import). Honors an explicit CUDA_VISIBLE_DEVICES already in the environment
# (bare-container passthrough / overrides).
_gpu = os.environ.get("DECISIONMAKER_SERVE_GPU_DEVICES")
if _gpu:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", _gpu)

import torch
import uvicorn
from transformers import AutoModel, AutoTokenizer

from server.src.app import create_app
from server.src.engine import DecisionEngine, EngineConfig
from server.src.model import DecisionHead


def build_engine():
    # Temperature precedence: explicit DECISIONMAKER_TEMPERATURE override > fitted
    # T from temperature.json beside the head checkpoint > 1.0 (no sharpening).
    temp_env = os.environ.get("DECISIONMAKER_TEMPERATURE")
    config = EngineConfig(
        max_length=int(os.environ.get("DECISIONMAKER_MAX_LENGTH", "512")),
        temperature=float(temp_env) if temp_env else 1.0,
        model=os.environ.get("DECISIONMAKER_MODEL", "Qwen/Qwen3-0.6B"),
        revision=os.environ.get("DECISIONMAKER_REVISION", "c1899de289a04d12100db370d81485cdf75e47ca"),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible; check GPU pinning (AddDevice + CUDA_VISIBLE_DEVICES)")

    tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # FP32 parameter storage; the forward uses bf16 autocast (bf16 convention).
    # transformers>=5 renamed torch_dtype -> dtype (verified against 5.17 in the
    # image; torch_dtype is deprecated). FP32 parameter storage; the forward uses
    # bf16 autocast (bf16 convention).
    backbone = AutoModel.from_pretrained(
        config.model, revision=config.revision,
        attn_implementation="sdpa", dtype=torch.float32,
    )
    backbone.config.use_cache = False
    backbone.to(torch.device("cuda"))
    head = DecisionHead(int(backbone.config.hidden_size))
    # Move the head to the SAME device as the backbone — a CPU head against a
    # CUDA backbone fails during embedding (device mismatch).
    head.to(torch.device("cuda"))
    # Load the FINE-TUNED head (Phase 5) if available. The backbone is byte-frozen
    # (head-only fine-tune) and served fresh from the HF pin above, so ONLY the
    # head weights are loaded. Without this the head is random-init (std=0.02) and
    # every /v1/decisionmaker answer collapses to ~uniform probabilities.
    head_checkpoint = os.environ.get(
        "DECISIONMAKER_HEAD_CHECKPOINT", "/app/runs/seed_ce/checkpoint.pt"
    )
    if os.path.isfile(head_checkpoint):
        ckpt = torch.load(head_checkpoint, weights_only=False)
        head.load_state_dict(ckpt["head"])
        print(f"head loaded from {head_checkpoint}", flush=True)
        # Auto-apply the fitted post-hoc temperature for this head if present.
        temp_json = os.path.join(os.path.dirname(head_checkpoint), "temperature.json")
        if not temp_env and os.path.isfile(temp_json):
            fitted = float(json.load(open(temp_json))["T"])
            config.temperature = fitted
            print(f"fitted temperature loaded from {temp_json}: T={fitted}", flush=True)
    else:
        print(
            f"WARNING: head checkpoint {head_checkpoint} not found; serving "
            f"random-init head (responses will be ~uniform)",
            flush=True,
        )
    return DecisionEngine(tokenizer, config, backbone=backbone, head=head), config


def main() -> None:
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?"
    print(f"loading Qwen3-0.6B on {device} ...", flush=True)
    engine, config = build_engine()
    print(f"engine ready (model={config.model}, revision={config.revision}); serving", flush=True)
    app = create_app(engine)
    uvicorn.run(app, host=os.environ.get("DECISIONMAKER_HOST", "0.0.0.0"),
                port=int(os.environ.get("DECISIONMAKER_PORT", "8090")), log_level="info")


if __name__ == "__main__":
    main()