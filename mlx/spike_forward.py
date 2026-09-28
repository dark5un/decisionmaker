#!/usr/bin/env python3
"""Spike: convert + one packed forward pass on MLX, mirroring server/src/engine._forward.

Not the real engine yet -- proves the seam works before building
mlx/model.py + mlx/forward.py for real:
  1. the pinned Qwen3-0.6B revision converts to MLX and loads,
  2. build_leaf_texts()/encode_leaves() (server/src/engine, torch-free) are
     reusable as-is to build real leaf token sequences,
  3. a packed, right-padded batch through Qwen3Model gives hidden states,
  4. the hidden state at each leaf's last REAL token feeds a fresh
     LayerNorm->Linear(1) head (mlx.nn) -- the same shape as
     server/src/model.DecisionHead -- and produces finite per-candidate scalars.

Run: uv run python spike_forward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from server.src.engine import build_leaf_texts  # torch-free; reused as-is

MLX_MODEL_PATH = Path(__file__).resolve().parent / "converted" / "qwen3-0.6b-c1899de"


def encode_leaves_mlx(tokenizer, leaf_texts: list[str]) -> list[list[int]]:
    """Same contract as engine.encode_leaves: tokenize + append EOS."""
    eos = tokenizer.eos_token_id
    return [tokenizer.encode(text, add_special_tokens=False) + [eos] for text in leaf_texts]


def pack_batch(leaf_tokens: list[list[int]], pad_token_id: int):
    lengths = [len(t) for t in leaf_tokens]
    width = max(lengths)
    tokens = mx.full((len(leaf_tokens), width), pad_token_id, dtype=mx.int32)
    for i, toks in enumerate(leaf_tokens):
        tokens[i, : len(toks)] = mx.array(toks, dtype=mx.int32)
    return tokens, mx.array(lengths, dtype=mx.int32)


def main() -> None:
    print(f"loading MLX model from {MLX_MODEL_PATH} ...")
    model, tokenizer = load(str(MLX_MODEL_PATH))
    hidden_size = model.args.hidden_size
    print(f"loaded: hidden_size={hidden_size}, layers={model.args.num_hidden_layers}")

    # Two questions of different type/length so the batch has real padding to exercise.
    boolean_q = {"type": "boolean", "instructions": "Does this convey urgency?",
                 "criteria": {"true": "Urgent", "false": "Not urgent"}}
    choice_q = {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments, invoicing, refunds",
                             "technical": "Bugs, outages, integrations",
                             "sales": "Pricing, upgrades, new accounts"}}
    state_text = "Customer wrote: 'Our invoice is overdue and we will cancel.'"

    leaf_texts = build_leaf_texts(state_text, boolean_q) + build_leaf_texts(state_text, choice_q)
    print(f"\n{len(leaf_texts)} leaves (1 boolean + 3 choice):")
    for t in leaf_texts:
        print("  ---")
        print("  " + t.replace("\n", "\n  "))

    leaf_tokens = encode_leaves_mlx(tokenizer, leaf_texts)
    print(f"\nleaf lengths: {[len(t) for t in leaf_tokens]}")

    tokens, lengths = pack_batch(leaf_tokens, tokenizer.pad_token_id)
    print(f"packed batch shape: {tokens.shape}")

    hidden = model.model(tokens)  # (P, width, H) -- final-normed hidden states, no LM head
    mx.eval(hidden)
    print(f"hidden state shape: {hidden.shape}")

    last = hidden[mx.arange(tokens.shape[0]), lengths - 1]  # (P, H)
    mx.eval(last)
    print(f"last-token hidden shape: {last.shape}")
    assert bool(mx.all(mx.isfinite(last)).item()), "non-finite hidden states"

    # Fresh (untrained) head, same shape as server/src/model.DecisionHead.
    norm = nn.LayerNorm(hidden_size)
    scalar = nn.Linear(hidden_size, 1)
    scores = scalar(norm(last)).squeeze(-1)
    mx.eval(scores)
    print(f"\nper-leaf scalars (random-init head): {scores.tolist()}")
    assert bool(mx.all(mx.isfinite(scores)).item()), "non-finite head output"

    print("\nSPIKE OK: convert -> load -> reused leaf-building -> packed forward -> "
          "last-token extraction -> head all produced finite, correctly-shaped output.")


if __name__ == "__main__":
    main()
