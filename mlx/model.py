"""MLX decision head + model -- mirrors server/src/model.py, no torch/CUDA.

Same architecture as the torch DecisionHead/DecisionModel (docs/architecture.md
§3): a plain per-candidate scalar scorer on top of a frozen, injectable
backbone.

    hidden_at_last_token -> LayerNorm -> Linear(hidden, 1) -> scalar per candidate
    softmax over a question's candidate scalars (mlx/forward.py); Boolean uses
    logits [0, z] (same convention as the torch path).

`backbone` here is the INNER transformer only -- e.g. an mlx_lm `Qwen3Model`,
reached via `mlx_lm.load(path).model` -- matching the torch path's `AutoModel`
(base model, no LM head). It exposes `.args.hidden_size` and, called with
token ids, returns final-normed hidden states shaped (P, width, H) (mlx_lm's
Qwen3Model applies its closing RMSNorm before returning -- verified in
mlx/spike_forward.py).
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


class DecisionHead(nn.Module):
    """Per-candidate scalar scorer: LayerNorm(hidden) -> Linear(hidden, 1).

    Nonzero random init on the scalar weight avoids a dead first step, same
    convention as server/src/model.DecisionHead. `set_head='attention'` is
    the same later cross-choice extension (not implemented here either);
    `'none'` is the flat baseline.
    """

    def __init__(self, hidden_size: int, set_head: str = "none"):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.scalar = nn.Linear(hidden_size, 1)
        self.scalar.weight = mx.random.normal((1, hidden_size), scale=0.02)
        self.scalar.bias = mx.zeros((1,))
        self.set_head = set_head
        if set_head == "attention":
            raise NotImplementedError("set_head='attention' is a Phase 5+ extension; use 'none'")

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.scalar(self.norm(hidden))


class DecisionModel(nn.Module):
    """Full model = backbone (injected) + head.

    Training convenience wrapper (one module for mlx.nn.value_and_grad +
    optimizer state), mirroring server/src/model.DecisionModel -- used by a
    future mlx/train.py, not by serving. The serving path (mlx/forward.py)
    calls backbone and head directly instead, matching how
    server/src/engine._forward keeps them separate for inference.
    """

    def __init__(self, backbone: Any, set_head: str = "none"):
        super().__init__()
        self.backbone = backbone
        hidden = int(backbone.args.hidden_size)
        self.head = DecisionHead(hidden, set_head=set_head)

    def __call__(self, tokens: mx.array, lengths: mx.array) -> mx.array:
        """tokens (P, width), lengths (P,) -> (P,) scalars."""
        hidden = self.backbone(tokens)  # (P, width, H); closing RMSNorm already applied
        leaves = hidden[mx.arange(tokens.shape[0]), lengths - 1]  # last real token
        return self.head(leaves).squeeze(-1).astype(mx.float32)
