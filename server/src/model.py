"""Decision head + model (Phase 3). Requires torch.

This is the trainable scoring surface on top of a frozen-injectable backbone.
The plan's engine is a pure function; this module is the tensor half. Imported
lazily by `engine._forward` so the pure engine logic never needs torch.

Architecture (documented in docs/architecture.md §3):
    hidden_at_last_token -> LayerNorm -> Linear(hidden, 1) -> scalar per candidate
    softmax over a question's candidate scalars; Boolean uses logits [0, z].
The head is a plain module so the Phase 5 harness can warm-start it (backbone
frozen) then fine-tune the whole model.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class DecisionHead(nn.Module):
    """Per-candidate scalar scorer: LayerNorm(hidden) -> Linear(hidden, 1).

    Nonzero random init on the scalar weight avoids a dead first step (bf16/scalar
    convention). The optional `set_head='attention'` variant is the later
    cross-choice set-attention (log_k feature); `'none'` is the flat baseline.
    """

    def __init__(self, hidden_size: int, set_head: str = "none"):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.scalar = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.scalar.weight, std=0.02)
        nn.init.zeros_(self.scalar.bias)
        self.set_head = set_head
        if set_head == "attention":
            raise NotImplementedError("set_head='attention' is a Phase 5+ extension; use 'none'")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.scalar(self.norm(hidden))


class DecisionModel(nn.Module):
    """Full model = backbone (injected) + head. Produces per-candidate scalars.

    The backbone is expected to expose `hidden_size` (via `.config.hidden_size`)
    and return `.last_hidden_state` from a call with `input_ids` + `attention_mask`
    (the HF `BaseModelOutput` shape handled in `engine._forward`).
    """

    def __init__(self, backbone: Any, set_head: str = "none"):
        super().__init__()
        self.backbone = backbone
        hidden = int(backbone.config.hidden_size)
        self.head = DecisionHead(hidden, set_head=set_head)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """tokens (P, width), attention (P, width), lengths (P,) -> (P,) scalars."""
        hidden = self.backbone(input_ids=tokens, attention_mask=attention_mask,
                               use_cache=False).last_hidden_state
        leaves = hidden[torch.arange(len(tokens)), lengths - 1]  # last real token
        return self.head(leaves).squeeze(-1).float()