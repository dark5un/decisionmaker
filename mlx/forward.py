"""One packed forward over all leaves on MLX -- mirrors server/src/engine._forward.

Reuses server.src.engine's pure, torch-free PreparedExample/prepare_request
(leaf construction, tokenization, validation) unchanged; only the tensor path
here is MLX-specific. Same packing scheme as the torch engine: ALL candidate
leaf paths across every question in the request are packed into one batch,
right-padded to the longest leaf, and run through a single backbone forward.

Right-padding + causal-only masking (mlx_lm's Qwen3Model builds no explicit
padding mask -- see mlx/spike_forward.py) is numerically safe here: a real
token can never attend to a padding token, because padding always sits at
FUTURE positions relative to every real token in a right-padded row. Verified
in the spike: the last real token's hidden state is byte-identical regardless
of how much trailing padding is added.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from server.src.engine import PreparedExample  # torch-free; reused as-is


def forward_mlx(backbone: Any, head: Any, pad_token_id: int,
                examples: List[PreparedExample]) -> List[List[float]]:
    """Pack every leaf across `examples` into ONE forward; split the resulting
    scalars back into per-question logit groups in engine semantics.

    `backbone` is the inner transformer (mlx/model.py's docstring); `head` is
    an mlx/model.DecisionHead instance. Returns one logit list per example, in
    `examples` order -- boolean groups are `[0.0, z]` (P(true)=sigmoid(z)),
    same convention as server/src/engine._forward.
    """
    paths = [toks for ex in examples for toks in ex.leaf_tokens]
    lengths_list = [len(p) for p in paths]
    width = max(lengths_list)

    tokens = mx.full((len(paths), width), pad_token_id, dtype=mx.int32)
    for i, p in enumerate(paths):
        tokens[i, : len(p)] = mx.array(p, dtype=mx.int32)
    lengths = mx.array(lengths_list, dtype=mx.int32)

    hidden = backbone(tokens)  # (P, width, H)
    leaves = hidden[mx.arange(tokens.shape[0]), lengths - 1]  # (P, H) last real token
    scalars = head(leaves).squeeze(-1).astype(mx.float32)  # (P,)
    mx.eval(scalars)
    scalars_list = scalars.tolist()

    out: List[List[float]] = []
    offset = 0
    for ex in examples:
        n = len(ex.leaf_tokens)
        group = scalars_list[offset : offset + n]
        offset += n
        if ex.type == "boolean":
            # logits [0, z] over the (false, true) pair: P(true)=sigmoid(z).
            group = [0.0, float(group[0])]
        out.append(group)
    return out
