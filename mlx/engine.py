"""MLX decision engine -- mirrors the serving role of server.src.engine.DecisionEngine
without its hardcoded torch `_forward`.

server/src/engine.py is untouched by this file (zero risk to the existing
torch/CUDA path): this class reuses its pure helpers (`EngineConfig`,
`EngineError`, `prepare_request`, `_answer_from_logits`) and swaps in
mlx/forward.forward_mlx for the tensor path, same split as
DecisionEngine.predict() -> engine._forward.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from server.src.engine import EngineConfig, EngineError, prepare_request, _answer_from_logits

from forward import forward_mlx


class MlxDecisionEngine:
    """Duck-type compatible with server.src.app.create_app's `engine` contract:
    `.predict(payload) -> dict`, `.config.model`, and a truthy `.backbone`
    once loaded -- so the existing FastAPI layer needs no changes at all.
    """

    def __init__(self, tokenizer: Any, config: EngineConfig, backbone: Any = None, head: Any = None):
        self.tokenizer = tokenizer
        self.config = config
        self.backbone = backbone
        self.head = head

    def predict(self, payload: Any, temperature: Optional[float] = None) -> Dict[str, Any]:
        temp = self.config.temperature if temperature is None else temperature
        if not isinstance(temp, (int, float)) or isinstance(temp, bool) or temp <= 0 or not math.isfinite(temp):
            raise EngineError("validation", "temperature must be a finite positive number")
        examples = prepare_request(self.tokenizer, payload, self.config.max_length)
        if self.backbone is None or self.head is None:
            raise EngineError("internal", "engine has no loaded backbone/head (MLX model not configured)")
        logits_per_question = forward_mlx(self.backbone, self.head, self.tokenizer.pad_token_id, examples)
        answers = {}
        input_tokens = 0
        for ex, logits in zip(examples, logits_per_question):
            answers[ex.qid] = _answer_from_logits(ex, logits, temp)
            input_tokens += sum(len(toks) for toks in ex.leaf_tokens)
        # No next-token decoding here either -- same "output_tokens always 0"
        # property as the torch engine, for the same reason (read from hidden
        # states in one pass, never decode).
        return {"model": self.config.model, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
