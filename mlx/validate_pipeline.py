#!/usr/bin/env python3
"""Integration check for mlx/model.py + mlx/forward.py against a real request.

Builds a full {state, questions} payload (boolean + choice + score -- all
three types), runs it through the actual reused pipeline:

    server.src.engine.prepare_request   (validation + leaf construction)
    -> mlx/forward.forward_mlx          (the MLX packed forward)
    -> server.src.engine._answer_from_logits  (softmax -> contract-shaped answer)

and checks every answer is contract-shaped (probabilities finite and sum to
1, boolean in [0,1], score legend present). This is the same shape the real
engine/server produces, just with an MLX backbone+head instead of torch/CUDA.

Run: uv run python validate_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from server.src.engine import prepare_request, _answer_from_logits  # torch-free

from mlx_lm import load

from model import DecisionHead
from forward import forward_mlx

MLX_MODEL_PATH = Path(__file__).resolve().parent / "converted" / "qwen3-0.6b-c1899de"

PAYLOAD = {
    "state": "Customer wrote: 'Our invoice is overdue and we will cancel.'",
    "questions": {
        "urgency": {"type": "boolean", "instructions": "Does this convey urgency?"},
        "routing": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        },
        "anger": {
            "type": "score",
            "instructions": "How angry does the customer sound?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}


def main() -> None:
    print(f"loading MLX model from {MLX_MODEL_PATH} ...")
    full_model, tokenizer = load(str(MLX_MODEL_PATH))
    backbone = full_model.model  # inner transformer only, no LM head
    head = DecisionHead(int(backbone.args.hidden_size))

    examples = prepare_request(tokenizer, PAYLOAD, max_length=512)
    print(f"prepared {len(examples)} examples: {[ex.qid for ex in examples]}")

    logits_per_question = forward_mlx(backbone, head, tokenizer.pad_token_id, examples)

    answers = {}
    for ex, logits in zip(examples, logits_per_question):
        answers[ex.qid] = _answer_from_logits(ex, logits, temperature=1.0)

    print()
    for qid, ans in answers.items():
        print(f"{qid}: {ans}")

    # Contract checks (docs/API.md): probabilities finite, sum to 1, bounds respected.
    b = answers["urgency"]
    assert b["type"] == "boolean" and 0.0 <= b["boolean"] <= 1.0

    c = answers["routing"]
    assert c["type"] == "choice"
    assert abs(sum(c["probabilities"].values()) - 1.0) < 1e-4
    assert c["choice"] in c["probabilities"]
    assert 0.0 <= c["confidence"] <= 1.0

    s = answers["anger"]
    assert s["type"] == "score"
    assert abs(sum(s["probabilities"].values()) - 1.0) < 1e-4
    assert set(s["legend"].keys()) == set(s["probabilities"].keys())
    assert 0.0 <= s["confidence"] <= 1.0

    print("\nVALIDATE OK: mlx/model.py + mlx/forward.py produce contract-shaped, "
          "sums-to-1 answers for boolean/choice/score via the reused engine "
          "validation + leaf-building + answer-shaping code.")


if __name__ == "__main__":
    main()
