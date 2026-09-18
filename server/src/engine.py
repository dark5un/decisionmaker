"""Candidate-leaf decision engine (Phase 3).

Pure, torch-free CORE plus an orchestrator that runs the ONE-forward pipeline over
an injected backbone + decision head. Interior monotone by construction: there is
no next-token decoding; we score candidate leaf paths in a single forward pass.

Design intent (see docs/architecture.md §3): leaves are built per the candidate-leaf
pattern; the qid is a mere response key and NEVER part
of any leaf (qid-invariance); Score leaves must not leak the ordinal index; the
Choice's option order is preserved because the model input is order-sensitive.

The pure functions here (`validate_request`, `build_leaf_texts`, `encode_leaves`)
do NOT import torch, so the full logic is unit-testable with a fake tokenizer and
no GPU. The tensor path imports torch lazily and surfaces a typed error when
torch or a backbone is absent.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Options & config
# ---------------------------------------------------------------------------

MODEL_DEFAULT = "Qwen/Qwen3-0.6B"
# Verified 2026-09-18 via HF API/raw config: the pinned revision resolves on
# Qwen/Qwen3-0.6B (the chat/instruction release, fine-tuned from Qwen3-0.6B-Base).
# "Qwen/Qwen3-0.6B-Instruct" is NOT used — it is a gated repo (401 without a
# token) and we train a decision head on hidden states, so the base checkpoint
# at the pinned revision is the reproducible choice.
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@dataclass
class EngineConfig:
    max_length: int = 512
    model: str = MODEL_DEFAULT
    revision: str = MODEL_REVISION
    temperature: float = 1.0
    set_head: str = "none"  # "none" | "attention" (attention added later)


# ---------------------------------------------------------------------------
# Request validation (mirrors spec/openapi.yaml structural rules)
# ---------------------------------------------------------------------------


class EngineError(Exception):
    """Typed failure surfaced by the engine. `code` maps at the HTTP layer so a
    contract violation is a 422 and an engine malfunction a 500."""

    def __init__(self, code: str, message: str, details: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def asdict(self) -> Dict[str, Any]:
        out = {"type": self.code, "message": self.message}
        if self.details:
            out["details"] = self.details
        return out


def _is_nonempty_string(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _serialize_state(state: Any) -> str:
    if _is_nonempty_string(state):
        return state
    # object / array -> compact canonical JSON (insertion order, no whitespace).
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def _text(value: Any) -> str:
    """StructuredText -> the exact text the model sees (string as-is, else compact JSON)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _is_nonempty_entry(value: Any) -> bool:
    """A usable StructuredText entry: non-empty string, or non-empty object/array."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list)):
        return len(value) > 0
    return False


def validate_request(payload: Mapping[str, Any]) -> None:
    """Enforce the wire contract described in docs/API.md. Raises EngineError(422)."""
    if not isinstance(payload, dict):
        raise EngineError("validation", "request must be a JSON object")
    if "state" not in payload:
        raise EngineError("validation", "missing required field", {"field": "state"})
    if "questions" not in payload:
        raise EngineError("validation", "missing required field", {"field": "questions"})

    state = payload["state"]
    if _is_nonempty_string(state):
        pass
    elif isinstance(state, (dict, list)):
        if len(state) == 0:
            raise EngineError("validation", "state must be non-empty", {"field": "state"})
    else:
        raise EngineError("validation", "state must be a non-empty string, object, or array")

    questions = payload["questions"]
    if not isinstance(questions, dict) or not questions:
        raise EngineError("validation", "questions must be a non-empty object")

    for qid, q in questions.items():
        if not _is_nonempty_string(qid):
            raise EngineError("validation", "question id must be a non-empty string")
        if not isinstance(q, dict):
            raise EngineError("validation", f"question '{qid}' must be an object")
        typ = q.get("type")
        if typ not in ("boolean", "choice", "score"):
            raise EngineError("validation", f"question '{qid}': type must be boolean|choice|score")
        extra = set(q) - {"type", "instructions", "criteria"}
        if extra:
            raise EngineError("validation", f"question '{qid}': unexpected fields {sorted(extra)}")

        criteria = q.get("criteria")
        if typ == "boolean":
            if criteria is not None:
                if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
                    raise EngineError(
                        "validation",
                        f"question '{qid}': boolean criteria may only contain true/false keys",
                    )
                for k, v in criteria.items():
                    if v is not None and not _is_nonempty_entry(v):
                        raise EngineError(
                            "validation",
                            f"question '{qid}': boolean '{k}' criterion must be a non-empty string|object|array",
                        )
        elif typ == "choice":
            if not isinstance(criteria, dict) or not (2 <= len(criteria) <= 255):
                raise EngineError("validation", f"question '{qid}': choice criteria must have 2..255 options")
            for key in criteria:
                if not _is_nonempty_string(key):
                    raise EngineError("validation", f"question '{qid}': choice option keys must be non-empty strings")
        else:  # score
            if not isinstance(criteria, list) or not (2 <= len(criteria) <= 10):
                raise EngineError("validation", f"question '{qid}': score criteria must have 2..10 levels")
            if not all(_is_nonempty_entry(level) for level in criteria):
                raise EngineError("validation", f"question '{qid}': score level descriptions must be non-empty string|object|array")

        instructions = q.get("instructions")
        if instructions is not None and not isinstance(instructions, (str, dict, list)):
            raise EngineError("validation", f"question '{qid}': instructions must be string|object|array|null")


# ---------------------------------------------------------------------------
# Leaf construction (the encoding the model sees) -- pure, torch-free
# ---------------------------------------------------------------------------


def _candidate_texts(question: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (ids, texts) for a question, in leaf order.

    - boolean: two leaves ids ["false","true"]; only the "true" path carries text
      (the model emits a scalar z; P(true) = sigmoid over logits [0, z]).
    - choice: one leaf per option, text "{key}: {description}" in criteria order;
      a null description collapses to just the key.
    - score: one leaf per level, text = the level description in order; the
      ordinal index is NEVER injected (the model must infer the order).
    """
    typ = question["type"]
    if typ == "boolean":
        return ["false", "true"], ["The proposition is true."]
    if typ == "choice":
        ids = list(question["criteria"].keys())
        texts = [
            f"{key}: {_text(desc)}" if (desc is not None and _is_nonempty_entry(desc)) else key
            for key, desc in ((k, question["criteria"][k]) for k in ids)
        ]
        return ids, texts
    return [str(i) for i in range(len(question["criteria"]))], [_text(level) for level in question["criteria"]]


def build_leaf_texts(state_text: str, question: Mapping[str, Any]) -> List[str]:
    """Pure: produce the exact leaf prompt strings for one question (no tokenizer).

    Format (documented in docs/architecture.md §3):
        State:\n{state}\n
        Question type: {type}\n
        Question:\n{instructions}\n
        [for boolean with criteria: "False criterion: ...\n" then "True criterion: ...\n"]
        Candidate:\n{candidate}\nDecision:
    """
    typ = question["type"]
    instructions = question.get("instructions")

    segments = [f"State:\n{state_text}\n", f"Question type: {typ}\n"]
    if instructions is not None:
        segments.append(f"Question:\n{_text(instructions)}\n")

    if typ == "boolean" and question.get("criteria"):
        criteria = question["criteria"]
        if "false" in criteria:
            segments.append(f"False criterion: {_text(criteria['false'])}\n")
        if "true" in criteria:
            segments.append(f"True criterion: {_text(criteria['true'])}\n")

    _, texts = _candidate_texts(question)
    return ["".join(segments) + f"Candidate:\n{t}\nDecision:" for t in texts]


# ---------------------------------------------------------------------------
# Tokenization / encoding -- torch-free; takes a tokenizer protocol
# ---------------------------------------------------------------------------


class TokenizerProtocol:
    """Minimal surface we need from a real PreTrainedTokenizer."""

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        raise NotImplementedError

    @property
    def eos_token_id(self) -> int:
        raise NotImplementedError

    @property
    def pad_token_id(self) -> int:
        raise NotImplementedError


def encode_leaves(tokenizer: TokenizerProtocol, leaf_texts: List[str], max_length: int) -> List[List[int]]:
    """Tokenize leaf texts and append EOS; reject (not truncate) oversized leaves.

    The EOS is appended AFTER the "Decision:" marker, as the candidate-leaf format
    requires. Oversized leaves raise instead of being silently truncated.
    """
    if max_length <= 0:
        raise EngineError("internal", "max_length must be positive")
    eos = tokenizer.eos_token_id
    leaves = []
    for text in leaf_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False) + [eos]
        if len(tokens) > max_length:
            raise EngineError(
                "validation",
                f"candidate path of {len(tokens)} tokens exceeds max_length={max_length}; input was NOT truncated",
            )
        leaves.append(tokens)
    return leaves


# ---------------------------------------------------------------------------
# Prepared examples + the one-forward orchestrator (lazy torch import)
# ---------------------------------------------------------------------------


@dataclass
class PreparedExample:
    """One question, fully prepared for scoring (and for the training harness).

    `state_id` is constant for this single-state engine; the training harness
    (Phase 5) extends this to many states by sharding requests.
    """

    qid: str
    state_id: str
    type: str
    candidate_ids: List[str]
    candidate_texts: List[str]
    leaf_tokens: List[List[int]]


def prepare_question(tokenizer: TokenizerProtocol, state_id: str, qid: str,
                     question: Mapping[str, Any], state_text: str, max_length: int) -> PreparedExample:
    """Validate + build leaves for a single question. The qid lands only here
    (in the metadata) and never in any leaf token."""
    ids, texts = _candidate_texts(question)
    leaf_texts = build_leaf_texts(state_text, question)
    leaves = encode_leaves(tokenizer, leaf_texts, max_length)
    return PreparedExample(qid=qid, state_id=state_id, type=question["type"],
                           candidate_ids=ids, candidate_texts=texts, leaf_tokens=leaves)


def prepare_request(tokenizer: TokenizerProtocol, payload: Mapping[str, Any],
                    max_length: int) -> List[PreparedExample]:
    """Validate a request and expand it into one PreparedExample per question.

    The state is serialized ONCE per request (shared across all its questions),
    so the same state text is reused — and no qid ever influences a token.
    """
    validate_request(payload)
    state_text = _serialize_state(payload["state"])
    examples = []
    for qid, question in payload["questions"].items():
        examples.append(prepare_question(tokenizer, "state", qid, question, state_text, max_length))
    return examples


def _distribution_confidence(probs_in_order: List[float]) -> float:
    """Spread-based confidence: how peaked the distribution is on its winner.

    Confidence = (p_max · k − 1) / (k − 1): linear between a uniform distribution
    over k candidates (0 = torn) and a point mass (1 = one option dominates).
    This is a property of the returned distribution, NOT a calibration guarantee.
    """
    k = len(probs_in_order)
    p_max = max(probs_in_order)
    if k <= 1:
        return 1.0
    return float(round((p_max * k - 1.0) / (k - 1.0), 6))


def _answer_from_logits(example: PreparedExample, logits: List[float], temperature: float) -> Dict[str, Any]:
    """Turn per-candidate logits into a contract-shaped answer; validates output."""
    k = len(example.candidate_ids)
    if len(logits) != k or not all(math.isfinite(v) for v in logits):
        raise EngineError("internal", "model produced non-finite or wrong-size logits")
    numeric = [v / temperature for v in logits]
    mx = max(numeric)
    exps = [math.exp(v - mx) for v in numeric]
    z = sum(exps)
    if not (z > 0 and all(math.isfinite(v) for v in exps)):
        raise EngineError("internal", "model produced invalid logits (softmax denominator)")
    probs = [e / z for e in exps]

    if example.type == "boolean":
        p_true = probs[1]
        return {"type": "boolean", "boolean": float(round(p_true, 6))}

    if example.type == "choice":
        probs_by_id = dict(zip(example.candidate_ids, probs))
        best = max(example.candidate_ids, key=probs_by_id.__getitem__)
        order = [probs_by_id[c] for c in example.candidate_ids]
        return {
            "type": "choice",
            "choice": best,
            "probabilities": probs_by_id,
            "confidence": _distribution_confidence(order),
        }

    # score
    score = sum(i * p for i, p in enumerate(probs))
    legend = {str(i): text for i, text in enumerate(example.candidate_texts)}
    return {
        "type": "score",
        "score": float(round(score, 6)),
        "legend": legend,
        "probabilities": {str(i): p for i, p in enumerate(probs)},
        "confidence": _distribution_confidence(probs),
    }


class DecisionEngine:
    """Persistent engine: constructed once, reused for serving and training.

    `backbone` and `head` are injected objects with the torch contract described
    in model.py. The pure gates work without them; `predict` demands them and
    raises a typed error otherwise.
    """

    def __init__(self, tokenizer: TokenizerProtocol, config: EngineConfig,
                 backbone: Any = None, head: Any = None):
        self.tokenizer = tokenizer
        self.config = config
        self.backbone = backbone
        self.head = head

    def build_leaves(self, payload: Mapping[str, Any]) -> List[PreparedExample]:
        return prepare_request(self.tokenizer, payload, self.config.max_length)

    def predict(self, payload: Mapping[str, Any], temperature: Optional[float] = None) -> Dict[str, Any]:
        temp = self.config.temperature if temperature is None else temperature
        if not isinstance(temp, (int, float)) or isinstance(temp, bool) or temp <= 0 or not math.isfinite(temp):
            raise EngineError("validation", "temperature must be a finite positive number")
        examples = self.build_leaves(payload)
        if self.backbone is None or self.head is None:
            raise EngineError("internal", "engine has no loaded backbone/head (torch runtime not configured)")
        logits_per_question = _forward(self.backbone, self.head, self.tokenizer, examples)
        answers = {}
        input_tokens = 0
        for ex, logits in zip(examples, logits_per_question):
            answers[ex.qid] = _answer_from_logits(ex, logits, temp)
            input_tokens += sum(len(toks) for toks in ex.leaf_tokens)
        # No next-token decoding anywhere, so output_tokens is always 0 for this
        # head-based engine: the answer is read from hidden states in one pass.
        return {"model": self.config.model, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}}


def _forward(backbone: Any, head: Any, tokenizer: TokenizerProtocol,
             examples: List[PreparedExample]) -> List[List[float]]:
    """One packed forward over all leaves -> per-question candidate logits."""
    import torch  # lazy: the pure tests never reach here.

    # Place all input tensors on the model's device. Derive it from the BACKBONE
    # (the authoritative compute target) — a CPU head against a CUDA backbone, or
    # CPU inputs against a CUDA model, both fail inside embed_tokens.
    device = next(backbone.parameters()).device
    paths = [toks for ex in examples for toks in ex.leaf_tokens]
    lengths = torch.tensor([len(p) for p in paths], dtype=torch.long, device=device)
    width = int(lengths.max())
    tokens = torch.full((len(paths), width), int(tokenizer.pad_token_id),
                        dtype=torch.long, device=device)
    for i, p in enumerate(paths):
        tokens[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=device)
    attention = torch.arange(width, device=device)[None, :] < lengths[:, None]

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        hidden = backbone(tokens, attention_mask=attention).last_hidden_state  # (P, width, H)
    leaves = hidden[torch.arange(len(paths), device=device), lengths - 1]  # last real token
    scalars = head(leaves).squeeze(-1).float()  # (P,)

    out = []
    offset = 0
    for ex in examples:
        n = len(ex.leaf_tokens)
        group = scalars[offset:offset + n].tolist()
        offset += n
        if ex.type == "boolean":
            # logits [0, z] over the (false, true) pair: P(true)=sigmoid(z).
            group = [0.0, float(group[0])]
        out.append(group)
    return out