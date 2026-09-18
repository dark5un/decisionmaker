"""Torch-free unit tests for the engine's pure core (Phase 3 gate).

These run with NO GPU and NO torch: they exercise the reversible tokenizer path
(FakeTokenizer encodes each char to its codepoint, so leaf tokens are
deterministic and character-level assertions are exact) plus validation and the
answer-shaping helpers.
"""
import json
import math
from pathlib import Path

import pytest

from server.src import engine as E
from server.src.engine import (
    EngineConfig,
    EngineError,
    DecisionEngine,
    _answer_from_logits,
    _candidate_texts,
    _distribution_confidence,
    _serialize_state,
    build_leaf_texts,
    encode_leaves,
    prepare_request,
    validate_request,
)


class FakeTokenizer:
    """Char->codepoint encoding: reversible, deterministic, no external dep."""

    def __init__(self):
        self.eos_token_id = 0
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def make_tok():
    t = FakeTokenizer()
    t.eos_token_id = 1
    t.pad_token_id = 0
    return t


# ---------------------------------------------------------------------------
# Leaf construction
# ---------------------------------------------------------------------------


def test_choice_leaf_format_exact():
    q = {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"technical": "Bugs, outages", "sales": "Pricing"},
    }
    leaves = build_leaf_texts("Some state.", q)
    assert leaves == [
        "State:\nSome state.\n"
        "Question type: choice\n"
        "Question:\nWhich team?\n"
        "Candidate:\ntechnical: Bugs, outages\nDecision:",
        "State:\nSome state.\n"
        "Question type: choice\n"
        "Question:\nWhich team?\n"
        "Candidate:\nsales: Pricing\nDecision:",
    ]


def test_choice_option_order_preserved():
    q = {"type": "choice", "criteria": {"z": "zz", "a": "aa", "m": "mm"}}
    ids, texts = _candidate_texts(q)
    assert ids == ["z", "a", "m"]  # as-written order, not sorted
    leaves = build_leaf_texts("s", q)
    assert "Candidate:\nz: zz\nDecision:" in leaves[0]
    assert "Candidate:\na: aa\nDecision:" in leaves[1]


def test_choice_null_description_collapses_to_key():
    q = {"type": "choice", "criteria": {"other": None, "spam": "unsolicited"}}
    ids, texts = _candidate_texts(q)
    assert texts[0] == "other"
    assert texts[1] == "spam: unsolicited"


def test_boolean_leaves_no_criteria():
    q = {"type": "boolean", "instructions": "Does this convey urgency?"}
    ids, texts = _candidate_texts(q)
    assert ids == ["false", "true"]  # two candidates for the ANSWER mapping...
    assert len(texts) == 1            # ...but only ONE semantic leaf (the true path)
    leaves = build_leaf_texts("Help!", q)
    assert len(leaves) == 1
    assert "False criterion:" not in leaves[0]
    assert "True criterion:" not in leaves[0]
    assert leaves[0].endswith("Candidate:\nThe proposition is true.\nDecision:")


def test_boolean_leaves_include_criteria_lines():
    q = {"type": "boolean", "criteria": {"true": "time-sensitive", "false": "no urgency"}}
    leaves = build_leaf_texts("Help!", q)
    assert "False criterion: no urgency\n" in leaves[0]
    assert "True criterion: time-sensitive\n" in leaves[0]


def test_score_never_leaks_ordinal_index():
    q = {"type": "score", "criteria": ["Calm", "Frustrated", "Very angry"]}
    ids, texts = _candidate_texts(q)
    assert ids == ["0", "1", "2"]  # IDs carry the index for the RESPONSE legend
    leaves = build_leaf_texts("s", q)
    # The model input must NOT contain "0:", "1:", "2:" or "level 1" etc.
    for leaf in leaves:
        assert "0:" not in leaf.split("Candidate:\n")[1]
        assert "Level" not in leaf and "level" not in leaf
    assert leaves[0].endswith("Candidate:\nCalm\nDecision:")
    assert leaves[1].endswith("Candidate:\nFrustrated\nDecision:")
    assert leaves[2].endswith("Candidate:\nVery angry\nDecision:")


def test_structured_instructions_serialized_as_json():
    q = {"type": "boolean", "instructions": {"focus": "compare fields", "fields": ["a", "b"]}}
    leaf = build_leaf_texts("s", q)[0]
    assert '{"focus":"compare fields","fields":["a","b"]}' in leaf


def test_structured_score_levels_serialized():
    q = {"type": "score", "criteria": [
        {"label": "Low", "description": "Cosmetic"},
        {"label": "High", "description": "Blocking"},
    ]}
    leaves = build_leaf_texts("s", q)
    assert '{"label":"Low","description":"Cosmetic"}' in leaves[0]
    assert '{"label":"High","description":"Blocking"}' in leaves[1]
    # ordinal index still never injected
    assert "0:" not in leaves[0].split("Candidate:\n")[1]


def test_structured_choice_description_serialized():
    q = {"type": "choice", "criteria": {"phishing": {"kind": "credential"}, "other": None}}
    ids, texts = _candidate_texts(q)
    assert texts[0] == "phishing: {\"kind\":\"credential\"}"
    assert texts[1] == "other"  # null description collapses to the key


def test_state_serialization():
    assert _serialize_state("plain text") == "plain text"
    assert _serialize_state({"a": 1, "b": [2]}) == '{"a":1,"b":[2]}'
    assert _serialize_state(["x", "y"]) == '["x","y"]'


# ---------------------------------------------------------------------------
# QID invariance (the character-level Phase 3 gate)
# ---------------------------------------------------------------------------


def test_qid_invariance_leaf_tokens_identical():
    tok = make_tok()
    q_template = {
        "is_urgent": {"type": "boolean", "instructions": "Is this urgent?"},
        "department": {"type": "choice", "criteria": {"billing": "Payments", "tech": "Bugs"}},
        "anger": {"type": "score", "criteria": ["Calm", "Very angry"]},
    }
    payload = {"state": "Help! payouts failing.", "questions": dict(q_template)}
    renamed = {
        "state": payload["state"],
        "questions": {
            "totally_different_id_1": q_template["is_urgent"],
            "totally_different_id_2": q_template["department"],
            "totally_different_id_3": q_template["anger"],
        },
    }
    a = prepare_request(tok, payload, 512)
    b = prepare_request(tok, renamed, 512)
    # Same leaf tokens, same order — only the qid metadata differs.
    assert [ex.leaf_tokens for ex in a] == [ex.leaf_tokens for ex in b]
    assert [ex.candidate_ids for ex in a] == [ex.candidate_ids for ex in b]
    assert [ex.qid for ex in a] != [ex.qid for ex in b]


def test_qid_string_never_in_leaf_tokens():
    tok = make_tok()
    payload = {
        "state": "s",
        "questions": {"supersecretqid": {"type": "boolean", "instructions": "x?"}},
    }
    examples = prepare_request(tok, payload, 512)
    # FakeTokenizer is char-level, so decode each leaf back to text and assert the
    # whole qid (not individual chars — those overlap ordinary English) is absent.
    for ex in examples:
        for toks in ex.leaf_tokens:
            text = "".join(chr(i) for i in toks if i != tok.eos_token_id)
            assert "supersecretqid" not in text


# ---------------------------------------------------------------------------
# Encoding / truncation
# ---------------------------------------------------------------------------


def test_encode_leaves_appends_eos_and_rejects_oversize():
    tok = make_tok()
    leaves = encode_leaves(tok, ["a", "ab"], max_length=3)
    assert leaves == [[ord("a"), 1], [ord("a"), ord("b"), 1]]
    with pytest.raises(EngineError) as ei:
        encode_leaves(tok, ["abc"], max_length=3)  # 3 tokens + eos = 4 > 3
    assert ei.value.code == "validation"
    assert "NOT truncated" in ei.value.message


# ---------------------------------------------------------------------------
# Validation (mirrors the schema's rejections)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"state": "", "questions": {"a": {"type": "boolean"}}},
        {"state": "x"},
        {"state": "x", "questions": {"q": {"type": "integer"}}},
        {"state": "x", "questions": {"q": {"type": "choice", "criteria": {"only": "one"}}}},
        {"state": "x", "questions": {"q": {"type": "score", "criteria": ["a"] * 1}}},
        {"state": "x", "questions": {"q": {"type": "score", "criteria": ["a"] * 11}}},
        {"state": "x", "questions": {"q": {"type": "boolean", "criteria": {"maybe": "x"}}}},
        {"state": "x", "questions": {}},
        {"state": "x", "questions": {"q": {"type": "choice", "criteria": {}, "bogus": 1}}},
    ],
)
def test_validate_request_rejects_invalid(payload):
    with pytest.raises(EngineError) as ei:
        validate_request(payload)
    assert ei.value.code == "validation"


def test_validate_request_accepts_reference_shape():
    p = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "domain_shape.request.json"
    payload = json.loads(p.read_text())
    validate_request(payload)  # does not raise


# ---------------------------------------------------------------------------
# Answer shaping + confidence
# ---------------------------------------------------------------------------


def test_distribution_confidence_edges():
    assert _distribution_confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert _distribution_confidence([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)
    assert _distribution_confidence([1.0, 0.0]) == pytest.approx(1.0)
    assert _distribution_confidence([0.7, 0.2, 0.1]) == pytest.approx((0.7 * 3 - 1) / 2)


def _ex(typ, ids, texts):
    return E.PreparedExample(qid="q", state_id="state", type=typ,
                             candidate_ids=ids, candidate_texts=texts,
                             leaf_tokens=[])


def test_answer_choice_shape():
    ex = _ex("choice", ["a", "b", "c"], ["A", "B", "C"])
    ans = _answer_from_logits(ex, [3.0, 2.0, 1.0], 1.0)
    assert ans["type"] == "choice"
    assert ans["choice"] == "a"
    assert abs(sum(ans["probabilities"].values()) - 1.0) < 1e-5
    assert 0 <= ans["confidence"] <= 1


def test_answer_choice_temperature_flattens():
    ex = _ex("choice", ["a", "b"], ["A", "B"])
    hot = _answer_from_logits(ex, [2.0, 1.0], 1.0)
    cold = _answer_from_logits(ex, [2.0, 1.0], 0.5)
    # cold (lower temperature -> sharper) must be more confident on the winner
    assert cold["probabilities"]["a"] > hot["probabilities"]["a"]


def test_answer_score_shape_and_value():
    ex = _ex("score", ["0", "1", "2"], ["Calm", "Frustrated", "Very angry"])
    ans = _answer_from_logits(ex, [0.0, 100.0, 0.0], 1.0)
    assert ans["score"] == pytest.approx(1.0)  # level 1 dominates
    assert ans["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert abs(sum(ans["probabilities"].values()) - 1.0) < 1e-5
    assert ans["confidence"] == pytest.approx(1.0)


def test_answer_score_fractional():
    ex = _ex("score", ["0", "1", "2"], ["a", "b", "c"])
    # uniform -> score = mean of {0,1,2} = 1.0
    ans = _answer_from_logits(ex, [0.0, 0.0, 0.0], 1.0)
    assert ans["score"] == pytest.approx(1.0)


def test_answer_boolean_shape_no_confidence():
    ex = _ex("boolean", ["false", "true"], ["The proposition is true."])
    # Boolean logits are assembled as [0, z]; P(true) = sigmoid(z).
    ans = _answer_from_logits(ex, [0.0, 1.0], 1.0)
    assert ans["type"] == "boolean"
    assert "confidence" not in ans
    assert ans["boolean"] == pytest.approx(1.0 / (1.0 + math.exp(-1)), abs=1e-5)

    ans2 = _answer_from_logits(ex, [0.0, -1.0], 1.0)
    assert ans2["boolean"] == pytest.approx(1.0 / (1.0 + math.exp(1)), abs=1e-5)


# ---------------------------------------------------------------------------
# Engine-level conduit (no torch path involved)
# ---------------------------------------------------------------------------


def test_engine_predict_requires_backbone():
    eng = DecisionEngine(tokenizer=make_tok(), config=EngineConfig())
    with pytest.raises(EngineError) as ei:
        eng.predict({"state": "x", "questions": {"q": {"type": "boolean"}}})
    assert ei.value.code == "internal"


def test_engine_build_leaves_qid_independent_tokens():
    eng = DecisionEngine(tokenizer=make_tok(), config=EngineConfig(max_length=512))
    a = eng.build_leaves({"state": "s", "questions": {"q1": {"type": "boolean"}}})
    b = eng.build_leaves({"state": "s", "questions": {"q9": {"type": "boolean"}}})
    assert a[0].leaf_tokens == b[0].leaf_tokens