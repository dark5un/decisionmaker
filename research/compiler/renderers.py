"""Pluggable prose renderers: state dict -> natural-language phrasings.

The synthesizer renders each enumerated state to 1..k prose phrasings before
running the rule interpreter to compute gold. Renderers are pluggable per skill
(and can be extended by registering into RENDERERS); skills without a bespoke
renderer fall back to a deterministic generic field-based phrasing.

The three shortlist skills reuse research/planA/synthesize_planA.RENDER so the
compiler reproduces planA_train.jsonl byte-for-byte where applicable.
"""
from __future__ import annotations

from research.planA.synthesize_planA import RENDER as _PLAN_A_RENDER  # noqa


def _generic(decision_id, st):
    """Deterministic generic phrasing: render each field as a declarative clause,
    then combine into the decision question. Returns a single phrasing."""
    clauses = []
    for k, v in st.items():
        if isinstance(v, bool):
            clauses.append(f"{k} is {'true' if v else 'false'}")
        else:
            clauses.append(f"{k} is {v}")
    return [". ".join(clauses) + f". Decision: {decision_id}."]


# skill -> decision_id -> renderer(state)->[phrasings]
EXTRA = {}


def render(skill, decision_id, st, k=None):
    """Return a list of natural-language phrasings for `st`.

    k (--k-phrasings) repeats/takes phrasings deterministically; the default is
    every phrasing the renderer provides.
    """
    fn = None
    try:
        fn = _PLAN_A_RENDER[skill]
    except KeyError:
        fn = EXTRA.get(skill, {}).get(decision_id, _generic)
    phrasings = fn(decision_id, st)
    if k is not None:
        phrasings = phrasings[:k] if k > 0 else phrasings[:1]
    return phrasings or _generic(decision_id, st)


def register(skill, decision_id, fn):
    EXTRA.setdefault(skill, {})[decision_id] = fn