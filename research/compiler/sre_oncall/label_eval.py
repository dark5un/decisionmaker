#!/usr/bin/env python3
"""Label the independent eval set via the RULE INTERPRETER (not author choice).

Anti-circularity: the 'correct' answer for each eval question is whatever the
compiled decision rules mechanically produce for the question's declared state.
The author is NOT allowed to pick the outcome; this script computes it. This is
the pre-registration step: outputs are fixed before training.

Usage: python3 research/compiler/sre_oncall/label_eval.py -> eval_labeled.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.compiler import core as c

HERE = Path(__file__).resolve().parent


def _load_tables():
    return [json.loads(l) for l in (HERE / "tables.jsonl").read_text().splitlines() if l.strip()]


def main() -> int:
    tables = _load_tables()
    by_did = {t["decision_id"]: t for t in tables}
    evals = [json.loads(l) for l in (HERE / "eval_questions_independent.jsonl").read_text().splitlines() if l.strip()]
    missing = sorted({e["decision_id"] for e in evals} - set(by_did))
    if missing:
        print("ERROR: eval decisions missing from tables:", missing)
        return 1

    label = []
    for e in evals:
        t = by_did[e["decision_id"]]
        decision = c._as_decision(t)
        outcome, fired_rid = __import__("research.planA.rules", fromlist=["rules"]).run_rules(decision, e["state"])
        dist, _winner = c.gold_for(t, e["state"])
        q = t["question"]
        label.append({
            "id": f"eval-{e['decision_id']}-{len(label)}",
            "decision_id": e["decision_id"],
            "prose": e["prose"],
            "state": e["state"],
            "question": q,
            "expected_outcome": outcome,
            "expected_label": (1.0 if q["type"] == "boolean" else None),
            "gold_probs": {e["decision_id"]: {k: round(float(v), 6) for k, v in dist.items()}},
            "fired_rule": fired_rid,
        })
    out = HERE / "eval_labeled.json"
    out.write_text("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in label) + "\n")
    from collections import Counter
    print(f"labeled {len(label)} eval questions -> {out}")
    print("per decision:", dict(Counter(x["decision_id"] for x in label)))
    for row in label:
        print(f"  {row['decision_id']:<22} expected={row['expected_outcome']:<8} rule={row['fired_rule']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())