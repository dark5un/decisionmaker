#!/usr/bin/env python3
"""Run the pre-registered 28 eval questions against the served decisionmaker.

Builds one {state, questions} body per eval record (state = the novel prose the
independent author wrote; questions = the decision's question contract), POSTs
to /v1/decisionmaker, and writes a scored table: model answer vs the
interpreter-computed gold label. Fully transparent — no answers are hidden.

Usage: python3 research/compiler/sre_oncall/run_eval.py [--url URL]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = "http://127.0.0.1:8090/v1/decisionmaker"


def _load_tables():
    return {json.loads(l)["decision_id"]: json.loads(l)
            for l in (HERE / "tables.jsonl").read_text().splitlines() if l.strip()}


def _gold_label(qtype, dist):
    """The interpreter-computed hard label for a question's gold distribution."""
    if qtype == "boolean":
        return 1.0 if dist.get("true", 0.0) >= dist.get("false", 0.0) else 0.0
    keys = list(dist.keys())
    return max(keys, key=lambda k: dist[k])


def _model_answer(ans):
    if ans.get("type") == "boolean":
        return ans.get("boolean")
    if ans.get("type") == "choice":
        return ans.get("choice")
    return ans.get("score")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT)
    ap.add_argument("--out", default=str(HERE / "eval_results.jsonl"))
    args = ap.parse_args()
    tables = _load_tables()
    evals = [json.loads(l) for l in (HERE / "eval_labeled.json").read_text().splitlines() if l.strip()]

    results = []
    for e in evals:
        t = tables[e["decision_id"]]
        q = t["question"]
        payload = {
            "state": e["prose"],
            "questions": {e["decision_id"]: {
                "type": q["type"],
                "instructions": q.get("instructions", ""),
                "criteria": q["criteria"],
            }},
        }
        req = urllib.request.Request(
            args.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {e['id']}: {exc}", file=sys.stderr)
            results.append({"id": e["id"], "decision_id": e["decision_id"],
                            "error": str(exc)})
            continue
        ans = body["answers"][e["decision_id"]]
        label = _gold_label(q["type"], e["gold_probs"][e["decision_id"]])
        model_ans = _model_answer(ans)
        if q["type"] == "boolean":
            correct = (bool(model_ans >= 0.5) == bool(label == 1.0))
        else:
            correct = (model_ans == label)
        results.append({
            "id": e["id"], "decision_id": e["decision_id"],
            "type": q["type"], "prose": e["prose"],
            "gold_label": label, "model_answer": model_ans,
            "raw": ans, "correct": correct,
        })

    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in results) + "\n")
    n = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    print(f"scored {n - sum(1 for r in results if 'error' in r)}/{n} via {args.url}")
    print(f"correct={correct} / total={sum(1 for r in results if 'correct' in r)}  ({correct/max(1,n):.0%})")
    from collections import Counter
    per = Counter(r["decision_id"] for r in results if r.get("correct"))
    tot = Counter(r["decision_id"] for r in results)
    for did in sorted(tot):
        print(f"  {did:<22} {per.get(did,0)}/{tot[did]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())