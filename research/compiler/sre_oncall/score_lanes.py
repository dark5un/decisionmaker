#!/usr/bin/env python3
"""Score the general-LLM baseline vs interpreter gold and cross-tabulate the
three lanes: {human (px, if provided), general LLM, trained head}.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

evals = [json.loads(l) for l in (HERE/"eval_labeled.json").read_text().splitlines() if l.strip()]
llm = json.loads((HERE/"llm_baseline_answers.json").read_text())
# llm list ordered id 1..28 == eval row order 0..27
ce = {json.loads(l)["id"]: json.loads(l) for l in (HERE/"eval_ce.jsonl").read_text().splitlines()}

def gold_hard(e):
    q = e["question"]["type"]
    dist = e["gold_probs"][e["decision_id"]]
    if q == "boolean":
        return "true" if dist.get("true",0) >= dist.get("false",0) else "false"
    return max(dist, key=dist.get)

def llm_answer(row):
    a = row["answer"]
    dt = row["decision_id"]
    if dt in ("severity-class", "fix-posture", "comms-escalation"):
        return a
    # boolean decisions: LLM used true/false strings
    return a.lower()

print("idx  decision_id        gold    llm    head-ce   agree(llm/head)")
llm_correct = head_correct = total = 0
for i, e in enumerate(evals):
    g = gold_hard(e)
    l = llm_answer(llm[i])
    h = ce[e["id"]]["model_answer"] if e["id"] in ce else None
    if e["question"]["type"] == "boolean":
        hc = "true" if (h is not None and h >= 0.5) else "false"
    else:
        hc = h
    total += 1
    lc = (l == g); hc_ok = (hc == g)
    llm_correct += lc; head_correct += hc_ok
    agree = "SAME" if l == hc else "diff"
    print(f"{i+1:<4} {e['decision_id']:<20} {g!s:<16} {l!s:<10} {hc!s:<10} {agree}")

print(f"\n=== SUMMARY (n={total}) ===")
print(f"general-LLM baseline: {llm_correct}/{total}  ({llm_correct/total:.0%})")
print(f"trained head (ce):    {head_correct}/{total}  ({head_correct/total:.0%})")

# per-decision
from collections import defaultdict

pc_l, pc_h, pc_t = defaultdict(int), defaultdict(int), defaultdict(int)
for i,e in enumerate(evals):
    g=gold_hard(e); l=llm_answer(llm[i])
    h=ce.get(e["id"],{}).get("model_answer")
    hc = ("true" if (h is not None and h>=0.5) else "false") if e["question"]["type"]=="boolean" else h
    pc_t[e["decision_id"]]+=1
    pc_l[e["decision_id"]]+= int(l==g); pc_h[e["decision_id"]] += int(hc==g)
print("\nper-decision  (llm / head / total):")
for d in sorted(pc_t):
    print(f"  {d:<22} {pc_l[d]:<3}/{pc_t[d]}  {pc_h[d]:<3}/{pc_t[d]}")