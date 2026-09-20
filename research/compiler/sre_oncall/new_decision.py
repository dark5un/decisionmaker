#!/usr/bin/env python3
"""Author a new boolean/choice decision for the sre-oncall corpus in one step.

You write a plain-text spec describing ONE decision; this script:

  1. builds a valid decision-table line (canonical_hash stamped) and appends it
     to tables.jsonl,
  2. AUTO-DERIVES one anchor witness state per rule (a state where the rule is
     the highest-priority match, found by running the interpreter), and appends
     those anchors to anchors.py,
  3. runs core.verify() with manual.md as the source text and prints the
     per-rule verdicts.

The anchor step is the human-in-loop part of the Gold Compiler. This generator
does it honestly: it does NOT hand-pick answers — it only finds a state in the
state space that the rule conditions actually fire, then asserts that the rule's
declared outcome is what the interpreter produces. If your rule never fires, or
fires a different outcome than you wrote, verify() will say 'contradicted'.

USAGE:
    python3 research/compiler/sre_oncall/new_decision.py spec.json
    python3 research/compiler/sre_oncall/new_decision.py -  # read spec from stdin

SPEC FORMAT (JSON):
{
  "decision_id": "force-restart",          # unique in the skill
  "question": "Should this node be force-restarted now?",
  "type": "boolean",                        # or "choice"
  "criteria": {                            # boolean: {"true":..,"false":..}
      "true":  "Force-restart now",        # choice:  {label: description}
      "false": "Hold"
  },
  "fields": ["crash_looping", "disk_full", "recent_deploy"],  # all boolean =false
  "source": "A forced restart of a node mends a hung or crash-looping process",
  "rules": [                               # evaluated in order; first match wins
    {"when": {"crash_looping": true, "disk_full": true}, "then": "false"},
    {"when": [{"disk_full": true}, {"recent_deploy": true}], "then": "false"},
        # "when" dict  = all fields AND;  list = any field OR
    {"then": "false", "default": true}     # exactly one default; condition null
  ]
}
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                      # repo root

sys.path.insert(0, str(ROOT))
from research.compiler import (
    core,
    corpus,
)


# --------------------------------------------------------------------------- #
# 1. read + validate the spec
# --------------------------------------------------------------------------- #
def load_spec(arg: str) -> dict:
    src = sys.stdin.read() if arg == "-" else Path(arg).read_text()
    spec = json.loads(src)
    need = {"decision_id", "question", "type", "criteria", "fields", "source", "rules"}
    missing = need - set(spec)
    if missing:
        raise SystemExit(f"spec missing: {sorted(missing)}")
    if spec["type"] not in ("boolean", "choice"):
        raise SystemExit(f"type must be 'boolean' or 'choice', got {spec['type']!r}")
    return spec


def _py_state(state: dict) -> str:
    """Render a state dict as a Python literal (True/False, not JSON true/false)
    so the generated anchors.py is valid Python."""
    items = ", ".join(f"\"{k}\": {('True' if v else 'False')}" for k, v in state.items())
    return "{" + items + "}"


def _condition(rule: dict):
    """'when' dict (all) / list (any) / absent (default->None)."""
    when = rule.get("when")
    if when is None:
        return None
    if isinstance(when, dict):
        return {"all": [_leaf(f, v) for f, v in when.items()]}
    if isinstance(when, list):
        return {"any": [_leaf(f, v) for item in when for f, v in item.items()]}
    raise SystemExit(f"bad 'when' (expect dict or list): {when!r}")


def _leaf(field: str, value):
    return {"field": field, "operator": "eq", "value": value}


def build_table(spec: dict) -> dict:
    fields = spec["fields"]
    # validate every field referenced in a condition exists in the schema
    for r in spec["rules"]:
        when = r.get("when") or {}
        items = when.items() if isinstance(when, dict) else \
            [kv for item in when for kv in item.items()]
        for f, _v in items:
            if f not in fields:
                raise SystemExit(
                    f"field {f!r} in a rule's 'when' is not in fields {fields!r}")
    table = {
        "skill": "sre-oncall",
        "decision_id": spec["decision_id"],
        "source": spec["source"],
        "input_schema": {
            "required": fields,
            "properties": {f: {"type": "boolean", "default": False} for f in fields},
        },
        "question": {
            "type": spec["type"],
            "instructions": spec["question"],
            "criteria": spec["criteria"],
        },
        "rules": [],
    }
    n = len(spec["rules"])
    defaults = 0
    for i, r in enumerate(spec["rules"]):
        cond = _condition(r)
        is_default = bool(r.get("default")) or cond is None
        defaults += 1 if is_default else 0
        if is_default and i != n - 1:
            raise SystemExit("the default rule must be the LAST rule")
        table["rules"].append({
            "id": f"r{i}",
            "priority": 100 - i,
            "condition": cond,
            "outcome": r["then"],
            "is_default": is_default,
        })
    if defaults != 1:
        raise SystemExit(f"need exactly ONE default rule; found {defaults}")
    # validate outcomes are criteria members
    crit = set(spec["criteria"].keys())
    for r in table["rules"]:
        if r["outcome"] not in crit:
            raise SystemExit(f"outcome {r['outcome']!r} not in criteria {sorted(crit)}")
    table["canonical_hash"] = corpus.canonical_hash(table)
    return table


# --------------------------------------------------------------------------- #
# 2. auto-derive one anchor witness state per rule (the honest part)
# --------------------------------------------------------------------------- #
def derive_anchors(spec: dict, table: dict) -> list:
    decision = core._as_decision(table)
    field_names = spec["fields"]

    # every distinct outcome a rule can fire, so we can drop unreachable rules
    seen_outcomes = {}
    for i, r in enumerate(table["rules"]):
        seen_outcomes[r["id"]] = r["outcome"], (_condition(spec["rules"][i]) is not None)

    anchors = []
    for rid, (outcome, _reachable) in seen_outcomes.items():
        # search the (bounded) state space for the first state where THIS rule
        # is the highest-priority match
        witness = _find_witness(decision, rid, field_names)
        if witness is None:
            print(f"  [warn] rule {rid}: no state in the space fires it -> rule is "
                  f"unreachable and will verify 'incomplete' unless proven default")
            continue
        anchors.append({"rule_id": rid, "state": witness, "expect": outcome})
    return anchors


def _find_witness(decision, target_rid, field_names):
    """Return a state where rules.run_rules fires target_rid and nothing higher."""
    from research.planA import rules as R
    for combo in itertools.product([True, False], repeat=len(field_names)):
        state = dict(zip(field_names, combo))
        _outcome, fired = R.run_rules(decision, state)
        if fired == target_rid:
            return state
    return None


# --------------------------------------------------------------------------- #
# 3. append to corpus + anchors, then VERIFY
# --------------------------------------------------------------------------- #
def append_and_verify(table: dict, anchors: list, manual_text: str, split: str = "train") -> int:
    did = table["decision_id"]

    # tables.jsonl
    tables_path = HERE / "tables.jsonl"
    existing = [json.loads(l) for l in tables_path.read_text().splitlines() if l.strip()]
    if any(t["decision_id"] == did for t in existing):
        print(f"  ERROR: decision_id {did!r} already in tables.jsonl; remove it first")
        return 1
    existing.append(table)
    tables_path.write_text("\n".join(json.dumps(t, ensure_ascii=False, sort_keys=True)
                                     for t in existing) + "\n")

    # anchors.py — insert an ANCHORS[<did>] block right before the closing '}'
    # of the ANCHORS dict (which is followed by blank line + 'def anchors_for').
    anchors_path = HERE / "anchors.py"
    anchors_src = anchors_path.read_text()
    block = (
        f"    # ---------------- {did} ----------------\n"
        f"    \"{did}\": [\n" +
        "".join(
            f"        {{\"rule_id\": \"{a['rule_id']}\", "
            f"\"state\": {_py_state(a['state'])}, "
            f"\"expect\": \"{a['expect']}\"}},\n"
            for a in anchors
        ) +
        "    ],\n"
    )
    marker = "}\n\n\ndef anchors_for(decision_id):"
    if marker not in anchors_src:
        raise SystemExit("anchors.py: could not find the ANCHORS closing brace marker")
    anchors_src = anchors_src.replace(marker, block + marker, 1)
    anchors_path.write_text(anchors_src)

    # verify
    rep = core.verify(table, anchors=anchors, source_text=manual_text)
    print(f"== {did} verified={rep['status']} ==")
    for rid, r in rep["per_rule"].items():
        print(f"  {rid:>4} {r['verdict']:>12} anchors_covered={r['anchors_covered']}")
    print("  (wrote table to tables.jsonl, anchors to anchors.py)")

    # write a starter renderer from the template into renderers/ (auto-discovered
    # by build.py — no RENDER edit needed) so the decision can be trained with
    # REALISTIC (not trivial) prose. The separator saves the edit instruction.
    fields = table["input_schema"]["required"]
    func = did.replace("-", "_")
    tpl = (HERE / "templates" / "renderer_template.py").read_text()
    starter = (tpl
               .replace("__DECISION__", did)
               .replace("__FIELDS__", '", "'.join(fields))
               .replace("__FUNC__", func))
    renderers_dir = HERE / "renderers"
    renderers_dir.mkdir(exist_ok=True)
    (renderers_dir / f"{func}.py").write_text(starter)

    # record the split plan for this decision so build.py trains it
    splits_path = HERE / "splits.json"
    splits = {}
    if splits_path.is_file():
        try:
            splits = json.loads(splits_path.read_text() or "{}")
        except (json.JSONDecodeError, OSError):
            splits = {}
    splits[did] = split
    splits_path.write_text(json.dumps(splits, indent=2) + "\n")
    print(f"  (wrote starter renderer: renderers/{func}.py)")
    print(f"  (recorded split '{split}' in splits.json)")

    # how to retrain with this decision
    print("\nTo train with this decision:")
    print("  python3 research/compiler/sre_oncall/build.py   # rebuild corpus")
    print("  # then run train.py on research/compiler/sre_oncall/out/train.jsonl")
    return 0 if rep["status"] == "verified" else 3


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    spec = load_spec(sys.argv[1])
    table = build_table(spec)
    anchors = derive_anchors(spec, table)
    manual_text = (HERE / "manual.md").read_text()
    return append_and_verify(table, anchors, manual_text, spec.get("split", "train"))


if __name__ == "__main__":
    sys.exit(main())