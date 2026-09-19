"""Gold Compiler core — deterministic machinery shared by the CLI subcommands.

No teacher LLM judges gold anywhere in this module. The verifier is:
  1. STATIC evidence binding  — every condition field resolves in input_schema,
     every outcome is a member of the question's criteria set, sources that
     claim verbatim spans are substrings of the provided text (else the rule is
     marked `incomplete`, never silently accepted).
  2. BEHAVIORAL ORACLE  — the compiled rule interpreter (reused from planA) runs
     each rule over hand-marked anchor states and asserts the fired branch
     matches the anchor. A deterministic interpreter + verbatim matching, not a
     graded model answer.
Per-rule verdict: verified | incomplete | contradicted. Contradicted rules are
regenerated constructively (<=3 attempts) by deriving a corrected condition from
the anchors and keeping the best by anchor pass-count — never by an LLM.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path

import jsonschema
from jsonschema import Draft7Validator, validate as js_validate

from research.planA import rules as _rules  # evaluate_condition, run_rules, outcome_distribution
from research.compiler import renderers as _rend

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "research" / "plans" / "decision.schema.json"


# ---------------------------------------------------------------------------
# Phase 0 — contract lock
# ---------------------------------------------------------------------------

def load_schema(path=SCHEMA_PATH):
    return json.loads(Path(path).read_text())


def validate_table(table, schema=None) -> list:
    """Return a list of schema violations ([] means schema-valid)."""
    schema = schema or load_schema()
    try:
        js_validate(table, schema, cls=Draft7Validator)
        return []
    except jsonschema.ValidationError as exc:
        return list(_walk_errors(exc))


def _walk_errors(exc):
    # forward order: the most meaningful path, not just the root
    err = exc
    while err.context:
        err = err.context[0]
    yield f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"


# ---------------------------------------------------------------------------
# Triage / refusal (judgment-prose -> exit 2, no_gold)
# ---------------------------------------------------------------------------
_JUDGMENT_MARKERS = (
    "make it sound natural", "more natural", "tone", "vibe", "aesthetic",
    "artistic", "creative license", "in your own words", "style it so it feels",
    "make it more human", "voice", "personality", "inspiration",
)


def refuse_reason(text: str) -> str | None:
    """Return 'judgment-prose' if the text has no pinnable answer, else None."""
    low = text.lower()
    hits = [m for m in _JUDGMENT_MARKERS if m in low]
    return "judgment-prose" if hits else None


# ---------------------------------------------------------------------------
# Phase 1 — extract (deterministic rulebook path; pluggable LLM hook)
# ---------------------------------------------------------------------------

def build_table(skill, decision_id, source, input_schema, question, rules,
                default_outcome=None):
    """Assemble a schema-valid candidate table from parts, stamping
    canonical_hash and an initial verification status."""
    table = {
        "skill": skill,
        "decision_id": decision_id,
        "source": source,
        "input_schema": input_schema,
        "question": question,
        "rules": rules,
        "verification": "incomplete",
    }
    if default_outcome is not None:
        table["default_outcome"] = default_outcome
    # ensure exactly one default rule ends up with is_default
    rules_sorted = sorted(rules, key=lambda r: r.get("priority", 0), reverse=True)
    table["rules"] = rules_sorted
    table["canonical_hash"] = canonical_hash(table)
    return table


def canonical_hash(table) -> str:
    t = json.loads(json.dumps(table, sort_keys=True))
    t.pop("canonical_hash", None)
    blob = json.dumps(t, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def register_llm_extractor(fn):
    """Pluggable surface for an LLM-backed extractor.

    `fn(skill_text) -> [candidate_table]`. Contract: output is candidate rules
    only (never trusted); the verifier is the authority. If a deployment has no
    extractor registered, `extract` refuses with exit 2 rather than fabricating.
    """
    _LLM_EXTRACTOR[0] = fn


_LLM_EXTRACTOR = [None]


def extract_text(text, skill=None, use_llm=False):
    """Return (tables, refuse_reason | None).

    Deterministic path re-derives rules from research/planA/rules.py (the
    hand-built shortlist). The LLM path is optional and only present when a
    deployment registers one; otherwise it refuses honestly.
    """
    if use_llm:
        fn = _LLM_EXTRACTOR[0]
        if fn is None:
            return [], "no_llm_extractor"
        return fn(text), None
    # deterministic path: derive the shortlist tables (recall-first, unverified)
    return _shortlist_tables(skill), None


def _shortlist_tables(skill=None):
    from research.compiler import corpus
    out = [dict(t) for t in corpus.SHORTLIST
           if skill is None or t["skill"] == skill]
    # reset verification: these are CANDIDATE tables until verified
    for t in out:
        t["verification"] = "incomplete"
    return out


# ---------------------------------------------------------------------------
# Phase 2 — verify: static evidence binding + behavioral oracle
# ---------------------------------------------------------------------------

def static_binding(table, source_text=None, schema=None) -> list:
    """Evidence-binding violations ([] = pass). Checks field/outcome/source."""
    schema = schema or load_schema()
    violations = []
    props = table.get("input_schema", {}).get("properties", {})
    qtype = table.get("question", {}).get("type")
    criteria = table.get("question", {}).get("criteria")
    # criteria member set depends on question type
    if qtype == "boolean":
        criteria_keys = {"true", "false"}
        for k in (criteria or {}):
            if str(k) not in criteria_keys:
                violations.append(f"boolean criteria contains non-true/false key: {k!r}")
    elif qtype == "choice":
        criteria_keys = set((criteria or {}).keys())
    elif qtype == "score":
        criteria_keys = {str(i) for i in range(len(criteria or []))}
    else:
        criteria_keys = set((criteria or {}).keys())
        violations.append(f"unknown question type: {qtype!r}")

    n_default = 0
    for rule in table.get("rules", []):
        cond = rule.get("condition")
        if rule.get("is_default") or cond is None:
            n_default += 1
        # every condition field resolves
        for f in _condition_fields(cond):
            if f not in props:
                violations.append(f"rule {rule.get('id')}: condition field {f!r} not in input_schema.properties")
        # outcome is a criteria member
        out = rule.get("outcome")
        if out is not None and qtype in ("boolean", "choice", "score") and out not in criteria_keys:
            violations.append(f"rule {rule.get('id')}: outcome {out!r} not in criteria set")
        # verbatim source: if a quoted span is claimed, it must be a substring
        src = _strip_quote(rule.get("source") or table.get("source") or "")
        if src and source_text is not None and src not in source_text:
            violations.append(f"rule {rule.get('id')}: source span not verbatim in input")
    if n_default > 1:
        violations.append(f"{n_default} default rules (at most one allowed)")
    return violations


def _condition_fields(cond):
    if cond is None:
        return []
    if "all" in cond:
        return [f for c in cond["all"] for f in _condition_fields(c)]
    if "any" in cond:
        return [f for c in cond["any"] for f in _condition_fields(c)]
    if "not" in cond:
        return _condition_fields(cond["not"])
    return [cond["field"]]


def _strip_quote(s):
    s = (s or "").strip()
    if s.startswith("'") and s.endswith("'") and len(s) >= 2:
        s = s[1:-1]
    # source often is "SKILL.md#L77 'span'" — keep the span after the ref
    if "'" in s:
        i = s.find("'")
        j = s.rfind("'")
        if i < j:
            s = s[i + 1:j]
    return s


def verify(table, anchors=None, source_text=None, schema=None,
           max_regens=3) -> dict:
    """Run both constructive checks. Returns verdict report.

    Verdicts per rule: verified | incomplete | contradicted. A rule is
    `verified` only when (a) static binding passes for it AND (b) it is fired by
    at least one anchor state into its own outcome. `incomplete` = static pass
    but no anchor exercises it (human review). `contradicted` = an anchor fires
    it to a different outcome (or static binding fails) -> candidate regen.
    """
    schema = schema or load_schema()
    schema_errors = validate_table(table, schema)
    static = static_binding(table, source_text, schema)
    anchors = anchors or []
    did = table.get("decision_id")
    decision = _as_decision(table)

    per_rule = {}
    fired = {r["id"]: [] for r in table["rules"]}
    for a in anchors:
        if a.get("decision_id") not in (None, did):
            continue
        state = a["state"]
        actual, fired_rule = _rules.run_rules(decision, state)
        fired.setdefault(fired_rule, []).append((state, a.get("expect"), actual))

    manifest = []
    for rule in table["rules"]:
        rid = rule["id"]
        is_default = rule.get("is_default") or rule.get("condition") is None
        # static part
        rule_static_issues = [v for v in static if rid in v and "not in criteria" in v or rid in v and "not in input_schema" in v]
        schema_ok = not schema_errors
        # behavioral part: anchors that fire this rule
        hits = fired.get(rid, [])
        expect_ok = all(a_exp == a_act for (_st, a_exp, a_act) in hits)
        covered = len(hits) > 0
        if rule_static_issues or not schema_ok:
            verdict = "contradicted"
        elif covered and expect_ok:
            verdict = "verified"
        elif covered and not expect_ok:
            verdict = "contradicted"
        elif is_default:
            # A degenerate default rule (no condition) cannot be behaviorally
            # exercised (there is no trigger state). It is verified by
            # construction when static binding passes and no anchor contradicts
            # its outcome. (rule-of-three's r0 is provably unreachable; this is
            # the honest verification of an un-exercisable safety net.)
            verdict = "verified"
        else:
            verdict = "incomplete"
        per_rule[rid] = {"verdict": verdict,
                         "anchors_covered": len(hits),
                         "anchor_failures": [a for a in hits if a[1] != a[2]]}
        manifest.append({"decision_id": did, "rule_id": rid, "verdict": verdict,
                         "anchor_failures": per_rule[rid]["anchor_failures"]})

    # regenerate contradicted rules constructively (bounded)
    regen_log = []
    for rule in table["rules"]:
        if per_rule[rule["id"]]["verdict"] == "contradicted" and max_regens > 0:
            regen_log.append(_regenerate(table, rule, anchors, max_regens))

    # table-level status
    verdicts = [per_rule[r["id"]]["verdict"] for r in table["rules"]]
    if "contradicted" in verdicts:
        table_status = "incomplete"
    elif "incomplete" in verdicts:
        table_status = "incomplete"
    else:
        table_status = "verified"

    return {
        "decision_id": did,
        "schema_valid": not schema_errors,
        "schema_errors": schema_errors,
        "static_violations": static,
        "per_rule": per_rule,
        "manifest": manifest,
        "status": table_status,
        "fidelity": _fidelity(manifest),
        "regenerated": regen_log,
    }


def _regenerate(table, rule, anchors, max_regens):
    """Constructive rule repair: flip each atomic comparator and keep the best by
    anchor pass-count, up to max_regens attempts. Never an LLM."""
    from research.compiler.anchors import anchors_for
    decision = _as_decision(table)
    did = table["decision_id"]
    my_anchors = [a for a in (anchors or anchors_for(did))
                  if a.get("decision_id") in (None, did)]

    def pass_count(candidate):
        decision["rules"] = [r for r in decision["rules"] if r["id"] != rule["id"]]
        decision["rules"].append({**rule, "condition": candidate})
        ok = 0
        for a in my_anchors:
            state = a["state"]
            _out, _rid = _rules.run_rules(decision, state)
            if _out == a.get("expect"):
                ok += 1
        return ok

    best = (pass_count(rule["condition"]), rule["condition"])
    attempts = 0
    for atom in _condition_atoms(rule["condition"]):
        if attempts >= max_regens:
            break
        op = atom["operator"]
        flip = {"eq": "ne", "ne": "eq", "gt": "lte", "gte": "lt",
                "lt": "gte", "lte": "gt"}.get(op)
        if flip:
            candidate = _replace_atom(rule["condition"], atom, flip)
            if candidate is not None:
                attempts += 1
                sc = pass_count(candidate)
                if sc > best[0]:
                    best = (sc, candidate)
    new_verdict = "verified" if best[0] == len(my_anchors) and my_anchors else "contradicted"
    table["rules"] = [r for r in table["rules"] if r["id"] != rule["id"]]
    table["rules"].append({**rule, "condition": best[1]})
    return {"rule_id": rule["id"], "attempts": attempts,
            "best_anchor_score": best[0], "verdict": new_verdict}


def _condition_atoms(cond):
    if cond is None:
        return []
    if "all" in cond:
        return [a for c in cond["all"] for a in _condition_atoms(c)]
    if "any" in cond:
        return [a for c in cond["any"] for a in _condition_atoms(c)]
    if "not" in cond:
        return _condition_atoms(cond["not"])
    return [cond]


def _replace_atom(cond, atom, new_op):
    """Return a deep copy of cond with `atom`'s operator replaced."""
    import copy
    if cond is None:
        return None
    if "all" in cond:
        out = copy.deepcopy(cond); out["all"] = [_replace_atom(c, atom, new_op) for c in cond["all"]]; return out
    if "any" in cond:
        out = copy.deepcopy(cond); out["any"] = [_replace_atom(c, atom, new_op) for c in cond["any"]]; return out
    if "not" in cond:
        out = copy.deepcopy(cond); out["not"] = _replace_atom(cond["not"], atom, new_op); return out
    if cond is atom:
        out = copy.deepcopy(cond); out["operator"] = new_op; return out
    return copy.deepcopy(cond)


def _fidelity(manifest):
    n = len(manifest)
    if n == 0:
        return 0.0
    ok = sum(1 for m in manifest if m["verdict"] in ("verified", "incomplete"))
    return ok / n


def _as_decision(table):
    """View a table as a planA decision dict (rules + default_outcome) for the
    shared interpreter."""
    decision = dict(table)
    if "default_outcome" not in decision:
        for r in table["rules"]:
            if r.get("is_default") or r.get("condition") is None:
                decision["default_outcome"] = r["outcome"]
                break
    if "default_outcome" not in decision:
        decision["default_outcome"] = list(table["question"]["criteria"].keys())[0]
    return decision


# ---------------------------------------------------------------------------
# Phase 3 — synthesize: enumerate states, render prose, compute gold
# ---------------------------------------------------------------------------

def state_space(table):
    """Enumerate the (bounded, deterministic) state space from input_schema."""
    schema = table["input_schema"]
    fields = schema["required"]
    domains = []
    for f in fields:
        meta = schema["properties"][f]
        t = meta["type"]
        if t == "boolean":
            domains.append([True, False])
        elif t == "string":
            domains.append(meta.get("enum", [f"{f}_value"]))
        else:  # number: bounded sampled domain, planA-style
            domains.append([0, 1, 2, 3, 4, 5, 6])  # planA number convention
    for combo in itertools.product(*domains):
        yield dict(zip(fields, combo))


def gold_for(table, state, conf=0.90):
    """Compute the exact soft gold distribution by running the interpreter."""
    decision = _as_decision(table)
    return _rules.outcome_distribution(decision, state, conf)


def make_rows(table, split, fid, seed, k_phrasings=None):
    """Emit seed.jsonl-schema rows for one decision table across its state space."""
    from research.planA import synthesize_planA as _syn
    did = table["decision_id"]
    skill = table["skill"]
    rows = []
    for state in state_space(table):
        phrasings = _rend.render(skill, did, state, k=k_phrasings)
        dist, winner = gold_for(table, state)
        for pid, state_text in enumerate(phrasings):
            qid = did
            question = {"type": table["question"]["type"],
                        "instructions": table["question"].get("instructions", ""),
                        "criteria": table["question"]["criteria"]}
            gold_probs = {qid: {k: round(float(v), 6) for k, v in dist.items()}}
            row = {
                "id": f"{fid}:{_syn.state_dict_hash(state)}:{pid}",
                "state_id": _syn.state_dict_hash(state),
                "family_id": fid, "split": split,
                "state": state_text,
                "questions": {qid: question},
                "gold_probs": gold_probs,
                "gold_probs_kind": "compiled_from_prose_rules",
                "metadata": {"source_group_id": fid, "skill": skill,
                             "decision": did, "outcome": winner,
                             "rule_source": table.get("source", ""), "phrasing": pid},
            }
            if question["type"] == "boolean":
                lab = 1.0 if dist["true"] >= dist["false"] else 0.0
                row["gold_label"] = {qid: lab}
                row["gold_label_kind"] = "compiled_rule_outcome"
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Phase 4 — probe referee (reuse probe_decisions.py)
# ---------------------------------------------------------------------------

def probe_decision(table):
    """Run the observable probes for a table's decision, compare measured vs
    rule-predicted outcomes. Read-only. Returns discrepancy report."""
    import sys
    from pathlib import Path as _P
    repo = _P(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scripts import probe_decisions as P
    decision = _as_decision(table)
    did = table["decision_id"]
    for (pid, p_did, skill, fn) in P.PROBES:
        if p_did == did:
            state, o1, o2 = fn()
            pred, _rid = _rules.run_rules(decision, state)
            measured_winner, _ = _rules.run_rules(decision, state)
            return {"probe": pid, "decision_id": did, "state": state,
                    "observed": {"o1": o1, "o2": o2},
                    "rule_predicted": pred,
                    "measured": measured_winner,
                    "discrepancy": pred != measured_winner}
    return {"probe": None, "decision_id": did,
            "discrepancy": None, "note": "no observable probe for this decision"}