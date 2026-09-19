#!/usr/bin/env python3
"""Plan A: Compile the Prose — evidence-bound decision tables for the 3-skill
training shortlist, compiled faithfully from each SKILL.md by the (human-in-loop)
agent, then synthesized into training rows via generate_seed-style exact gold.

Each decision table matches research/plans/decision.schema.json:
  input_schema (fields) + question (type + criteria) + ordered rules
  (condition tree over fields -> outcome) + default outcome.

Gold for a synthetic state = the soft distribution induced by running the rules
against that state (matches the soft-target training recipe). Grounding: every
outcome label is a member of the question's criteria set and every rule encodes
a conditional statement present in the skill prose (source noted).

This module is the "compiler." The synthesizer (synthesize_planA) enumerates
meaningful state variations per skill, runs the rules, and emits train rows in
the exact seed.jsonl schema so train.py is unchanged.
"""
from __future__ import annotations
import re


# ---------------------------------------------------------------------------
# 1) podman-quadlet-deploy  (source: software-development/podman-quadlet-deploy)
# ---------------------------------------------------------------------------
PODMAN_RULES = {
    "skill": "podman-quadlet-deploy",
    "decisions": [
        {
            "decision_id": "is-service-working",
            "source": "SKILL.md 'A listening front-end port is not a working service' + verify-empirically-in-order",
            "input_schema": {
                "required": ["front_port_up", "backend_container_up", "logs_show_crash"],
                "properties": {
                    "front_port_up": {"type": "boolean", "default": True},
                    "backend_container_up": {"type": "boolean", "default": True},
                    "logs_show_crash": {"type": "boolean", "default": False},
                },
            },
            "question": {
                "type": "choice",
                "instructions": "Is the service actually working, or is the front port masking a dead backend?",
                "criteria": {
                    "working": "service is genuinely serving",
                    "backend_dead": "front port up but backend dead/not serving",
                    "investigate": "need to check logs/container state before judging",
                },
            },
            "default_outcome": "investigate",
            "rules": [
                {"id": "r1", "priority": 100,
                 "condition": {"all": [{"field": "front_port_up", "operator": "eq", "value": True},
                                       {"field": "backend_container_up", "operator": "eq", "value": False}]},
                 "outcome": "backend_dead"},
                {"id": "r2", "priority": 90,
                 "condition": {"all": [{"field": "front_port_up", "operator": "eq", "value": True},
                                       {"field": "backend_container_up", "operator": "eq", "value": True},
                                       {"field": "logs_show_crash", "operator": "eq", "value": False}]},
                 "outcome": "working"},
                {"id": "r0", "priority": 0, "condition": None, "outcome": "investigate", "is_default": True},
            ],
        },
        {
            "decision_id": "rootless-podman-broken",
            "source": "SKILL.md 'Rootless podman not working' -> setuid bit",
            "input_schema": {
                "required": ["newuidmap_setuid", "rootless_info"],
                "properties": {
                    "newuidmap_setuid": {"type": "boolean", "default": True},
                    "rootless_info": {"type": "string", "enum": ["true", "false"]},
                },
            },
            "question": {
                "type": "choice",
                "instructions": "Rootless podman commands are failing. What is the root cause / fix?",
                "criteria": {
                    "ok": "rootless podman works",
                    "fix_setuid": "chmod u+s /usr/bin/newuidmap /usr/bin/newgidmap",
                    "other": "some other root cause; investigate",
                },
            },
            "default_outcome": "other",
            "rules": [
                {"id": "r1", "priority": 100,
                 "condition": {"all": [{"field": "newuidmap_setuid", "operator": "eq", "value": False}]},
                 "outcome": "fix_setuid"},
                {"id": "r0", "priority": 0, "condition": None, "outcome": "ok", "is_default": True},
            ],
        },
        {
            "decision_id": "gpu-cdi-presence",
            "source": "SKILL.md 'GPU / CDI passthrough' -> toolkit/CDI required",
            "input_schema": {
                "required": ["nvidia_ctk_present", "cdi_yaml_absent"],
                "properties": {
                    "nvidia_ctk_present": {"type": "boolean", "default": False},
                    "cdi_yaml_absent": {"type": "boolean", "default": True},
                },
            },
            "question": {
                "type": "choice",
                "instructions": "A quadlet uses AddDevice=nvidia.com/gpu=all. What must be true for GPU passthrough to work?",
                "criteria": {
                    "ok": "toolkit + CDI devices present, passthrough works",
                    "install_toolkit": "install nvidia-container-toolkit and generate CDI",
                    "regenerate_cdi": "regenerate the CDI spec (nvidia-ctk cdi generate)",
                },
            },
            "default_outcome": "ok",
            "rules": [
                {"id": "r1", "priority": 100,
                 "condition": {"all": [{"field": "nvidia_ctk_present", "operator": "eq", "value": False}]},
                 "outcome": "install_toolkit"},
                {"id": "r2", "priority": 80,
                 "condition": {"all": [{"field": "nvidia_ctk_present", "operator": "eq", "value": True},
                                       {"field": "cdi_yaml_absent", "operator": "eq", "value": True}]},
                 "outcome": "regenerate_cdi"},
                {"id": "r0", "priority": 0, "condition": None, "outcome": "ok", "is_default": True},
            ],
        },
        {
            "decision_id": "gpu-uuid-pinning",
            "source": "SKILL.md 'GPU / CDI passthrough' -> durable fix (CUDA-level UUID pinning)",
            "input_schema": {
                "required": ["cdi_stale", "uuid_pinned_cuda"],
                "properties": {
                    "cdi_stale": {"type": "boolean", "default": True},
                    "uuid_pinned_cuda": {"type": "boolean", "default": False},
                },
            },
            "question": {
                "type": "choice",
                "instructions": "How should a two-GPU box pin services to the right physical card durably?",
                "criteria": {
                    "cuda_uuid": "AddDevice=gpu=all + Environment=CUDA_VISIBLE_DEVICES=GPU-<uuid>",
                    "cdi_uuid": "pin via AddDevice=nvidia.com/gpu=GPU-<uuid> in CDI",
                    "none": "no pinning needed",
                },
            },
            "default_outcome": "cuda_uuid",
            "rules": [
                {"id": "r1", "priority": 100,
                 "condition": {"all": [{"field": "cdi_stale", "operator": "eq", "value": True},
                                       {"field": "uuid_pinned_cuda", "operator": "eq", "value": False}]},
                 "outcome": "cuda_uuid"},
                {"id": "r0", "priority": 0, "condition": None, "outcome": "none", "is_default": True},
            ],
        },
        {
            "decision_id": "temp-sharpening-catch",
            "source": "SKILL.md 'Sharpening a SOFT-target head' -> compute data ceiling, don't deploy T that breaks ECE",
            "input_schema": {
                "required": ["soft_target_head", "temp_below1", "boolean_ece_worse"],
                "properties": {
                    "soft_target_head": {"type": "boolean", "default": False},
                    "temp_below1": {"type": "boolean", "default": False},
                    "boolean_ece_worse": {"type": "boolean", "default": False},
                },
            },
            "question": {
                "type": "choice",
                "instructions": "Should post-hoc temperature sharpening (T<1) be deployed for this head?",
                "criteria": {
                    "no_t_sharp": "don't deploy T that worsens ECE against real labels — train CE vs hard targets instead",
                    "deploy": "deploy the temperature",
                    "check_ceiling": "first compute the data ceiling (gold itself may be soft)",
                },
            },
            "default_outcome": "check_ceiling",
            "rules": [
                {"id": "r1", "priority": 100,
                 "condition": {"all": [{"field": "temp_below1", "operator": "eq", "value": True},
                                       {"field": "boolean_ece_worse", "operator": "eq", "value": True}]},
                 "outcome": "no_t_sharp"},
                {"id": "r0", "priority": 0, "condition": None, "outcome": "check_ceiling", "is_default": True},
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# 2) systematic-debugging  (source: software-development/systematic-debugging)
# ---------------------------------------------------------------------------
DEBUGGING_RULES = {
    "skill": "systematic-debugging",
    "decisions": [
        {
            "decision_id": "root-cause-before-fix",
            "source": "SKILL.md 'The Iron Law': NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST",
            "input_schema": {
                "required": ["root_cause_known"],
                "properties": {"root_cause_known": {"type": "boolean", "default": False}},
            },
            "question": {
                "type": "boolean",
                "instructions": "A fix is about to be proposed. May we propose it now?",
                "criteria": {"false": "no — finish root-cause investigation first", "true": "yes — root cause is known"},
            },
            "default_outcome": "false",
            "rules": [
                {"id": "r1", "priority": 100,
                 "condition": {"all": [{"field": "root_cause_known", "operator": "eq", "value": True}]},
                 "outcome": "true"},
                {"id": "r0", "priority": 0, "condition": None, "outcome": "false", "is_default": True},
            ],
        },
        {
            "decision_id": "tight-loop-before-theory",
            "source": "SKILL.md 'The Feedback Loop Rule': before reading code to build a theory, create an identifiable tight command that goes red",
            "input_schema": {
                "required": ["tight_loop_exists"],
                "properties": {"tight_loop_exists": {"type": "boolean", "default": False}},
            },
            "question": {
                "type": "boolean",
                "instructions": "Should you build a theory about the bug before establishing a tight red-capable loop?",
                "criteria": {"false": "no — build/identify the tight feedback loop first", "true": "yes — theory first is fine"},
            },
            "default_outcome": "false",
            "rules": [
                {"id": "r1", "priority": 100,
                 "condition": {"all": [{"field": "tight_loop_exists", "operator": "eq", "value": True}]},
                 "outcome": "true"},
                {"id": "r0", "priority": 0, "condition": None, "outcome": "false", "is_default": True},
            ],
        },
        {
            "decision_id": "rule-of-three",
            "source": "SKILL.md 'Rule of Three': if >=3 fixes failed, STOP and question the architecture",
            "input_schema": {
                "required": ["fixes_failed"],
                "properties": {"fixes_failed": {"type": "number", "default": 0}},
            },
            "question": {
                "type": "choice",
                "instructions": "Several fixes have failed. What should happen next?",
                "criteria": {
                    "question_arch": "STOP and question the architecture / fundamentals",
                    "return_phase1": "return to Phase 1 re-analysis with new info",
                    "keep_fixing": "try another fix",
                },
            },
            "default_outcome": "return_phase1",
            "rules": [
                {"id": "r1", "priority": 100,
                 "condition": {"all": [{"field": "fixes_failed", "operator": "gte", "value": 3}]},
                 "outcome": "question_arch"},
                {"id": "r2", "priority": 40,
                 "condition": {"all": [{"field": "fixes_failed", "operator": "lt", "value": 3}]},
                 "outcome": "return_phase1"},
                {"id": "r0", "priority": 0, "condition": None, "outcome": "return_phase1", "is_default": True},
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# 3) ryoku-desktop-ops  (source: ryoku-desktop-ops)
# ---------------------------------------------------------------------------
RYOKU_RULES = {
    "skill": "ryoku-desktop-ops",
    "decisions": [
        {
            "decision_id": "idle-pipeline-running",
            "source": "SKILL.md pitfall: ryoku-idle start only launches hypridle on laptops (device=desktop disabled)",
            "input_schema": {
                "required": ["is_desktop", "hypridle_running"],
                "properties": {
                    "is_desktop": {"type": "boolean", "default": True},
                    "hypridle_running": {"type": "boolean", "default": False},
                },
            },
            "question": {
                "type": "choice",
                "instructions": "Idle/lock/screensaver don't run on this tower. Correct approach?",
                "criteria": {
                    "user_unit_hypridle": "run hypridle -c <conf> from a user systemd unit (ryoku-idle start is laptop-only)",
                    "ryoku_idle_start": "use ryoku-idle start",
                    "config_edit": "edit the ship-laid hypridle config directly",
                },
            },
            "default_outcome": "ryoku_idle_start",
            "rules": [
                {"id": "r1", "priority": 100,
                 "condition": {"all": [{"field": "is_desktop", "operator": "eq", "value": True},
                                       {"field": "hypridle_running", "operator": "eq", "value": False}]},
                 "outcome": "user_unit_hypridle"},
                {"id": "r0", "priority": 0, "condition": None, "outcome": "ryoku_idle_start", "is_default": True},
            ],
        },
        {
            "decision_id": "dpms-wake-keys",
            "source": "SKILL.md pitfall: misc key_press/mouse_move_enables_dpms default false -> black lock after DPMS",
            "input_schema": {
                "required": ["dpms_wake_keys_true"],
                "properties": {"dpms_wake_keys_true": {"type": "boolean", "default": False}},
            },
            "question": {
                "type": "boolean",
                "instructions": "Screen wakes black after DPMS-off + input. Should key_press/mouse_move_enables_dpms be set true?",
                "criteria": {"false": "no — leave defaults", "true": "yes — set both true in user.lua"},
            },
            "default_outcome": "false",
            "rules": [
                {"id": "r1", "priority": 100,
                 "condition": {"all": [{"field": "dpms_wake_keys_true", "operator": "eq", "value": False}]},
                 "outcome": "true"},
                {"id": "r0", "priority": 0, "condition": None, "outcome": "false", "is_default": True},
            ],
        },
    ],
}

ALL = {"podman-quadlet-deploy": PODMAN_RULES, "systematic-debugging": DEBUGGING_RULES,
       "ryoku-desktop-ops": RYOKU_RULES}


def evaluate_condition(cond, state):
    if cond is None:
        return True
    if "all" in cond:
        return all(evaluate_condition(c, state) for c in cond["all"])
    if "any" in cond:
        return any(evaluate_condition(c, state) for c in cond["any"])
    if "not" in cond:
        return not evaluate_condition(cond["not"], state)
    field, op, value = cond["field"], cond["operator"], cond["value"]
    actual = state.get(field)
    try:
        if op == "eq": return actual == value
        if op == "ne": return actual != value
        if op == "gt": return actual > value
        if op == "gte": return actual >= value
        if op == "lt": return actual < value
        if op == "lte": return actual <= value
        if op == "in": return actual in value
    except TypeError:
        return False
    return False


def run_rules(decision, state):
    """Return (matched_outcome, which_rule) by descending priority, else default."""
    for rule in sorted(decision["rules"], key=lambda r: r["priority"], reverse=True):
        if evaluate_condition(rule["condition"], state):
            return rule["outcome"], rule["id"]
    return decision["default_outcome"], "default"


def outcome_distribution(decision, state, conf=0.90):
    """Map an outcome to a soft gold distribution over the criteria (exact, so the
    soft-target trainer has a deterministic target). Winner gets 'conf', the rest
    share 1-conf - winner may also be default. Deterministic."""
    criteria = decision["question"]["criteria"]
    if decision["question"]["type"] == "boolean":
        keys = ["false", "true"]
    elif decision["question"]["type"] == "choice":
        keys = list(criteria.keys())
    else:  # score
        keys = [str(i) for i in range(len(criteria))]
    outcome, _ = run_rules(decision, state)
    winner = outcome
    dist = {k: 0.0 for k in keys}
    # winner key must be in criteria keys; if not, fall back to default outcome
    if winner not in keys and decision["question"]["type"] in ("score", "choice"):
        winner = decision["default_outcome"]
    if winner not in keys:
        winner = keys[0]
    remainder = 1.0 - conf
    n_others = len(keys) - 1
    dist[winner] = conf
    if n_others > 0:
        for k in keys:
            if k != winner:
                dist[k] = remainder / n_others
    return dist, winner


if __name__ == "__main__":
    # sanity: each skill decision runs on a set of representative states
    import json
    total_q = 0
    for skill, pkg in ALL.items():
        print(f"\n== {skill} ==")
        for d in pkg["decisions"]:
            total_q += 1
            schema = d["input_schema"]
            # build a few representative states
            states = []
            if d["decision_id"] == "rule-of-three":
                states = [{"fixes_failed": 0}, {"fixes_failed": 2}, {"fixes_failed": 3}, {"fixes_failed": 5}]
            else:
                reps = []
                for f, meta in schema["properties"].items():
                    if meta["type"] == "boolean":
                        reps.append({f: True}); reps.append({f: False})
                for rep in reps:
                    st = {g: (True if schema["properties"][g]["type"] == "boolean" else
                              (schema["properties"][g].get("enum", ["true"])[0] if g == "rootless_info" else 0))
                          for g in schema["required"]}
                    st.update(rep)
                    states.append(st)
            for st in states:
                dist, winner = outcome_distribution(d, st)
                # validate winner in criteria for choice
                print(f"  {d['decision_id']}: {st} -> {winner} dist={ {k: round(v,2) for k,v in dist.items()} }")
    print(f"\nTotal decision tables: {total_q}")
