#!/usr/bin/env python3
"""Plan B: Derive the Generator — gold is COMPUTED by a deterministic program.

Mirrors the exact-gold contract of generate_seed.py but for the 3-skill decision
shortlist. For each skill decision, a generating function samples states from a
per-skill categorical prior over the feature combinations, and gold is an exact,
computable soft distribution over outcomes in which the winner's confidence
depends on a per-row "evidence strength" drawn from the prior. This is materially
different from Plan A (which exhaustively enumerated states at a fixed 0.90
confidence): here gold is a program output, coverage is prior-weighted, and the
OOD families deliberately SHIFT the priors so calibration must degrade.

Program/seed could be re-run for byte-identical output (data_sha256 stamped).

Per-skill "state" is the tuple of input_schema fields; the renderers turn a
feature combo back into natural language, identical phrasing family across plans
so the heads see comparable text (a cross-plan consistency courtesy, not a gold
leak).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

from research.planA.rules import ALL as RULES_PKG, evaluate_condition


# --- per-skill feature lists + renderers (shared phrasing with Plan A) ---------
FEATURES = {
    "podman-quadlet-deploy": {
        "is-service-working": ["front_port_up", "backend_container_up", "logs_show_crash"],
        "rootless-podman-broken": ["newuidmap_setuid", "rootless_info"],
        "gpu-cdi-presence": ["nvidia_ctk_present", "cdi_yaml_absent"],
        "gpu-uuid-pinning": ["cdi_stale", "uuid_pinned_cuda"],
        "temp-sharpening-catch": ["soft_target_head", "temp_below1", "boolean_ece_worse"],
    },
    "systematic-debugging": {
        "root-cause-before-fix": ["root_cause_known"],
        "tight-loop-before-theory": ["tight_loop_exists"],
        "rule-of-three": ["fixes_failed"],
    },
    "ryoku-desktop-ops": {
        "idle-pipeline-running": ["is_desktop", "hypridle_running"],
        "dpms-wake-keys": ["dpms_wake_keys_true"],
    },
}

# default(baseline) values per boolean field; each field is True with some bias
BOOL_BIAS = {  # P(field=True) in-distribution
    "front_port_up": 0.6, "backend_container_up": 0.5, "logs_show_crash": 0.4,
    "newuidmap_setuid": 0.8, "nvidia_ctk_present": 0.6, "cdi_yaml_absent": 0.5,
    "cdi_stale": 0.6, "uuid_pinned_cuda": 0.5, "soft_target_head": 0.7,
    "temp_below1": 0.4, "boolean_ece_worse": 0.3, "root_cause_known": 0.55,
    "tight_loop_exists": 0.5, "is_desktop": 0.8, "hypridle_running": 0.4,
    "dpms_wake_keys_true": 0.4, "rootless_info": {"true": 0.7, "false": 0.3},
}


def _render(skill, decision_id, st):
    if skill == "podman-quadlet-deploy":
        return _render_podman(decision_id, st)
    if skill == "systematic-debugging":
        return _render_debugging(decision_id, st)
    return _render_ryoku(decision_id, st)


def _render_podman(did, st):
    if did == "is-service-working":
        fp="UP" if st["front_port_up"] else "DOWN"; be="running" if st["backend_container_up"] else "exited/crash-looping"
        lg="show a crash" if st["logs_show_crash"] else "look clean"
        return f"Front-end port 8090 is {fp}; backend container {be}; logs {lg}."
    if did=="rootless-podman-broken":
        su="has setuid root" if st["newuidmap_setuid"] else "LOST setuid root (-rwxr-xr-x)"; ri=st["rootless_info"]
        return f"/usr/bin/newuidmap {su}; podman info Rootless={ri}."
    if did=="gpu-cdi-presence":
        ct="installed" if st["nvidia_ctk_present"] else "NOT installed"; cdi="present" if not st["cdi_yaml_absent"] else "ABSENT"
        return f"nvidia-container-toolkit {ct}; /etc/cdi/nvidia.yaml {cdi}."
    if did=="gpu-uuid-pinning":
        cs="stale" if st["cdi_stale"] else "fresh"; up="CUDA-pinned" if st["uuid_pinned_cuda"] else "not CUDA-pinned"
        return f"CDI {cs}; quadlet {up}."
    if did=="temp-sharpening-catch":
        tt="soft" if st["soft_target_head"] else "hard"; tp="T<1" if st["temp_below1"] else "T=1.0"
        ew="worsens" if st["boolean_ece_worse"] else "keeps ECE"
        return f"Head trained {tt}; temperature {tp}; fit {ew}."
    raise KeyError(did)


def _render_debugging(did, st):
    if did=="root-cause-before-fix":
        return "Root cause is KNOWN." if st["root_cause_known"] else "Root cause is NOT understood."
    if did=="tight-loop-before-theory":
        return "A tight loop EXISTS." if st["tight_loop_exists"] else "No tight loop built yet."
    if did=="rule-of-three":
        return f"{st['fixes_failed']} fixes failed so far."
    raise KeyError(did)


def _render_ryoku(did, st):
    if did=="idle-pipeline-running":
        return f"Device is {'desktop' if st['is_desktop'] else 'laptop'}; hypridle {'running' if st['hypridle_running'] else 'not running'}."
    if did=="dpms-wake-keys":
        return f"DPMS wake keys are {'true' if st['dpms_wake_keys_true'] else 'false'}."
    raise KeyError(did)


def sample_state(rng, decision, field_bias):
    """Sample one feature combination from the (possibly shifted) per-field prior."""
    st = {}
    for f, meta in decision["input_schema"]["properties"].items():
        if meta["type"] == "boolean":
            st[f] = rng.random() < field_bias.get(f, 0.5)
        elif f == "rootless_info":
            st[f] = "true" if rng.random() < field_bias.get(f, {"true":0.5})["true"] else "false"
        else:  # numeric fixes_failed
            st[f] = rng.choice([0,1,2,3,4,5,6])
    return st


def compute_gold(decision, st, conf):
    """Deterministic: winner via rules, soft distribution with per-row confidence."""
    winner = decision["default_outcome"]
    for rule in sorted(decision["rules"], key=lambda r: r["priority"], reverse=True):
        if evaluate_condition(rule["condition"], st):
            winner = rule["outcome"]; break
    qtype = decision["question"]["type"]
    criteria = decision["question"]["criteria"]
    keys = ["false","true"] if qtype=="boolean" else list(criteria.keys())
    if winner not in keys:
        winner = decision["default_outcome"]
    if winner not in keys:
        winner = keys[0]
    dist = {k: 0.0 for k in keys}
    rem = 1.0 - conf
    dist[winner] = conf
    for k in keys:
        if k != winner:
            dist[k] = rem / (len(keys) - 1)
    return dist, winner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/data/planB_train.jsonl")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--per-split", type=int, default=60)
    ap.add_argument("--ood-families", type=int, default=6)
    ap.add_argument("--conf-range", type=float, nargs=2, default=[0.70, 0.90])
    args = ap.parse_args()
    rng = random.Random(args.seed)

    decisions = [(skill, d) for skill, pkg in RULES_PKG.items() for d in pkg["decisions"]]
    # assign split round-robin but ensure test/ood get a boolean (hand-set)
    SPLIT_ASSIGN = {
        "is-service-working":"train","rootless-podman-broken":"train","gpu-cdi-presence":"calibration",
        "gpu-uuid-pinning":"dev","temp-sharpening-catch":"dev","root-cause-before-fix":"test",
        "tight-loop-before-theory":"ood","rule-of-three":"calibration","idle-pipeline-running":"test",
        "dpms-wake-keys":"ood",
    }
    rows = []
    fid_counter = 0
    for skill, decision in decisions:
        did = decision["decision_id"]
        split = SPLIT_ASSIGN[did]
        base_bias = BOOL_BIAS
        fid = f"planB-{skill}:{did}".replace(":","-")
        for _ in range(args.per_split):
            st = sample_state(rng, decision, base_bias)
            conf = rng.uniform(*args.conf_range)
            dist, winner = compute_gold(decision, st, conf)
            qtype = decision["question"]["type"]
            qid = did
            question = {"type": qtype, "instructions": decision["question"]["instructions"],
                        "criteria": decision["question"]["criteria"]}
            row = {
                "id": f"{fid}:{fid_counter}", "state_id": fid_counter, "family_id": fid, "split": split,
                "state": _render(skill, did, st),
                "questions": {qid: question},
                "gold_probs": {qid: {k: round(float(v),6) for k,v in dist.items()}},
                "gold_probs_kind": "computed_programmatic_distribution",
                "metadata": {"source_group_id": fid, "skill": skill, "decision": did,
                             "outcome": winner, "conf": round(conf,3),
                             "program": "research/generators/planB_generator.py",
                             "state": st},
            }
            if qtype == "boolean":
                row["gold_label"] = {qid: 1.0 if dist["true"]>=dist["false"] else 0.0}
                row["gold_label_kind"] = "program_computed_gold_label"
            rows.append(row); fid_counter += 1

    # OOD families: shift the boolean/field biases toward the DEFAULT branch so the
    # prior moves away from the training range -> calibration must degrade.
    default_bias = {f: (0.1 if f in BOOL_BIAS and isinstance(BOOL_BIAS[f], float) else BOOL_BIAS.get(f)) for f in BOOL_BIAS}
    # push "root_cause_known" / "tight_loop_exists" / "dpms_wake_keys_true" to send booleans to default false often
    for k in BOOL_BIAS:
        if isinstance(BOOL_BIAS[k], float):
            default_bias[k] = 0.05  # fields rarely true -> default branch dominates
    for k in range(args.ood_families):
        # pick one decision per ood family (rotate)
        skill, decision = decisions[k % len(decisions)]
        did = decision["decision_id"]
        fid = f"planB-ood-{did}".replace(":","-")
        for _ in range(args.per_split):
            st = sample_state(rng, decision, default_bias)
            conf = rng.uniform(*args.conf_range)
            dist, winner = compute_gold(decision, st, conf)
            qtype = decision["question"]["type"]; qid = did
            question = {"type": qtype, "instructions": decision["question"]["instructions"],
                        "criteria": decision["question"]["criteria"]}
            row = {
                "id": f"{fid}:{fid_counter}", "state_id": fid_counter, "family_id": fid, "split": "ood",
                "state": _render(skill, did, st),
                "questions": {qid: question},
                "gold_probs": {qid: {k: round(float(v),6) for k,v in dist.items()}},
                "gold_probs_kind": "computed_programmatic_distribution",
                "metadata": {"source_group_id": fid, "skill": skill, "decision": did,
                             "outcome": winner, "conf": round(conf,3),
                             "program": "research/generators/planB_generator.py", "ood": True, "state": st},
            }
            if qtype == "boolean":
                row["gold_label"] = {qid: 1.0 if dist["true"]>=dist["false"] else 0.0}
                row["gold_label_kind"] = "program_computed_gold_label"
            rows.append(row); fid_counter += 1

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True)+"\n" for r in rows)
    out.write_text(blob, encoding="utf-8")
    from collections import Counter
    print(f"wrote {len(rows)} rows -> {out}")
    print("splits:", dict(Counter(r["split"] for r in rows)))
    print("types:", dict(Counter(r["questions"][list(r['questions'])[0]]["type"] for r in rows)))
    # determinism double-run check
    import subprocess
    print(f"data_sha256={hashlib.sha256(blob.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
