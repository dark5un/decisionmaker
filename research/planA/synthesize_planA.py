#!/usr/bin/env python3
"""Plan A synthesizer: compile rule-tables (rules.py) into seed.jsonl-schema rows.

For each skill decision, enumerate the state space described by its input_schema,
render each state into natural-language scenario text (several prose phrasings per
state — the same decision described the way the actual skill prose phrases it),
build the question, and compute the EXACT soft gold distribution by running the
compiled rules. Splits are group-separated: each decision is one source_group_id
(no leakage across splits); the same decision never appears in two splits.

Row schema matches generate_seed.py / train.py exactly. Deterministic + seeded;
data_sha256 stamped. Priority-based rule evaluation maps each state to exactly one
winner; soft gold = winner .90, rest shared (honest soft-target ceiling ~0.90).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

from research.planA.rules import ALL, outcome_distribution

# --- natural-language state renderers (each returns a LIST of phrasings) --------

def render_podman(decision_id, st):
    if decision_id == "is-service-working":
        fp = "UP" if st['front_port_up'] else "DOWN"
        be = "running" if st['backend_container_up'] else "exited/crash-looping"
        lg = "show a crash" if st['logs_show_crash'] else "look clean"
        return [
            f"Front-end port 8090 is {fp}. The backend container is {be}. Service logs {lg}.",
            f"Health returns 200 on {fp} port; the backend is {be}; logs {lg}.",
            f"A reverse proxy listens ({fp}) but the backend container is {be} and logs {lg}.",
        ]
    if decision_id == "rootless-podman-broken":
        su = "has its setuid root bit (mode -rwsr-xr-x)" if st['newuidmap_setuid'] else "LOST its setuid root bit (mode -rwxr-xr-x)"
        ri = st['rootless_info']
        return [
            f"/usr/bin/newuidmap {su}. podman info Rootless reports {ri}.",
            f"Rootless checks: newuidmap {su}; Rootless={ri}.",
        ]
    if decision_id == "gpu-cdi-presence":
        ctk = "installed (nvidia-ctk present)" if st['nvidia_ctk_present'] else "NOT installed (nvidia-ctk absent)"
        cdi = "is present" if not st['cdi_yaml_absent'] else "is ABSENT"
        return [
            f"nvidia-container-toolkit is {ctk}. The CDI spec /etc/cdi/nvidia.yaml {cdi}.",
            f"AddDevice=nvidia.com/gpu=all is used. Toolkit: {ctk}. /etc/cdi/nvidia.yaml {cdi}.",
        ]
    if decision_id == "gpu-uuid-pinning":
        cs = "stale (device minors may have moved)" if st['cdi_stale'] else "fresh"
        up = "pins via CUDA_VISIBLE_DEVICES=GPU-<uuid>" if st['uuid_pinned_cuda'] else "does NOT pin at CUDA level"
        return [
            f"CDI spec is {cs}. The quadlet {up}.",
            f"A two-GPU box; the service {up}. CDI is {cs}.",
        ]
    if decision_id == "temp-sharpening-catch":
        stgt = "soft" if st['soft_target_head'] else "hard"
        tp = "T<1 (sharpening)" if st['temp_below1'] else "T=1.0 (no sharpening)"
        ew = "WORSENS" if st['boolean_ece_worse'] else "keeps stable"
        return [
            f"The head was trained with {stgt} targets. Post-hoc temperature {tp} is proposed; it {ew} boolean ECE.",
            f"Soft-target head? {stgt}. Temperature {tp}; boolean ECE {ew} against real labels.",
        ]
    raise KeyError(decision_id)

def render_debugging(decision_id, st):
    if decision_id == "root-cause-before-fix":
        rc = "COMPLETE - the bug is understood" if st['root_cause_known'] else "NOT done - the WHY is still unknown"
        return [
            f"Root-cause investigation is {rc}. A fix is about to be proposed.",
            f"Phase 1 root-cause is {rc}; you are tempted to propose a fix now.",
        ]
    if decision_id == "tight-loop-before-theory":
        tl = "EXISTS and has been run" if st['tight_loop_exists'] else "has NOT been built yet"
        return [
            f"A tight red-capable feedback loop {tl}. Theory is being formed.",
            f"Before coding a theory: the tight repro loop {tl}.",
        ]
    if decision_id == "rule-of-three":
        n = st["fixes_failed"]
        return [
            f"{n} fix attempt(s) failed so far; each revealed new shared-state coupling.",
            f"You have tried {n} fixes and none stuck.",
        ]
    raise KeyError(decision_id)

def render_ryoku(decision_id, st):
    if decision_id == "idle-pipeline-running":
        dt = "desktop tower" if st['is_desktop'] else "laptop"
        hr = "already running" if st['hypridle_running'] else "NOT running; idle/lock/screensaver silent"
        return [
            f"This machine is a {dt}. hypridle is {hr}.",
            f"Idle pipeline: device is a {dt}; hypridle {hr}.",
        ]
    if decision_id == "dpms-wake-keys":
        dk = "already true" if st['dpms_wake_keys_true'] else "at default false"
        return [
            f"Screen wakes BLACK after DPMS-off + input. key_press/mouse_move_enables_dpms are {dk}.",
            f"The compositor stays black on first input after lock; DPMS wake keys are {dk}.",
        ]
    raise KeyError(decision_id)

RENDER = {"podman-quadlet-deploy": render_podman, "systematic-debugging": render_debugging,
          "ryoku-desktop-ops": render_ryoku}


def state_space(decision):
    schema = decision["input_schema"]
    fields = schema["required"]
    combos = []
    for f in fields:
        meta = schema["properties"][f]
        if meta["type"] == "boolean":
            combos.append([True, False])
        elif f == "rootless_info":
            combos.append(["true", "false"])
        else:  # numeric fixes_failed
            combos.append([0, 1, 2, 3, 4, 5, 6])
    for combo in itertools.product(*combos):
        yield dict(zip(fields, combo))


def build_question(decision):
    return {"type": decision["question"]["type"],
            "instructions": decision["question"]["instructions"],
            "criteria": decision["question"]["criteria"]}


def make_rows(decision, skill, state_dict, split, fid, seed):
    """Return all phrasing-rows for one state (one per renderer phrasing)."""
    phrasings = RENDER[skill](decision["decision_id"], state_dict)
    out = []
    for pid, state_text in enumerate(phrasings):
        rng = random.Random(seed + pid)
        qid = decision["decision_id"]
        question = build_question(decision)
        dist, winner = outcome_distribution(decision, state_dict)
        gold_probs = {qid: {k: round(float(v), 6) for k, v in dist.items()}}
        row = {
            "id": f"{fid}:{state_dict_hash(state_dict)}:{pid}",
            "state_id": f"{state_dict_hash(state_dict)}",
            "family_id": fid, "split": split,
            "state": state_text,
            "questions": {qid: question},
            "gold_probs": gold_probs,
            "gold_probs_kind": "compiled_from_prose_rules",
            "metadata": {"source_group_id": fid, "skill": skill,
                         "decision": decision["decision_id"], "outcome": winner,
                         "rule_source": decision.get("source", ""), "phrasing": pid},
        }
        if question["type"] == "boolean":
            lab = 1.0 if dist["true"] >= dist["false"] else 0.0
            row["gold_label"] = {qid: lab}
            row["gold_label_kind"] = "compiled_rule_outcome"
        out.append(row)
    return out


def state_dict_hash(st):
    import hashlib
    return hashlib.sha1(json.dumps(st, sort_keys=True).encode()).hexdigest()[:8]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/data/planA_train.jsonl")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--conf", type=float, default=0.90)
    args = ap.parse_args()

    rows = []
    decisions = [(skill, d) for skill, pkg in ALL.items() for d in pkg["decisions"]]
    # Hand-assign each decision to a split so that BOTH the gated splits (test, ood)
    # contain a boolean decision (the boolean ECE gate is computed on test + ood).
    # Group separation is preserved: each decision is in exactly one split.
    SPLIT_ASSIGN = {
        "is-service-working": "train",
        "rootless-podman-broken": "train",
        "gpu-cdi-presence": "calibration",
        "gpu-uuid-pinning": "dev",
        "temp-sharpening-catch": "dev",
        "root-cause-before-fix": "test",        # boolean -> test
        "tight-loop-before-theory": "ood",      # boolean -> ood
        "rule-of-three": "calibration",
        "idle-pipeline-running": "test",
        "dpms-wake-keys": "ood",                # boolean -> ood
    }
    for skill, decision in decisions:
        split = SPLIT_ASSIGN[decision["decision_id"]]
        fid = f"{skill}:{decision['decision_id']}".replace(":", "-")
        for state_dict in state_space(decision):
            rows.extend(make_rows(decision, skill, state_dict, split, fid, args.seed))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    out.write_text(blob, encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {out}")
    from collections import Counter
    print("splits:", dict(Counter(r["split"] for r in rows)))
    print("types:", dict(Counter(r["questions"][list(r['questions'])[0]]["type"] for r in rows)))
    # verify determinism
    print(f"data_sha256={hashlib.sha256(blob.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
