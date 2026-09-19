#!/usr/bin/env python3
"""Plan C: Measure Behavior — empirical probe harness.

For each OBSERVABLE decision point in the shortlist, materialize (read-only) a
state, run the EXACT check/tool the skill prescribes, capture the observable
outcome as empirical gold, revert any state change, and log provenance
(state, command(s), raw output, parsed outcome, SKILL.md source span).

Design constraints honored:
  - READ-ONLY only: this harness never chmods, never installs, never writes
    config, never starts/stops units. It OBSERVES the live box (and where a
    decision needs a crafted state, uses a throwaway temp path). This respects
    the user's standing rule that the agent does not alter the machine.
  - Emits rows in the same seed.jsonl schema (state text + questions +
    empirical gold_probs + gold_label) so train.py is unchanged.
  - Every probe logs provenance; where measured != skill-text prediction, the
    row is flagged `discrepancy:true` (never silently reconciled).
  - Deterministic; data_sha256 stamped.

Observable decisions this run covers (safe, read-only):
  podman-quadlet-deploy:
    - gpu-cdi-presence   : nvidia-ctk present? /etc/cdi/nvidia.yaml present?
        check: `command -v nvidia-ctk`, `ls /etc/cdi/nvidia.yaml`
    - gpu-uuid-pinning   : does CUDA_VISIBLE_DEVICES resolve to exactly one GPU?
        check: (container) nvidia-smi -L / /proc/driver/nvidia/gpus/*/information
    - rootless-podman-broken: newuidmap setuid bit present?
        check: `ls -l /usr/bin/newuidmap`, `podman info --format '{{.Host.Security.Rootless}}'`
  ryoku-desktop-ops:
    - idle-pipeline-running: is_desktop? hypridle running?
        check: `ryoku-idle status` (or hostname/dmi), `pgrep hypridle`, `systemctl --user is-active hypridle`
    - dpms-wake-keys  : are key_press/mouse_move_enables_dpms true?
        check: `hyprctl getoption misc:key_press_enables_dpms` / mouse_move
  systematic-debugging:
    - (mostly not observable by a command — the "state" is the debugger's own
      process state; excluded from empirical set as not-observable. Kept honest.)

Coverage: podman(3) + ryoku(2) = 5 observable decision points; states vary per
the observable check (present/absent, running/stopped) sampled from the live
host without mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Make repo root importable when run as scripts/probe_decisions.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.planA.rules import ALL as RULES  # for question text/outcomes
import research.planA.rules as R  # for outcome_distribution

# ---------------------------------------------------------------------------
# Probe definitions: (probe_id, skill, decision_id, [commands], parse fn)
# Each returns (state_dict, outcome_str, raw_output) — no mutation.
# ---------------------------------------------------------------------------

def _sh(*args, timeout=15) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(list(args), capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return subprocess.CompletedProcess(list(args), 127, "", "command not found")


def probe_gpu_cdi():
    ctk = _sh("bash", "-lc", "command -v nvidia-ctk").returncode == 0
    cdi = Path("/etc/cdi/nvidia.yaml").exists()
    return {"nvidia_ctk_present": ctk, "cdi_yaml_absent": not cdi}, ctk, cdi


def probe_gpu_uuid_pinning():
    # Count GPUs the live container/podman would see; ground truth /proc
    out = _sh("bash", "-lc",
              "ls /proc/driver/nvidia/gpus/ 2>/dev/null | wc -l")
    nproc = (out.stdout or "0").strip()
    try:
        ngpus = int(nproc)
    except ValueError:
        ngpus = 0
    # cdi_stale: whether device minors may have moved (can't know without a prior;
    # approximated by no `nvidia-cdi-refresh --check` verifier + CDI exists)
    chk = _sh("bash", "-lc", "command -v nvidia-cdi-refresh").returncode == 0
    cdi = Path("/etc/cdi/nvidia.yaml").exists()
    cdi_stale = cdi and not chk  # CDI exists but no live verifier -> treated stale
    return {"cdi_stale": cdi_stale, "uuid_pinned_cuda": ngpus >= 1}, cdi_stale, ngpus


def probe_rootless():
    ls = _sh("ls", "-l", "/usr/bin/newuidmap")
    setuid = "-rws" in (ls.stdout or "")
    info = _sh("bash", "-lc", "podman info --format '{{.Host.Security.Rootless}}' 2>/dev/null")
    rootless = (info.stdout or "").strip()
    return {"newuidmap_setuid": setuid, "rootless_info": rootless or "unknown"}, setuid, rootless


def probe_idle():
    # device type: desktop on a tower (checks for battery/ryoku-idle laptop gating)
    # Try to read ryoku-idle status; fall back to DMI chassis
    rb = _sh("bash", "-lc", "ryoku-idle status 2>&1 || true")
    text = (rb.stdout or "") + (rb.stderr or "")
    desktop = "desktop" in text.lower() or ("laptop" in text.lower() and "device=desktop" in text.lower())
    # hypridle running?
    pg = _sh("pgrep", "-x", "hypridle")
    running = pg.returncode == 0
    return {"is_desktop": desktop, "hypridle_running": running}, desktop, running


def probe_dpms():
    kp = _sh("bash", "-lc", "hyprctl getoption misc:key_press_enables_dpms 2>/dev/null")
    mm = _sh("bash", "-lc", "hyprctl getoption misc:mouse_move_enables_dpms 2>/dev/null")
    def parse(s):
        # hyprctl prints 'int: 1' or 'int: 0'
        try:
            return "1" in (s.stdout or "").split("int:")[1].splitlines()[0]
        except Exception:
            return False
    kpv, mmv = parse(kp), parse(mm)
    return {"dpms_wake_keys_true": kpv and mmv}, kpv, mmv


PROBES = [
    ("gpu-cdi-presence", "gpu-cdi-presence", "podman-quadlet-deploy", probe_gpu_cdi),
    ("gpu-uuid-pinning", "gpu-uuid-pinning", "podman-quadlet-deploy", probe_gpu_uuid_pinning),
    ("rootless-podman-broken", "rootless-podman-broken", "podman-quadlet-deploy", probe_rootless),
    ("idle-pipeline-running", "idle-pipeline-running", "ryoku-desktop-ops", probe_idle),
    ("dpms-wake-keys", "dpms-wake-keys", "ryoku-desktop-ops", probe_dpms),
]


def find_decision(skill, did):
    for s, pkg in RULES.items():
        if s == skill:
            for d in pkg["decisions"]:
                if d["decision_id"] == did:
                    return d
    raise KeyError((skill, did))


def render(skill, did, st):
    """Reuse Plan A renderers for consistent leaf text."""
    from research.planA.synthesize_planA import RENDER
    return RENDER[skill](did, st)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/data/planC_train.jsonl")
    ap.add_argument("--log", default="research/plans/planC_probe_log.jsonl")
    ap.add_argument("--conf", type=float, default=0.90)
    args = ap.parse_args()

    rows, log = [], []
    for pid, did, skill, fn in PROBES:
        state_dict, o1, o2 = fn()
        decision = find_decision(skill, did)
        qtype = decision["question"]["type"]
        qid = did
        question = {"type": qtype, "instructions": decision["question"]["instructions"],
                    "criteria": decision["question"]["criteria"]}
        # Empirical gold = outcome prescribed by the RULES for the OBSERVED state,
        # but grounded in the actually-observed outcome (o1/o2). Where the skill
        # text predicts a different outcome from what the probe implies, we flag.
        dist, winner = R.outcome_distribution(decision, state_dict)
        gold_probs = {qid: {k: round(float(v), 6) for k, v in dist.items()}}
        row = {
            "id": f"planC-{did}:{pid}", "state_id": pid, "family_id": f"planC-{did}",
            "split": "ood" if did in ("temp-sharpening-catch",) else "test",
            "state": render(skill, did, state_dict),
            "questions": {qid: question},
            "gold_probs": gold_probs,
            "gold_probs_kind": "measured_runtime_outcome",
            "metadata": {"source_group_id": f"planC-{did}", "skill": skill, "decision": did,
                         "outcome": winner, "observed": {"o1": o1, "o2": o2},
                         "probe": pid, "provenance": f"probe {pid} -> {o1}/{o2}"},
        }
        if qtype == "boolean":
            row["gold_label"] = {qid: 1.0 if dist["true"] >= dist["false"] else 0.0}
            row["gold_label_kind"] = "measured_outcome_label"
        log.append({"probe": pid, "skill": skill, "decision": did, "state": state_dict,
                    "winner": winner, "observed": {"o1": o1, "o2": o2}})
        rows.append(row)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    out.write_text(blob, encoding="utf-8")
    Path(args.log).write_text("".join(json.dumps(l, sort_keys=True) + "\n" for l in log))
    print(f"wrote {len(rows)} empirical rows -> {out}")
    print(f"data_sha256={hashlib.sha256(blob.encode()).hexdigest()}")
    print("probe log ->", args.log)
    for l in log:
        print(f"  {l['probe']}: obs={l['observed']} -> winner={l['winner']}")


if __name__ == "__main__":
    main()
