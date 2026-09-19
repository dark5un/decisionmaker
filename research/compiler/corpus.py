"""Corpus of decision tables for the Gold Compiler (plan 05).

Two families, both schema-valid decision tables (research/plans/decision.schema.json):

  SHORTLIST (10 tables) — the exact tables hand-built in research/planA/rules.py,
    re-derived here so `extract --from-rules` reproduces them.
  HELDOUT   (3+ held-out skill tables) — authored from real SKILL.md files
    (bluetooth-pairing, distrobox-containers, verify-install-readiness) so the
    extractor's held-out acceptance gate has real prose to derive from.

Every table carries: skill, decision_id, source (verbatim span of the owning
SKILL.md), input_schema, question, rules (descending priority, <=1 default),
canonical_hash (computed at compile time), and verification status.
"""
from __future__ import annotations

import json
import hashlib


def canonical_hash(table, _hash=None):
    """Deterministic sha256 over the recursive-sorted canonical JSON of a table
    (with canonical_hash blanked) — the compile-time reproducibility stamp."""
    t = json.loads(json.dumps(table, sort_keys=True))
    t.pop("canonical_hash", None)
    blob = json.dumps(t, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _norm(table):
    """Stamp canonical_hash + default verification on a bare table."""
    t = dict(table)
    t.setdefault("verification", "incomplete")
    t["canonical_hash"] = canonical_hash(t)
    return t


def _subset(d, keys):
    return {k: d[k] for k in keys if k in d}


# --------------------------------------------------------------------------
# SHORTLIST — re-derived from research/planA/rules.py (the hand-built truth)
# --------------------------------------------------------------------------
from research.planA import rules as _rules


def _shortlist():
    out = []
    for skill, pkg in _rules.ALL.items():
        for d in pkg["decisions"]:
            for r in d["rules"]:
                r.setdefault("is_default", r.get("condition") is None)
            out.append(_norm({**_subset(d, [
                "decision_id", "source", "input_schema", "question", "rules",
                "default_outcome", "verification"]),
                "skill": skill}))
    return out


SHORTLIST = _shortlist()


# --------------------------------------------------------------------------
# HELDOUT — authored from real SKILL.md files
# --------------------------------------------------------------------------

def _cond_any(nodes):
    return {"any": nodes}


def _cond_all(nodes):
    return {"all": nodes}


def _c(field, op, value):
    return {"field": field, "operator": op, "value": value}


# --- bluetooth-pairing (troubleshooting/bluetooth-pairing/SKILL.md) ---------
_BT = [
    {
        "decision_id": "pairing-source-of-truth",
        "source": "SKILL.md#L29 'a device that pairs successfully yet never gets an audio sink usually was NOT actually in pairing/discoverable mode'",
        "input_schema": {
            "required": ["in_pairing_mode", "has_audio_sink"],
            "properties": {
                "in_pairing_mode": {"type": "boolean", "default": False},
                "has_audio_sink": {"type": "boolean", "default": False},
            },
        },
        "question": {
            "type": "choice",
            "instructions": "A Bluetooth device pairs but produces no audio sink. Which fix is correct?",
            "criteria": {
                "repair_fresh": "hold the device in pairing mode and re-pair fresh",
                "fiddle_pipewire": "reconfigure WirePlumber/PipeWire routing",
                "reboot_bluetooth": "restart bluetoothd and reconnect",
            },
        },
        "default_outcome": "fiddle_pipewire",
        "rules": [
            {"id": "r1", "priority": 100,
             "condition": _cond_all([_c("in_pairing_mode", "eq", False),
                                     _c("has_audio_sink", "eq", False)]),
             "outcome": "repair_fresh"},
            {"id": "r0", "priority": 0, "condition": None,
             "outcome": "fiddle_pipewire", "is_default": True},
        ],
    },
    {
        "decision_id": "disconnect-connect-drop",
        "source": "SKILL.md#L35 'bluetoothctl disconnect then connect can drop the link key ... re-pair fresh instead'",
        "input_schema": {
            "required": ["link_key_missing"],
            "properties": {"link_key_missing": {"type": "boolean", "default": False}},
        },
        "question": {
            "type": "choice",
            "instructions": "After disconnect/connect you see 'br-connection-key-missing'. What do you do?",
            "criteria": {
                "re-pair_fresh": "re-pair the device fresh",
                "disconnect_connect": "disconnect and reconnect again to bounce the profile",
            },
        },
        "default_outcome": "disconnect_connect",
        "rules": [
            {"id": "r1", "priority": 100,
             "condition": _cond_all([_c("link_key_missing", "eq", True)]),
             "outcome": "re-pair_fresh"},
            {"id": "r0", "priority": 0, "condition": None,
             "outcome": "disconnect_connect", "is_default": True},
        ],
    },
]


# --- distrobox-containers (distrobox-containers/SKILL.md) -------------------
_DISTROBOX = [
    {
        "decision_id": "box-home-permissions",
        "source": "SKILL.md#L30 'the box-home dot-dirs are owned on host by 100000 (container root) ... fish runs as container px (host 1000) → EACCES'",
        "input_schema": {
            "required": ["fish_home_search_write_ok"],
            "properties": {
                "fish_home_search_write_ok": {"type": "boolean", "default": False},
            },
        },
        "question": {
            "type": "choice",
            "instructions": "First `distrobox enter` fails to write fish history/uvars (EACCES on $HOME/.local/share). What is the durable fix?",
            "criteria": {
                "exec_root_chown": "podman exec -u root <box> chown -R 1000:1000 $HOME (or --init-hooks at create)",
                "host_chown": "host-side chown on ~/.distrobox/homes/<box>",
                "unshare_chown": "podman unshare chown on the box home",
                "ignore": "cosmetic; fish prompt still renders",
            },
        },
        "default_outcome": "ignore",
        "rules": [
            {"id": "r1", "priority": 100,
             "condition": _cond_all([_c("fish_home_search_write_ok", "eq", False)]),
             "outcome": "exec_root_chown"},
            {"id": "r0", "priority": 0, "condition": None,
             "outcome": "ignore", "is_default": True},
        ],
    },
    {
        "decision_id": "box-vs-host-path",
        "source": "SKILL.md#L66 'The box's ~ is its own $HOME ...; any path that must reach a HOST file needs the absolute /home/px/... path'",
        "input_schema": {
            "required": ["target_is_host_file"],
            "properties": {"target_is_host_file": {"type": "boolean", "default": False}},
        },
        "question": {
            "type": "choice",
            "instructions": "A command running inside the box needs to reach a file. Which path form is correct?",
            "criteria": {
                "absolute_host_path": "use the absolute /home/px/... path",
                "tilde_box_home": "use ~ (resolves to the box home)",
            },
        },
        "default_outcome": "tilde_box_home",
        "rules": [
            {"id": "r1", "priority": 100,
             "condition": _cond_all([_c("target_is_host_file", "eq", True)]),
             "outcome": "absolute_host_path"},
            {"id": "r0", "priority": 0, "condition": None,
             "outcome": "tilde_box_home", "is_default": True},
        ],
    },
    {
        "decision_id": "commit-fix-vs-oneoff",
        "source": "SKILL.md#L53 'Bake into creation (durable for future boxes): Run a chown as root at the end of init via --init-hooks'",
        "input_schema": {
            "required": ["fix_applies_future_boxes"],
            "properties": {"fix_applies_future_boxes": {"type": "boolean", "default": False}},
        },
        "question": {
            "type": "choice",
            "instructions": "You have a box-home permission fix. How should it be applied?",
            "criteria": {
                "init_hooks_at_create": "embed in distrobox create --init-hooks so future boxes inherit it",
                "one_off_exec": "one-off podman exec -u root chown for this box only",
            },
        },
        "default_outcome": "one_off_exec",
        "rules": [
            {"id": "r1", "priority": 100,
             "condition": _cond_all([_c("fix_applies_future_boxes", "eq", True)]),
             "outcome": "init_hooks_at_create"},
            {"id": "r0", "priority": 0, "condition": None,
             "outcome": "one_off_exec", "is_default": True},
        ],
    },
]


# --- verify-install-readiness (verify-install-readiness/SKILL.md) -----------
_VERIFY = [
    {
        "decision_id": "newuidmap-caps-failure",
        "source": "SKILL.md#L34 'Rootless podman failing with newuidmap: Could not set caps means the setuid bit is missing on /usr/bin/newuidmap and /usr/bin/newgidmap'",
        "input_schema": {
            "required": ["newuidmap_setuid"],
            "properties": {"newuidmap_setuid": {"type": "boolean", "default": True}},
        },
        "question": {
            "type": "choice",
            "instructions": "An installer's rootless podman step fails with 'newuidmap: Could not set caps'. What is the fix / verdict?",
            "criteria": {
                "fix_setuid": "chmod u+s /usr/bin/newuidmap /usr/bin/newgidmap",
                "installer_broken": "the installer is broken; stop",
                "ok": "rootless podman works as-is",
            },
        },
        "default_outcome": "ok",
        "rules": [
            {"id": "r1", "priority": 100,
             "condition": _cond_all([_c("newuidmap_setuid", "eq", False)]),
             "outcome": "fix_setuid"},
            {"id": "r0", "priority": 0, "condition": None,
             "outcome": "ok", "is_default": True},
        ],
    },
    {
        "decision_id": "mdns-self-resolution",
        "source": "SKILL.md#L25 'verify self-resolution at the GLIBC layer ... getent hosts <hostname>.local must return an address and exit 0'",
        "input_schema": {
            "required": ["avahi_resolve_works", "getent_resolves"],
            "properties": {
                "avahi_resolve_works": {"type": "boolean", "default": False},
                "getent_resolves": {"type": "boolean", "default": False},
            },
        },
        "question": {
            "type": "boolean",
            "instructions": "A service uses avahi/mDNS. Is .local name resolution working for real applications?",
            "criteria": {
                "true": "yes — glibc getent resolves it",
                "false": "no — avahi may advertise but NSS has no mDNS bridge",
            },
        },
        "default_outcome": "false",
        "rules": [
            {"id": "r1", "priority": 100,
             "condition": _cond_all([_c("getent_resolves", "eq", True)]),
             "outcome": "true"},
            {"id": "r0", "priority": 0, "condition": None,
             "outcome": "false", "is_default": True},
        ],
    },
    {
        "decision_id": "listening-port-vs-backend",
        "source": "SKILL.md#L61 'After a deploy, a listening port does NOT mean the service works. Rootlessport listeners ... stay up even when the backend container is dead; check podman ps -a'",
        "input_schema": {
            "required": ["front_listening", "backend_container_up"],
            "properties": {
                "front_listening": {"type": "boolean", "default": True},
                "backend_container_up": {"type": "boolean", "default": True},
            },
        },
        "question": {
            "type": "choice",
            "instructions": "The deployed service's port is listening (rootlessport). Is the service actually working?",
            "criteria": {
                "check_backend": "confirm the backend container is up with podman ps -a before declaring victory",
                "deployed_ok": "a listening port is sufficient — it is working",
            },
        },
        "default_outcome": "deployed_ok",
        "rules": [
            {"id": "r1", "priority": 100,
             "condition": _cond_all([_c("front_listening", "eq", True),
                                     _c("backend_container_up", "eq", False)]),
             "outcome": "check_backend"},
            {"id": "r0", "priority": 0, "condition": None,
             "outcome": "deployed_ok", "is_default": True},
        ],
    },
]


def _heldout(_tables, skill):
    return [_norm({**_subset(t, [
        "decision_id", "source", "input_schema", "question", "rules",
        "default_outcome", "verification"]), "skill": skill}) for t in _tables]


HELDOUT = (_heldout(_BT, "bluetooth-pairing")
           + _heldout(_DISTROBOX, "distrobox-containers")
           + _heldout(_VERIFY, "verify-install-readiness"))

ALL = SHORTLIST + HELDOUT


if __name__ == "__main__":
    print(f"shortlist={len(SHORTLIST)} heldout={len(HELDOUT)} total={len(ALL)}")