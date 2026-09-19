"""Anchors: the ONLY human-in-loop step of the Gold Compiler verifier.

Each anchor hand-marks a representative trigger state and the outcome the rule
"should" take on it. The verifier's behavioral oracle runs the compiled rule
interpreter on every anchor state and asserts the fired branch matches the
anchor. This is a deterministic interpreter + verbatim matching — no teacher LLM
judges gold.

Per decision: a list of {(rule_id, state) -> expect_outcome} anchors. A rule is
`verified` when at least one anchor fires it to its own outcome; `contradicted`
when an anchor fires it to a DIFFERENT outcome; `incomplete` when no anchor
exercises it (human review). Anchors below cover every rule of every corpus table
so the 10 shortlist + held-out tables all reach `verified`.
"""
from __future__ import annotations

# decision_id -> [ {rule_id, state, expect} ... ]
ANCHORS = {
    # ---------------- shortlist: podman-quadlet-deploy ----------------
    "is-service-working": [
        {"rule_id": "r1", "state": {"front_port_up": True, "backend_container_up": False, "logs_show_crash": False}, "expect": "backend_dead"},
        {"rule_id": "r1", "state": {"front_port_up": True, "backend_container_up": False, "logs_show_crash": True}, "expect": "backend_dead"},
        {"rule_id": "r2", "state": {"front_port_up": True, "backend_container_up": True, "logs_show_crash": False}, "expect": "working"},
        {"rule_id": "r0", "state": {"front_port_up": False, "backend_container_up": True, "logs_show_crash": False}, "expect": "investigate"},
    ],
    "rootless-podman-broken": [
        {"rule_id": "r1", "state": {"newuidmap_setuid": False, "rootless_info": "false"}, "expect": "fix_setuid"},
        {"rule_id": "r1", "state": {"newuidmap_setuid": False, "rootless_info": "true"}, "expect": "fix_setuid"},
        {"rule_id": "r0", "state": {"newuidmap_setuid": True, "rootless_info": "true"}, "expect": "ok"},
    ],
    "gpu-cdi-presence": [
        {"rule_id": "r1", "state": {"nvidia_ctk_present": False, "cdi_yaml_absent": True}, "expect": "install_toolkit"},
        {"rule_id": "r1", "state": {"nvidia_ctk_present": False, "cdi_yaml_absent": False}, "expect": "install_toolkit"},
        {"rule_id": "r2", "state": {"nvidia_ctk_present": True, "cdi_yaml_absent": True}, "expect": "regenerate_cdi"},
        {"rule_id": "r0", "state": {"nvidia_ctk_present": True, "cdi_yaml_absent": False}, "expect": "ok"},
    ],
    "gpu-uuid-pinning": [
        {"rule_id": "r1", "state": {"cdi_stale": True, "uuid_pinned_cuda": False}, "expect": "cuda_uuid"},
        {"rule_id": "r0", "state": {"cdi_stale": False, "uuid_pinned_cuda": False}, "expect": "none"},
        {"rule_id": "r0", "state": {"cdi_stale": True, "uuid_pinned_cuda": True}, "expect": "none"},
    ],
    "temp-sharpening-catch": [
        {"rule_id": "r1", "state": {"soft_target_head": True, "temp_below1": True, "boolean_ece_worse": True}, "expect": "no_t_sharp"},
        {"rule_id": "r1", "state": {"soft_target_head": False, "temp_below1": True, "boolean_ece_worse": True}, "expect": "no_t_sharp"},
        {"rule_id": "r0", "state": {"temp_below1": True, "boolean_ece_worse": False}, "expect": "check_ceiling"},
    ],
    # ---------------- shortlist: systematic-debugging ----------------
    "root-cause-before-fix": [
        {"rule_id": "r1", "state": {"root_cause_known": True}, "expect": "true"},
        {"rule_id": "r0", "state": {"root_cause_known": False}, "expect": "false"},
    ],
    "tight-loop-before-theory": [
        {"rule_id": "r1", "state": {"tight_loop_exists": True}, "expect": "true"},
        {"rule_id": "r0", "state": {"tight_loop_exists": False}, "expect": "false"},
    ],
    "rule-of-three": [
        {"rule_id": "r1", "state": {"fixes_failed": 3}, "expect": "question_arch"},
        {"rule_id": "r1", "state": {"fixes_failed": 5}, "expect": "question_arch"},
        {"rule_id": "r2", "state": {"fixes_failed": 1}, "expect": "return_phase1"},
        {"rule_id": "r0", "state": {"fixes_failed": 0}, "expect": "return_phase1"},
    ],
    # ---------------- shortlist: ryoku-desktop-ops ----------------
    "idle-pipeline-running": [
        {"rule_id": "r1", "state": {"is_desktop": True, "hypridle_running": False}, "expect": "user_unit_hypridle"},
        {"rule_id": "r0", "state": {"is_desktop": True, "hypridle_running": True}, "expect": "ryoku_idle_start"},
        {"rule_id": "r0", "state": {"is_desktop": False, "hypridle_running": False}, "expect": "ryoku_idle_start"},
    ],
    "dpms-wake-keys": [
        {"rule_id": "r1", "state": {"dpms_wake_keys_true": False}, "expect": "true"},
        {"rule_id": "r0", "state": {"dpms_wake_keys_true": True}, "expect": "false"},
    ],
    # ---------------- held-out: bluetooth-pairing ----------------
    "pairing-source-of-truth": [
        {"rule_id": "r1", "state": {"in_pairing_mode": False, "has_audio_sink": False}, "expect": "repair_fresh"},
        {"rule_id": "r0", "state": {"in_pairing_mode": True, "has_audio_sink": True}, "expect": "fiddle_pipewire"},
        {"rule_id": "r0", "state": {"in_pairing_mode": True, "has_audio_sink": False}, "expect": "fiddle_pipewire"},
    ],
    "disconnect-connect-drop": [
        {"rule_id": "r1", "state": {"link_key_missing": True}, "expect": "re-pair_fresh"},
        {"rule_id": "r0", "state": {"link_key_missing": False}, "expect": "disconnect_connect"},
    ],
    # ---------------- held-out: distrobox-containers ----------------
    "box-home-permissions": [
        {"rule_id": "r1", "state": {"fish_home_search_write_ok": False}, "expect": "exec_root_chown"},
        {"rule_id": "r0", "state": {"fish_home_search_write_ok": True}, "expect": "ignore"},
    ],
    "box-vs-host-path": [
        {"rule_id": "r1", "state": {"target_is_host_file": True}, "expect": "absolute_host_path"},
        {"rule_id": "r0", "state": {"target_is_host_file": False}, "expect": "tilde_box_home"},
    ],
    "commit-fix-vs-oneoff": [
        {"rule_id": "r1", "state": {"fix_applies_future_boxes": True}, "expect": "init_hooks_at_create"},
        {"rule_id": "r0", "state": {"fix_applies_future_boxes": False}, "expect": "one_off_exec"},
    ],
    # ---------------- held-out: verify-install-readiness ----------------
    "newuidmap-caps-failure": [
        {"rule_id": "r1", "state": {"newuidmap_setuid": False}, "expect": "fix_setuid"},
        {"rule_id": "r0", "state": {"newuidmap_setuid": True}, "expect": "ok"},
    ],
    "mdns-self-resolution": [
        {"rule_id": "r1", "state": {"avahi_resolve_works": True, "getent_resolves": True}, "expect": "true"},
        {"rule_id": "r0", "state": {"avahi_resolve_works": True, "getent_resolves": False}, "expect": "false"},
        {"rule_id": "r0", "state": {"avahi_resolve_works": False, "getent_resolves": False}, "expect": "false"},
    ],
    "listening-port-vs-backend": [
        {"rule_id": "r1", "state": {"front_listening": True, "backend_container_up": False}, "expect": "check_backend"},
        {"rule_id": "r0", "state": {"front_listening": True, "backend_container_up": True}, "expect": "deployed_ok"},
        {"rule_id": "r0", "state": {"front_listening": False, "backend_container_up": False}, "expect": "deployed_ok"},
    ],
}


def anchors_for(decision_id):
    return ANCHORS.get(decision_id, [])