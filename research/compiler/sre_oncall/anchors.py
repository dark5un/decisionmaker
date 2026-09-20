"""sre-oncall anchors — the only human-in-loop step of the Gold Compiler.

Each anchor hand-marks a trigger state and the outcome the rule should take on
it. The verifier runs the rule interpreter on every anchor state and asserts
the fired branch matches `expect`. One anchor per rule (at least), including
the defaults. Fields match the input_schema of each table in tables.jsonl.
"""

from __future__ import annotations

ANCHORS = {
    # ---------------- severity-class ----------------
    "severity-class": [
        # r1: data corrupted -> sev1 no matter what else
        {"rule_id": "r1", "state": {"data_corrupted": True, "primary_path_down": False,
                                    "workaround_verified": False, "sla_breach_risk": False},
         "expect": "sev1"},
        # r2: fully down, no workaround -> sev1
        {"rule_id": "r2", "state": {"data_corrupted": False, "primary_path_down": True,
                                    "workaround_verified": False, "sla_breach_risk": False},
         "expect": "sev1"},
        # r3: fully down, workaround verified -> sev2
        {"rule_id": "r3", "state": {"data_corrupted": False, "primary_path_down": True,
                                    "workaround_verified": True, "sla_breach_risk": False},
         "expect": "sev2"},
        # r4: NOT down, SLA at risk, no workaround -> sev1 (the start-up quirk)
        {"rule_id": "r4", "state": {"data_corrupted": False, "primary_path_down": False,
                                    "workaround_verified": False, "sla_breach_risk": True},
         "expect": "sev1"},
        # r5: NOT down, SLA at risk, workaround verified -> sev2
        {"rule_id": "r5", "state": {"data_corrupted": False, "primary_path_down": False,
                                    "workaround_verified": True, "sla_breach_risk": True},
         "expect": "sev2"},
        # r0 default: nothing triggering -> sev3
        {"rule_id": "r0", "state": {"data_corrupted": False, "primary_path_down": False,
                                    "workaround_verified": False, "sla_breach_risk": False},
         "expect": "sev3"},
    ],
    # ---------------- fix-posture ----------------
    "fix-posture": [
        # r1: migration irreversible -> forward_fix even if fix not ready, incident quiet
        {"rule_id": "r1", "state": {"migration_irreversible": True,
                                    "forward_fix_ready": False, "incident_active": False},
         "expect": "forward_fix"},
        # r1 again: irreversible + active + ready
        {"rule_id": "r1", "state": {"migration_irreversible": True,
                                    "forward_fix_ready": True, "incident_active": True},
         "expect": "forward_fix"},
        # r2: reversible, fix ready (quiet) -> forward_fix ("calmer ... still forward")
        {"rule_id": "r2", "state": {"migration_irreversible": False,
                                    "forward_fix_ready": True, "incident_active": False},
         "expect": "forward_fix"},
        # r2: reversible, fix ready, active -> forward_fix (active -> forward when ready)
        {"rule_id": "r2", "state": {"migration_irreversible": False,
                                    "forward_fix_ready": True, "incident_active": True},
         "expect": "forward_fix"},
        # r3: reversible, fix NOT ready -> rollback
        {"rule_id": "r3", "state": {"migration_irreversible": False,
                                    "forward_fix_ready": False, "incident_active": True},
         "expect": "rollback"},
        # r0 default
        {"rule_id": "r0", "state": {"migration_irreversible": False,
                                    "forward_fix_ready": False, "incident_active": False},
         "expect": "rollback"},
    ],
    # ---------------- page-second-responder ----------------
    "page-second-responder": [
        # r1: data loss -> page regardless
        {"rule_id": "r1", "state": {"data_loss": True, "primary_fully_down": False,
                                    "sla_breach_risk": False, "workaround_verified": True},
         "expect": "true"},
        # r2: primary fully down -> page
        {"rule_id": "r2", "state": {"data_loss": False, "primary_fully_down": True,
                                    "sla_breach_risk": False, "workaround_verified": False},
         "expect": "true"},
        # r3: SLA at risk -> page
        {"rule_id": "r3", "state": {"data_loss": False, "primary_fully_down": False,
                                    "sla_breach_risk": True, "workaround_verified": True},
         "expect": "true"},
        # r0 default: nothing -> solo (even with verified workaround)
        {"rule_id": "r0", "state": {"data_loss": False, "primary_fully_down": False,
                                    "sla_breach_risk": False, "workaround_verified": True},
         "expect": "false"},
    ],
    # ---------------- canary-promote ----------------
    "canary-promote": [
        # r1: forward-only schema + no back-compat -> hold, even if tests green
        {"rule_id": "r1", "state": {"tests_green": True, "backcompat_proven": False,
                                    "rollback_drill_passed": True, "forward_only_schema": True,
                                    "traffic_spike": True},
         "expect": "false"},
        # r2: tests green + backcompat + drill passed -> promote (non-fwd schema)
        {"rule_id": "r2", "state": {"tests_green": True, "backcompat_proven": True,
                                    "rollback_drill_passed": True, "forward_only_schema": False,
                                    "traffic_spike": False},
         "expect": "true"},
        # r2: tests green + backcompat + drill, forward-only but backcompat PROVEN -> promote
        {"rule_id": "r2", "state": {"tests_green": True, "backcompat_proven": True,
                                    "rollback_drill_passed": True, "forward_only_schema": True,
                                    "traffic_spike": False},
         "expect": "true"},
        # r0 default: tests not all green -> hold
        {"rule_id": "r0", "state": {"tests_green": False, "backcompat_proven": True,
                                    "rollback_drill_passed": True, "forward_only_schema": False,
                                    "traffic_spike": False},
         "expect": "false"},
    ],
    # ---------------- comms-escalation ----------------
    "comms-escalation": [
        # r1: breach -> full escalation even if customer requested internal
        {"rule_id": "r1", "state": {"is_breach": True, "data_incident": False,
                                    "customer_internal_only_request": True,
                                    "fixed_contract": False, "is_smb_or_consumer": True,
                                    "sev2_or_above": False},
         "expect": "full_escalation"},
        # r2: data incident, no internal request -> full escalation (any tier)
        {"rule_id": "r2", "state": {"is_breach": False, "data_incident": True,
                                    "customer_internal_only_request": False,
                                    "fixed_contract": False, "is_smb_or_consumer": True,
                                    "sev2_or_above": True},
         "expect": "full_escalation"},
        # r3: data incident + internal requested, NOT breach -> internal_only_ok
        {"rule_id": "r3", "state": {"is_breach": False, "data_incident": True,
                                    "customer_internal_only_request": True,
                                    "fixed_contract": False, "is_smb_or_consumer": True,
                                    "sev2_or_above": True},
         "expect": "internal_only_ok"},
        # r4: fixed contract, non-smb, sev2+ -> account_owner_only
        {"rule_id": "r4", "state": {"is_breach": False, "data_incident": False,
                                    "customer_internal_only_request": False,
                                    "fixed_contract": True, "is_smb_or_consumer": False,
                                    "sev2_or_above": True},
         "expect": "account_owner_only"},
        # r0 default: no triggers -> internal_only_ok
        {"rule_id": "r0", "state": {"is_breach": False, "data_incident": False,
                                    "customer_internal_only_request": False,
                                    "fixed_contract": False, "is_smb_or_consumer": True,
                                    "sev2_or_above": False},
         "expect": "internal_only_ok"},
    ],
    # ---------------- force-restart ----------------
    "force-restart": [
        # r1: crash-looping + disk full -> HOLD, even with everything else clean
        {"rule_id": "r1", "state": {"crash_looping": True, "disk_full": True,
                                    "recent_deploy": False, "in_flight_batch": False,
                                    "cpu_pegged": True},
         "expect": "false"},
        # r2: crash-looping + recent deploy -> HOLD (roll back, don't bounce)
        {"rule_id": "r2", "state": {"crash_looping": True, "disk_full": False,
                                    "recent_deploy": True, "in_flight_batch": False,
                                    "cpu_pegged": False},
         "expect": "false"},
        # r3: crash-looping + in-flight batch -> HOLD (let it drain)
        {"rule_id": "r3", "state": {"crash_looping": True, "disk_full": False,
                                    "recent_deploy": False, "in_flight_batch": True,
                                    "cpu_pegged": False},
         "expect": "false"},
        # r4: crash-looping, NO confounders -> restart (cpu peg is a distractor)
        {"rule_id": "r4", "state": {"crash_looping": True, "disk_full": False,
                                    "recent_deploy": False, "in_flight_batch": False,
                                    "cpu_pegged": True},
         "expect": "true"},
        # r4 again: restart with everything clean, cpu not pegged
        {"rule_id": "r4", "state": {"crash_looping": True, "disk_full": False,
                                    "recent_deploy": False, "in_flight_batch": False,
                                    "cpu_pegged": False},
         "expect": "true"},
        # r0 default: not crash-looping -> hold, even with cpu peg noise
        {"rule_id": "r0", "state": {"crash_looping": False, "disk_full": True,
                                    "recent_deploy": True, "in_flight_batch": True,
                                    "cpu_pegged": True},
         "expect": "false"},
    ],
    # ---------------- log-ci-verdict ----------------
    "log-ci-verdict": [
        # r1: retried and passed -> noise, regardless of how loud first attempt was
        {"rule_id": "r1", "state": {"exit_nonzero": True, "timeout": True,
                                    "known_flaky": False, "retried_passed": True,
                                    "verbose_logs": True},
         "expect": "false"},
        # r2: known-flaky + non-zero -> noise by design (already triaged)
        {"rule_id": "r2", "state": {"exit_nonzero": True, "timeout": False,
                                    "known_flaky": True, "retried_passed": False,
                                    "verbose_logs": True},
         "expect": "false"},
        # r2: known-flaky + timeout -> noise by design
        {"rule_id": "r2", "state": {"exit_nonzero": False, "timeout": True,
                                    "known_flaky": True, "retried_passed": False,
                                    "verbose_logs": False},
         "expect": "false"},
        # r3: non-zero, not flaky, not retried -> flag for review
        {"rule_id": "r3", "state": {"exit_nonzero": True, "timeout": False,
                                    "known_flaky": False, "retried_passed": False,
                                    "verbose_logs": True},
         "expect": "true"},
        # r4: timeout, not flaky, not retried -> flag for review
        {"rule_id": "r4", "state": {"exit_nonzero": False, "timeout": True,
                                    "known_flaky": False, "retried_passed": False,
                                    "verbose_logs": True},
         "expect": "true"},
        # r0 default: clean run -> noise (verbose logs are a distractor)
        {"rule_id": "r0", "state": {"exit_nonzero": False, "timeout": False,
                                    "known_flaky": False, "retried_passed": False,
                                    "verbose_logs": True},
         "expect": "false"},
    ],
}


def anchors_for(decision_id):
    return ANCHORS.get(decision_id, [])