"""sre-oncall bespoke renderers.

Turn an enumerated state dict (bool fields) into realistic, natural on-call
scenario prose. The DECISIVE fields are described in realistic operator
language (not "data_corrupted is true"), and distractor fields are woven into
the narrative so a pattern-matching model is tempted, but a human applying the
runbook gets the right signal. Deterministic: given the same state, the same
texts are produced.

Compiler renderer contract: fn(decision_id, st) -> list[str] (>=1 phrasings).
"""

from __future__ import annotations


def _sev(decision_id, st):
    data = st.get("data_corrupted")
    down = st.get("primary_path_down")
    work = st.get("workaround_verified")
    sla = st.get("sla_breach_risk")

    if data:
        lead = ("A growing number of write paths are reporting checksum "
                "mismatches and an operator has confirmed corrupted records "
                "now sitting read-side in the production store.")
        lead2 = ("We are seeing repeated storage-layer integrity failures and "
                 "a spot-check turned up real records that no longer match "
                 "their index; the corruption is not cosmetic.")
    else:
        lead = ("No checksum or integrity failures have been reported anywhere "
                "in the store, and a full consistency pass on the last hour "
                "came back clean.")
        lead2 = ("Storage integrity looks fine — no bad records, no checksum "
                 "skips, the data layer is reporting healthy on every node.")

    if down:
        serve = ("The primary decision-serving path is completely unavailable "
                 "to everyone right now — requests are failing at the edge.")
        serve2 = ("The main serving route is down hard; traffic is erroring "
                  "out at 100% and nothing is answering.")
    else:
        serve = ("The primary serving route is still up and answering requests "
                 "at normal latency.")
        serve2 = ("The main path is serving fine — no outage on the serving "
                  "route, requests are completing normally.")

    if work:
        mitig = ("A mitigation is already published and one customer has "
                 "independently confirmed it restores correct behaviour, so "
                 "the exposure is contained for a meaningful subset of users.")
        mitig2 = ("A workaround is out there and an external user verified it "
                  "resolves the symptoms — it is a real, confirmed mitigation, "
                  "not a runbook guess.")
    else:
        mitig = ("No workaround exists yet; nothing has been found that "
                 "restores correctness for the affected users.")
        mitig2 = ("There is no published mitigation so far — we have nothing "
                  "verified that would bring things back to normal.")

    if sla:
        sla_clause = ("Separately, this event is already counting against a "
                      "contractual SLA and a breach looks probable in the "
                      "current window.")
        sla_clause2 = ("A customer-facing SLA is at genuine risk — the "
                       "contractual response time is going to be missed if "
                       "this continues.")
    else:
        sla_clause = ("No contractual SLA is at risk; the response-time "
                      "commitments are not threatened by this.")
        sla_clause2 = ("SLAs are safe — none of the response targets are in "
                       "real danger from this event.")

    core = (
        f"{lead} {serve} {mitig} {sla_clause} "
        f"There is noise in the dashboards but the useful signal is above. "
        f"What severity class is this incident?"
    )
    core2 = (
        f"{lead2} {serve2} {mitig2} {sla_clause2} "
        f"A couple of monitors are flapping, but the above is the substance. "
        f"Classify the severity of this incident."
    )
    return [core, core2]


def _fix(decision_id, st):
    irr = st.get("migration_irreversible")
    ready = st.get("forward_fix_ready")
    active = st.get("incident_active")

    if irr:
        mig = ("The offending change shipped with a schema and data migration "
               "that is one-way; the store has already been moved past the old "
               "structure and the previous application build no longer "
               "understands it, so reverting would leave a live process "
               "pointing at a shape it cannot read.")
        mig2 = ("A forward-only migration rode in with the change: the "
                "database has already advanced beyond the prior code and "
                "rolling the binaries back would desync them from the data.")
    else:
        mig = ("The change is fully reversible — no forward-only migration, "
               "so an old build can come back safely onto the same store.")
        mig2 = ("Nothing irreversible shipped; the schema is backward "
                "compatible and an earlier version can be reinstated cleanly.")

    if ready:
        fixc = ("A correct forward fix is already built and has been "
                "exercised against the actual failure mode, so it is genuinely "
                "ready rather than assumed.")
        fixc2 = ("The forward fix exists, landed, and was tested against a "
                 "reproduction of this exact incident — it is verified ready.")
    else:
        fixc = ("No forward fix is ready yet; the candidate fix has not been "
                "exercised against the real failure, so it cannot be trusted.")
        fixc2 = ("There is no verified forward fix on hand — the potential "
                "patch has not been proven against this incident.")

    if active:
        act = ("The production incident is currently active and customers are "
               "impacted right now, and an immediate fleet-bounce would itself "
               "be operationally risky mid-distribution.")
        act2 = ("We are mid-incident with real traffic affected, and flipping "
                "the whole fleet at once right now would add more risk than "
                "it removes.")
    else:
        act = ("The incident is quiet at the moment — it is resolved enough "
               "that there is no live fire, the field is stable.")
        act2 = ("Things are calm now; the outage is not actively burning and "
               "the system is in a stable posture.")

    core = (
        f"{mig} {fixc} {act} "
        f"The runbook says to decide on fix posture from reversibility and "
        f"readiness, not from which is faster to type. Should we roll back or "
        f"ship a forward fix?"
    )
    core2 = (
        f"{mig2} {fixc2} {act2} "
        f"Choose the fix posture the runbook prescribes for this state."
    )
    return [core, core2]


def _page(decision_id, st):
    dl = st.get("data_loss")
    down = st.get("primary_fully_down")
    sla = st.get("sla_breach_risk")
    work = st.get("workaround_verified")

    if dl:
        a = ("Data has been lost or leaked — records are gone or exposed, "
             "which the manual treats as immediate and non-negotiable.")
        a2 = ("There is confirmed data impact: records are missing or "
              "compromised. That is the category that most overrides defaults.")
    else:
        a = ("No data loss or leak — nothing is missing, nothing was exposed.")
        a2 = ("Data is intact; no loss, no exposure, storage is safe.")

    if down:
        b = ("Primary serving is fully down — the main path is dead for "
             "everyone.")
        b2 = ("The primary serving path is completely offline.")
    else:
        b = ("Primary serving is up — the main path is alive.")
        b2 = ("The serving route is still up.")

    if sla:
        c = ("A contractual SLA is at real risk of breach.")
        c2 = ("We are at risk of missing a contractual response commitment.")
    else:
        c = ("No SLA is at risk.")
        c2 = ("Contractual response targets are not threatened.")

    if work:
        d = ("A workaround exists and a user confirmed it, which in a quiet "
             "situation would argue for staying solo even if the dashboards "
             "look noisy.")
        d2 = ("There is a verified mitigation already, so under normal "
              "conditions the inclination would be to keep handling it alone.")
    else:
        d = ("There is no verified workaround in place yet.")
        d2 = ("Nothing has been mitigated — no confirmed workaround exists.")

    core = (
        f"{a} {b} {c} {d} "
        f"The dashboards are noisy and the alerts talk over each other, but "
        f"re-read the manual's paging conditions. Should a second responder "
        f"be paged, or do we stay solo?"
    )
    core2 = (
        f"{a2} {b2} {c2} {d2} "
        f"Decide: page a second responder, or handle it solo?"
    )
    return [core, core2]


def _canary(decision_id, st):
    tests = st.get("tests_green")
    bc = st.get("backcompat_proven")
    drill = st.get("rollback_drill_passed")
    fwd = st.get("forward_only_schema")
    spike = st.get("traffic_spike")

    if tests:
        t = ("The unit and integration suites are all green — every test we "
             "run passed.")
        t2 = ("Tests are green across the board; the suite reports no "
              "failures.")
    else:
        t = ("The test suites are NOT green — there are failing tests in the "
             "run.")
        t2 = ("Tests are red; the suite has failures.")

    if bc:
        b = ("Backward compatibility is proven: an old client was verified "
             "surviving against this new build.")
        b2 = ("We demonstrated the schema is backward compatible — a prior "
             "version still works against the new head.")
    else:
        b = ("Backward compatibility has NOT been demonstrated for this "
             "build's schema.")
        b2 = ("No back-compat evidence exists; we have not shown an old "
             "client works against this.")

    if drill:
        r = ("A rollback drill for this exact build has passed at least once.")
        r2 = ("The rollback drill for this build has been run and passed.")
    else:
        r = ("The rollback drill for this build has never been run.")
        r2 = ("No rollback drill has been executed for this build.")

    if fwd:
        f = ("The schema evolution is forward-only, meaning once the new head "
             "is out there it does not go back.")
        f2 = ("This is a one-way schema change — it is forward-only and "
              "cannot be reverted by reinstating the old build.")
    else:
        f = ("The schema evolution is not forward-only.")
        f2 = ("The schema can come back; it is not a one-way change.")

    if spike:
        s = ("Incoming traffic to the canary is a spike rather than a steady "
             "soak.")
        s2 = ("The canary is seeing a traffic spike, not a steady ramp.")
    else:
        s = ("Traffic to the canary is a normal steady soak.")
        s2 = ("The canary load is steady and ordinary.")

    core = (
        f"{t} {b} {r} {f} {s} "
        f"The runbook says promotion is gated on evidence, not on a test "
        f"count, and warns against treating traffic shape as an error. Is "
        f"this build safe to promote to the canary lane?"
    )
    core2 = (
        f"{t2} {b2} {r2} {f2} {s2} "
        f"Decide whether promotion to canary is safe."
    )
    return [core, core2]


def _comms(decision_id, st):
    breach = st.get("is_breach")
    data = st.get("data_incident")
    internal = st.get("customer_internal_only_request")
    fixed = st.get("fixed_contract")
    smb = st.get("is_smb_or_consumer")
    sev2 = st.get("sev2_or_above")

    if breach:
        a = ("The company itself has classified this as a reportable breach, "
             "which the manual states is a legal/regulatory disclosure that "
             "is not overridable.")
        a2 = ("This is an actual breach, and per policy a reportable breach is "
              "never suppressed.")
    else:
        a = ("This is not a breach — there is no reportable security "
             "disclosure here.")
        a2 = ("No breach; this is a routine incident category, not a legal "
              "disclosure.")

    if data:
        b = ("There is a confirmed data incident affecting a customer's "
             "records.")
        b2 = ("A real data incident is in play involving customer data.")
    else:
        b = ("There is no data incident — no customer data is implicated.")
        b2 = ("No data incident; customer data is untouched.")

    if internal:
        c = ("The customer has explicitly requested this be kept strictly "
             "internal.")
        c2 = ("The customer asked us not to bring outside parties in.")
    else:
        c = ("No request has been made to keep the incident internal.")
        c2 = ("The customer has not asked for internal-only handling.")

    if fixed:
        d = ("This is a fixed multi-year contract, not an SMB or consumer "
             "account.")
        d2 = ("The customer is on a long-term fixed contract.")
    else:
        d = ("This is not a fixed multi-year contract.")
        d2 = ("No fixed contract is in place for this customer.")

    if smb:
        e = ("The account is small-business or consumer-facing.")
        e2 = ("This is an SMB or consumer account.")
    else:
        e = ("The account is not SMB or consumer-facing — it is an "
             "enterprise-grade relationship.")
        e2 = ("Not an SMB/consumer account; this is a larger enterprise "
              "footprint.")

    if sev2:
        g = ("The incident is at severity two or above.")
        g2 = ("This is severity two or higher.")
    else:
        g = ("The incident is below severity two.")
        g2 = ("Severity is below the sev-two line.")

    core = (
        f"{a} {b} {c} {d} {e} {g} "
        f"Re-read section 5's escalation rule carefully — there is a known "
        f"trap around what customer preference may and may not suppress. "
        f"Who must be on the comms thread for this incident?"
    )
    core2 = (
        f"{a2} {b2} {c2} {d2} {e2} {g2} "
        f"Apply the escalation policy: who has to be included, and what is "
        f"the correct escalation category?"
    )
    return [core, core2]


def _restart(decision_id, st):
    loop = st.get("crash_looping")
    disk = st.get("disk_full")
    deploy = st.get("recent_deploy")
    batch = st.get("in_flight_batch")
    cpu = st.get("cpu_pegged")

    if loop:
        a = ("The node is crash-looping — the process comes up, dies, and "
             "comes up again on a tight cycle, so nothing is stable.")
        a2 = ("We are seeing the classic crash-loop: the service repeatedly "
              "starts and then falls over within seconds.")
    else:
        a = ("The node is not crash-looping; the process is running and "
             "staying up.")
        a2 = ("No crash-loop here — the process is alive and stable.")

    if disk:
        b = ("The disk is at capacity — a full filesystem is staring back at "
             "us, and restarting would just make the boot spin on a store it "
             "cannot write.")
        b2 = ("Storage is full; the filesystem cannot take more writes, and "
              "a restart will not clear that.")
    else:
        b = ("The disk has free space; storage is not the constraint.")
        b2 = ("Disk is fine — plenty of free capacity.")

    if deploy:
        c = ("This node crashed immediately after a recent deploy, which "
             "makes the deploy the prime suspect and rolling back the right "
             "move rather than bouncing it again.")
        c2 = ("The box went down right behind a new deploy; the deploy is "
              "the prime suspect, so restarting just hides the evidence.")
    else:
        c = ("There has been no recent deploy on this node.")
        c2 = ("No fresh deploy is in the picture here.")

    if batch:
        d = ("The process was mid-way through writing a batch, so a restart "
             "would destroy in-flight work — better to hold and let it drain.")
        d2 = ("A batch write is currently in flight; killing the node now "
              "would corrupt or drop it.")
    else:
        d = ("No batch write is in flight; nothing important is mid-write.")
        d2 = ("There is no in-flight batch to lose.")

    if cpu:
        e = ("A pegged-CPU counter is spiking loudly on the dashboard.")
        e2 = ("Some monitor is screaming about CPU being pinned.")
    else:
        e = ("CPU counters look ordinary.")
        e2 = ("No CPU alarm is going off.")

    core = (
        f"{a} {b} {c} {d} {e} "
        f"The runbook warns that the default is HOLD and restart is a "
        f"narrow, high-context exception — rule out the disk, the deploy, "
        f"and the batch first. Should this node be force-restarted now?"
    )
    core2 = (
        f"{a2} {b2} {c2} {d2} {e2} "
        f"Decide: force-restart this node, or hold?"
    )
    return [core, core2]


def _lci(decision_id, st):
    nz = st.get("exit_nonzero")
    tmo = st.get("timeout")
    flaky = st.get("known_flaky")
    retry = st.get("retried_passed")
    verbose = st.get("verbose_logs")

    if retry:
        a = ("The run was re-executed and has already passed — the replay "
             "succeeded, so the matter is resolved regardless of how loud the "
             "first attempt was.")
        a2 = ("We re-ran it and it passed this time; there is nothing left "
              "to review.")
    else:
        a = ("This run has not been re-executed to a pass — no retry has "
             "come back green yet.")
        a2 = ("There is no successful re-run; the original attempt is all "
              "we have.")

    if nz:
        b = ("The step exited non-zero.")
        b2 = ("The invocation returned a non-zero exit code.")
    else:
        b = ("The step completed with a zero exit code.")
        b2 = ("The invocation exited cleanly — zero.")

    if tmo:
        c = ("The run hit its timeout.")
        c2 = ("The invocation timed out before finishing.")
    else:
        c = ("The run did not time out.")
        c2 = ("No timeout was hit.")

    if flaky:
        d = ("This failure belongs to a KNOWN-FLAKY signal, which the "
             "runbook is designed to ignore — flagging it would burn a human "
             "on already-triaged noise.")
        d2 = ("The source is on the known-flaky list, so it is supposed to "
              "be filtered out by design.")
    else:
        d = ("Nothing here is on the known-flaky list; no explanation "
             "absolves it.")
        d2 = ("The signal is not known-flaky — there is no triaged excuse.")

    if verbose:
        e = ("The log is extremely verbose, spraying thousands of lines.")
        e2 = ("The log output is huge and noisy.")
    else:
        e = ("The log output is concise.")
        e2 = ("The log is quiet.")

    core = (
        f"{a} {b} {c} {d} {e} "
        f"Per the manual, only a red run with no known-flaky explanation "
        f"needs a human; a success-on-replay and a known-flaky signature "
        f"are both noise. Does this CI or log run need a human to review "
        f"it?"
    )
    core2 = (
        f"{a2} {b2} {c2} {d2} {e2} "
        f"Decide: flag this run for human review, or treat it as noise?"
    )
    return [core, core2]


RENDER = {
    "severity-class": _sev,
    "fix-posture": _fix,
    "page-second-responder": _page,
    "canary-promote": _canary,
    "comms-escalation": _comms,
    "force-restart": _restart,
    "log-ci-verdict": _lci,
}