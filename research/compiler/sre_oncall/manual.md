# sre-oncall — An On-Call Incident-Response Runbook

REALISM: this manual mirrors a production on-call discipline for a hosted
service: an event-driven decision-classification service whose only product is
trust. Human readable; every section below is a source of truth for the
decision tables compiled against it.

=========================================================================
1. SEVERITY AND IMPACT — what class of incident is this?
=========================================================================

Severity is a function of *observed* impact, not of which component happens to
be loudest. Never let monitoring noise or a single datapoint decide. Wait until
two independent signals agree unless the very first signal is catastrophic.

Any incident that corrupts, loses, or leaks production data is severity one,
even if the blast radius looks small today. Data integrity outranks everything
else: an event classified as severity two but causing data loss is an
under-classification and must be re-raised. Trust is the product; a leak is not
an availability event.

A complete outage of the primary decision-serving path, with no workaround, is
severity one regardless of how many customers can name the symptom right now.
If a workaround exists that restores correctness for a meaningful subset of
users, the incident is severity two: the exposure is contained, but the primary
path is still down. Once the workaround is *verified* (a user confirms it, not
just a runbook guess), the residual event is usually severity three.

A partial or degraded-but-serving state — slow, throttled, or a non-primary
path — is severity three unless a contractual SLA is actually at risk of
breach, in which case it climbs to severity two. Severity three does not page a
second responder by default.

During a declared incident, the runbook's own commands may be used read-only to
confirm hypotheses. Never mutate production state during classification.

Potential quirk to notice when reading: an incident can be severity one because
of *risk of* breach to a contractual SLA together with no workaround, even
though nothing is actually down — the threat of the breach is the impact.

=========================================================================
2. ROLLBACK vs FORWARD-FIX — the fix posture
=========================================================================

The moment you believe a recent change caused the incident, decide between
rolling back and shipping a forward fix. The decision is driven by whether the
damage is reversible, not by which is faster to type.

If the offending change carried a schema or data migration that cannot be
reversed, you cannot roll back cleanly: the database has already moved past the
old code, and reverting the application would leave it pointing at a structure
it does not understand. In that case the fix must be forward. This holds even
when a rollback would otherwise be trivial and tempting.

If the change is reversible (no forward-only migration) and a forward fix is
not yet ready and verified, roll back first to restore a known-good state, then
re-investigate on stable ground. Rolling back was never about being fast; it is
about recovering a correct, explainable state.

If the change is reversible AND a correct forward fix is already built and
tested, prefer the forward fix only when the production incident is active and
an immediate rollback would itself be operationally risky (e.g. mid-distribution
to a broad fleet). Otherwise, with the incident quiet, the calmer and safer
choice is still to land the forward fix deliberately rather than bounce the
fleet.

Edge: a forward fix that is "ready" but has not been exercised against the real
incident is not actually ready. Unverified forward fixes never beat a clean
rollback.

=========================================================================
3. HOW MUCH TO PAGE — when a second responder joins
=========================================================================

Default posture for an on-call engineer is solo triage. A second responder is
paged only when one of a small number of conditions is true.

Data loss / leak, or active corruption — page immediately, do not wait to
confirm. If primary serving is fully down, page. If the incident risks breaching
a contractual SLA, page. If the incident is contained to a severity three with a
verified workaround, do not page.

The one subtlety: if the on-call has *already* confirmed a workaround that
restores correctness AND the incident is still only severity three AND no SLA
is at risk, stay solo even if the logs are noisy. Two signals still count as one
when the second signal is just re-echoing the first.

Reading that back: "solo triage" is the default *only* when all of {no data
impact, not fully down, no SLA risk} hold. Any single one of those conditions
flips you to paging.

=========================================================================
4. CANARY AND RELEASE GATE — is this build safe to promote?
=========================================================================

Promotion to the canary lane is gated on evidence, not on a test count. The
unit and integration suite must be green, the Card/Contract schema must be
backward compatible (an old client must survive a new head), and a rollback
drill for this build must have passed at least once. Green tests alone do not
promote a build whose schema evolution is forward-only AND untested for
back-compat.

If the incoming traffic to the canary is a spike rather than a steady soak, and
nothing has regressed, that is not a failure — do not treat traffic shape as an
error signal. But if the schema is forward-only and the back-compat drill has
never been run, promotion is blocked even if every test is green.

A safe canary promote requires: tests green, schema back-compat demonstrated,
and a passed rollback drill. If all three hold, promote. If any is missing,
hold. The judgment call that trips people up: a green suite plus a passed
rollback drill but a forward-only schema with NO back-compat evidence is still
a hold — the schema risk is the deciding factor, not the tests.

=========================================================================
5. CONTACT-PROXIMITY and COMMS — who must be on the thread
=========================================================================

A pay-as-you-go customer with a confirmed data incident is always escalated to
the account owner and the security contact, regardless of revenue tier. A
free-tier customer gets the same data-incident escalation — data is data.

The only exception that downgrades an otherwise mandatory escalation is an
explicit customer request to keep it internal, and even that does not apply
when the incident is a breach (a legal/regulatory disclosure is not overridable
by customer preference).

By default, a fixed multi-year contract (non SMB, non consumer) includes the
account owner on the thread for any severity two or above. A fixed-contract
customer who is small-business or consumer-facing does not get the account
owner by default unless severity one.

There is a known trap in the manual. Re-read it carefully: "the security
contact is always copied for data incidents" is stated as unconditional in
section 1, but section 5 makes even a breach report subject to the customer's
explicit internal-only request ONLY when it is not a breach. A breach that the
company itself classifies as reportable is never suppressed. Customer
preference can suppress routine data incidents but never a reportable breach.
=========================================================================

=========================================================================
6. FORCE-RESTART vs HOLD — should this node be restarted now?
=========================================================================

A forced restart of a node mends a hung or crash-looping process, but it is a
blunt instrument that can destroy in-flight work and mask the real cause. When
primary serving points at a node that is crash-looping, the first question is
NOT "can I restart it" but "is a restart the right tool, or will it burn the
house down".

A restart is the move only when the node is genuinely crash-looping AND none of
the three known confounders is present. The confounders are: a full disk, a
very recent deploy, and an in-flight batch write. All three independently say
HOLD, not restart.

If the disk is at capacity, a restart will not clear it — the process will
simply fail to boot again, or worse, spin on a filesystem it cannot write. Hold
and address storage first.

If the node crashed right after a deploy, a restart only masks the symptom: the
deploy is the prime suspect and the correct move is a rollback, not a bouncing
the process. Restarting hides the evidence you need.

If the process was mid-way through writing a batch, a restart destroys that
batch. Never restart during a write you can let finish; hold and let it drain.

Only when the node is crash-looping AND none of {full disk, recent deploy,
in-flight batch} is true is a force-restart the right call. A pegged-CPU
counter spiking loudly is a distractor: it explains nothing on its own and
must not push you to restart a node that is otherwise healthy or confounded.

Reading that back: the default is HOLD. Restart is a narrow, high-context
exception, not a reflex. If you are reaching for the restart button, first
rule out the disk, the deploy, and the batch.

=========================================================================
7. LOG/CI VERDICT — does this run need a human?
=========================================================================

Every automated log line and CI invocation ends in a yes/no question that the
pipeline answers before a human ever looks: does this run genuinely need an
eyeball, or is it part of the noise the system is designed to filter?

A non-zero exit code on its own does not mean a human must look. A non-zero
exit that belongs to a KNOWN-FLAKY test is by design ignored — flagging it
would burn an on-call on a signal the runbook has already triaged. The same
holds for a timeout that belongs to a known-flaky source.

A re-run that has already PASSED resolves the matter: once the run succeeds on
replay, there is nothing left to review, regardless of how loud the first
attempt was.

A human must be flagged when the evidence points at a genuine failure and no
explanation absolves it: a run that exits non-zero OR times out, from a signal
that is NOT known-flaky. One of {non-zero exit, timeout} plus not-known-flaky
is the flag condition. Two independent symptoms echo the same condition; they
still count as one flag.

The clause that matters: compilers and linters returning non-zero from a
noisy-but-benign source (a strict-new warning turned on mid-suite, a flaky
network call) are classified relative to the flaky list — to a human
eyeballing it, a red run can be "reviewed" yet still be noise. Only a red run
with no known-flaky explanation needs a human.
=========================================================================