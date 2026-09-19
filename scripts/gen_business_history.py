"""Generate the plan-06 business-history CSV fixture deterministically.

Columns: segment, campaign, dealsize, quarter (time block), decision, candidate
(the choice being scored: renew | lapse), outcome (1 = account won/retained),
arm (treatment). The head never sees the LAST quarter (test_block).

Synthetic but structured: close/win rate depends on segment/campaign/dealsize and
the treatment arm. Choosing 'lapse' almost never wins (low, non-zero, so both
candidate arms carry real empirical mass). Seeded -> stable fixture.
"""
import csv
import random
from pathlib import Path

rng = random.Random(20260919)

SEGMENTS = ["enterprise", "midmarket", "sme"]
CAMPAIGNS = ["brand", "product", "perf", "partner"]
QUARTERS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1"]  # last = held-out
ARMS = ["control", "trial"]


def win_p(segment, campaign, dealsize, arm, candidate):
    p = {"enterprise": 0.55, "midmarket": 0.42, "sme": 0.30}[segment]
    p += {"brand": 0.05, "product": 0.0, "perf": -0.03, "partner": 0.08}[campaign]
    p += 0.02 * (dealsize - 2)          # larger accounts win more
    if arm == "trial":
        p += 0.12                       # trial arm lifts win rate
    if candidate == "lapse":
        p = 0.03                        # lapsing rarely wins the account back
    return max(0.02, min(0.95, p))


rows = []
for q in QUARTERS:
    for _ in range(2600):
        seg = rng.choice(SEGMENTS)
        camp = rng.choice(CAMPAIGNS)
        size = rng.randint(1, 4)
        arm = rng.choice(ARMS)
        cand = rng.choice(["renew", "renew", "lapse"])  # 2:1 renew bias
        p = win_p(seg, camp, size, arm, cand)
        outcome = 1 if rng.random() < p else 0
        rows.append({"segment": seg, "campaign": camp, "dealsize": size,
                     "quarter": q, "decision": "renew", "candidate": cand,
                     "outcome": outcome, "arm": arm})

out = Path(__file__).resolve().parents[1] / "research" / "data" / "business_history.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["segment", "campaign", "dealsize", "quarter",
                                      "decision", "candidate", "outcome", "arm"])
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows -> {out}")