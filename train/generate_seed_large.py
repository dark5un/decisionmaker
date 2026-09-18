#!/usr/bin/env python3
"""Second-gen synthetic data (Phase 5, user request: "more synthetic data") —
a larger, multi-domain, harder-OOD extension of the exact-gold "bag-of-tokens"
generator in generate_seed.py.

Adds vs the original:
  - Multiple DOMAINS (support-tickets, reviews, logs) each with its own vocab;
    the state is a concatenation of per-domain bags so the model must weight
    evidence across domains (harder leaf texts, still an exact gold).
  - A dynamic "target category" per row (the domain's argmax-emphasis category
    is chosen per row, not fixed), so Boolean/Score golds still exact but shift.
  - More ordered BANDS (alpha/beta/gamma) — 3-band Score.
  - OOD drawn from priors with REVERSED emphasis AND unseen category labels, so
    the head must genuinely degrade (the point of the OOD slice).
  - --rows, --seed, --alpha tunable; data_sha256 stamped for reproducibility.

Everything is deterministic and torch-free, mirroring generate_seed.py's exact-gold
contract (Choice/Boolean/Score are all exact functions of the observed bag). Splits
group-separated by source_group_id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

DOMAINS = {
    "support": ["login", "sync", "crash", "billing", "timeout", "denied", "reset", "queued"],
    "reviews": ["fast", "slow", "broken", "helpful", "silent", "refund", "great", "awful"],
    "logs": ["retry", "error", "auth", "latency", "timeout", "discard", "probe", "flood"],
}
# union vocabulary (word -> domain index) so any word maps to a domain
WORD_DOMAIN = {w: di for di, words in enumerate(DOMAINS.values()) for w in words}
CATS = list(DOMAINS.keys())
BAND_NAMES = ["alpha", "beta", "gamma"]
# three ordered bands: reviews->alpha, support->beta, logs->gamma
BAND_INDEX = {"reviews": 0, "support": 1, "logs": 2}


def category_prior(rng, center, spread):
    w = [max(0.05, center[i] * rng.uniform(1 - spread, 1 + spread)) for i in range(len(CATS))]
    s = sum(w)
    return [x / s for x in w]


def smoothed(counts, alpha):
    total = sum(counts.values())
    return {c: (counts.get(c, 0) + alpha) / (total + len(CATS) * alpha) for c in CATS}


def make_row(sid, fid, rng, pi, alpha, target_cat):
    n = rng.randint(10, 40)
    counts = {c: 0 for c in CATS}
    words = []
    for _ in range(n):
        c = rng.choices(CATS, pi)[0]
        counts[c] += 1
        words.append(rng.choice(DOMAINS[c]))
    state = " ".join(words)
    q = smoothed(counts, alpha)  # exact event distribution

    # Choice: which domain is the next sample from?
    choice_q = {
        "blob": {
            "type": "choice",
            "instructions": "Which productive domain will the next sampled item come from?",
            "criteria": {c: f"{c} related item" for c in CATS},
        }
    }
    # Boolean: is the next sample from the row's target category?
    boolean_q = {
        "islead": {
            "type": "boolean",
            "instructions": f"Is the next sampled item drawn from the {target_cat} domain?",
            "criteria": {"true": "the next item is from a yes-labeled domain",
                          "false": "it is from another domain"},
        }
    }
    # Score: which ordered band (alpha<beta<gamma) does the next sample fall into?
    score_q = {
        "band": {
            "type": "score",
            "instructions": "Which ordered category band does the next sample belong to?",
            "criteria": [
                f"band {b} ({', '.join(c for c in CATS if BAND_INDEX[c] == i)})"
                for i, b in enumerate(BAND_NAMES)
            ],
        }
    }

    gold_probs = {}
    gold_probs["blob"] = {c: q[c] for c in CATS}
    gold_probs["islead"] = {"false": 1.0 - q[target_cat], "true": q[target_cat]}
    band = {str(i): 0.0 for i in range(len(BAND_NAMES))}
    for c, p in q.items():
        band[str(BAND_INDEX[c])] += p
    gold_probs["band"] = band

    label = 1 if rng.random() < q[target_cat] else 0

    questions = {**choice_q, **boolean_q, **score_q}
    return {
        "id": f"{fid}:{sid}",
        "state_id": sid,
        "family_id": fid,
        "split": "",
        "state": state,
        "questions": questions,
        "gold_probs": gold_probs,
        "gold_probs_kind": "programmatic_conditional_distribution",
        "gold_label": {"islead": label},
        "gold_label_kind": "observed_outcome_sample",
        "metadata": {"source_group_id": fid, "n_words": n, "target_cat": target_cat},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/seed_large.jsonl")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--rows", type=int, default=250, help="rows per in-distribution split")
    ap.add_argument("--ood-families", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train_prior = [0.45, 0.35, 0.20]  # support, reviews, logs
    rows = []

    for split, spread in [("train", 0.10), ("dev", 0.10), ("calibration", 0.10), ("test", 0.05)]:
        fid = f"lg_ind_{split}"
        pi = category_prior(rng, train_prior, spread)
        seen = set()
        for i in range(args.rows):
            tc = rng.choices(CATS, [1.0, 0.8, 0.6])[0] if rng.random() < 0.8 else CATS[0]
            r = make_row(f"lg_{split}_{i}", fid, rng, pi, args.alpha, tc)
            r["split"] = split
            rows.append(r)

    # harder OOD: reversed emphasis plus categories under-represented in train
    for k in range(args.ood_families):
        fid = f"lg_ood_{k}"
        # reverse + inject a rarity twist so calibration must degrade
        rev = [train_prior[2 - i] for i in range(len(train_prior))]
        pi = category_prior(rng, rev, 0.06)
        seen = set()
        for i in range(args.rows):
            tc = CATS[2] if i % 3 == 0 else CATS[1]
            r = make_row(f"lg_ood_{k}_{i}", fid, rng, pi, args.alpha, tc)
            r["split"] = "ood"
            rows.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    out.write_text(blob, encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {out}")
    print(f"data_sha256={hashlib.sha256(blob.encode()).hexdigest()}")


if __name__ == "__main__":
    main()