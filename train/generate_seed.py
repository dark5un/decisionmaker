#!/usr/bin/env python3
"""Synthetic seed-data generator (Phase 5) with EXACT ground-truth distributions.

A deterministic, torch-free generator so the calibration story is checkable with
pure stdlib. Task: "bag-of-tokens" — each state is a multiset of words drawn from
a known category prior pi; every question's gold distribution is an exact,
computable function of the observed bag, so it is a ground-truth reference.

Event model: after observing the bag, the "next sampled word's category" has
probability q_c = (count_c + alpha) / (N + K*alpha) (additive smoothing). All
three question types are defined as functions of this exact q:

  - Choice over categories: gold[c] = q_c
  - Boolean "is the next sample from category X?": gold P(yes) = q_X
  - Score over ordered category BANDS: gold[band] = sum of q_c over the band

Splits are GROUP-SEPARATED by source_group_id (one generating family never leaks
across two splits). OOD families use category priors sampled outside the
training range, so calibration must degrade there — that is the point of the OOD
slice.

Determinism: a seed drives every random draw; the output file carries a
data_sha256 so a run is reproducible from the exact bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

VOCAB = {
    "vehicles": ["sedan", "truck", "bus", "moped", "ferry", "tram", "rail", "bike"],
    "food": ["rice", "bread", "soup", "curry", "salad", "stew", "tofu", "eggs"],
    "tools": ["hammer", "chisel", "rasp", "clamp", "auger", "lathe", "file", "vice"],
    "rigs": ["drone", "rover", "gantry", "winch", "pylon", "crane", "torque", "axle"],
}
CATS = list(VOCAB.keys())          # ordered; used for deterministic band definitions
BAND_NAMES = ["alpha", "beta"]     # two ordered bands: first half, second half
BAND_INDEX = {c: (0 if i < len(CATS) // 2 else 1) for i, c in enumerate(CATS)}


def smoothed(state_counts, alpha):
    """Exact event distribution q over categories given observed counts."""
    total = sum(state_counts.values())
    k = len(CATS)
    out = {c: (state_counts.get(c, 0) + alpha) / (total + k * alpha) for c in CATS}
    return out


def choose_prior(rng, center, spread):
    """Sample a categorical prior pi over categories, ~center with noise."""
    weights = [max(0.05, center[i] * rng.uniform(1 - spread, 1 + spread)) for i in range(len(CATS))]
    s = sum(weights)
    return [w / s for w in weights]


def make_row(sid, fid, rng, pi, alpha):
    """One JSONL row: a state bag + choice/boolean/score questions with exact gold."""
    n = rng.randint(8, 32)
    counts = {c: 0 for c in CATS}
    words = []
    for _ in range(n):
        c = rng.choices(CATS, pi)[0]
        counts[c] += 1
        words.append(rng.choice(VOCAB[c]))
    state = " ".join(words)
    q = smoothed(counts, alpha)  # exact event distribution

    # Choice: which domain will the next sample come from?
    choice_questions = {
        "blob": {
            "type": "choice", "instructions": "Which domain will the next sampled item come from?",
            "criteria": {c: f"{c} domain" for c in CATS},
        }
    }
    # Boolean: is the next sample from the taget (most-loaded) category?
    target = max(CATS, key=lambda c: counts[c])
    boolean_questions = {
        "islead": {
            "type": "boolean",
            "instructions": f"Is the next sampled item drawn from the {target} domain?",
            "criteria": {"true": "the next item is from a category labeled 'yes'", "false": "it is from another category"},
        }
    }
    # Score: which ordered band does the next sample fall into?
    score_questions = {
        "band": {
            "type": "score",
            "instructions": "Which ordered category band does the next sampled item belong to?",
            "criteria": [f"band {b} ({', '.join(c for c in CATS if BAND_INDEX[c] == i)})" for i, b in enumerate(BAND_NAMES)],
        }
    }

    # gold_probs keyed by qid, in candidate-id order
    gold_probs = {}
    gold_probs["blob"] = {c: q[c] for c in CATS}
    gold_probs["islead"] = {"false": 1.0 - q[target], "true": q[target]}
    band_probs = {str(i): 0.0 for i in range(len(BAND_NAMES))}
    for c, p in q.items():
        band_probs[str(BAND_INDEX[c])] += p
    gold_probs["band"] = band_probs

    # Observed binary outcome for the boolean, sampled once from the exact q and
    # frozen in the file (reproducible). ECE needs real outcomes.
    label = 1 if rng.random() < q[target] else 0

    questions = {**choice_questions, **boolean_questions, **score_questions}
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
        "metadata": {"source_group_id": fid, "n_words": n},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/seed.jsonl", help="output JSONL path")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--per-split", type=int, default=200, help="rows per in-distribution split")
    ap.add_argument("--ood-families", type=int, default=6, help="families drawn from out-of-distribution priors")
    ap.add_argument("--alpha", type=float, default=0.1, help="additive smoothing in the exact q")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train_prior = [0.4, 0.35, 0.15, 0.1]
    rows = []

    # In-distribution splits, each from DISTINCT families (group separation).
    for split, spread in [("train", 0.10), ("dev", 0.10), ("calibration", 0.10), ("test", 0.05)]:
        fid = f"ind_{split}"
        pi = choose_prior(rng, train_prior, spread)
        for i in range(args.per_split):
            r = make_row(f"{split}_{i}", fid, rng, pi, args.alpha)
            r["split"] = split
            rows.append(r)

    # OOD: families whose priors lean on categories the training distribution under-weights.
    for k in range(args.ood_families):
        fid = f"ood_{k}"
        shifted = [train_prior[2 - i] for i in range(len(train_prior))]  # reverse emphasis
        pi = choose_prior(rng, shifted, 0.08)
        for i in range(args.per_split):
            r = make_row(f"ood_{k}_{i}", fid, rng, pi, args.alpha)
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