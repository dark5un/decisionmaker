#!/usr/bin/env python3
"""Oracle evaluation of the seed set (Phase 5) — pure stdlib.

Uses the EXACT gold distribution AS the model prediction (the best possible
oracle). Establishes the baseline the trained head should approach: NLL=0,
Brier=0, and binary boolean ECE ~ 0 (the label was sampled from that same q). Wrong
calibration tooling or a broken generator would show up here as nonzero values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.ece import (
    brier, cross_entropy, expected_calibration_error, reliability_diagram, render_reliability,
)


def load(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/seed.jsonl")
    args = ap.parse_args()
    rows = load(Path(args.input))
    splits = sorted({r["split"] for r in rows})

    print(f"{len(rows)} rows; splits: {splits}\n")
    for split in splits:
        sub = [r for r in rows if r["split"] == split]
        noll, briers, nlls = 0.0, 0.0, 0.0
        n_boolean = 0
        ptrue, plabel = [], []
        n_q = 0
        for r in sub:
            for qid, q in r["questions"].items():
                gold = r["gold_probs"][qid]
                if q["type"] == "boolean":
                    ptrue.append(gold["true"])
                    plabel.append(r["gold_label"][qid])
                    n_boolean += 1
                # candidate order for gold vector
                if q["type"] == "choice":
                    order = list(q["criteria"].keys())
                elif q["type"] == "score":
                    order = [str(i) for i in range(len(q["criteria"]))]
                else:
                    order = ["false", "true"]
                gvec = [gold[k] for k in order]
                nlls += cross_entropy(gvec, gvec)
                briers += brier(gvec, gvec)
                n_q += 1
        ece = expected_calibration_error(ptrue, plabel) if ptrue else float("nan")
        reli = reliability_diagram(ptrue, plabel) if ptrue else None
        print(f"[{split}] rows={len(sub)} questions={n_q} "
              f"NLL={nlls / n_q:.5f} Brier={briers / n_q:.5f} boolean_ECE={ece:.5f}")
        if reli and split in ("test", "ood"):
            print(render_reliability(reli))


if __name__ == "__main__":
    main()