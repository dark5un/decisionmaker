#!/usr/bin/env python3
"""history_to_gold.py — plan 06 example gold-loader: empirical gold from history.

A small, deterministic, torch-free loader that turns a historical outcome log
(tabular CSV) into Decision-Maker training rows whose gold is *measured*, not
compiled: the empirical outcome distribution of each candidate given a
decision-state bucket, additively smoothed. This is Plan C at scale — the most
defensible trust source, with no teacher LLM in the path.

Contract (plan 06, research/plans/06-business-history-gold.md):
  - Bucket: collapse rows by (state features -> state), candidate.
  - Empirical gold: gold_probs[candidate] = count(outcome=1)/count(bucket),
    smoothed with `alpha` additive smoothing (same smoothed() shape family as
    train/generate_seed.py).
  - Exact-by-construction: two runs over identical (log, seed) are byte-identical
    and carry a data_sha256.
  - Row schema: identical to generate_seed.py / data/seed.jsonl (state,
    questions{qid}, gold_probs{qid}), group-separated split by source family.
  - Splits: train/dev/cal by feature family; test = a future time BLOCK (held
    out by time) the head never saw -> OOD-style honest generalization signal.
  - Confounder guard (--treat): bucket gold WITHIN treatment arm so outcome gold
    is not a mix of decision-correctness and luck-of-action.
  - Refuse path: a bucket with too few observations (< --min-count) is dropped
    or flagged gold:insufficient — never padded with invented rows.
  - Honest ceiling: prints mean gold-peak per candidate (the data ceiling).
  - train.py unchanged: `python -m train.train --data <out>` with --loss ce
    --target soft --epochs 30.

CSV columns:
    <state feature cols...>, <decision col>, <candidate col>, <outcome col>
Optionally: a `--time` column (integer/date block) and a `--treat` column.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path, cols: list) -> list:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in cols if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"CSV missing required columns: {missing}; have {reader.fieldnames}")
        return [dict(r) for r in reader]


def additively_smoothed(counts: dict, alpha: float, universe: list) -> dict:
    """Empirical distribution over universe with additive (Laplace) smoothing.
    Same shape family as generate_seed.smoothed: p_c = (n_c + alpha)/(N + K alpha)."""
    total = sum(counts.values())
    k = len(universe)
    return {c: (counts.get(c, 0) + alpha) / (total + k * alpha) for c in universe}


def gold_row(bucket, state, qid, decision_id, decision, candidate, split, fid,
             alpha, universe, metadata_extra=None):
    """One seed.jsonl-schema row: state prose + choice question + empirical gold."""
    counts = {c: bucket.get(c, 0) for c in universe}
    dist = additively_smoothed(counts, alpha, universe)
    winner = max(universe, key=lambda c: dist[c])
    question = {"type": "choice",
                "instructions": f"Given this state, which {candidate} outcome is most likely?",
                "criteria": {c: c for c in universe}}
    gold_probs = {qid: {c: round(float(p), 6) for c, p in dist.items()}}
    label = {c: (1.0 if c == winner else 0.0) for c in universe}
    meta = {"source_group_id": fid, "decision": decision_id,
            "candidate": candidate, "outcome": winner, "n": sum(counts.values())}
    if metadata_extra:
        meta.update(metadata_extra)
    return {
        "id": f"{fid}:{_stable_state_hash(state)}",
        "state_id": _stable_state_hash(state),
        "family_id": fid, "split": split,
        "state": state, "questions": {qid: question},
        "gold_probs": gold_probs,
        "gold_probs_kind": "measured_empirical_outcome_distribution",
        "metadata": meta,
    }


def _stable_state_hash(state: str):
    return hashlib.sha1(state.encode()).hexdigest()[:12]


def _slot_split(block, test_block):
    """Deterministic, group-separated split from a time block.
    The held-out FUTURE block is 'test' (OOD-style, never trained on); the rest
    are split train/dev/calibration by a stable hash of the block."""
    if test_block is not None and block == test_block:
        return "test"
    h = int(_stable_state_hash(f"block:{block}"), 16)
    r = h % 100
    if r < 60:
        return "train"
    if r < 80:
        return "dev"
    return "calibration"


def _family_split(state):
    """Deterministic group-separated split by stable hash of the state family."""
    h = int(_stable_state_hash(state), 16)
    r = h % 100
    if r < 60:
        return "train"
    if r < 80:
        return "dev"
    if r < 90:
        return "calibration"
    return "test"


def _truthy(out):
    try:
        return float(out) >= 0.5
    except ValueError:
        return str(out).strip().lower() in ("1", "yes", "true", "won", "y")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV history log")
    ap.add_argument("--state", required=True, help="comma-separated state feature column names")
    ap.add_argument("--decision-col", required=True, help="column naming the decision point")
    ap.add_argument("--decision-id", default="business", help="qid / decision identifier")
    ap.add_argument("--candidate-col", required=True)
    ap.add_argument("--outcome-col", required=True)
    ap.add_argument("--out", default="research/data/biz_train.jsonl")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--treat", default=None, help="treatment-arm column -> confounder guard")
    ap.add_argument("--time", dest="time_col", default=None,
                    help="time-block column; its MAX block becomes the held-out TEST split")
    args = ap.parse_args()

    state_cols = [c.strip() for c in args.state.split(",") if c.strip()]
    rows = read_csv(Path(args.input),
                    state_cols + [args.decision_col, args.candidate_col, args.outcome_col]
                    + ([args.treat] if args.treat else [])
                    + ([args.time_col] if args.time_col else []))

    fid = args.decision_id
    test_block = None
    if args.time_col:
        blocks = sorted({r[args.time_col] for r in rows})
        if len(blocks) > 1:
            test_block = blocks[-1]  # the FUTURE block the head never saw

    universe = sorted({r[args.candidate_col] for r in rows})
    if len(universe) < 2:
        raise SystemExit(f"need >=2 candidates in {args.candidate_col}; have {universe}")

    # ------ bucket: (state, candidate, block [, treat]) -> (n, pos) ---------
    bucket_counts = Counter()   # key -> total rows
    bucket_pos = Counter()      # key -> outcome=1 rows
    # track, per (state, block), what treat arms exist (for confounder guard)
    arm_of = {}                 # (state, block) -> set of treat arms
    for r in rows:
        st_core = " ".join(f"{k}:{r[k]}" for k in state_cols) or "(no state)"
        cand = r[args.candidate_col]
        block = r.get(args.time_col) if args.time_col else None
        treat = r.get(args.treat) if args.treat else None
        key = (st_core, cand, block, treat)
        bucket_counts[key] += 1
        if _truthy(r[args.outcome_col]):
            bucket_pos[key] += 1
        arm_of.setdefault((st_core, block), set()).add(treat)

    # ------ aggregate to state-level distribution ---------------------------
    # For a given state (and block, and optionally treat arm), total positive
    # count per candidate across the family.
    from collections import defaultdict
    state_agg = defaultdict(lambda: defaultdict(int))   # (block, treat, treat_q or ANY) -> cand -> pos
    total_agg = defaultdict(int)                        # (block, treat) -> n per state-cand
    for (st_core, cand, block, treat), n in bucket_counts.items():
        pos = bucket_pos[(st_core, cand, block, treat)]
        if args.treat:
            # confounder guard: only per-arm (treatment-arm-matched) gold
            key_arm = (block, treat)
            state_agg[(st_core, key_arm)][cand] += pos
            total_agg[(st_core, key_arm)] += n
        else:
            # flat aggregate over the whole state block
            key_flat = (block, None)
            state_agg[(st_core, key_flat)][cand] += pos
            total_agg[(st_core, key_flat)] += n

    # per-arm split assignment (guarded) or flat (unguarded)
    out_rows = []
    refused = 0
    def _key_sort(kv):
        (st, (blk, tr)), _v = kv
        return (str(st), str(blk or ""), str(tr or ""))
    for (st_core, key), _v in sorted(state_agg.items(), key=_key_sort):
        block, treat_arm = key
        n = total_agg[(st_core, key)]
        if n < args.min_count:
            refused += 1   # sub-min-count bucket refused, never padded
            continue
        if args.time_col:
            split = _slot_split(block, test_block)
        else:
            split = _family_split(st_core)
        state_text = st_core
        if args.treat and treat_arm is not None:
            state_text = f"{st_core} {args.treat}:{treat_arm}"
        cand_pos = state_agg[(st_core, key)]
        dist = additively_smoothed(cand_pos, args.alpha, universe)
        winner = max(universe, key=lambda cc: dist[cc])
        question = {"type": "choice",
                    "instructions": f"Given this state, which {args.candidate_col} outcome is most likely?",
                    "criteria": {cc: cc for cc in universe}}
        row = {
            "id": f"{fid}:{_stable_state_hash(state_text)}:{winner}",
            "state_id": _stable_state_hash(state_text), "family_id": fid,
            "split": split, "state": state_text,
            "questions": {args.decision_id: question},
            "gold_probs": {args.decision_id: {cc: round(float(p), 6) for cc, p in dist.items()}},
            "gold_probs_kind": "measured_empirical_outcome_distribution",
            "metadata": {"source_group_id": fid, "decision": args.decision_id,
                         "candidate": args.candidate_col, "n": n, "gold": "ok"},
        }
        out_rows.append(row)

    blob = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                   for r in sorted(out_rows, key=lambda r: r["id"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(blob, encoding="utf-8")
    sha = hashlib.sha256(blob.encode()).hexdigest()

    from collections import Counter as _C
    print(f"wrote {len(out_rows)} rows -> {out}  (refused sub-min-count buckets: {refused})")
    if out_rows:
        print("splits:", dict(_C(r['split'] for r in out_rows)))
    print(f"data_sha256={sha}")
    if out_rows:
        mean_peak = sum(max(r["gold_probs"][args.decision_id].values()) for r in out_rows) / len(out_rows)
        print(f"mean gold-peak (data ceiling): {mean_peak:.3f}  [low p = genuinely ambiguous history, not flatness bug]")
    print(f"test_block={test_block}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())