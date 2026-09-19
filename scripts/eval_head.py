#!/usr/bin/env python3
"""Shared evaluator (Step 0.4) — the ONE tool all three plans are measured with.

Loads a trained head checkpoint exactly like run_server.build_engine (fp32
backbone, bf16 autocast forward), runs the SHARED held-out question set, and
prints the cross-plan comparable metrics:

  - per-question predicted distribution + confidence
  - boolean ECE on any held-out boolean questions (vs human gold)
  - per-question-type pred-peak (max prob) and the resulting margin
    (pred-peak vs gold-peak ceiling)
  - mean confidence per question type

The held-out set is fully-held-out skills whose decisions no training set saw.
Question rows use the same schema as train data (state + questions + optional
gold_probs/gold_label). Optional temperature loaded from temperature.json beside
the checkpoint (mirrors serve precedence via T).

Usage:
    python scripts/eval_head.py <checkpoint> --questions holdout_questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_SKILL = Path(__file__).resolve().parent.parent  # repo root (script in scripts/)


def resolve_gpu_index(spec: str) -> str:
    if not spec:
        return os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if spec.startswith("GPU-"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout
        for line in out.splitlines():
            idx, uuid = [p.strip() for p in line.split(",")]
            if uuid == spec:
                return idx
        raise ValueError(f"GPU UUID {spec!r} not present")
    return spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path, help="path to a runs/<arm>/checkpoint.pt")
    ap.add_argument("--questions", type=Path, required=True, help="holdout_questions.jsonl")
    ap.add_argument("--gpu", default="", help="index or GPU-<uuid>; default CUDA_VISIBLE_DEVICES or 0")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override fitted temperature (default: from temperature.json beside checkpoint, else 1.0)")
    args = ap.parse_args()

    gpu = resolve_gpu_index(args.gpu)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)

    import torch
    from transformers import AutoModel, AutoTokenizer
    from server.src.model import DecisionModel
    import train.train as T

    ckpt_path = Path(args.checkpoint)
    cfg_dir = ckpt_path.parent
    cfg = json.loads((cfg_dir / "config.json").read_text())

    # Load head exactly like serve (fp32 params, bf16 autocast forward).
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    backbone = AutoModel.from_pretrained(
        cfg["model"], revision=cfg["revision"],
        attn_implementation="sdpa", dtype=torch.float32,
    ).cuda().eval()
    backbone.config.use_cache = False
    model = DecisionModel(backbone).cuda().eval()
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.head.load_state_dict(ckpt["head"])
    del backbone

    # Temperature: explicit override > temperature.json beside checkpoint > 1.0.
    temp = args.temperature
    if temp is None:
        temp_json = cfg_dir / "temperature.json"
        temp = float(json.load(open(temp_json))["T"]) if temp_json.is_file() else 1.0
    print(f"head={ckpt_path} temperature={temp:.4f}", flush=True)

    rows = T.load_rows(args.questions)

    def evaluate_rows(rows):
        """Run one forward per (row, qid), return per-question records."""
        out = []
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for r in rows:
                    for qid, q in r["questions"].items():
                        leaves = T.build_leaf_texts_question(r, qid)
                        toks = [tok.encode(t, add_special_tokens=False) + [tok.eos_token_id] for t in leaves]
                        width = max(len(x) for x in toks)
                        P = len(toks)
                        t = torch.full((P, width), tok.pad_token_id, dtype=torch.long, device="cuda")
                        lens = torch.tensor([len(x) for x in toks], dtype=torch.long, device="cuda")
                        for i, x in enumerate(toks):
                            t[i, : len(x)] = torch.tensor(x, dtype=torch.long, device="cuda")
                        attn = torch.arange(width, device="cuda")[None, :] < lens[:, None]
                        sc = model(t, attn, lens).squeeze(-1).float().cpu().numpy()
                        gold = r.get("gold_probs", {}).get(qid)
                        label = r.get("gold_label", {}).get(qid)
                        if q["type"] == "boolean":
                            z = float(np.atleast_1d(sc)[0]) / temp
                            pt = 1.0 / (1.0 + math.exp(-z))
                            rec = {
                                "row": r["id"], "qid": qid, "type": "boolean",
                                "dist": {"false": 1.0 - pt, "true": pt},
                                "p_true": round(pt, 6), "confidence": max(pt, 1.0 - pt),
                                "gold": gold, "gold_label": label,
                                "pred_peak": max(pt, 1.0 - pt),
                            }
                            if gold:
                                rec["gold_peak"] = max(gold.values())
                        else:
                            order = (list(q["criteria"].keys()) if q["type"] == "choice"
                                     else [str(i) for i in range(len(q["criteria"]))])
                            z = sc / temp
                            z = z - z.max()
                            e = np.exp(z)
                            p = e / e.sum()
                            dist = {k: float(round(float(v), 6)) for k, v in zip(order, p)}
                            rec = {
                                "row": r["id"], "qid": qid, "type": q["type"],
                                "dist": dist, "confidence": float(round(float(p.max()), 6)),
                                "gold": gold, "orders": order,
                                "pred_peak": float(round(float(p.max()), 6)),
                            }
                            if gold:
                                gold_vec = [float(gold[k]) for k in order]
                                rec["gold_peak"] = float(max(gold_vec))
                        out.append(rec)
        return out

    # Optional boolean split of the held-out rows into the set.
    recs = evaluate_rows(rows)

    from train.ece import expected_calibration_error
    print("\n=== per-question records ===")
    for rec in recs:
        print(json.dumps(rec, sort_keys=True))

    # Aggregate: boolean ECE, pred-peak vs gold-peak ceiling per type, mean conf.
    print("\n=== aggregate ===")
    by_type = {}
    for rec in recs:
        by_type.setdefault(rec["type"], []).append(rec)

    for typ in ("boolean", "choice", "score"):
        if typ not in by_type:
            continue
        sub = by_type[typ]
        peaks = [r["pred_peak"] for r in sub]
        gpeaks = [r["gold_peak"] for r in sub if r.get("gold_peak") is not None]
        mean_conf = sum(r["confidence"] for r in sub) / len(sub)
        mean_pred_peak = float(np.mean(peaks))
        gap = ""
        if gpeaks:
            mean_gold_peak = float(np.mean(gpeaks))
            ceil = mean_gold_peak
            gap = f" gold-peak-ceiling={ceil:.4f} gap={ceil - mean_pred_peak:.4f}"
        print(f"[{typ}] n={len(sub)} mean_conf={mean_conf:.4f} "
              f"mean_pred_peak={mean_pred_peak:.4f}{gap}")

    bools = [r for r in recs if r["type"] == "boolean" and r.get("gold_label") is not None]
    if bools:
        ptrue = [r["p_true"] for r in bools]
        lab = [int(r["gold_label"] >= 0.5) for r in bools]
        ece = expected_calibration_error(ptrue, lab)
        print(f"[boolean] n={len(bools)} ECE(vs human gold)={ece:.4f}")
    else:
        print("[boolean] no held-out boolean questions with labels; ECE not computed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
