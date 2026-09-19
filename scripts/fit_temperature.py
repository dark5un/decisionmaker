#!/usr/bin/env python3
"""Fit a single post-hoc softmax temperature T for the decision head.

Why: the head is trained with CE against the SOFT gold distribution (= heavy label
smoothing), which the literature shows yields well-calibrated but under-confident
(flat) predictions. Post-hoc temperature scaling (Guo et al. 2017) is the standard
accuracy-preserving sharpener: divide logits by T before softmax. T < 1 sharpens.

T is fit by minimizing NLL against the SAME gold distribution used for training
(a proper scoring rule), on the hold-out `calibration` split — exactly the splits
the generator already produces. So this script never touches train/test/ood.

The model is built exactly like the deployed serve entrypoint (fp32 backbone +
bf16 autocast forward) so the fitted T matches what /v1/decisionmaker serves.

Usage (inside a torch image; CUDA_VISIBLE_DEVICES set before import):
    CUDA_VISIBLE_DEVICES=<idx> python scripts/fit_temperature.py \
        --run-dir runs/seed_ce --data data/seed.jsonl

Writes <run-dir>/temperature.json:
    { "T": float, "ce_at_T1": float, "ce_at_T": float,
      "boolean_ece_at_T1": float, "boolean_ece_at_T": float,
      "method": "posthoc_temperature_scaling", "minimized": "nll_vs_gold", ... }
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


def ternary_min(f, lo: float, hi: float, iters: int = 80) -> tuple[float, float]:
    """Golden-section search over log T minimizing f (convex in log-logits scale)."""
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if f(math.exp(m1)) < f(math.exp(m2)):
            hi = m2
        else:
            lo = m1
    t = math.exp((lo + hi) / 2)
    return t, f(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/seed_ce")
    ap.add_argument("--data", default="data/seed.jsonl")
    ap.add_argument("--gpu", default="", help="index or GPU-<uuid>; default CUDA_VISIBLE_DEVICES or 0")
    args = ap.parse_args()

    gpu = resolve_gpu_index(args.gpu)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)

    # torch import AFTER CUDA_VISIBLE_DEVICES
    import torch
    from transformers import AutoModel, AutoTokenizer

    from server.src.model import DecisionModel
    import train.train as T  # pure helpers only (load_rows, leaf builder, gold_vector)

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())

    # Backbone EXACTLY like serve's build_engine: fp32 params, bf16 autocast forward.
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    backbone = AutoModel.from_pretrained(
        cfg["model"], revision=cfg["revision"],
        attn_implementation="sdpa", dtype=torch.float32,
    ).cuda().eval()
    backbone.config.use_cache = False
    model = DecisionModel(backbone).cuda().eval()          # constructs fresh head internally
    ckpt = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    model.head.load_state_dict(ckpt["head"])               # inject the trained head
    del backbone

    rows = T.load_rows(Path(args.data))
    cal = [r for r in rows if r["split"] == "calibration"]
    print(f"calibration rows: {len(cal)}", flush=True)

    tokenize = lambda toks: tok.encode(toks, add_special_tokens=False) + [tok.eos_token_id]

    groups, golds, types = [], [], []
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for r in cal:
                for qid, q in r["questions"].items():
                    leaves = T.build_leaf_texts_question(r, qid)
                    toks = [tokenize(x) for x in leaves]
                    width = max(len(x) for x in toks)
                    P = len(toks)
                    t = torch.full((P, width), tok.pad_token_id, dtype=torch.long, device="cuda")
                    lens = torch.tensor([len(x) for x in toks], dtype=torch.long, device="cuda")
                    for i, x in enumerate(toks):
                        t[i, : len(x)] = torch.tensor(x, dtype=torch.long, device="cuda")
                    attn = torch.arange(width, device="cuda")[None, :] < lens[:, None]
                    sc = model(t, attn, lens).squeeze(-1).float().cpu().numpy()  # (P,) or ()
                    if q["type"] == "boolean":
                        z = float(np.atleast_1d(sc)[0])
                        grp = np.array([0.0, z])
                    else:
                        grp = np.atleast_1d(sc)
                    gv = T.gold_vector(q["type"], q.get("criteria"), r["gold_probs"][qid])
                    groups.append(grp); golds.append(np.array(gv)); types.append(q["type"])

    def nll(temp: float) -> float:
        total, n = 0.0, 0
        for grp, gv in zip(groups, golds):
            z = grp / temp
            z = z - z.max()
            p = np.exp(z); p = p / p.sum()
            total -= float((gv * np.log(np.clip(p, 1e-12, 1.0))).sum())
            n += 1
        return total / n

    def ece(temp: float) -> float:
        """Boolean-only ECE on calibration split (real labels exist only for boolean)."""
        # Discover the boolean question id dynamically (not hardcoded 'islead'),
        # so this works for any plan's schema.
        bobool = None
        for r in cal:
            for qid, q in r["questions"].items():
                if q["type"] == "boolean":
                    bobool = qid
                    break
            if bobool:
                break
        if bobool is None:
            print("no boolean question in calibration split; boolean ECE skipped")
            return float("nan")
        bin_edges = np.linspace(0, 1, 11)
        confs, accs = [], []
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for r in cal:
                    leaves = T.build_leaf_texts_question(r, bobool)
                    toks = [tokenize(x) for x in leaves]
                    width = max(len(x) for x in toks)
                    t = torch.full((len(toks), width), tok.pad_token_id, dtype=torch.long, device="cuda")
                    lens = torch.tensor([len(x) for x in toks], dtype=torch.long, device="cuda")
                    for i, x in enumerate(toks):
                        t[i, : len(x)] = torch.tensor(x, dtype=torch.long, device="cuda")
                    attn = torch.arange(width, device="cuda")[None, :] < lens[:, None]
                    z = float(np.atleast_1d(model(t, attn, lens).squeeze(-1).item())[0])
                    p = 1.0 / (1.0 + math.exp(-z / temp))
                    confs.append(p); accs.append(float(r["gold_label"][bobool] >= 0.5))
        confs, accs = np.array(confs), np.array(accs)
        s = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            m = (confs >= lo) & (confs < hi)
            if m.sum() > 0:
                s += abs(accs[m].mean() - confs[m].mean()) * (m.sum() / len(confs))
        return float(s)

    ce1, ece1 = nll(1.0), ece(1.0)
    Tfit, ceT = ternary_min(nll, lo=math.log(0.05), hi=math.log(5.0))
    eceT = ece(Tfit)

    print(f"T=1.0   : NLL(vs gold)={ce1:.4f}  boolean_ECE={ece1:.4f}")
    print(f"T={Tfit:.4f}: NLL={ceT:.4f}  boolean_ECE={eceT:.4f}")
    print(f"-> sharpen confidence by factor 1/T = {1 / Tfit:.3f}")

    out = {
        "T": round(float(Tfit), 6),
        "ce_at_T1": round(ce1, 6),
        "ce_at_T": round(ceT, 6),
        "boolean_ece_at_T1": round(ece1, 6),
        "boolean_ece_at_T": round(eceT, 6),
        "method": "posthoc_temperature_scaling",
        "minimized": "nll_vs_gold_calibration_split",
        "samples": len(cal),
        "head_checkpoint": str(run_dir / "checkpoint.pt"),
    }
    (run_dir / "temperature.json").write_text(json.dumps(out, indent=2))
    print("wrote", run_dir / "temperature.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())