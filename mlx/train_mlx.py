#!/usr/bin/env python3
"""MLX training harness -- mirrors train/train.py's role (Phase 5), no torch/CUDA.

Named train_mlx.py, not train.py: a plain module file here named exactly
`train.py` would shadow the repo root's `train/` namespace package (it has no
__init__.py) for any `import train.x` done from within this directory --
CPython gives a regular module priority over a namespace-package portion
regardless of sys.path order, so `from train.train import ...` would resolve
to this file instead of the real one. Confirmed by hitting exactly that
failure before renaming.

Reuses train/train.py's pure top section (RunConfig, load_rows, data_sha256,
group_splits, gold_vector, target_vector, build_leaf_texts_question) and
train/losses.py + train/ece.py unchanged (both already pure stdlib, verified
torch-free at import time in mlx/spike_forward.py's investigation). Only the
tensor path -- model build, batch packing, forward/backward, optimizer step --
is MLX-native here.

Boolean semantics match the engine/torch trainer exactly: a boolean question
has ONE real leaf (the "true" path) whose scalar z yields P(true)=sigmoid(z);
the loss treats gold as the 2-vector ["false","true"] against logits [0, z].

Loss arms (--loss): ce | brier | paired -- same three arms, same public-math
(train/losses.py); the paired arm is still the experiment, not a given.

Head-only warm start freezes the backbone via mlx's `Module.freeze()`:
`model.trainable_parameters()` then contains only the head's ~3k params, so
`nn.value_and_grad` never needs to backprop through the 28-layer backbone
(the head's params are downstream of the backbone's output, not upstream of
it -- same memory motivation as the torch trainer's `requires_grad_(False)`,
verified directly: see the freeze check this file's build was validated with).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

MLX_DIR = Path(__file__).resolve().parent
REPO_ROOT = MLX_DIR.parent
for p in (MLX_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from train.train import (  # pure top section; torch only imported inside its own functions
    RunConfig, load_rows, data_sha256, group_splits, gold_vector, target_vector,
    build_leaf_texts_question,
)
from train.losses import paired_reward_single, sample_cat  # pure stdlib
from train.ece import expected_calibration_error, reliability_diagram, render_reliability

from mlx_lm import load as mlx_load
from model import DecisionModel

DEFAULT_MLX_MODEL_PATH = MLX_DIR / "converted" / "qwen3-0.6b-c1899de"


def _build_model_and_tokenizer(cfg: RunConfig, mlx_model_path: str):
    full_model, tok = mlx_load(mlx_model_path)
    backbone = full_model.model
    model = DecisionModel(backbone)
    if not cfg.finetune:
        # FREEZE the backbone (head warm-start): trainable_parameters() then
        # excludes it entirely, so value_and_grad below never builds a
        # backward graph through the 28 transformer layers.
        model.backbone.freeze()
    cfg.device = str(mx.default_device())
    cfg.dtype = str(backbone.embed_tokens.weight.dtype)
    cfg.n_head_params = sum(v.size for _, v in tree_flatten(model.head.trainable_parameters()))
    return model, tok


class _Batch:
    """One packed forward's arrays + per-group bookkeeping (mirrors train/train.py's _Batch,
    minus the `attention` field -- mlx_lm's Qwen3Model needs none; see mlx/forward.py)."""

    __slots__ = ("tokens", "lengths", "types", "gold_vecs", "target_vecs", "labels")

    def __init__(self, tokens, lengths, types, gold_vecs, target_vecs, labels):
        self.tokens, self.lengths = tokens, lengths
        self.types = types
        self.gold_vecs = gold_vecs
        self.target_vecs = target_vecs
        self.labels = labels


def _pack_batch(rows_batch, tok, cfg: RunConfig) -> _Batch:
    paths, types, gold_vecs, target_vecs, labels = [], [], [], [], []
    for r in rows_batch:
        for qid, q in r["questions"].items():
            leaves = build_leaf_texts_question(r, qid)
            for text in leaves:
                toks = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]
                if len(toks) > cfg.max_length:
                    raise ValueError(
                        f"leaf too long ({len(toks)} > {cfg.max_length}) in row {r['id']} "
                        f"question {qid}; input NOT truncated"
                    )
                paths.append(toks)
            types.append(q["type"])
            gv = gold_vector(q["type"], q.get("criteria"), r["gold_probs"][qid])
            gold_vecs.append(gv)
            target_vecs.append(target_vector(gv, cfg.target))
            if q["type"] == "boolean":
                labels.append(int(r.get("gold_label", {}).get(qid, 1.0) >= 0.5))
            else:
                labels.append(None)

    width = max(len(p) for p in paths)
    tokens = mx.full((len(paths), width), tok.pad_token_id, dtype=mx.int32)
    for i, p in enumerate(paths):
        tokens[i, : len(p)] = mx.array(p, dtype=mx.int32)
    lengths = mx.array([len(p) for p in paths], dtype=mx.int32)
    return _Batch(tokens, lengths, types, gold_vecs, target_vecs, labels)


def _group_logits(scalars: mx.array, batch: _Batch) -> List[mx.array]:
    """Split the packed (P,) scalars into per-group logit vectors (engine semantics)."""
    groups, offset = [], 0
    for typ, gold in zip(batch.types, batch.gold_vecs):
        n = 1 if typ == "boolean" else len(gold)
        group = scalars[offset : offset + n]
        offset += n
        if typ == "boolean":
            z = group[0]
            groups.append(mx.stack([mx.zeros_like(z), z]))
        else:
            groups.append(group)
    return groups


def loss_ce(logits_group: mx.array, gold_vec) -> mx.array:
    z = logits_group - logits_group.max()
    logp = z - mx.logsumexp(z)
    goldt = mx.array(gold_vec, dtype=mx.float32)
    return -(goldt * logp).sum()


def loss_brier(logits_group: mx.array, gold_vec) -> mx.array:
    p = mx.softmax(logits_group)
    goldt = mx.array(gold_vec, dtype=mx.float32)
    return ((p - goldt) ** 2).sum()


def loss_paired(logits_group: mx.array, gold_vec, rng: random.Random, M: int, trials: int) -> mx.array:
    """RLCD paired proper-reward (train/losses.py) -- same construction as
    train/train.py's torch arm: sample on the CPU with plain Python (the
    stochastic-gradient estimator is exact public math, nothing MLX needs to
    trace through), then build a differentiable surrogate
    -sum_i (r_i - b_i) log p[A_i] so the gradient flows through p=softmax(z).
    """
    p_list = mx.softmax(logits_group).tolist()
    gold_list = list(gold_vec)
    prng = random.Random(rng.randrange(1 << 30))
    K = len(p_list)
    acc = [0.0] * K
    for _ in range(trials):
        Y = sample_cat(prng, gold_list)
        _, terms = paired_reward_single(prng, p_list, Y, M)
        for a, w in terms:
            if 0 <= a < K:
                acc[a] += w
    acc = [a / trials for a in acc]
    weights = mx.array(acc, dtype=mx.float32)
    p = mx.softmax(logits_group)
    return -(weights * mx.log(p + 1e-12)).sum()


def _train_step(model: DecisionModel, optimizer: optim.Optimizer, loss_name: str,
                batch: _Batch, rng: random.Random, cfg: RunConfig) -> float:
    def compute_loss(batch: _Batch) -> mx.array:
        scalars = model(batch.tokens, batch.lengths)
        groups = _group_logits(scalars, batch)
        total = mx.array(0.0)
        for g, tgt in zip(groups, batch.target_vecs):
            if loss_name == "paired":
                total = total + loss_paired(g, tgt, rng, cfg.paired_m, cfg.paired_trials)
            elif loss_name == "brier":
                total = total + loss_brier(g, tgt)
            else:
                total = total + loss_ce(g, tgt)
        return total

    loss_and_grad = nn.value_and_grad(model, compute_loss)
    loss, grads = loss_and_grad(batch)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return float(loss.item())


def evaluate(model: DecisionModel, tok, splits: dict, cfg: RunConfig) -> dict:
    predictions: dict = {}
    for split, rows in splits.items():
        preds = []
        for r in rows:
            for qid, q in r["questions"].items():
                leaves = build_leaf_texts_question(r, qid)
                toks = [tok.encode(t, add_special_tokens=False) + [tok.eos_token_id] for t in leaves]
                width = max(len(t) for t in toks)
                tokens = mx.full((len(toks), width), tok.pad_token_id, dtype=mx.int32)
                for i, x in enumerate(toks):
                    tokens[i, : len(x)] = mx.array(x, dtype=mx.int32)
                lengths = mx.array([len(x) for x in toks], dtype=mx.int32)
                scalars = model(tokens, lengths)
                mx.eval(scalars)
                raw = scalars.tolist()
                if q["type"] == "boolean":
                    z = raw[0]
                    pt = 1.0 / (1.0 + math.exp(-z))
                    preds.append({
                        "row": r["id"], "qid": qid, "type": "boolean", "p_true": float(round(pt, 6)),
                        "label": int(r.get("gold_label", {}).get(qid, 1.0) >= 0.5),
                        "gold": r["gold_probs"][qid],
                    })
                else:
                    order = (list(q["criteria"].keys()) if q["type"] == "choice"
                             else [str(i) for i in range(len(q["criteria"]))])
                    m = max(raw)
                    exps = [math.exp(v - m) for v in raw]
                    s = sum(exps)
                    p = [e / s for e in exps]
                    preds.append({
                        "row": r["id"], "qid": qid, "type": q["type"],
                        "pred": [float(round(v, 6)) for v in p], "labels": order,
                        "gold": [float(r["gold_probs"][qid][k]) for k in order],
                    })
        predictions[split] = preds
    return predictions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/seed.jsonl")
    ap.add_argument("--run-dir", default="runs/mlx_seed_ce")
    ap.add_argument("--mlx-model-path", default=str(DEFAULT_MLX_MODEL_PATH))
    ap.add_argument("--loss", choices=["ce", "brier", "paired"], default="ce")
    ap.add_argument("--target", choices=["soft", "hard"], default="soft")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-candidates", type=int, default=128)
    ap.add_argument("--ece-threshold", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--paired-m", type=int, default=6)
    ap.add_argument("--paired-trials", type=int, default=8)
    args = ap.parse_args()

    data_path = Path(args.data)
    rows = load_rows(data_path)
    splits = group_splits(rows)
    print(f"loaded {len(rows)} rows; splits: { {k: len(v) for k, v in splits.items()} }")

    cfg = RunConfig(
        loss=args.loss, target=args.target, finetune=args.finetune, epochs=args.epochs, lr=args.lr,
        batch_candidates=args.batch_candidates, ece_threshold=args.ece_threshold,
        seed=args.seed, paired_m=args.paired_m, paired_trials=args.paired_trials,
        run_dir=args.run_dir, data_sha256=data_sha256(data_path),
    )
    rng = random.Random(cfg.seed)

    if not Path(args.mlx_model_path).is_dir():
        print(f"ERROR: no MLX model at {args.mlx_model_path}; convert it first (see mlx/spike_forward.py)")
        return 1

    model, tok = _build_model_and_tokenizer(cfg, args.mlx_model_path)

    run_path = Path(cfg.run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print("run config:", json.dumps(asdict(cfg), indent=2))

    train_rows = splits.get("train", [])
    if not train_rows:
        print("ERROR: no train split in the data")
        return 1

    optimizer = optim.AdamW(learning_rate=cfg.lr, weight_decay=cfg.wd)

    start = time.time()
    step = 0
    for epoch in range(cfg.epochs):
        rng.shuffle(train_rows)
        # greedy candidate-packing: accumulate rows until the batch has enough leaves
        batch_rows: list = []
        cur_leaves = 0
        for r in train_rows:
            n_leaves = sum(len(build_leaf_texts_question(r, qid)) for qid in r["questions"])
            if batch_rows and cur_leaves + n_leaves > cfg.batch_candidates:
                batch = _pack_batch(batch_rows, tok, cfg)
                _train_step(model, optimizer, cfg.loss, batch, rng, cfg)
                step += 1
                batch_rows = [r]
                cur_leaves = n_leaves
            else:
                batch_rows.append(r)
                cur_leaves += n_leaves
        if batch_rows:
            batch = _pack_batch(batch_rows, tok, cfg)
            _train_step(model, optimizer, cfg.loss, batch, rng, cfg)
            step += 1
        print(f"[epoch {epoch+1}/{cfg.epochs}] steps={step} elapsed={time.time()-start:.1f}s", flush=True)

    # MLX-native weights, not a torch state_dict -- load with nn.Module.load_weights
    # (see mlx/run_server.py's DECISIONMAKER_HEAD_CHECKPOINT).
    model.head.save_weights(str(run_path / "checkpoint.safetensors"))
    (run_path / "train.log").write_text(
        f"loss={cfg.loss} epochs={cfg.epochs} lr={cfg.lr} steps={step} "
        f"elapsed={time.time()-start:.1f}s data_sha256={cfg.data_sha256}\n"
    )
    print("checkpoint:", run_path / "checkpoint.safetensors")

    predictions = evaluate(model, tok, splits, cfg)
    (run_path / "predictions.json").write_text(json.dumps(predictions, indent=2))

    paper = []
    gate_ok = True
    for split in ("test", "ood"):
        sub = predictions.get(split, [])
        ptrue = [x["p_true"] for x in sub if x["type"] == "boolean"]
        plab = [x["label"] for x in sub if x["type"] == "boolean"]
        ece = expected_calibration_error(ptrue, plab) if ptrue else float("nan")
        reli = reliability_diagram(ptrue, plab) if ptrue else None
        paper.append(f"[{split}] rows={len(sub)} boolean ECE={ece:.4f} (threshold {cfg.ece_threshold})")
        paper.append(render_reliability(reli) if reli else "(no boolean predictions)")
        if not (ece < cfg.ece_threshold):
            gate_ok = False
    (run_path / "ece_report.txt").write_text("\n".join(paper))
    print("\n".join(paper))
    print(f"\nECE gate {'PASS' if gate_ok else 'FAIL'}")
    return 0 if gate_ok else 2


if __name__ == "__main__":
    sys.exit(main())
