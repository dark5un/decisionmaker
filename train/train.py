#!/usr/bin/env python3
"""Phase 5 training harness — fine-tune the decision head, then calibrate.

REAL training entrypoint (no stubs). Loads the actual pinned backbone
Qwen/Qwen3-0.6B@c1899de, packs each batch's candidate-leaves into ONE forward
(mirroring server/src/engine._forward), optimizes the decision head (head
warm-start with frozen backbone, or full-model with `--finetune`), and writes a
FROZEN run dir: checkpoint + config (incl. data_sha256) + train log + per-split
predictions. Calibration metrics (ECE/NLL/Brier + reliability diagram) are
computed from the SAVED predictions, never by re-running inference.

Boolean semantics (must match the engine): a boolean question has ONE real leaf (the
"true" path) whose scalar z yields P(true)=sigmoid(z). The loss treats the gold
as the 2-vector ["false","true"] against logits [0, z], so the gradient flows
through z. Choice/Score use one leaf per candidate with a 2..K-way softmax.

Loss arms (--loss):
  ce     softmax cross-entropy on the gold distribution (PRIMARY; proper)
  brier  vector Brier = sum_k (p_k - gold_k)^2 (recovers the same q)
  paired RLCD-style paired proper-reward (train/losses.py), a stochastic-gradient
         estimator of expected Brier. THE EXPERIMENT: does it help calibration?
         Record honestly if it does not beat CE.

GPU: pinned hotfix-plane RTX 4070 Ti (see scripts/verify_gpu.sh; re-verified
live at load). fp32 weights ~2.4 GiB, +optimizer ~5 GiB — fits the 4070 Ti's
~8 GiB free, but STOP the serve container first (it holds the same card).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Pure config + data utilities (no torch; unit-testable here)
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    model: str = "Qwen/Qwen3-0.6B"
    revision: str = "c1899de289a04d12100db370d81485cdf75e47ca"
    loss: str = "ce"                       # ce | brier | paired
    target: str = "soft"                   # "soft" = CE vs gold distribution; "hard" = CE vs one-hot(argmax gold)
    finetune: bool = False                 # head-only unless --finetune
    epochs: int = 3
    lr: float = 3e-4
    batch_candidates: int = 128            # candidate leaves in one packed forward
    max_length: int = 512
    paired_m: int = 6                      # RLCD draws per candidate
    paired_trials: int = 8                 # decorrelated PG draws per group
    ece_threshold: float = 0.20            # gate on test+ood boolean ECE
    seed: int = 17
    wd: float = 1e-4
    data_sha256: str = ""
    run_dir: str = ""
    # filled at run time
    device: str = ""
    dtype: str = ""
    n_head_params: int = 0


def load_rows(path: Path) -> list:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def data_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_splits(rows: list) -> dict:
    """Split group->list-of-rows. Group separation holds because the generator
    assigns each source_group_id to exactly one split."""
    splits: dict = {}
    for r in rows:
        splits.setdefault(r["split"], []).append(r)
    return splits


def gold_vector(qtype: str, criteria, gold: dict) -> list:
    """gold_probs -> vector in candidate-id order (== engine._candidate_texts order)."""
    if qtype == "choice":
        order = list(criteria.keys())
    elif qtype == "score":
        order = [str(i) for i in range(len(criteria))]
    else:  # boolean -> (false, true)
        order = ["false", "true"]
    return [float(gold[k]) for k in order]


def target_vector(gold_vec: list, target: str) -> list:
    """Training target per question. 'soft' = the gold distribution itself (current
    behavior, equivalent to heavy label smoothing -> calibrated but flat). 'hard' =
    one-hot on the gold argmax -- the recipe that builds real logit margin and that
    the temp-scaling literature says calibrates best AFTER post-hoc sharpening."""
    if target == "soft":
        return gold_vec
    if target == "hard":
        v = [0.0] * len(gold_vec)
        v[int(max(range(len(gold_vec)), key=lambda i: gold_vec[i]))] = 1.0
        return v
    raise ValueError(f"unknown target {target!r} (expected soft|hard)")


def leaf_count_for_question(qtype: str) -> int:
    """Real leaves the engine builds per question type (boolean builds ONE: the
    "true" path; the false side is the sigmoid complement)."""
    if qtype == "boolean":
        return 1
    if qtype == "choice":
        raise ValueError("choice leaf count depends on criteria size; use build_leaf_texts")
    raise ValueError("score leaf count depends on criteria size")


# ---------------------------------------------------------------------------
# Torch runtime (imported only inside real runs; the container has torch/numpy)
# ---------------------------------------------------------------------------


def _build_model_and_tokenizer(cfg: RunConfig):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("training requires a CUDA device; re-verify with scripts/verify_gpu.sh")

    tok = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    # Backbone memory: bf16 halves weight memory vs fp32 (good for the frozen
    # head-only case); when full fine-tuning we want fp32 master weights for the
    # optimizer state. Head always stays fp32 (it's the trainable surface).
    bb_dtype = torch.float32 if cfg.finetune else torch.bfloat16
    backbone = AutoModel.from_pretrained(cfg.model, revision=cfg.revision,
                                         dtype=bb_dtype,
                                         attn_implementation="sdpa").cuda()

    from server.src.model import DecisionModel
    model = DecisionModel(backbone).cuda()
    if not cfg.finetune:
        # FREEZE the backbone (the plan's "head warm-start"): no autograd graph
        # for the 24 layers, so activations are not retained for backprop and
        # memory stays ~weights + head-only activations. Without this the
        # full-model backprop graph blows memory (observed OOM ~8 GiB).
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        model.backbone.eval()
    cfg.device = "cuda"
    cfg.dtype = str(model.backbone.dtype)
    cfg.n_head_params = sum(p.numel() for p in model.head.parameters())
    return model, tok


def build_leaf_texts_question(row, qid):
    """Exactly what server/src/engine.build_leaf_texts produces for one question."""
    from server.src import engine as E
    return E.build_leaf_texts(row["state"], row["questions"][qid])


class _Batch:
    """One packed forward's tensors + per-group bookkeeping."""

    __slots__ = ("tokens", "attention", "lengths", "types", "gold_vecs", "target_vecs", "labels")

    def __init__(self, tokens, attention, lengths, types, gold_vecs, target_vecs, labels):
        self.tokens, self.attention, self.lengths = tokens, attention, lengths
        self.types = types                  # ["choice"|"score"|"boolean"] per group
        self.gold_vecs = gold_vecs           # gold vector per group
        self.target_vecs = target_vecs       # training target vector per group (soft|hard)
        self.labels = labels                 # boolean 0/1 observed outcome per group (or None)


def _pack_batch(rows_batch, tok, cfg, rng) -> _Batch:
    """Tokenize the batch's candidate leaves into one packed tensor + group maps.

    Returns _Batch. sizes(n_leaves) and qtype are derived per example so the
    trainer can reconstruct each question's logit group exactly like the engine.
    """
    import torch
    dev = tok.pad_token_id is not None and "cuda"
    # We can't know the device before model construct in this pure fn; pack on
    # the same device by reading an env hint set by the run. Default cuda.
    device = torch.device("cuda")

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
    P = len(paths)
    tokens = torch.full((P, width), tok.pad_token_id, dtype=torch.long, device=device)
    lengths = torch.tensor([len(p) for p in paths], dtype=torch.long, device=device)
    for i, p in enumerate(paths):
        tokens[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=device)
    attention = torch.arange(width, device=device)[None, :] < lengths[:, None]
    return _Batch(tokens, attention, lengths, types, gold_vecs, target_vecs, labels)


def _group_logits(model, batch: _Batch, dev) -> list:
    """Run the ONE packed forward and split the per-leaf scalars into per-group
    logit vectors in engine semantics (boolean: [0, z]). Returns list of tensors."""
    import torch
    with torch.autocast(device_type=dev, dtype=torch.bfloat16):
        scalars = model(batch.tokens, batch.attention, batch.lengths)  # (P,)
    scalars = scalars.float()
    groups, offset = [], 0
    for typ, _gold in zip(batch.types, batch.gold_vecs):
        n = 1 if typ == "boolean" else len(_gold)
        group = scalars[offset:offset + n]
        offset += n
        if typ == "boolean":
            # single real leaf z -> logits [0, z] (P(true)=sigmoid(z)).
            z = group[0]
            groups.append(torch.stack([torch.zeros_like(z), z]))
        else:
            groups.append(group)
    return groups


def _softmax_dist(logits):
    """torch softmax with bf16 input tending to be fp32-returned; keep fp32."""
    return logits.float().softmax(0)


def loss_ce(logits_group, gold_vec):
    import torch
    z = logits_group.float() - logits_group.float().max()
    logp = z.log_softmax(0)
    goldt = torch.tensor(gold_vec, device=z.device, dtype=z.dtype)
    return -(goldt * logp).sum()


def loss_brier(logits_group, gold_vec):
    import torch
    p = _softmax_dist(logits_group)
    goldt = torch.tensor(gold_vec, device=p.device, dtype=p.dtype)
    return ((p - goldt) ** 2).sum()


def loss_paired(logits_group, gold_vec, rng, M, trials):
    """RLCD paired proper-reward (train/losses.py). Stochastic-gradient estimate
    of expected Brier with a detached debiased baseline:
        surrogate = -sum_i (r_i - b_i) log p[A_i],  A_i ~ p
    Gradient flows through p=softmax(z). M is the number of paired draws per
    policy sample; trials decorrelates the estimator (exact math, see the module).
    """
    import torch
    from train.losses import paired_reward_single, sample_cat
    p = _softmax_dist(logits_group)
    p_np = p.detach().cpu().numpy()
    gold_np = list(gold_vec)
    prng = random.Random(rng.randrange(1 << 30))
    acc = torch.zeros_like(p)
    for _ in range(trials):
        Y = sample_cat(prng, gold_np)
        _, terms = paired_reward_single(prng, p_np.tolist(), Y, M)
        for a, w in terms:
            if 0 <= a < len(acc):
                acc[a] += w
    acc.div_(trials)
    return -(acc * torch.log(p + 1e-12)).sum()


def _train_step(model, opt, loss_fn, batch_rows, tok, cfg, rng, dev):
    import torch
    batch = _pack_batch(batch_rows, tok, cfg, rng)
    opt.zero_grad()
    groups = _group_logits(model, batch, dev)
    total = torch.tensor(0.0, device=dev, requires_grad=True)
    for g, tgt in zip(groups, batch.target_vecs):
        if cfg.loss == "paired":
            total = total + loss_paired(g, tgt, rng, cfg.paired_m, cfg.paired_trials)
        else:
            total = total + loss_fn(g, tgt)
    total.backward()
    opt.step()
    return float(total.item())


def evaluate(model, tok, splits, cfg, dev):
    import torch
    import numpy as np  # lazy: the pure/eval helpers stay importable w/o numpy
    predictions = {}
    for split, rows in splits.items():
        preds = []
        for r in rows:
            for qid, q in r["questions"].items():
                leaves = build_leaf_texts_question(r, qid)
                toks = [tok.encode(t, add_special_tokens=False) + [tok.eos_token_id] for t in leaves]
                width = max(len(t) for t in toks)
                P = len(toks)
                t = torch.full((P, width), tok.pad_token_id, dtype=torch.long, device=dev)
                lens = torch.tensor([len(x) for x in toks], dtype=torch.long, device=dev)
                for i, x in enumerate(toks):
                    t[i, : len(x)] = torch.tensor(x, dtype=torch.long, device=dev)
                attn = torch.arange(width, device=dev)[None, :] < lens[:, None]
                model.eval()
                with torch.no_grad():
                    with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                        sc = model(t, attn, lens)
                probs = sc.cpu().numpy()
                if q["type"] == "boolean":
                    z = float(probs[0])
                    pt = 1.0 / (1.0 + math.exp(-z))
                    preds.append({
                        "row": r["id"], "qid": qid, "type": "boolean", "p_true": float(round(pt, 6)),
                        "label": int(r.get("gold_label", {}).get(qid, 1.0) >= 0.5),
                        "gold": r["gold_probs"][qid],
                    })
                else:
                    order = (list(q["criteria"].keys()) if q["type"] == "choice"
                             else [str(i) for i in range(len(q["criteria"]))])
                    z = probs - probs.max()
                    e = np.exp(z)
                    p = e / e.sum()
                    preds.append({
                        "row": r["id"], "qid": qid, "type": q["type"],
                        "pred": [float(round(v, 6)) for v in p], "labels": order,
                        "gold": [float(r["gold_probs"][qid][k]) for k in order],
                    })
        predictions[split] = preds
    return predictions


# ---------------------------------------------------------------------------
# GPU selection (torch-free; runs BEFORE torch import)
# ---------------------------------------------------------------------------


def _gpu_inventory() -> list[dict]:
    """Enumerate CUDA GPUs via nvidia-smi: [{index, uuid, mem_free, name}]."""
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,name",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    inv = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        inv.append({
            "index": int(parts[0]), "uuid": parts[1],
            "mem_free": int(parts[2]), "name": parts[3],
        })
    return inv


def resolve_gpu(spec: str) -> str:
    """Resolve a --gpu/--DECISIONMAKER_GPU spec to a CUDA_VISIBLE_DEVICES value.

    Accepts (case-insensitive):
      "auto"      -> the GPU with the most free memory (best default for training)
      "0","1",... -> the physical device index
      "GPU-..."   -> a specific device UUID (nvidia-smi naming)
    Returns a CUDA_VISIBLE_DEVICES string. Raises ValueError if nothing matches.
    """
    spec = (spec or "auto").strip()
    if spec.lower() == "auto":
        inv = sorted(_gpu_inventory(), key=lambda g: g["mem_free"], reverse=True)
        if not inv:
            raise ValueError("no CUDA GPU enumerated by nvidia-smi")
        chosen = inv[0]
        return f"{chosen['index']}"
    if spec.startswith("GPU-"):
        inv = _gpu_inventory()
        for g in inv:
            if g["uuid"] == spec:
                return f"{g['index']}"
        raise ValueError(f"GPU UUID {spec!r} not present (have: "
                         f"{', '.join(g['uuid'] for g in inv)})")
    # bare index
    idx = int(spec)
    inv = _gpu_inventory()
    if not any(g["index"] == idx for g in inv):
        raise ValueError(f"GPU index {idx} not present (have indices "
                         f"{[g['index'] for g in inv]})")
    return str(idx)


def _apply_gpu(spec: str) -> str:
    """Set CUDA_VISIBLE_DEVICES from a spec. MUST run before `import torch`.
    Returns the value applied. Honors an explicit CUDA_VISIBLE_DEVICES already
    in the environment when the spec is empty (bare-container passthrough)."""
    if spec:
        value = resolve_gpu(spec)
        os.environ["CUDA_VISIBLE_DEVICES"] = value
        return value
    # no --gpu given: respect the environment (quadlet pin) if set
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]
    value = resolve_gpu("auto")
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/seed.jsonl")
    ap.add_argument("--run-dir", default="runs/seed_ce")
    ap.add_argument("--loss", choices=["ce", "brier", "paired"], default="ce")
    ap.add_argument("--target", choices=["soft", "hard"], default="soft",
                    help="CE target: 'soft'=gold distribution (calibrated but flat); "
                         "'hard'=one-hot on gold argmax (builds logit margin; pair with temp scaling)")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-candidates", type=int, default=128)
    ap.add_argument("--ece-threshold", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--paired-m", type=int, default=6)
    ap.add_argument("--paired-trials", type=int, default=8)
    ap.add_argument(
        "--gpu", default=os.environ.get("DECISIONMAKER_GPU")
                 or os.environ.get("DECISIONMAKER_TRAIN_GPU_DEVICES", ""),
        help="GPU for training: 'auto' (freest) | index | GPU-<uuid>. "
             "Default: DECISIONMAKER_GPU or DECISIONMAKER_TRAIN_GPU_DEVICES env, "
             "else CUDA_VISIBLE_DEVICES, else auto.",
    )
    args = ap.parse_args()

    applied = _apply_gpu(args.gpu)

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

    import torch  # noqa: F401  (fail fast before any directory writes if torch missing)
    model, tok = _build_model_and_tokenizer(cfg)

    run_path = Path(cfg.run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print("run config:", json.dumps(asdict(cfg), indent=2))

    train_rows = splits.get("train", [])
    if not train_rows:
        print("ERROR: no train split in the data")
        return 1

    head_params = list(model.head.parameters())
    if cfg.finetune:
        opt = torch.optim.AdamW(
            [{"params": head_params}, {"params": model.backbone.parameters()}],
            lr=cfg.lr, weight_decay=cfg.wd,
        )
    else:
        opt = torch.optim.AdamW(head_params, lr=cfg.lr, weight_decay=cfg.wd)

    loss_fn = {"ce": loss_ce, "brier": loss_brier}.get(cfg.loss)
    model.train()
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
                _train_step(model, opt, loss_fn, batch_rows, tok, cfg, rng, "cuda")
                step += 1
                batch_rows = [r]
                cur_leaves = n_leaves
            else:
                batch_rows.append(r)
                cur_leaves += n_leaves
        if batch_rows:
            _train_step(model, opt, loss_fn, batch_rows, tok, cfg, rng, "cuda")
            step += 1
        print(f"[epoch {epoch+1}/{cfg.epochs}] steps={step} elapsed={time.time()-start:.1f}s",
              flush=True)

    torch.save({"head": model.head.state_dict(), "backbone": model.backbone.state_dict()},
               run_path / "checkpoint.pt")
    (run_path / "train.log").write_text(
        f"loss={cfg.loss} epochs={cfg.epochs} lr={cfg.lr} steps={step} "
        f"elapsed={time.time()-start:.1f}s data_sha256={cfg.data_sha256}\n"
    )
    print("checkpoint:", run_path / "checkpoint.pt")

    predictions = evaluate(model, tok, splits, cfg, "cuda")
    (run_path / "predictions.json").write_text(json.dumps(predictions, indent=2))

    from train.ece import expected_calibration_error, reliability_diagram, render_reliability
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