"""Calibration harness (Phase 5) — pure stdlib, no torch.

Uses TEN FIXED bins over p(true) (binary/boolean events), NOT top-label ECE (see
docs/architecture.md §4). A reliability
diagram is produced alongside ECE, NLL, and Brier. Metrics are computed from
SAVED predictions, never by re-running inference.

"Calibrated" gate: ~90% of the 0.9-predictions are correct, etc. ECE is not a
guarantee — it is the reported, inspectable quantity.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def expected_calibration_error(preds_probs: List[float], labels: List[int],
                               n_bins: int = 10) -> float:
    """ECE over n fixed bins of p(true); weighted |bin_accuracy - bin_confidence|.

    preds_probs: predicted P(true) in [0,1]; labels: 0/1 observed outcomes.
    Bins are [0/n,1/n), [1/n,2/n), ..., with the final bin closed at 1.0.
    """
    if len(preds_probs) != len(labels):
        raise ValueError("preds_probs and labels must be same length")
    if not preds_probs:
        return 0.0
    n = n_bins
    ece = 0.0
    for i in range(n):
        lo, hi = i / n, (i + 1) / n
        idxs = [j for j in range(len(preds_probs))
                if (lo <= preds_probs[j] < hi) or (i == n - 1 and preds_probs[j] == 1.0)]
        if not idxs:
            continue
        bin_conf = _mean([preds_probs[j] for j in idxs])
        bin_acc = _mean([float(labels[j]) for j in idxs])
        ece += (len(idxs) / len(preds_probs)) * abs(bin_acc - bin_conf)
    return ece


def reliability_diagram(preds_probs: List[float], labels: List[int],
                        n_bins: int = 10) -> List[Optional[Dict[str, float]]]:
    """Per-bin records for the report/reliability diagram playback."""
    per_bin: List[Optional[Dict[str, float]]] = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idxs = [j for j in range(len(preds_probs))
                if (lo <= preds_probs[j] < hi) or (i == n_bins - 1 and preds_probs[j] == 1.0)]
        if not idxs:
            per_bin.append(None)
        else:
            per_bin.append({
                "low": lo, "high": hi, "count": len(idxs),
                "accuracy": _mean([float(labels[j]) for j in idxs]),
                "confidence": _mean([preds_probs[j] for j in idxs]),
            })
    return per_bin


def render_reliability(reli: List[Optional[Dict[str, float]]], max_bar: int = 40) -> str:
    """ASCII reliability diagram: ideal diagonal vs observed accuracy per bin."""
    lines = ["Calibrated accuracy=confidence (perfect) should lie on the diagonal."]
    for i, cell in enumerate(reli):
        low = i / 10
        if cell is None:
            lines.append(f"[0.{low:.0f}] empty bin")
            continue
        acc = cell["accuracy"]
        conf = cell["confidence"]
        bar = int(round(acc * max_bar))
        gap = int(round(abs(acc - conf) * max_bar))
        mark = "|" if abs(acc - conf) < 0.03 else ("v" if acc < conf else "^")
        lines.append(f"[{low:>4.1f}] acc {acc:.3f} conf {conf:.3f} "
                     f"{'#'*bar:<{max_bar}} {mark}{' '*gap} n={cell['count']}")
    return "\n".join(lines)


def cross_entropy(gold: List[float], pred: List[float], eps: float = 1e-12) -> float:
    """Softmax CE over a gold distribution (proper scoring). gold,pred same K."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred must be same length")
    return -sum(g * math.log(max(p, eps)) for g, p in zip(gold, pred))


def brier(gold: List[float], pred: List[float]) -> float:
    """Vector Brier = sum_k (pred_k - gold_k)^2 (proper scoring). Doubles the
    scalar Berbooleanli Brier for a binary event over two classes."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred must be same length")
    return sum((p - g) ** 2 for p, g in zip(pred, gold))


def report(gold_rows, pred_rows) -> Dict[str, float]:
    """Aggregate NLL/Brier over exact-gold rows (optionally per-split caller)."""
    nlls, briers = [], []
    for gold, pred in zip(gold_rows, pred_rows):
        nlls.append(cross_entropy(gold, pred))
        briers.append(brier(gold, pred))
    return {"nll": _mean(nlls), "brier": _mean(briers), "rows": len(nlls)}