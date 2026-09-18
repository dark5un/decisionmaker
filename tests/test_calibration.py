"""Calibration + loss-math verification (Phase 5) — pure stdlib, no torch/GPU."""
import math
import random

import pytest

from train import ece as ECE
from train import losses as LOSS


# ---------------------------------------------------------------------------
# ECE harness
# ---------------------------------------------------------------------------


def test_ece_perfectly_calibrated_is_near_zero():
    preds, labels = [], []
    for i in range(10):  # 10 fixed bins
        center = (i + 0.5) / 10
        n = 20
        ones = round(center * n)
        preds += [center] * n
        labels += [1] * ones + [0] * (n - ones)
    val = ECE.expected_calibration_error(preds, labels)
    assert val < 0.02


def test_ece_overconfident_is_large():
    preds = [0.9] * 100
    labels = [1 if i % 2 else 0 for i in range(100)]  # 50/50 despite conf 0.9
    val = ECE.expected_calibration_error(preds, labels)
    assert val > 0.3


def test_ece_asserts_length_mismatch():
    with pytest.raises(ValueError):
        ECE.expected_calibration_error([0.5], [0, 1])


def test_reliability_diagram_counts():
    reli = ECE.reliability_diagram([0.05, 0.15, 0.95], [0, 0, 1])
    filled = [c for c in reli if c is not None]
    assert sum(c["count"] for c in filled) == 3


def test_proper_scoring_metrics_well_defined():
    gold = [0.7, 0.3]
    pred = [0.7, 0.3]
    assert LOSS.cross_entropy(gold, pred) < LOSS.cross_entropy(gold, [0.5, 0.5])
    assert LOSS.vector_brier(pred, gold) < LOSS.vector_brier([0.5, 0.5], gold)
    assert ECE.brier(gold, pred) == pytest.approx(LOSS.vector_brier(pred, gold))


# ---------------------------------------------------------------------------
# Paired proper-reward identities (public math, numeric Monte Carlo check)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p,q,M", [
        ([0.5, 0.5], [0.5, 0.5], 4),
        ([0.8, 0.2], [0.6, 0.4], 4),
        ([0.7, 0.2, 0.1], [0.3, 0.5, 0.2], 5),
        ([0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25], 6),
    ],
)
def test_paired_expected_reward_equals_analytic(p, q, M):
    rng = random.Random(7)
    est = LOSS.expected_reward_mc(p, q, M, trials=20000, rng=rng)
    analy = LOSS.analytic_expected_reward(p, q)
    assert est == pytest.approx(analy, abs=0.02)


@pytest.mark.parametrize(
    "p,q", [
        ([0.5, 0.5], [0.5, 0.5]),
        ([0.8, 0.2], [0.6, 0.4]),
        ([0.7, 0.2, 0.1], [0.3, 0.5, 0.2]),
    ],
)
def test_pg_gradient_equals_grad_brier(p, q):
    rng = random.Random(11)
    M = 6
    est = LOSS.pg_gradient_mc(p, q, M, trials=200000, rng=rng)
    want = LOSS.analytic_grad_z(p, q)
    for a, b in zip(est, want):
        assert a == pytest.approx(b, abs=0.05)


def test_softmax_and_ce_optimum():
    logits = [0.0, 1.0, 2.0]
    p = LOSS.softmax(logits)
    assert abs(sum(p) - 1.0) < 1e-12
    assert LOSS.cross_entropy(logits, p) < LOSS.cross_entropy(logits, [0.0, 0.5, 0.5])
    assert LOSS.one_hot(2, 3) == [0.0, 0.0, 1.0]