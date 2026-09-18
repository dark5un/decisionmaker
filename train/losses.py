"""Loss objectives for the decision head (Phase 5) — pure stdlib.

These are the PUBLIC-math loss terms (proper-scoring theory, Gneiting & Raftery
2007; the paired proper-reward estimator is a stochastic-gradient estimator of
expected Brier). This module is torch-free so the MATH is verifiable by numeric
Monte Carlo before it runs on GPU. The torch harness (train/train.py) mirrors
these formulas exactly with autograd.

Key identity tested here (from the derivation notes): for student distribution p,
observed label Y~q, and M draws A_i~p with replacement,
    E[R] = 2 p·q - ||p||^2          and        E[grad L_PG] = grad ||p - q||^2
"""
from __future__ import annotations

import math
import random
from typing import List, Sequence


def softmax(z: Sequence[float]) -> List[float]:
    m = max(z)
    ex = [math.exp(v - m) for v in z]
    s = sum(ex)
    return [e / s for e in ex]


def cross_entropy(logits: Sequence[float], gold: Sequence[float], eps: float = 1e-12) -> float:
    """-sum(gold * log_softmax(logits)); gold is the soft target distribution."""
    p = softmax(list(logits))
    return -sum(g * math.log(max(pi, eps)) for g, pi in zip(gold, p))


def vector_brier(pred: Sequence[float], gold: Sequence[float]) -> float:
    """Vector Brier = sum_k (pred_k - gold_k)^2. Twice the scalar Berbooleanli Brier
    for a two-class event. Both proper scoring."""
    return sum((a - b) ** 2 for a, b in zip(pred, gold))


def one_hot(k: int, size: int) -> List[float]:
    return [1.0 if i == k else 0.0 for i in range(size)]


def sample_cat(rng: random.Random, probs: Sequence[float]) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1


def paired_reward_single(rng: random.Random, p: Sequence[float], Y: int, M: int) -> tuple:
    """One paired draw: returns (reward R, list of (A_i, r_i - b_i) debiased terms).

    A_i ~ p with replacement; c[k] counts occurrences. Assumes p is a proper
    distribution (sums to 1).
    """
    K = len(p)
    A = [sample_cat(rng, p) for _ in range(M)]
    c = [0] * K
    for a in A:
        c[a] += 1
    R = (2.0 / M) * sum(1 for a in A if a == Y) - sum(ci * (ci - 1) for ci in c) / (M * (M - 1))

    terms = []
    # Conditional, A_i-detached baseline.
    base_top = (2.0 / M) * p[Y]
    for ii, a in enumerate(A):
        r_i = (2.0 / M) * (1.0 if a == Y else 0.0) - 2.0 * (c[a] - 1) / (M * (M - 1))
        # sum over j != i of p[A_j]
        others = sum(p[A[j]] for j in range(M) if j != ii)
        b_i = base_top - 2.0 * others / (M * (M - 1))
        terms.append((a, r_i - b_i))
    return R, terms


def pg_gradient_mc(p: Sequence[float], q: Sequence[float], M: int, trials: int,
                   rng: random.Random) -> List[float]:
    """Monte Carlo estimate of E[grad_z L_PG] over the LOGITS z (p=softmax(z)).

    L_PG = -sum_i (r_i - b_i) log p[A_i]; with the baseline detached. The softmax
    gradient is d/dz_b log p[a] = onehot_b(a) - p_b, giving
        grad_z L_PG = sum_i (r_i - b_i) * (p_b - onehot_b(A_i))
    which by policy-gradient has expectation grad_z ||p - q||^2.
    """
    K = len(p)
    grad = [0.0] * K
    for _ in range(trials):
        Y = sample_cat(rng, q)  # observed label ~ target q
        _, terms = paired_reward_single(rng, p, Y, M)
        for a, w in terms:
            for b in range(K):
                grad[b] += w * (p[b] - (1.0 if a == b else 0.0))
    return [g / trials for g in grad]


def expected_reward_mc(p: Sequence[float], q: Sequence[float], M: int, trials: int,
                       rng: random.Random) -> float:
    total = 0.0
    for _ in range(trials):
        Y = sample_cat(rng, q)
        R, _ = paired_reward_single(rng, p, Y, M)
        total += R
    return total / trials


def analytic_grad_z(p: Sequence[float], q: Sequence[float]) -> List[float]:
    """grad_z ||p - q||^2 for p=softmax(z) = J_softmax(z)^T * 2(p - q).

    J_{a,b} = p_a (1[a==b] - p_b) is the softmax Jacobian.
    """
    K = len(p)
    delta = [2 * (p[i] - q[i]) for i in range(K)]
    grad = [0.0] * K
    for a in range(K):
        for b in range(K):
            jac = p[a] * ((1.0 if a == b else 0.0) - p[b])
            grad[b] += jac * delta[a]
    return grad


def analytic_expected_reward(p: Sequence[float], q: Sequence[float]) -> float:
    return 2 * sum(a * b for a, b in zip(p, q)) - sum(a * a for a in p)