"""Synthetic expert of OpenL2D (Alves et al., 2025), reduced to what the experiments use.

Adapted from the OpenL2D code. On an instance x with true label y, the expert errs with probability

    P(FP | x) = sigmoid(fpr_beta + alpha * w.x / ||w||)   if y = 0,
    P(FN | x) = sigmoid(fnr_beta - alpha * w.x / ||w||)   if y = 1,

and fpr_beta, fnr_beta are set by bisection so that the mean error probability over the data equals
the target false positive and false negative rates. The expert's decisions flip y with that
probability, with a generator seeded by `seed`.
"""

import math

import numpy as np
import pandas as pd


def _sigmoid(value):
    if value < -15:
        return 0
    elif value > 15:
        return 1
    return 1 / (1 + math.exp(-value))


def _bisect(rate, mean_error, tolerance=0.00001):
    """The beta in [-200, 200] whose mean error probability equals `rate`."""
    a, b = -200, 200
    mean_a, mean_b = mean_error(a), mean_error(b)
    assert (mean_a - rate) * (mean_b - rate) < 0
    beta = (b + a) / 2
    mean = mean_error(beta)
    while np.abs(mean - rate) > tolerance:
        if (mean_a - rate) * (mean - rate) < 0:
            b, mean_b = beta, mean
        elif (mean_b - rate) * (mean - rate) < 0:
            a, mean_a = beta, mean
        beta = (b + a) / 2
        mean = mean_error(beta)
    return beta


def simulate(x: pd.DataFrame, y: np.ndarray, w: np.ndarray, fpr: float, fnr: float, alpha: float, seed: int):
    """Fit fpr_beta and fnr_beta to the target rates and draw the expert's decisions.

    Returns fpr_beta, fnr_beta and the decisions.
    """
    projection = (x * w / np.linalg.norm(w)).sum(axis=1)
    p_fp = lambda beta: (y == 0) * (beta + (alpha * projection)).apply(_sigmoid)  # noqa: E731
    p_fn = lambda beta: (y == 1) * (beta - (alpha * projection)).apply(_sigmoid)  # noqa: E731
    fpr_beta = _bisect(fpr, lambda beta: p_fp(beta)[y == 0].mean())
    fnr_beta = _bisect(fnr, lambda beta: p_fn(beta)[y == 1].mean())
    mask = np.random.default_rng(seed=seed).binomial(n=1, p=p_fn(fnr_beta) + p_fp(fpr_beta)).astype(bool)
    decisions = y.copy()
    decisions[mask] = np.abs(decisions[mask] - 1)
    return fpr_beta, fnr_beta, decisions
