"""Inferência pareada para a diferença de Gini entre dois modelos."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class GiniBootstrap:
    """Diferença observada, intervalo percentil e réplicas pareadas."""

    observed: float
    lower: float
    upper: float
    samples: NDArray


def paired_gini_bootstrap(
    y_true: ArrayLike,
    reference: ArrayLike,
    challenger: ArrayLike,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> GiniBootstrap:
    """Reamostra eventos e não eventos separadamente com os mesmos índices por modelo."""
    y = np.asarray(y_true)
    p_reference = np.asarray(reference)
    p_challenger = np.asarray(challenger)
    if n_resamples < 1:
        raise ValueError("O número de réplicas deve ser positivo.")
    if y.ndim != 1 or p_reference.shape != y.shape or p_challenger.shape != y.shape:
        raise ValueError("A target e as duas previsões devem ser vetores de mesmo comprimento.")
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("O bootstrap estratificado requer eventos (1) e não eventos (0).")
    events = np.flatnonzero(y == 1)
    non_events = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        ids = np.concatenate(
            [
                rng.choice(events, size=len(events), replace=True),
                rng.choice(non_events, size=len(non_events), replace=True),
            ]
        )
        samples[i] = 2 * (
            roc_auc_score(y[ids], p_challenger[ids]) - roc_auc_score(y[ids], p_reference[ids])
        )
    observed = (2 * roc_auc_score(y, p_challenger) - 1) - (2 * roc_auc_score(y, p_reference) - 1)
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return GiniBootstrap(observed, lower, upper, samples)
