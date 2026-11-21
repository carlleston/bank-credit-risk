"""Comparação entre previsões calculadas sobre as mesmas observações."""

from collections.abc import Mapping

import pandas as pd
from numpy.typing import ArrayLike

from .calibration import probability_metrics
from .discrimination import RocPoints, discrimination_metrics, roc_points


def compare_predictions(y_true: ArrayLike, predictions: Mapping[str, ArrayLike]) -> pd.DataFrame:
    """Compara discriminação e qualidade probabilística por modelo na ordem fornecida."""
    rows = {}
    for name, proba in predictions.items():
        probability = probability_metrics(y_true, proba)
        rows[name] = {
            **discrimination_metrics(y_true, proba),
            **{key: probability[key] for key in ("Log_Loss", "Brier_Score", "PD_Media", "O_E")},
        }
    return pd.DataFrame(rows).T


def comparison_roc(y_true: ArrayLike, predictions: Mapping[str, ArrayLike]) -> dict[str, RocPoints]:
    """Prepara curvas ROC com as probabilidades de cada modelo no mesmo teste."""
    return {name: roc_points(y_true, p) for name, p in predictions.items()}
