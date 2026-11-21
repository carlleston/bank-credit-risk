"""Contribuições aditivas do LightGBM na escala de margem bruta (log-odds)."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

if TYPE_CHECKING:
    from shap import Explanation


def explain_sample(
    model: LGBMClassifier, features: pd.DataFrame, *, size: int = 1000, seed: int = 42
) -> "Explanation":
    """Amostre sem reposição e explique os valores originais apresentados ao modelo."""
    from shap import Explanation

    sample = features.sample(n=min(size, len(features)), random_state=seed)
    contributions = model.booster_.predict(sample, pred_contrib=True)
    return Explanation(
        values=contributions[:, :-1],
        base_values=contributions[:, -1],
        data=sample.to_numpy(),
        feature_names=sample.columns.tolist(),
    )


def mean_absolute_importance(explanation: "Explanation") -> pd.DataFrame:
    """Ordene as variáveis pela média do valor absoluto das contribuições."""
    return pd.DataFrame(
        {
            "Variavel": explanation.feature_names,
            "SHAP_Medio_Absoluto": np.abs(explanation.values).mean(axis=0),
        }
    ).sort_values("SHAP_Medio_Absoluto", ascending=False)


def dominant_features(explanation: "Explanation", *, top: int = 3) -> pd.DataFrame:
    """Partição da contribuição absoluta agregada atribuída às top variáveis."""
    if top < 1:
        raise ValueError("top deve ser positivo.")
    table = mean_absolute_importance(explanation)
    total = table["SHAP_Medio_Absoluto"].sum()
    table["Participacao_Absoluta"] = table["SHAP_Medio_Absoluto"] / total if total else 0.0
    table["Participacao_Acumulada"] = table["Participacao_Absoluta"].cumsum()
    return table.head(top)
