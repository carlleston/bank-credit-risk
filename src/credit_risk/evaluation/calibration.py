"""Avaliação da qualidade das probabilidades, no nível da carteira e dos decis."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.special import logit
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def probability_metrics(y_true: ArrayLike, probabilities: ArrayLike) -> dict[str, float]:
    """Calcula taxa de evento, PD média, O/E, Brier e log loss da amostra."""
    y, p = np.asarray(y_true), np.asarray(probabilities)
    return {
        "Taxa_Evento": y.mean(),
        "PD_Media": p.mean(),
        "O_E": y.sum() / p.sum(),
        "Brier_Score": brier_score_loss(y, p),
        "Log_Loss": log_loss(y, p),
    }


def calibration_summary(
    y_train: ArrayLike,
    p_train: ArrayLike,
    y_test: ArrayLike,
    p_test: ArrayLike,
    *,
    extended: bool = False,
) -> pd.DataFrame:
    """Resume treino e teste. A tabela ampliada do LightGBM inclui O/E e AUC."""
    rows = {}
    for name, y, p in (("Treino", y_train, p_train), ("Teste", y_test, p_test)):
        metrics = probability_metrics(y, p)
        if extended:
            metrics["AUC"] = roc_auc_score(y, p)
        else:
            del metrics["O_E"]
        rows[name] = metrics
    return pd.DataFrame(rows).T


def decile_data(y_true: ArrayLike, probabilities: ArrayLike, n_deciles: int = 10) -> pd.DataFrame:
    """Ordena a PD crescente e desempata pela ordem das linhas, como no notebook."""
    frame = pd.DataFrame({"target": np.asarray(y_true), "PD": np.asarray(probabilities)})
    frame["Decil"] = pd.qcut(frame["PD"].rank(method="first"), q=n_deciles, labels=False) + 1
    return frame


def calibration_deciles(
    y_true: ArrayLike, probabilities: ArrayLike, *, extended: bool = False
) -> pd.DataFrame:
    """Compara PD prevista e taxa observada em cada decil de risco crescente."""
    table = (
        decile_data(y_true, probabilities)
        .groupby("Decil")
        .agg(
            Clientes=("target", "size" if extended else "count"),
            Eventos=("target", "sum"),
            PD_Media=("PD", "mean"),
            Taxa_Observada=("target", "mean"),
        )
    )
    table["Diferenca_pp"] = (table["PD_Media"] - table["Taxa_Observada"]) * 100
    if extended:
        table["O_E"] = table["Taxa_Observada"] / table["PD_Media"]
    return table


def calibration_curves(
    y_train: ArrayLike,
    p_train: ArrayLike,
    y_test: ArrayLike,
    p_test: ArrayLike,
) -> Mapping[str, tuple[NDArray, NDArray]]:
    """Calcula os pontos (PD média, frequência observada) por quantil em cada amostra."""
    curves = {}
    for name, y, p in (("Treino", y_train, p_train), ("Teste", y_test, p_test)):
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        curves[name] = (predicted, observed)
    return curves


def calibration_regression(y_true: ArrayLike, probabilities: ArrayLike) -> pd.DataFrame:
    """Ajusta logit(Y) ~ 1 + logit(PD) e retorna parâmetros e IC de Wald de 95%."""
    import statsmodels.api as sm

    bounded = np.clip(np.asarray(probabilities), 1e-8, 1 - 1e-8)
    design = sm.add_constant(logit(bounded))
    result = sm.Logit(np.asarray(y_true), design).fit(disp=False)
    intervals = np.asarray(result.conf_int())
    return pd.DataFrame(
        {
            "Parametro": ["Intercept", "Slope"],
            "Estimativa": result.params,
            "IC_95_Inferior": intervals[:, 0],
            "IC_95_Superior": intervals[:, 1],
            "Valor_Ideal": [0, 1],
        }
    )
