"""Concentração dos eventos por decil e dos clientes por intervalo fixo de PD."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .calibration import decile_data

PD_BANDS = (0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)


def hhi(shares: ArrayLike) -> float:
    """Soma dos quadrados das participações normalizadas. Total nulo produz NaN."""
    p = np.asarray(shares, dtype=float)
    if p.sum() == 0:
        return np.nan
    return float(np.sum((p / p.sum()) ** 2))


def event_hhi(
    y_true: ArrayLike, y_proba: ArrayLike, n_deciles: int = 10
) -> tuple[float, pd.DataFrame]:
    """Distribuição dos eventos entre os decis formados pela PD de um modelo."""
    table = (
        decile_data(y_true, y_proba, n_deciles)
        .groupby("Decil")
        .agg(
            Clientes=("target", "size"),
            Eventos=("target", "sum"),
            PD_Media=("PD", "mean"),
        )
    )
    table["Participacao_Eventos"] = table["Eventos"] / table["Eventos"].sum()
    return hhi(table["Participacao_Eventos"]), table


def portfolio_hhi(
    probabilities: ArrayLike, bins: Sequence[float] = PD_BANDS
) -> tuple[float, pd.Series]:
    """Conta clientes em intervalos (a, b], incluindo o limite inferior da primeira faixa."""
    groups = pd.cut(np.asarray(probabilities), bins=bins, include_lowest=True)
    counts = pd.Series(groups).value_counts(sort=False)
    shares = counts / counts.sum()
    return hhi(shares), shares


def event_concentration(
    y_true: ArrayLike, predictions: Mapping[str, ArrayLike]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara HHIs e devolve as participações por decil utilizadas no gráfico."""
    rows, shares = [], {}
    for name, proba in predictions.items():
        value, table = event_hhi(y_true, proba)
        rows.append({"Modelo": name, "HHI_Eventos": value})
        shares[name] = table["Participacao_Eventos"]
    return pd.DataFrame(rows), pd.DataFrame(shares)


def portfolio_concentration(
    event_summary: pd.DataFrame, predictions: Mapping[str, ArrayLike]
) -> pd.DataFrame:
    """Acrescenta o HHI da distribuição da carteira à tabela de HHI de eventos."""
    table = event_summary.copy()
    table["HHI_Carteira"] = [portfolio_hhi(predictions[name])[0] for name in table["Modelo"]]
    return table
