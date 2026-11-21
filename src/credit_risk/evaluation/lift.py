"""Desempenho e captura por classe, ordenados do maior para o menor risco."""

import numpy as np
import pandas as pd

from credit_risk.rating.scale import GROUP_COUNT


def risk_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Agrega os sete ratings, mantendo posições vazias e taxas sem arredondar."""
    table = (
        scored.groupby("Grupo_Risco")
        .agg(
            Clientes=("target", "size"),
            Eventos=("target", "sum"),
            PD_Min_Observada=("PD", "min"),
            PD_Max_Observada=("PD", "max"),
            PD_Media=("PD", "mean"),
            Taxa_Evento=("target", "mean"),
        )
        .reindex(range(1, GROUP_COUNT + 1))
    )
    table["Clientes"] = table["Clientes"].fillna(0)
    table["Eventos"] = table["Eventos"].fillna(0)
    customers = table["Clientes"].sum()
    events = table["Eventos"].sum()
    if events <= 0:
        raise ValueError("A amostra não possui eventos.")

    table["Lift"] = table["Taxa_Evento"] / (events / customers)
    table["Captura_Eventos"] = table["Eventos"] / events
    table["Captura_Acumulada"] = table["Eventos"].cumsum() / events
    table["Populacao_Acumulada"] = table["Clientes"].cumsum() / customers
    table["Lift_Acumulado"] = table["Captura_Acumulada"] / table["Populacao_Acumulada"]
    return table


def is_monotonic(table: pd.DataFrame, tolerance: float = 1e-12) -> bool:
    """Confere a ordem decrescente das taxas nos grupos com clientes."""
    rates = table.loc[table["Clientes"] > 0, "Taxa_Evento"].to_numpy()
    return bool(np.all(np.diff(rates) <= tolerance))


def validate_train_monotonicity(table: pd.DataFrame) -> bool:
    """Interrompe o fluxo se as taxas de treino quebrarem a monotonicidade."""
    if not is_monotonic(table):
        raise ValueError(
            "A régua proposta não manteve monotonicidade no treino. Revise os agrupamentos."
        )
    return True


def compare_ratings(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Alinha os grupos e calcula a diferença teste menos treino em pontos percentuais."""
    comparison = pd.DataFrame(
        {
            "Clientes_Treino": train["Clientes"],
            "Clientes_Teste": test["Clientes"],
            "PD_Treino": train["PD_Media"],
            "PD_Teste": test["PD_Media"],
            "Taxa_Evento_Treino": train["Taxa_Evento"],
            "Taxa_Evento_Teste": test["Taxa_Evento"],
            "Lift_Treino": train["Lift"],
            "Lift_Teste": test["Lift"],
        }
    )
    comparison["Diferenca_Evento_pp"] = (
        comparison["Taxa_Evento_Teste"] - comparison["Taxa_Evento_Treino"]
    ) * 100
    return comparison
