"""Figuras de calibração, concentração e diferença de desempenho entre modelos."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..evaluation.bootstrap import GiniBootstrap
from ..evaluation.discrimination import RocPoints


def plot_calibration(
    curves: Mapping[str, tuple[NDArray, NDArray]],
    title: str,
    *,
    linewidth: float | None = None,
) -> tuple[Figure, Axes]:
    """Mostra as curvas de calibração por quantil e a referência de calibração perfeita."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, (predicted, observed) in curves.items():
        ax.plot(predicted, observed, marker="o", linewidth=linewidth, label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibração perfeita")
    ax.set(xlabel="Probabilidade prevista", ylabel="Taxa de evento observada", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_calibration_deciles(table: pd.DataFrame) -> tuple[Figure, Axes]:
    """Compara a frequência de eventos e a PD média dos decis de teste do LightGBM."""
    ax = table[["PD_Media", "Taxa_Observada"]].plot(kind="bar", figsize=(11, 6), rot=0)
    ax.set(
        title="Calibração por Decil - LightGBM",
        xlabel="Decil de risco",
        ylabel="Probabilidade / Taxa de evento",
    )
    ax.legend(["PD Média", "Taxa Observada"])
    ax.grid(axis="y", alpha=0.3)
    ax.figure.tight_layout()
    return ax.figure, ax


def plot_discrimination_comparison(table: pd.DataFrame) -> tuple[Figure, Axes]:
    """Compara AUC, Gini e KS dos modelos na amostra de teste."""
    ax = table[["AUC", "Gini", "KS"]].T.plot(kind="bar", figsize=(10, 6), rot=0)
    ax.set(
        title="Comparação de Performance - Teste",
        ylabel="Valor da métrica",
        ylim=(0, 1),
    )
    ax.legend(title="Modelo")
    ax.grid(axis="y", alpha=0.3)
    ax.figure.tight_layout()
    return ax.figure, ax


def plot_quality_comparison(table: pd.DataFrame) -> tuple[Figure, Axes]:
    """Compara as perdas logarítmica e de Brier no teste."""
    ax = table[["Log_Loss", "Brier_Score"]].T.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set(title="Comparação de Qualidade Probabilística", ylabel="Valor")
    ax.grid(axis="y", alpha=0.3)
    ax.figure.tight_layout()
    return ax.figure, ax


def plot_test_roc(curves: Mapping[str, RocPoints]) -> tuple[Figure, Axes]:
    """Exibe as curvas ROC das previsões no mesmo conjunto de teste."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, curve in curves.items():
        ax.plot(curve.fpr, curve.tpr, linewidth=2, label=f"{name} - AUC: {curve.auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Aleatório")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Comparação das Curvas ROC",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_bootstrap(result: GiniBootstrap) -> tuple[Figure, Axes]:
    """Mostra a distribuição das réplicas e as diferenças nula e observada."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(result.samples, bins=40, alpha=0.75)
    ax.axvline(0, color="red", linestyle="--", label="Diferença nula")
    ax.axvline(result.observed, color="black", linestyle="-", label="Diferença observada")
    ax.set(
        xlabel="Gini LightGBM - Gini Regressão Logística",
        ylabel="Frequência",
        title="Bootstrap Pareado - Diferença de Gini",
    )
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_event_shares(table: pd.DataFrame) -> tuple[Figure, Axes]:
    """Exibe a participação de cada decil no total de eventos do modelo."""
    ax = table.plot(kind="bar", figsize=(11, 6), rot=0)
    ax.set(
        title="Concentração dos Eventos por Decil",
        xlabel="Decil de Risco",
        ylabel="Participação dos Eventos",
    )
    ax.legend(title="Modelo")
    ax.grid(axis="y", alpha=0.3)
    ax.figure.tight_layout()
    return ax.figure, ax
