"""Figuras de discriminação, busca de hiperparâmetros e performance."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from credit_risk.evaluation.discrimination import RocPoints


def plot_roc(train: RocPoints, test: RocPoints, *, logistic: bool) -> tuple[Figure, Axes]:
    """Compare treino e teste mantendo o formato de cada gráfico original."""
    fig, ax = plt.subplots(figsize=(10, 7 if logistic else 6))
    for label, points in (("Treino", train), ("Teste", test)):
        prefix = "AUC:" if logistic else "AUC"
        ax.plot(
            points.fpr,
            points.tpr,
            linewidth=2 if logistic else None,
            label=f"{label} - {prefix} {points.auc:.4f}",
        )
    ax.plot(
        [0, 1], [0, 1], "--", color="gray", label="Classificador aleatório" if logistic else None
    )
    ax.set(
        xlabel="False Positive Rate (FPR)" if logistic else "False Positive Rate",
        ylabel="True Positive Rate (TPR)" if logistic else "True Positive Rate",
        title="Curva ROC - Regressão Logística com WoE Monotônico"
        if logistic
        else "Curva ROC - LightGBM",
    )
    ax.legend(loc="lower right" if logistic else "best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_ks(points: RocPoints, title: str) -> tuple[Figure, Axes]:
    """Mostre as acumuladas e a primeira distância máxima entre os cortes finitos."""
    thresholds, tpr, fpr, peak = points.finite_ks_points()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, tpr, label="TPR - Eventos")
    ax.plot(thresholds, fpr, label="FPR - Não eventos")
    ax.axvline(
        thresholds[peak],
        linestyle="--",
        color="gray",
        label=f"KS máximo = {tpr[peak] - fpr[peak]:.4f}",
    )
    ax.set(xlabel="Threshold", ylabel="Taxa acumulada", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_search_history(history: pd.DataFrame) -> tuple[Figure, Axes]:
    """AUC média das tentativas e máximo cumulativo; o critério de busca é log loss."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["Iteracao"], history["AUC_CV"], marker="o", alpha=0.5, label="AUC da avaliação")
    ax.plot(history["Iteracao"], history["Melhor_AUC"], linewidth=2, label="Melhor AUC acumulado")
    ax.set(
        xlabel="Iteração Hyperopt",
        ylabel="AUC médio - Cross Validation",
        title="Otimização de Hiperparâmetros - LightGBM",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_discrimination(metrics: pd.DataFrame) -> tuple[Figure, Axes]:
    """Mostre AUC, Gini e KS lado a lado no treino e no teste."""
    ax = metrics.loc[["Treino", "Teste"]].T.plot(kind="bar", figsize=(10, 6), rot=0)
    ax.set(title="Performance - LightGBM Otimizado", ylabel="Valor da métrica", ylim=(0, 1))
    ax.legend(title="Amostra")
    ax.grid(axis="y", alpha=0.3)
    ax.figure.tight_layout()
    return ax.figure, ax
