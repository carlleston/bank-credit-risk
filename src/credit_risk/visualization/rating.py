"""Gráficos da régua de rating, lift e captura de eventos."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_rates(table: pd.DataFrame, title: str) -> tuple[Figure, Axes]:
    """Sobrepõe PD média prevista e taxa de eventos observada por grupo."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        table.index,
        table["Taxa_Evento"] * 100,
        marker="o",
        linewidth=2,
        label="Taxa de eventos observada",
    )
    ax.plot(
        table.index,
        table["PD_Media"] * 100,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="PD média prevista",
    )
    ax.set_title(title)
    ax.set_xlabel("Grupo de risco (1 = maior risco)")
    ax.set_ylabel("Taxa de eventos (%)")
    ax.set_xticks(table.index)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_lift(table: pd.DataFrame, title: str) -> tuple[Figure, Axes]:
    """Compara o lift de cada grupo com a referência de seleção aleatória."""
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(table.index, table["Lift"])
    ax.axhline(1, linestyle="--", color="red", label="Seleção aleatória (Lift = 1)")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_title(title)
    ax.set_xlabel("Grupo de risco (1 = maior risco)")
    ax.set_ylabel("Lift")
    ax.set_xticks(table.index)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_capture(train: pd.DataFrame, test: pd.DataFrame) -> tuple[Figure, Axes]:
    """Mostra a captura de eventos pela parcela acumulada de clientes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, table in (("Treino", train), ("Teste", test)):
        ax.plot(
            [0, *(table["Populacao_Acumulada"] * 100)],
            [0, *(table["Captura_Acumulada"] * 100)],
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.plot([0, 100], [0, 100], "--", color="gray", label="Seleção aleatória")
    ax.set_title("Captura Acumulada de Eventos - LightGBM")
    ax.set_xlabel("População acumulada (%)")
    ax.set_ylabel("Eventos capturados (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax
