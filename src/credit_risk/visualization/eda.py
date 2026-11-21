"""Figuras e tabelas para exploração, comparação dos bins e estabilidade."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_correlations(
    table: pd.DataFrame, *, by_target: bool = False, save_to: Path | None = None
) -> tuple[Figure, Axes]:
    """Desenha a matriz de Pearson ou a associação da target por dois métodos."""
    fig, ax = plt.subplots(figsize=(9, 6) if by_target else (12, 8))
    sns.heatmap(
        table,
        ax=ax,
        annot=True,
        cmap="coolwarm",
        fmt=".3f" if by_target else ".2f",
        center=0,
        vmin=-1,
        vmax=1,
    )
    ax.set_title(
        "Correlação das Variáveis com a Target" if by_target else "Matriz de Correlação - Pearson"
    )
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to)
    return fig, ax


def plot_psi(table: pd.DataFrame, *, save_to: Path | None = None) -> tuple[Figure, Axes]:
    """Desenha o PSI calculado por variável com a escala vertical do original."""
    values = table.set_index("Variavel")[["PSI_Monotonico", "PSI_Livre"]]
    ax = values.plot(kind="bar", figsize=(11, 6), rot=0, width=0.75)
    fig = ax.get_figure()
    highest = values.max().max()
    if highest > 0:
        ax.set_ylim(0, highest * 5)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3, fontsize=9)
    ax.set(
        title="PSI das Variáveis WoE - Treino vs Teste",
        xlabel="Variável",
        ylabel="Population Stability Index",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to)
    return fig, ax


def show_binning_tables(
    tables: Iterable[tuple[str, pd.DataFrame, pd.DataFrame]],
) -> None:
    """Apresenta as duas tabelas de cada variável na ordem de treinamento."""
    for name, monotonic, free in tables:
        print(f"\n{'=' * 60}\nVARIÁVEL: {name}\n{'=' * 60}")
        print("\nMONOTÔNICO")
        display(monotonic)
        print("\nLIVRE")
        display(free)
