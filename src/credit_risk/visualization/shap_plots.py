"""Visualizações de explicações SHAP na escala de margem bruta do modelo."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from shap import Explanation


def plot_beeswarm(explanation: "Explanation", *, max_display: int = 15) -> tuple[Figure, Axes]:
    """Distribuição global de contribuições com cores nos valores de entrada."""
    import shap

    fig = plt.figure()
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    return fig, fig.axes[0]


def plot_waterfall(
    explanation: "Explanation", row: int, *, max_display: int = 15
) -> tuple[Figure, Axes]:
    """Decomposição da margem para uma única observação da amostra explicada."""
    import shap

    fig = plt.figure()
    shap.plots.waterfall(explanation[row], max_display=max_display, show=False)
    return fig, fig.axes[0]


def plot_dependence(explanation: "Explanation", feature: str) -> tuple[Figure, Axes]:
    """Relação entre o valor de um preditor e sua contribuição para a margem."""
    import shap

    fig, ax = plt.subplots()
    shap.plots.scatter(explanation[:, feature], ax=ax, show=False)
    return fig, ax
