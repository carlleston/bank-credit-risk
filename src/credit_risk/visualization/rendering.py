"""Exibição de figuras no notebook sem duplicar a saída do backend inline."""

import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def show_plot(plot: tuple[Figure, Axes]) -> None:
    """Exiba a figura da função de plot uma vez e libere o estado do pyplot."""
    fig, _ = plot
    try:
        display(fig)
    finally:
        plt.close(fig)
