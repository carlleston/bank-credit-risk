"""Correlações exploratórias e filtro de multicolinearidade do case."""

import pandas as pd


def correlation_tables(
    frame: pd.DataFrame, target: str = "target"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula Pearson entre campos e a associação de Pearson e Spearman com a target."""
    numeric = frame.select_dtypes(include=["number"])
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    associations = pd.DataFrame({"Pearson": pearson[target], "Spearman": spearman[target]})
    ordered = associations.drop(index=target).sort_values("Pearson", key=abs, ascending=False)
    return pearson, ordered


def select_features(
    frame: pd.DataFrame, target: str = "target", threshold: float = 0.80
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prioriza |Pearson com a target| e descarta pares com |r| >= limiar.

    Recebe a base antes da divisão para reproduzir a seleção do notebook.
    O teste de correlação é feito contra cada preditor já selecionado.
    """
    predictors = frame.drop(columns=[target])
    priority = frame.corr(method="pearson")[target].drop(target).abs().sort_values(ascending=False)
    correlations = predictors.corr(method="pearson").abs()
    selected: list[str] = []
    removed: list[str] = []
    for name in priority.index:
        if any(correlations.loc[name, other] >= threshold for other in selected):
            removed.append(name)
        else:
            selected.append(name)
    return frame[selected + [target]].copy(), selected, removed


def show_selection(
    frame: pd.DataFrame, filtered: pd.DataFrame, selected: list[str], removed: list[str]
) -> None:
    """Exibe nomes e dimensões na mesma ordem da análise original."""
    print("Variáveis selecionadas:", selected)
    print("Variáveis removidas:", removed)
    print("Dimensão original:", frame.shape)
    print("Dimensão final:", filtered.shape)
