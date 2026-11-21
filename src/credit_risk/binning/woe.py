"""Estratégias monotônica e livre de WoE implementadas pelo OptBinning."""

from collections.abc import Iterable, Iterator
from typing import Literal

import pandas as pd
from optbinning import BinningProcess

Trend = Literal["auto_asc_desc"] | None


def make_woe_transformer(
    columns: Iterable[str],
    trend: Trend = "auto_asc_desc",
    *,
    max_n_bins: int = 5,
    min_bin_size: float = 0.02,
    min_prebin_size: float = 0.01,
) -> BinningProcess:
    """Constrói o transformador supervisionado; zero é um código especial.

    O transformador expõe fit(X, y), transform(X, metric="woe") e summary().
    O case mantém o valor padrão de metric_special na transformação, inclusive
    quando a tabela de binning exibe uma estimativa empírica para esse grupo.
    """
    names = list(columns)
    fit_params = {
        name: {
            "dtype": "numerical",
            "solver": "cp",
            "special_codes": [0],
            "monotonic_trend": trend,
        }
        for name in names
    }
    return BinningProcess(
        variable_names=names,
        max_n_bins=max_n_bins,
        min_bin_size=min_bin_size,
        min_prebin_size=min_prebin_size,
        binning_fit_params=fit_params,
    )


def compare_iv(monotonic: BinningProcess, free: BinningProcess) -> pd.DataFrame:
    """Compara Information Value por nome e ordena pelo IV da estratégia livre."""
    mono_iv = monotonic.summary()[["name", "iv"]].rename(columns={"iv": "IV_Monotonico"})
    free_iv = free.summary()[["name", "iv"]].rename(columns={"iv": "IV_Livre"})
    comparison = mono_iv.merge(free_iv, on="name", how="outer")
    comparison["Diferenca_IV"] = comparison["IV_Livre"] - comparison["IV_Monotonico"]
    return comparison.sort_values("IV_Livre", ascending=False)


def binning_tables(
    columns: Iterable[str], monotonic: BinningProcess, free: BinningProcess
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Produz as tabelas completas, incluindo especiais, ausentes e totais."""
    for column in columns:
        mono_table = monotonic.get_binned_variable(column).binning_table.build()
        free_table = free.get_binned_variable(column).binning_table.build()
        yield column, mono_table, free_table


def is_monotonic_regular_rate(process: BinningProcess, column: str) -> bool:
    """Verifica a monotonicidade em qualquer direção, nos intervalos numéricos regulares."""
    fitted = process.get_binned_variable(column)
    table = fitted.binning_table.build()
    rates = table.iloc[: len(fitted.splits) + 1]["Event rate"]
    return bool(rates.is_monotonic_increasing or rates.is_monotonic_decreasing)
