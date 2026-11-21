"""Persistência das matrizes, tabelas e transformadores WoE do experimento."""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from optbinning import BinningProcess

logger = logging.getLogger(__name__)


def save_woe(
    directory: Path,
    *,
    monotonic: BinningProcess,
    free: BinningProcess,
    train_mono: pd.DataFrame,
    test_mono: pd.DataFrame,
    train_free: pd.DataFrame,
    test_free: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    iv: pd.DataFrame,
) -> None:
    """Grava os nove artefatos que o notebook original disponibilizava."""
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "X_train_woe_mono": train_mono,
        "X_test_woe_mono": test_mono,
        "X_train_woe_livre": train_free,
        "X_test_woe_livre": test_free,
        "y_train": y_train,
        "y_test": y_test,
        "comparacao_iv": iv,
    }
    for name, table in tables.items():
        table.to_pickle(directory / f"{name}.pkl")
    monotonic.save(str(directory / "binning_mono.pkl"))
    free.save(str(directory / "binning_livre.pkl"))
    logger.info("Transformadores e matrizes WoE gravados em %s", directory)


def load_woe(directory: Path, variant: Literal["mono", "livre"]) -> BinningProcess:
    """Carrega um transformador criado por save_woe em um diretório confiável."""
    if variant not in ("mono", "livre"):
        raise ValueError("A variante de WoE deve ser 'mono' ou 'livre'.")
    return BinningProcess.load(str(directory / f"binning_{variant}.pkl"))
