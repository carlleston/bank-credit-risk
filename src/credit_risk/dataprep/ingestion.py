"""Leitura e auditoria da base do case sem transformar os dados."""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

CASE_PREDICTORS = tuple(f"v{i}" for i in range(1, 9))


def validate_case_schema(frame: pd.DataFrame) -> None:
    """Valida a presença dos campos do case e o domínio binário do desfecho."""
    required = (*CASE_PREDICTORS, "target")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}. Confira o separador ';'.")
    if not frame.columns.is_unique:
        raise ValueError("A base contém nomes de colunas repetidos.")
    non_numeric = [column for column in frame if not is_numeric_dtype(frame[column])]
    if non_numeric:
        raise ValueError(
            f"Colunas com tipo não numérico: {non_numeric}. Confira a vírgula decimal."
        )
    target = frame["target"]
    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise ValueError("A target deve conter as duas classes 0 e 1, sem valores ausentes.")


def read_case_csv(path: Path | str) -> pd.DataFrame:
    """Lê o CSV original, preservando ordem, índices e inferência de tipos do pandas."""
    frame = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    validate_case_schema(frame)
    return frame


def show_dataset_info(frame: pd.DataFrame) -> None:
    """Exibe dimensões e tipos na apresentação do notebook."""
    print(f"Linhas: {frame.shape[0]:,}")
    print(f"Colunas: {frame.shape[1]}")
    frame.info()


def audit_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Conta ausências, zeros e observações além das cercas de Tukey por coluna."""
    quartiles = frame.quantile([0.25, 0.75])
    lower, upper = quartiles.loc[0.25], quartiles.loc[0.75]
    spread = upper - lower
    outliers = frame.lt(lower - 1.5 * spread) | frame.gt(upper + 1.5 * spread)
    return pd.DataFrame(
        {
            "Tipo": frame.dtypes.astype(str),
            "Ausentes": frame.isna().sum(),
            "Zeros": frame.eq(0).sum(),
            "Distintos": frame.nunique(dropna=True),
            "Fora_das_cercas_IQR": outliers.sum(),
        }
    )


def count_duplicates(frame: pd.DataFrame) -> int:
    """Conta repetições de linhas completas além da primeira ocorrência."""
    return int(frame.duplicated().sum())
