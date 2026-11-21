"""Persistência local dos modelos e tabelas produzidos no case."""

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier
    from statsmodels.discrete.discrete_model import BinaryResultsWrapper


def save_logistic(directory: Path, result: "BinaryResultsWrapper") -> None:
    """Salve o ajuste completo do Logit com parâmetros e metadados da inferência."""
    directory.mkdir(parents=True, exist_ok=True)
    result.save(str(directory / "modelo_lr.pkl"))


def load_logistic(directory: Path) -> "BinaryResultsWrapper":
    """Carregue um ajuste salvo com save_logistic a partir de origem confiável."""
    import statsmodels.api as sm

    return sm.load(str(directory / "modelo_lr.pkl"))


def save_lightgbm(
    directory: Path,
    model: "LGBMClassifier",
    params: Mapping[str, int | float],
    metrics: pd.DataFrame,
    history: pd.DataFrame,
    importance: pd.DataFrame,
    *,
    trials: pd.DataFrame | None = None,
) -> None:
    """Grave modelo, parâmetros, métricas, histórico e importância com nomes originais."""
    directory.mkdir(parents=True, exist_ok=True)
    for filename, value in (("modelo_lgb.pkl", model), ("best_params.pkl", dict(params))):
        with (directory / filename).open("wb") as stream:
            pickle.dump(value, stream)
    metrics.to_pickle(directory / "metricas_lgb.pkl")
    history.to_pickle(directory / "historico_hyperopt.pkl")
    importance.to_pickle(directory / "importancia_shap.pkl")
    if trials is not None:
        trials.to_pickle(directory / "trials_hyperopt.pkl")


def load_lightgbm(directory: Path) -> "LGBMClassifier":
    """Carregue o classificador LightGBM salvo localmente por uma origem confiável."""
    with (directory / "modelo_lgb.pkl").open("rb") as stream:
        return pickle.load(stream)
