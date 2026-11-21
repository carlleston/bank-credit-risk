"""Preparação e treinamento do classificador LightGBM."""

from collections.abc import Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def prepare_samples(
    train: pd.DataFrame, test: pd.DataFrame, y_train: ArrayLike, y_test: ArrayLike
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Copie os preditores e preserve o alinhamento por posição da target original."""
    x_train = train.copy()
    x_test = test.copy().loc[:, x_train.columns]
    return (
        x_train,
        x_test,
        pd.Series(np.asarray(y_train), index=x_train.index),
        pd.Series(np.asarray(y_test), index=x_test.index),
    )


def make_classifier(
    params: Mapping[str, float | int],
    n_estimators: int,
    *,
    seed: int = 42,
    validation: bool = False,
    n_jobs: int = -1,
) -> lgb.LGBMClassifier:
    """Construa o estimador final ou o estimador para early stopping nos folds."""
    evaluation = {"metric": "binary_logloss"} if validation else {}
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        subsample_freq=1,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
        **evaluation,
        **params,
    )


def train_lightgbm(
    features: pd.DataFrame,
    target: pd.Series,
    params: Mapping[str, float | int],
    n_estimators: int,
    *,
    seed: int = 42,
    n_jobs: int = -1,
) -> lgb.LGBMClassifier:
    """Reajuste com a mediana de árvores dos folds do melhor trial por log loss."""
    classifier = make_classifier(params, max(1, n_estimators), seed=seed, n_jobs=n_jobs)
    return classifier.fit(features, target)


def predict_pd(model: lgb.LGBMClassifier, features: pd.DataFrame) -> NDArray:
    """Retorne a probabilidade da classe 1, como no treinamento original."""
    return model.predict_proba(features)[:, 1]
