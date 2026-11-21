"""TPE, validação estratificada e seleção por log loss no LightGBM."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from inspect import signature
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from credit_risk.modeling.boosting import make_classifier, predict_pd


class FoldTransformer(Protocol):
    """Transformação supervisionada opcional, reconstruída a cada fold."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Any: ...
    def transform(self, features: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class SearchConfig:
    """Parâmetros fixos da busca do case. O TPE e o split recebem a mesma seed."""

    seed: int = 42
    n_jobs: int = -1
    folds: int = 3
    max_evals: int = 40
    max_trees: int = 1000
    stopping_rounds: int = 50


@dataclass(frozen=True)
class SearchResult:
    """Tentativas, parâmetros convertidos e número de árvores do trial escolhido."""

    params: dict[str, int | float]
    n_estimators: int
    best_trial: dict[str, Any]
    trials: Trials
    space: Mapping[str, Any]

    def auc_history(self) -> pd.DataFrame:
        """Série utilizada na figura do case; a linha superior acompanha a AUC."""
        history = pd.DataFrame(
            [
                {"Iteracao": i, "AUC_CV": item["auc_cv"]}
                for i, item in enumerate(self.trials.results, start=1)
            ]
        )
        history["Melhor_AUC"] = history["AUC_CV"].cummax()
        return history

    def trial_table(self) -> pd.DataFrame:
        """Uma linha por tentativa, com hiperparâmetros e todas as métricas de CV."""
        rows = []
        for trial in self.trials.trials:
            assignment = {
                name: values[0] for name, values in trial["misc"]["vals"].items() if values
            }
            params = typed_params(space_eval(self.space, assignment))
            rows.append({"Iteracao": trial["tid"] + 1, **params, **trial["result"]})
        return pd.DataFrame(rows)


def search_space() -> dict[str, Any]:
    """Construa as mesmas distribuições contínuas e quantizadas da busca inicial."""
    return {
        "num_leaves": hp.quniform("num_leaves", 4, 31, 1),
        "max_depth": hp.quniform("max_depth", 2, 8, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.15)),
        "min_child_samples": hp.quniform("min_child_samples", 20, 150, 5),
        "subsample": hp.uniform("subsample", 0.65, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.65, 1.0),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-3), np.log(10)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-3), np.log(10)),
    }


def typed_params(params: Mapping[str, float]) -> dict[str, int | float]:
    """Converta os três parâmetros quantizados para o tipo exigido pelo LightGBM."""
    result = dict(params)
    for name in ("num_leaves", "max_depth", "min_child_samples"):
        if name in result:
            result[name] = int(result[name])
    return result


def evaluate_trial(
    params: Mapping[str, float],
    *,
    features: pd.DataFrame,
    target: pd.Series,
    config: SearchConfig,
    transformer_factory: Callable[[], FoldTransformer] | None = None,
) -> dict[str, Any]:
    """Avalie um trial nos folds. Por padrão o modelo usa os preditores brutos."""
    cv = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    losses, aucs, iterations = [], [], []
    for train, validation in cv.split(features, target):
        x_tr, x_val = features.iloc[train], features.iloc[validation]
        y_tr, y_val = target.iloc[train], target.iloc[validation]
        if transformer_factory is not None:
            transformer = transformer_factory()
            transformer.fit(x_tr, y_tr)
            x_tr, x_val = transformer.transform(x_tr), transformer.transform(x_val)

        model = make_classifier(
            typed_params(params),
            config.max_trees,
            seed=config.seed,
            validation=True,
            n_jobs=config.n_jobs,
        )
        # O notebook usa a API eval_X/eval_y; releases anteriores recebem eval_set.
        if "eval_X" in signature(model.fit).parameters:
            eval_data = {"eval_X": x_val, "eval_y": y_val}
        else:
            eval_data = {"eval_set": [(x_val, y_val)]}
        model.fit(
            x_tr,
            y_tr,
            **eval_data,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=config.stopping_rounds, first_metric_only=True, verbose=False
                )
            ],
        )
        pd_validation = predict_pd(model, x_val)
        losses.append(log_loss(y_val, pd_validation, labels=[0, 1]))
        aucs.append(roc_auc_score(y_val, pd_validation))
        iterations.append(model.best_iteration_)

    mean_loss = float(np.mean(losses))
    return {
        "loss": mean_loss,
        "status": STATUS_OK,
        "logloss_cv": mean_loss,
        "auc_cv": float(np.mean(aucs)),
        "best_iteration": int(np.median(iterations)),
    }


def optimize(
    features: pd.DataFrame,
    target: pd.Series,
    space: Mapping[str, Any],
    config: SearchConfig,
    *,
    transformer_factory: Callable[[], FoldTransformer] | None = None,
) -> SearchResult:
    """Minimize log loss média e recupere a mediana de árvores do melhor trial."""
    trials = Trials()
    objective = partial(
        evaluate_trial,
        features=features,
        target=target,
        config=config,
        transformer_factory=transformer_factory,
    )
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=config.max_evals,
        trials=trials,
        rstate=np.random.default_rng(config.seed),
    )
    best_trial = min(trials.results, key=lambda item: item["loss"])
    return SearchResult(
        params=typed_params(space_eval(space, best)),
        n_estimators=best_trial["best_iteration"],
        best_trial=best_trial,
        trials=trials,
        space=space,
    )
