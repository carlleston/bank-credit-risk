"""Ensaio independente da base do case, incluindo inferência a partir dos artefatos."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from credit_risk.binning.woe import compare_iv, make_woe_transformer
from credit_risk.dataprep.ingestion import read_case_csv
from credit_risk.dataprep.selection import select_features
from credit_risk.dataprep.splitting import split_stratified
from credit_risk.evaluation.discrimination import discrimination_metrics
from credit_risk.evaluation.lift import compare_ratings, is_monotonic, risk_table
from credit_risk.explainability.contributions import explain_sample, mean_absolute_importance
from credit_risk.modeling.boosting import predict_pd, train_lightgbm
from credit_risk.modeling.logistic import fit_logistic, predict_logistic
from credit_risk.optimization.hyperopt_search import SearchConfig, optimize, search_space
from credit_risk.persistence.models import (
    load_lightgbm,
    load_logistic,
    save_lightgbm,
    save_logistic,
)
from credit_risk.persistence.rating import load_rating_scale, save_rating_artifacts
from credit_risk.persistence.woe import load_woe, save_woe
from credit_risk.rating.scale import RatingScale
from credit_risk.visualization.shap_plots import plot_beeswarm, plot_dependence, plot_waterfall


@pytest.fixture
def case_csv(tmp_path):
    """Três atributos binários geram oito perfis; as cópias exercitam a seleção."""
    per_profile = 300
    profiles = np.repeat(np.arange(8), per_profile)
    data = pd.DataFrame()
    for column in range(1, 9):
        bit = (column - 1) % 3
        data[f"v{column}"] = ((profiles >> bit) % 2 + 1) / 2
    rate = 0.05 + 0.12 * profiles
    within_profile = np.tile(np.arange(per_profile), 8)
    data["target"] = (within_profile < np.rint(per_profile * rate)).astype(int)
    path = tmp_path / "DADOS_CASE_modelagem .csv"
    data.to_csv(path, sep=";", decimal=",", encoding="utf-8-sig", index=False)
    return path


def test_case_from_csv_through_persisted_predictions(case_csv: Path, tmp_path: Path):
    data = read_case_csv(case_csv)
    assert data.at[0, "v1"] == 0.5
    filtered, retained, removed = select_features(data)
    assert len(retained) == 3 and len(removed) == 5
    x_train, x_test, y_train, y_test = split_stratified(filtered)
    assert len(x_test) == len(data) // 5
    assert x_train.index.intersection(x_test.index).empty
    assert x_train.index.union(x_test.index).equals(data.index)
    assert y_train.mean() == pytest.approx(data["target"].mean(), abs=1 / len(y_train))

    mono = make_woe_transformer(retained).fit(x_train, y_train)
    free = make_woe_transformer(retained, trend=None).fit(x_train, y_train)
    mono_train, mono_test = (mono.transform(x, metric="woe") for x in (x_train, x_test))
    free_train, free_test = (free.transform(x, metric="woe") for x in (x_train, x_test))
    for transformed, raw in (
        (mono_train, x_train),
        (mono_test, x_test),
        (free_train, x_train),
        (free_test, x_test),
    ):
        assert transformed.index.equals(raw.index)
        assert transformed.columns.equals(raw.columns)
        assert np.isfinite(transformed.to_numpy()).all()

    woe_dir = tmp_path / "woe"
    save_woe(
        woe_dir,
        monotonic=mono,
        free=free,
        train_mono=mono_train,
        test_mono=mono_test,
        train_free=free_train,
        test_free=free_test,
        y_train=y_train,
        y_test=y_test,
        iv=compare_iv(mono, free),
    )
    for variant, expected in (("mono", mono_test), ("livre", free_test)):
        restored = load_woe(woe_dir, variant)
        pd.testing.assert_frame_equal(restored.transform(x_test, metric="woe"), expected)

    logit = fit_logistic(mono_train, y_train)
    pd_logit = predict_logistic(logit, mono_test)
    logit_dir = tmp_path / "logistic"
    save_logistic(logit_dir, logit)
    pd.testing.assert_series_equal(
        predict_logistic(load_logistic(logit_dir), mono_test[retained[::-1]]), pd_logit
    )
    assert pd_logit.index.equals(y_test.index)
    assert discrimination_metrics(y_test, pd_logit)["AUC"] > 0.5

    params = {"num_leaves": 8, "max_depth": 3, "min_child_samples": 10, "learning_rate": 0.1}
    model = train_lightgbm(x_train, y_train, params, n_estimators=40, n_jobs=1)
    train_pd, test_pd = predict_pd(model, x_train), predict_pd(model, x_test)
    metrics = pd.DataFrame([discrimination_metrics(y_test, test_pd)], index=["Teste"])
    assert metrics.at["Teste", "AUC"] > 0.5
    model_dir = tmp_path / "lightgbm"
    explanation = explain_sample(model, x_test, size=32)
    importance = mean_absolute_importance(explanation)
    history = pd.DataFrame(columns=["Iteracao", "AUC_CV", "Melhor_AUC"])
    shap_margin = explanation.base_values + explanation.values.sum(axis=1)
    expected_margin = model.booster_.predict(x_test.sample(n=32, random_state=42), raw_score=True)
    np.testing.assert_allclose(shap_margin, expected_margin, atol=1e-12)
    # Confere a escala original usada nas cores dos gráficos de explicabilidade.
    np.testing.assert_array_equal(explanation.data, x_test.sample(n=32, random_state=42).to_numpy())
    save_lightgbm(model_dir, model, params, metrics, history, importance)
    np.testing.assert_array_equal(predict_pd(load_lightgbm(model_dir), x_test), test_pd)

    scale = RatingScale.fit(train_pd, y_train)
    train = scale.classify(train_pd, y_train)
    test = scale.classify(test_pd, y_test)
    train_table, test_table = (scale.limits().join(risk_table(frame)) for frame in (train, test))
    assert is_monotonic(train_table)
    assert set(train["Grupo_Risco"]) == set(range(1, 8))
    assert test_table["Clientes"].sum() == len(y_test)
    assert test_table["Eventos"].sum() == y_test.sum()
    assert test_table["Captura_Acumulada"].iloc[-1] == pytest.approx(1)
    rating_dir = tmp_path / "rating"
    comparison = compare_ratings(train_table, test_table)
    save_rating_artifacts(rating_dir, scale, train_table, test_table, train, test, comparison)
    reloaded = load_rating_scale(rating_dir)
    pd.testing.assert_frame_equal(reloaded.classify(test_pd), test.drop(columns="target"))
    pd.testing.assert_frame_equal(pd.read_pickle(rating_dir / "comparacao_rating.pkl"), comparison)


def test_shap_figures_render(case_csv: Path, tmp_path: Path):
    x_train, x_test, y_train, _ = split_stratified(select_features(read_case_csv(case_csv))[0])
    params = {"num_leaves": 8, "max_depth": 3, "min_child_samples": 10}
    model = train_lightgbm(x_train, y_train, params, n_estimators=8, n_jobs=1)
    sample = explain_sample(model, x_test, size=32)
    for name, plot in (
        ("beeswarm", plot_beeswarm(sample)),
        ("waterfall", plot_waterfall(sample, row=0)),
        ("dependence", plot_dependence(sample, sample.feature_names[0])),
    ):
        figure, axes = plot
        try:
            assert axes.figure is figure
            path = tmp_path / f"{name}.png"
            figure.savefig(path)
            assert path.read_bytes().startswith(b"\x89PNG")
        finally:
            plt.close(figure)


def test_tpe_optimizes_log_loss_and_keeps_selected_tree_count(case_csv: Path):
    x_train, _, y_train, _ = split_stratified(select_features(read_case_csv(case_csv))[0])
    config = SearchConfig(folds=3, max_evals=2, max_trees=12, stopping_rounds=2, n_jobs=1)
    result = optimize(x_train, y_train, search_space(), config)
    trials = result.trial_table()
    assert len(trials) == len(result.auc_history()) == 2
    assert result.best_trial["loss"] == pytest.approx(trials["logloss_cv"].min())
    assert result.n_estimators == result.best_trial["best_iteration"]
    assert 1 <= result.n_estimators <= config.max_trees
    np.testing.assert_allclose(result.auc_history()["Melhor_AUC"], trials["auc_cv"].cummax())
