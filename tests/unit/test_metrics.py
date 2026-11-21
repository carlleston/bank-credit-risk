"""Propriedades de discriminação, concentração, calibração e incerteza."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.binning.stability import calcular_psi_woe
from credit_risk.evaluation.bootstrap import paired_gini_bootstrap
from credit_risk.evaluation.calibration import calibration_deciles, calibration_summary
from credit_risk.evaluation.concentration import event_hhi, hhi, portfolio_hhi
from credit_risk.evaluation.discrimination import discrimination_metrics


@pytest.mark.parametrize(
    ("scores", "expected_auc", "expected_ks"),
    [
        ([0.1, 0.2, 0.8, 0.9], 1.0, 1.0),
        ([0.1, 0.8, 0.2, 0.9], 0.75, 0.5),
        ([0.4, 0.4, 0.4, 0.4], 0.5, 0.0),
        ([0.9, 0.8, 0.2, 0.1], 0.0, 0.0),
    ],
)
def test_auc_gini_and_one_sided_ks(scores, expected_auc, expected_ks):
    metrics = discrimination_metrics([0, 0, 1, 1], scores)
    assert 0 <= metrics["AUC"] <= 1
    assert metrics["AUC"] == pytest.approx(expected_auc)
    assert metrics["Gini"] == pytest.approx(2 * expected_auc - 1)
    assert metrics["KS"] == pytest.approx(expected_ks)


def test_categorical_psi_uses_the_same_woe_buckets_on_both_sides():
    base = pd.Series([-0.8, -0.8, 0.5, 0.5])
    assert calcular_psi_woe(base, base.iloc[::-1]) == pytest.approx(0.0, abs=1e-12)
    shifted = pd.Series([-0.8, 0.5, 0.5, 0.5])
    assert calcular_psi_woe(base, shifted) > 0
    assert np.isfinite(calcular_psi_woe(pd.Series([0, 0]), pd.Series([1, 1])))


def test_hhi_distinguishes_clients_and_events():
    assert np.isnan(hhi([0, 0]))
    assert hhi([3, 3, 3]) == pytest.approx(1 / 3)
    assert hhi([10, 0]) == pytest.approx(1)
    probabilities = [0.005, 0.01, 0.7, 0.9]
    concentration, counts = event_hhi([0, 0, 1, 1], probabilities, n_deciles=2)
    assert concentration == pytest.approx(1)
    assert counts["Clientes"].sum() == 4
    assert counts["Eventos"].sum() == 2
    concentration, shares = portfolio_hhi(probabilities)
    assert concentration == pytest.approx(0.5)
    assert shares.sum() == pytest.approx(1)


def test_brier_log_loss_oe_and_tie_order():
    y = np.tile([0, 1], 10)
    p = np.full(len(y), 0.25)
    summary = calibration_summary(y, p, y, p, extended=True)
    assert summary.at["Teste", "Brier_Score"] == pytest.approx(0.3125)
    assert summary.at["Teste", "Log_Loss"] == pytest.approx(-np.log(0.25 * 0.75) / 2)
    assert summary.at["Teste", "O_E"] == pytest.approx(2)
    table = calibration_deciles(y, p)
    assert table["Clientes"].tolist() == [2] * 10
    assert table["Eventos"].tolist() == [1] * 10
    assert np.allclose(table["Diferenca_pp"], -25)


def test_paired_bootstrap_keeps_both_scores_on_each_draw():
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.6, 0.8, 0.5, 0.4])
    result = paired_gini_bootstrap(y, p, p, n_resamples=32, seed=11)
    np.testing.assert_array_equal(result.samples, np.zeros(32))
    assert (result.observed, result.lower, result.upper) == (0, 0, 0)
    reverse = paired_gini_bootstrap(y, p, 1 - p, n_resamples=32, seed=11)
    forward = paired_gini_bootstrap(y, 1 - p, p, n_resamples=32, seed=11)
    np.testing.assert_allclose(reverse.samples, -forward.samples, atol=1e-15)
