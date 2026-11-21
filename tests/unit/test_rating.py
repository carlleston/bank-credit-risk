"""Protege a consolidação 3/4, os limites dos intervalos e o sentido do risco."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.evaluation.lift import is_monotonic, risk_table
from credit_risk.rating.scale import RatingScale


@pytest.fixture(scope="module")
def fitted_scale():
    # Nenhuma amostragem: a taxa em cada um dos oito patamares é conhecida.
    levels = np.arange(1, 9) / 10
    observations = 100
    probabilities = np.repeat(levels, observations)
    target = np.concatenate(
        [np.arange(observations) < round(p * observations) for p in levels]
    ).astype(int)
    return RatingScale.fit(probabilities, target)


def test_training_scale_merges_3_and_4_and_orders_risk(fitted_scale):
    scored = fitted_scale.classify(np.arange(1, 9) / 10)
    assert scored["Bin_Original"].tolist() == list(range(8))
    assert scored["Grupo_Original"].tolist() == list(range(8, 0, -1))
    assert scored["Grupo_Risco"].tolist() == [7, 6, 5, 4, 3, 3, 2, 1]
    assert "target" not in scored
    limits = fitted_scale.limits()
    assert limits.index.tolist() == list(range(1, 8))
    assert limits.at[3, "Grupos_Originais"] == "3, 4"
    assert limits.at[7, "PD_Limite_Inferior"] == 0
    assert limits.at[1, "PD_Limite_Superior"] == 1


def test_new_scores_match_the_training_boundaries(fitted_scale):
    limits = fitted_scale.limits()
    for group in range(1, 7):
        boundary = limits.at[group, "PD_Limite_Inferior"]
        scores = [np.nextafter(boundary, 0), boundary, np.nextafter(boundary, 1)]
        assert fitted_scale.classify(scores)["Grupo_Risco"].tolist() == [group + 1, group, group]
    assert fitted_scale.classify([0.0, 1.0])["Grupo_Risco"].tolist() == [7, 1]


def test_historical_scale_rejects_a_different_number_of_bins():
    p = np.repeat([0.1, 0.9], 100)
    y = np.r_[np.arange(100) < 10, np.arange(100) < 90].astype(int)
    with pytest.raises(ValueError, match="8"):
        RatingScale.fit(p, y)


def test_every_group_has_unit_lift_for_uniform_event_rates():
    groups = np.repeat(np.arange(1, 8), 4)
    scored = pd.DataFrame(
        {"Grupo_Risco": groups, "PD": 1 - groups / 8, "target": np.tile([0, 0, 0, 1], 7)}
    )
    table = risk_table(scored)
    assert table.index.tolist() == list(range(1, 8))
    assert np.allclose(table["Lift"], 1)
    assert np.allclose(table["Lift_Acumulado"], 1)
    assert table["Clientes"].sum() == len(scored)
    assert table["Eventos"].sum() == scored["target"].sum()
    assert table["Captura_Acumulada"].iloc[-1] == pytest.approx(1)
    assert table["Populacao_Acumulada"].iloc[-1] == pytest.approx(1)
    assert is_monotonic(table)


def test_monotonicity_checks_order_and_skips_empty_groups():
    scored = pd.DataFrame(
        {
            "Grupo_Risco": [1, 1, 4, 4, 6, 6],
            "PD": [0.7, 0.7, 0.4, 0.4, 0.1, 0.1],
            "target": [1, 1, 1, 0, 0, 0],
        }
    )
    table = risk_table(scored)
    assert table.at[2, "Clientes"] == 0
    assert is_monotonic(table)
    table.at[6, "Taxa_Evento"] = 0.8
    assert not is_monotonic(table)
    with pytest.raises(ValueError):
        risk_table(scored.assign(target=0))
