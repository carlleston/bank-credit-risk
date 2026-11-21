"""Population Stability Index sobre categorias de WoE."""

import numpy as np
import pandas as pd


def calcular_psi_woe(train: pd.Series, test: pd.Series, epsilon: float = 1e-6) -> float:
    """Calcula PSI na união dos valores de WoE, mantendo NaNs e suavização aditiva."""
    dist_train = train.value_counts(normalize=True, dropna=False)
    dist_test = test.value_counts(normalize=True, dropna=False)
    categories = dist_train.index.union(dist_test.index, sort=False)
    p_train = dist_train.reindex(categories, fill_value=0).to_numpy(dtype=float)
    p_test = dist_test.reindex(categories, fill_value=0).to_numpy(dtype=float)
    denominator = 1 + epsilon * len(categories)
    p_train = (p_train + epsilon) / denominator
    p_test = (p_test + epsilon) / denominator
    return float(np.sum((p_test - p_train) * np.log(p_test / p_train)))


def compare_psi(
    train_mono: pd.DataFrame,
    test_mono: pd.DataFrame,
    train_free: pd.DataFrame,
    test_free: pd.DataFrame,
) -> pd.DataFrame:
    """Compara distribuições de treino e teste por variável e estratégia."""
    rows = []
    for name in train_mono.columns:
        mono = calcular_psi_woe(train_mono[name], test_mono[name])
        free = calcular_psi_woe(train_free[name], test_free[name])
        rows.append(
            {
                "Variavel": name,
                "PSI_Monotonico": mono,
                "PSI_Livre": free,
                "Diferenca": free - mono,
            }
        )
    return pd.DataFrame(rows).sort_values("PSI_Livre", ascending=False)


def classificar_psi(psi: float) -> str:
    """Classifica o valor com as fronteiras 0,10 e 0,25 do notebook."""
    if psi < 0.10:
        return "Baixa mudança"
    if psi < 0.25:
        return "Mudança moderada"
    return "Mudança elevada"


def classify_psi_table(table: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta os rótulos às duas variantes de PSI."""
    return table.assign(
        Status_Monotonico=table["PSI_Monotonico"].apply(classificar_psi),
        Status_Livre=table["PSI_Livre"].apply(classificar_psi),
    )
