"""Regressão logística de máxima verossimilhança sobre o WoE monotônico."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import BinaryResultsWrapper


def logistic_design(features: pd.DataFrame) -> pd.DataFrame:
    """Adicione o intercepto, inclusive quando outro preditor for constante."""
    return sm.add_constant(features, has_constant="add")


def fit_logistic(
    features: pd.DataFrame, target: pd.Series, *, maxiter: int = 2000
) -> BinaryResultsWrapper:
    """Ajuste o Logit sem penalização, com intercepto e otimizador original."""
    return sm.Logit(target, logistic_design(features)).fit(disp=False, maxiter=maxiter)


def predict_logistic(result: BinaryResultsWrapper, features: pd.DataFrame) -> pd.Series:
    """Prediga a PD com o mesmo intercepto e ordem de colunas usados no treino."""
    design = logistic_design(features).loc[:, result.params.index]
    return result.predict(design)


def coefficient_table(result: BinaryResultsWrapper) -> pd.DataFrame:
    """Resuma coeficientes, erro padrão, IC 95%, p-valor e odds ratio."""
    intervals = result.conf_int(alpha=0.05)
    table = pd.DataFrame(
        {
            "Coeficiente": result.params,
            "Erro_Padrao": result.bse,
            "P_Value": result.pvalues,
            "IC_95_Inferior": intervals.iloc[:, 0],
            "IC_95_Superior": intervals.iloc[:, 1],
        }
    )
    table["Odds_Ratio"] = np.exp(table["Coeficiente"])
    table["Significativo_5pct"] = table["P_Value"] < 0.05
    return table
