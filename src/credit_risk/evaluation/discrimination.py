"""Discriminação de um classificador binário com PD crescente com o risco."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class RocPoints:
    """Coordenadas e cortes da curva ROC, na ordem retornada pelo scikit-learn."""

    fpr: NDArray
    tpr: NDArray
    thresholds: NDArray
    auc: float

    @property
    def ks(self) -> float:
        """Maior distância vertical entre TPR e FPR, no sentido do evento."""
        return float(np.max(self.tpr - self.fpr))

    @property
    def gini(self) -> float:
        """Gini binário calculado a partir da área sob a ROC."""
        return 2 * self.auc - 1

    def finite_ks_points(self) -> tuple[NDArray, NDArray, NDArray, int]:
        """Cortes finitos, TPR, FPR e índice do primeiro KS máximo para o gráfico."""
        finite = np.isfinite(self.thresholds)
        tpr, fpr = self.tpr[finite], self.fpr[finite]
        return self.thresholds[finite], tpr, fpr, int(np.argmax(tpr - fpr))

    def metrics(self) -> dict[str, float]:
        """Métricas usadas em todos os quadros de discriminação."""
        return {"AUC": self.auc, "Gini": self.gini, "KS": self.ks}


def roc_points(y_true: ArrayLike, y_proba: ArrayLike) -> RocPoints:
    """Calcule a ROC e a AUC para o evento codificado como 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return RocPoints(fpr, tpr, thresholds, float(roc_auc_score(y_true, y_proba)))


def discrimination_metrics(y_true: ArrayLike, y_proba: ArrayLike) -> dict[str, float]:
    """Calcule AUC, Gini e KS com a definição compartilhada entre os modelos."""
    return roc_points(y_true, y_proba).metrics()


def train_test_discrimination(train: RocPoints, test: RocPoints, gap_label: str) -> pd.DataFrame:
    """Apresente treino, teste e a diferença treino menos teste."""
    result = pd.DataFrame({"Treino": train.metrics(), "Teste": test.metrics()}).T
    result.loc[gap_label] = result.loc["Treino"] - result.loc["Teste"]
    return result
