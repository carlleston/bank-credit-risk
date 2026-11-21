"""Discretização supervisionada da PD e consolidação nas sete classes do case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from optbinning import OptimalBinning


# Índice 0 = grupo original 1 (maior risco).
FINAL_GROUPS = (1, 2, 3, 3, 4, 5, 6, 7)
GROUP_COUNT = max(FINAL_GROUPS)


def _probabilities(values: ArrayLike) -> NDArray:
    probabilities = np.asarray(values)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError("A PD deve ser um vetor unidimensional de valores finitos.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("A PD deve pertencer ao intervalo [0, 1].")
    return probabilities


@dataclass(frozen=True)
class RatingScale:
    """Aplica ao teste e a novos clientes os cortes aprendidos no treino."""

    binning: OptimalBinning

    def __post_init__(self) -> None:
        n_bins = len(self.binning.splits) + 1
        if n_bins != len(FINAL_GROUPS):
            raise ValueError(
                f"O OptimalBinning gerou {n_bins} bins, mas a régua validada exige 8. "
                "Verifique os cortes antes de consolidar."
            )

    @classmethod
    def fit(
        cls,
        pd_train: ArrayLike,
        target_train: ArrayLike,
        *,
        max_bins: int = 10,
        min_bin_size: float = 0.05,
    ) -> RatingScale:
        """Estima os cortes no treino com o solver e a tendência do notebook."""
        from optbinning import OptimalBinning

        probabilities = _probabilities(pd_train)
        binning = OptimalBinning(
            name="PD_LightGBM",
            dtype="numerical",
            solver="cp",
            max_n_bins=max_bins,
            min_bin_size=min_bin_size,
            min_prebin_size=0.01,
            monotonic_trend="ascending",
        )
        binning.fit(probabilities, np.asarray(target_train))
        if binning.status not in ("OPTIMAL", "FEASIBLE"):
            raise RuntimeError(f"Binning sem solução válida: {binning.status}")
        return cls(binning)

    @property
    def mapping(self) -> dict[int, int]:
        """Mapeia cada grupo original ao final na ordem decrescente de risco."""
        return dict(enumerate(FINAL_GROUPS, start=1))

    def classify(self, pd_pred: ArrayLike, target: ArrayLike | None = None) -> pd.DataFrame:
        """Pontua por posição, inclusive PD exatamente sobre o corte superior.

        Em um corte interno, o cliente entra no bin de maior PD. O índice
        posicional reproduz as tabelas históricas, mesmo com um índice de
        entrada não sequencial. O desfecho é opcional na inferência.
        """
        probabilities = _probabilities(pd_pred)
        bins = np.asarray(self.binning.transform(probabilities, metric="indices")).astype(int)
        n_bins = len(FINAL_GROUPS)
        if not np.isin(bins, np.arange(n_bins)).all():
            raise ValueError("Índices de bins inesperados.")
        original_groups = n_bins - bins
        data = {} if target is None else {"target": np.asarray(target)}
        data.update(
            PD=probabilities,
            Bin_Original=bins,
            Grupo_Original=original_groups,
            Grupo_Risco=np.asarray(FINAL_GROUPS)[original_groups - 1],
        )
        return pd.DataFrame(data)

    def limits(self) -> pd.DataFrame:
        """Constrói os intervalos por grupo em precisão integral, do pior ao melhor."""
        boundaries = np.concatenate(([0.0], self.binning.splits, [1.0]))
        rows = []
        mapping = self.mapping
        for group in range(1, GROUP_COUNT + 1):
            original = [key for key, final in mapping.items() if final == group]
            bins = [len(FINAL_GROUPS) - key for key in original]
            rows.append(
                {
                    "Grupo_Risco": group,
                    "Grupos_Originais": ", ".join(map(str, original)),
                    "PD_Limite_Inferior": boundaries[min(bins)],
                    "PD_Limite_Superior": boundaries[max(bins) + 1],
                }
            )
        return pd.DataFrame(rows).set_index("Grupo_Risco")
