"""Localização do CSV do case e dos artefatos gerados pelo notebook."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Caminhos imutáveis; cada experimento pode apontar para outro diretório."""

    root: Path
    raw_csv: Path
    artifacts: Path

    @classmethod
    def discover(cls, root: Path | None = None) -> "ProjectPaths":
        """Localiza o projeto instalado em modo editável ou aberto pelo Jupyter."""
        if root is None:
            cwd = Path.cwd().resolve()
            candidates = (cwd, *cwd.parents, Path(__file__).resolve().parents[3])
            root = next(
                (
                    p
                    for p in candidates
                    if (p / "src/credit_risk").is_dir() and (p / "pyproject.toml").is_file()
                ),
                None,
            )
        if root is None:
            raise FileNotFoundError("Abra o notebook na pasta do projeto ou informe root.")
        root = Path(root).expanduser().resolve()
        csv = os.environ.get("CREDIT_RISK_CSV", "data/raw/DADOS_CASE_modelagem .csv")
        artifacts = os.environ.get("CREDIT_RISK_ARTIFACTS", "artifacts")
        return cls(root, root / Path(csv).expanduser(), root / Path(artifacts).expanduser())

    @property
    def woe(self) -> Path:
        """Transformadores ajustados e matrizes de treino e teste."""
        return self.artifacts / "woe"

    @property
    def logistic(self) -> Path:
        """Resultados ajustados da regressão logística."""
        return self.artifacts / "logistic"

    @property
    def lightgbm(self) -> Path:
        """Modelo LightGBM e resultados da otimização."""
        return self.artifacts / "lightgbm"

    @property
    def rating(self) -> Path:
        """Discretizador, régua final e populações classificadas."""
        return self.artifacts / "rating"
