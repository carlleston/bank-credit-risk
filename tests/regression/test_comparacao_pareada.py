"""Compara o estado científico dos dois notebooks na base do case e em um controle."""

import os
import pickle
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

# Os nomes são a interface histórica compartilhada pelos dois notebooks.
FIELDS = (
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "X_train_woe_mono",
    "X_test_woe_mono",
    "X_train_woe_livre",
    "X_test_woe_livre",
    "y_proba_train",
    "y_proba_test",
    "y_proba_train_lgb",
    "y_proba_test_lgb",
    "best_params",
    "best_n_estimators",
    "resultados",
    "resultados_lgb",
    "tabela_coeficientes",
    "comparacao_iv",
    "df_psi",
    "comparacao_modelos",
    "tabela_calibracao",
    "comparacao_hhi_final",
    "regua_pd",
    "df_train_grupos",
    "df_test_grupos",
    "tabela_train",
    "tabela_test",
    "comparacao_rating",
)


def _write_control_csv(path: Path) -> None:
    """Oito perfis conhecidos, três atributos independentes e preditores redundantes."""
    per_profile = 800
    profile = np.repeat(np.arange(8), per_profile)
    rates = np.array([0.02, 0.08, 0.16, 0.28, 0.43, 0.59, 0.75, 0.91])
    events = np.concatenate([np.arange(per_profile) < round(rate * per_profile) for rate in rates])
    a, b, c = [1 + ((profile >> j) % 2) for j in range(3)]
    frame = pd.DataFrame(
        {
            "v1": a,
            "v2": 15 * b + 20,
            "v3": c,
            "v4": 2.0 * a,
            "v5": 3.0 * b,
            "v6": 2.0 * c,
            "v7": 1.0 * a,
            "v8": 1.0 * b,
            "target": events.astype(int),
        }
    )
    frame.sample(frac=1, random_state=349).to_csv(path, sep=";", decimal=",", index=False)


def _execute(path: Path, run_dir: Path, csv: Path, root: Path) -> dict:
    """Executa desde o início e coleta arrays antes de qualquer arredondamento."""
    run_dir.mkdir()
    # O código original localiza o CSV pelo nome no diretório atual do kernel.
    (run_dir / "DADOS_CASE_modelagem .csv").write_bytes(csv.read_bytes())
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "import pickle\n"
            "from pathlib import Path\n"
            f"estado = {{name: globals()[name] for name in {FIELDS!r}}}\n"
            "estado.update(shap_values=shap_exp.values, shap_base=shap_exp.base_values)\n"
            "with Path('estado.pkl').open('wb') as stream:\n"
            "    pickle.dump(estado, stream)"
        )
    )
    working_copy = run_dir / "execucao.ipynb"
    nbformat.write(nb, working_copy)
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("run_kernel.py")), str(working_copy)],
        cwd=run_dir,
        env={
            **os.environ,
            "PYTHONPATH": str(root / "src"),
            "CREDIT_RISK_CSV": str(csv),
            "CREDIT_RISK_ARTIFACTS": str(run_dir / "artifacts"),
        },
        check=True,
        timeout=3600,
    )
    with (run_dir / "estado.pkl").open("rb") as stream:
        return pickle.load(stream)


@pytest.mark.parametrize("base", ["controle_sintetico", "csv_do_case"])
def test_equivalencia_pareada(tmp_path: Path, base: str) -> None:
    """Compara linha a linha as decisões da régua e as previsões nos dois splits."""
    original = os.environ.get("CREDIT_RISK_ORIGINAL_NOTEBOOK")
    if original is None:
        pytest.skip("CREDIT_RISK_ORIGINAL_NOTEBOOK habilita a comparação com o código recebido.")
    root = Path(__file__).resolve().parents[2]
    if base == "controle_sintetico":
        csv = tmp_path / "DADOS_CASE_modelagem .csv"
        _write_control_csv(csv)
    else:
        csv = Path(os.environ.get("CREDIT_RISK_CSV", root / "data/raw/DADOS_CASE_modelagem .csv"))
        csv = csv.expanduser().resolve(strict=True)
    before = _execute(Path(original).resolve(strict=True), tmp_path / "original", csv, root)
    after = _execute(root / "notebooks/credit_risk_modeling.ipynb", tmp_path / "modular", csv, root)
    assert before.keys() == after.keys()
    for key, expected in before.items():
        actual = after[key]
        if isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(expected, actual, check_exact=True, obj=key)
        elif isinstance(expected, pd.Series):
            pd.testing.assert_series_equal(expected, actual, check_exact=True, obj=key)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(expected, actual, err_msg=key)
        else:
            assert expected == actual, key
