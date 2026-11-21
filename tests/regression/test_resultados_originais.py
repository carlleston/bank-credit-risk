"""Reexecuta o notebook em kernel novo e compara suas saídas com o case histórico."""

import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal

import nbformat
import pytest


class DataFrameTable(HTMLParser):
    """Lê o conteúdo visível das células das tabelas HTML emitidas pelo pandas."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row:
            self.rows.append(self._row)

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def table_rows(html: str) -> list[list[str]]:
    """Extrai cabeçalhos, índices e valores sem inferir tipos de strings como '3, 4'."""
    table = DataFrameTable()
    table.feed(html)
    table.close()
    return table.rows


# A coleta roda somente depois de todas as células do notebook, no mesmo kernel.
# Cada campo tem uma referência externa na saída gravada pelo notebook recebido.
COLLECT = """
import json
print(json.dumps({
    "features": list(selecionadas),
    "removed": list(removidas),
    "train_shape": list(X_train.shape),
    "test_shape": list(X_test.shape),
    "train_events": int(y_train.sum()),
    "test_events": int(y_test.sum()),
    "trials": len(busca.trials.results),
    "best_params": best_params,
    "cv_trees": int(best_n_estimators),
    "cv_auc": float(melhor_trial["auc_cv"]),
    "cv_loss": float(melhor_trial["loss"]),
    "gini_difference": float(bootstrap.observed),
    "bootstrap_lower": float(bootstrap.lower),
    "bootstrap_upper": float(bootstrap.upper)
}, ensure_ascii=False))
"""


def _compare_table(actual: list, reference: dict) -> str | None:
    expected = reference["rows"]
    if len(actual) != len(expected):
        return f"quantidade de linhas {len(actual)} (esperado {len(expected)})"
    for i, (actual_row, expected_row) in enumerate(zip(actual, expected, strict=True)):
        if len(actual_row) != len(expected_row):
            return f"linha {i} contém {len(actual_row)} valores (esperado {len(expected_row)})"
        for j, (found, wanted) in enumerate(zip(actual_row, expected_row, strict=True)):
            if found == wanted:
                continue
            # Cabeçalhos, índices, intervalos formatados, categorias e textos são exatos.
            if i and j:
                try:
                    value, target = float(found), float(wanted)
                except ValueError:
                    pass
                else:
                    tolerance = reference["atol"]
                    if (
                        expected[0][j]
                        in {
                            "Clientes",
                            "Eventos",
                            "Clientes_Treino",
                            "Clientes_Teste",
                            "Count",
                            "Non-event",
                            "Event",
                            "Grupos_Originais",
                        }
                        or expected_row[0] == "count"
                    ):
                        tolerance = 0.0
                    if math.isclose(value, target, rel_tol=0, abs_tol=tolerance):
                        continue
            return f"linha {expected_row[0]!r}, coluna {expected[0][j]!r}: {found!r} (esperado {wanted!r})"
    return None


def compare_outputs(notebook: nbformat.NotebookNode, metrics: dict, reference: dict) -> list[str]:
    """Retorna as divergências por tabela, figura ou indicador."""
    tables = defaultdict(list)
    images = Counter()
    for cell in notebook.cells:
        origin = cell.metadata.get("original_cell")
        if cell.cell_type != "code" or origin is None:
            continue
        for output in cell.outputs:
            data = output.get("data", {})
            if "text/html" in data:
                rows = table_rows(data["text/html"])
                if rows:
                    tables[str(origin)].append(rows)
            if "image/png" in data:
                images[str(origin)] += 1

    errors = []
    for cell, expected_tables in reference["tables"].items():
        actual = tables[cell]
        if len(actual) != len(expected_tables):
            errors.append(
                f"Célula {cell}: {len(actual)} tabelas, esperadas {len(expected_tables)}."
            )
            continue
        for table, expected in zip(actual, expected_tables, strict=True):
            diff = _compare_table(table, expected)
            if diff:
                errors.append(f"Célula {cell}, saída original {expected['output']}: {diff}")
    for cell, count in reference["figures"].items():
        if images[cell] < count:
            errors.append(
                f"Célula {cell}: {images[cell]} figuras PNG, esperadas pelo menos {count}."
            )
    for field, expected in reference["metrics"].items():
        actual = metrics[field]
        if field == "best_params":
            if set(actual) != set(expected):
                errors.append(
                    f"Parâmetros vencedores: chaves {list(actual)} (esperadas {list(expected)})."
                )
                continue
            for key, value in expected.items():
                if not math.isclose(actual[key], value, rel_tol=0, abs_tol=1e-12):
                    errors.append(f"{field}.{key}: {actual[key]} (esperado {value}).")
        elif field in reference["metric_atol"]:
            if not math.isclose(
                actual, expected, rel_tol=0, abs_tol=reference["metric_atol"][field]
            ):
                errors.append(f"{field}: {actual} (esperado {expected}).")
        elif actual != expected:
            errors.append(f"{field}: {actual} (esperado {expected}).")
    return errors


def execute_and_compare(
    root: Path,
    csv: Path,
    artifact_dir: Path,
    *,
    executed_path: Path | None = None,
    kernel: Literal["inprocess", "jupyter"] = "inprocess",
) -> None:
    """Executa uma vez, grava opcionalmente o notebook fresco e verifica o histórico."""
    root = root.resolve()
    original = root / "notebooks" / "credit_risk_modeling.ipynb"
    notebook = nbformat.read(original, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.cells.append(nbformat.v4.new_code_cell(COLLECT))
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(root / "src"), os.environ.get("PYTHONPATH")))
        ),
        "CREDIT_RISK_CSV": str(csv.resolve(strict=True)),
        "CREDIT_RISK_ARTIFACTS": str(artifact_dir.resolve()),
    }
    if kernel == "jupyter":
        from jupyter_client import AsyncKernelManager
        from nbclient import NotebookClient

        manager = AsyncKernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        NotebookClient(notebook, km=manager, timeout=2400).execute(
            cwd=str(original.parent), env=env, cleanup_kc=True
        )
    elif kernel == "inprocess":
        with tempfile.TemporaryDirectory() as temporary:
            working_copy = Path(temporary) / "execucao.ipynb"
            nbformat.write(notebook, working_copy)
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name("run_kernel.py")), str(working_copy)],
                cwd=original.parent,
                env=env,
                check=True,
                timeout=3600,
            )
            notebook = nbformat.read(working_copy, as_version=4)
    else:
        raise ValueError(f"Kernel desconhecido: {kernel}")
    collection = notebook.cells.pop()
    metrics = json.loads(
        "".join(
            o.text for o in collection.outputs if o.output_type == "stream" and o.name == "stdout"
        )
    )
    if executed_path is not None:
        nbformat.write(notebook, executed_path)
    reference = json.loads(
        (root / "reports" / "referencia_original.json").read_text(encoding="utf-8")
    )
    differences = compare_outputs(notebook, metrics, reference)
    assert not differences, "Diferenças em relação ao notebook recebido:\n" + "\n".join(differences)


def test_regressao_contra_notebook_recebido(tmp_path: Path) -> None:
    """O dataset do case habilita a comparação numérica dos dois modelos e da régua."""
    root = Path(__file__).resolve().parents[2]
    csv = Path(os.environ.get("CREDIT_RISK_CSV", root / "data/raw/DADOS_CASE_modelagem .csv"))
    if not csv.is_file() and "CREDIT_RISK_CSV" not in os.environ:
        pytest.skip(f"Disponibilize o CSV original em {csv} ou configure CREDIT_RISK_CSV.")
    execute_and_compare(root, csv, tmp_path / "artefatos")
