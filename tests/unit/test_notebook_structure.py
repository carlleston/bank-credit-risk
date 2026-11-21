"""Contrato de apresentação e orquestração do notebook entregue."""

import ast
from pathlib import Path

import nbformat


def test_notebook_contains_only_orchestration() -> None:
    notebook = Path(__file__).resolve().parents[2] / "notebooks" / "credit_risk_modeling.ipynb"
    contents = nbformat.read(notebook, as_version=4)
    nbformat.validate(contents)
    business_logic = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Lambda,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.If,
        ast.Match,
        ast.ListComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.With,
    )
    for index, cell in enumerate(contents.cells):
        if cell.cell_type == "code":
            nodes = list(ast.walk(ast.parse(cell.source)))
            assert not any(isinstance(node, business_logic) for node in nodes), index
            assert not any(isinstance(node, ast.Import) for node in nodes), index
            for node in nodes:
                if isinstance(node, ast.ImportFrom):
                    assert node.module and node.module.split(".")[0] == "credit_risk", index
