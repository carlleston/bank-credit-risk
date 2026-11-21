"""Executa células sequenciais em um novo ipykernel, sem precisar abrir portas."""

import sys
from pathlib import Path

import nbformat
from ipykernel.inprocess.manager import InProcessKernelManager
from nbformat.v4 import output_from_msg


def execute_cell(client, source: str, *, store_history: bool = True) -> tuple[int, list]:
    """Consome a resposta de execução e todas as mensagens de saída até o estado idle."""
    request = client.execute(source, store_history=store_history)
    reply = client.get_shell_msg(timeout=30)
    outputs = []
    while True:
        message = client.get_iopub_msg(timeout=30)
        kind = message["msg_type"]
        content = message["content"]
        if (
            kind == "status"
            and content["execution_state"] == "idle"
            and message["parent_header"].get("msg_id") == request
        ):
            break
        if kind == "clear_output":
            outputs.clear()
        elif kind in {"display_data", "execute_result", "stream", "error"}:
            outputs.append(output_from_msg(message))
    if reply["content"]["status"] != "ok":
        raise RuntimeError("\n".join(reply["content"].get("traceback", [])))
    return reply["content"]["execution_count"], outputs


def run(notebook_file: Path) -> None:
    """Executa e atualiza apenas a cópia temporária fornecida pelo teste."""
    notebook = nbformat.read(notebook_file, as_version=4)
    manager = InProcessKernelManager()
    manager.start_kernel()
    client = manager.client()
    client.start_channels()
    try:
        execute_cell(client, "%matplotlib inline", store_history=False)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                try:
                    cell.execution_count, cell.outputs = execute_cell(client, cell.source)
                except Exception as exc:
                    raise RuntimeError(f"Falha na célula {index}, {cell.source[:100]!r}.") from exc
    finally:
        client.stop_channels()
        manager.shutdown_kernel()
    nbformat.write(notebook, notebook_file)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
