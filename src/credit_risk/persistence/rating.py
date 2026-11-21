"""Leitura e escrita dos oito artefatos originais da régua em formato pickle."""

import pickle
from pathlib import Path

import pandas as pd

from credit_risk.rating.scale import RatingScale


def save_rating_artifacts(
    directory: Path,
    scale: RatingScale,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_scored: pd.DataFrame,
    test_scored: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Salva objetos de binning, mapeamento, tabelas e classificações individuais."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "binning_pd_monotonico.pkl").open("wb") as file:
        pickle.dump(scale.binning, file)
    with (directory / "mapa_rating_7_grupos.pkl").open("wb") as file:
        pickle.dump(scale.mapping, file)
    frames = {
        "regua_pd_7_grupos.pkl": scale.limits(),
        "tabela_rating_train.pkl": train,
        "tabela_rating_test.pkl": test,
        "comparacao_rating.pkl": comparison,
        "clientes_rating_train.pkl": train_scored,
        "clientes_rating_test.pkl": test_scored,
    }
    for filename, frame in frames.items():
        frame.to_pickle(directory / filename)


def load_rating_scale(directory: Path) -> RatingScale:
    """Carrega uma régua de origem confiável e confere o mapa e os limites gravados."""
    directory = Path(directory)
    with (directory / "binning_pd_monotonico.pkl").open("rb") as file:
        scale = RatingScale(pickle.load(file))
    with (directory / "mapa_rating_7_grupos.pkl").open("rb") as file:
        saved_mapping = pickle.load(file)
    if scale.mapping != saved_mapping:
        raise ValueError("O mapeamento salvo diverge da consolidação dos grupos 3 e 4.")
    limits = pd.read_pickle(directory / "regua_pd_7_grupos.pkl")
    if not scale.limits().equals(limits):
        raise ValueError("Os limites salvos divergem dos cortes do discretizador.")
    return scale
