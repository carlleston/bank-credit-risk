"""Divisão estratificada e distribuição observada do desfecho."""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_stratified(
    frame: pd.DataFrame, target: str = "target", test_size: float = 0.20, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide linhas sem reindexar e estratifica com o vetor original de classes."""
    x, y = frame.drop(columns=[target]), frame[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    return x_train, x_test, y_train, y_test


def show_split(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Apresenta as dimensões e frequências relativas em cada amostra."""
    print(f"Treino: {x_train.shape}")
    print(f"Teste:  {x_test.shape}")
    print("\nDistribuição da target:")
    print(f"Base original:\n{y.value_counts(normalize=True)}")
    print(f"\nTreino:\n{y_train.value_counts(normalize=True)}")
    print(f"\nTeste:\n{y_test.value_counts(normalize=True)}")
