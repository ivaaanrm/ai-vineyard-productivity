"""Base model protocol for training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Regressor(Protocol):
    """Contract for all regression models in the training pipeline."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def get_params(self) -> Dict[str, Any]: ...

    def save(self, path: Path) -> Path: ...

    @classmethod
    def load(cls, path: Path) -> Regressor: ...
