"""Random Forest regressor wrapper."""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestModel:
    """Thin wrapper around sklearn RandomForestRegressor."""

    name = "random_forest"

    def __init__(self, **params: Any) -> None:
        self._params = params
        self._model = RandomForestRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return {"name": self.name, **self._params}

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_

    def save(self, path: Path) -> Path:
        path = Path(path)
        joblib.dump(self._model, path)
        return path

    @classmethod
    def load(cls, path: Path) -> RandomForestModel:
        instance = cls.__new__(cls)
        instance._model = joblib.load(path)
        instance._params = instance._model.get_params()
        instance.name = "random_forest"
        return instance
