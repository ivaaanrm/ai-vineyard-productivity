"""XGBoost regressor wrapper."""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any, Dict

import numpy as np
from xgboost import XGBRegressor


class XGBoostModel:
    """Thin wrapper around XGBRegressor."""

    name = "xgboost"

    def __init__(self, **params: Any) -> None:
        self._params = params
        self._model = XGBRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        self._model.fit(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return {"name": self.name, **self._params}

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_

    @property
    def evals_result(self) -> Dict[str, Any] | None:
        """Return per-round eval metrics if eval_set was used during fit."""
        result = self._model.evals_result()
        return result if result else None

    def save(self, path: Path) -> Path:
        path = Path(path)
        joblib.dump(self._model, path)
        return path

    @classmethod
    def load(cls, path: Path) -> XGBoostModel:
        instance = cls.__new__(cls)
        instance._model = joblib.load(path)
        instance._params = instance._model.get_params()
        instance.name = "xgboost"
        return instance
