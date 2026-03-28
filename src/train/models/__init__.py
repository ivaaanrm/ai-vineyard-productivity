"""Model registry — maps config names to model classes."""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import Regressor
from .random_forest import RandomForestModel
from .xgboost import XGBoostModel

MODEL_REGISTRY: Dict[str, Type] = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
}


def build_model(name: str, params: Dict[str, Any]) -> Regressor:
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**params)


__all__ = [
    "MODEL_REGISTRY",
    "Regressor",
    "RandomForestModel",
    "XGBoostModel",
    "build_model",
]
