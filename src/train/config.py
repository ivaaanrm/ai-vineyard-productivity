"""Training pipeline configuration — load from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration for a single model type."""

    enabled: bool = True
    params: Dict[str, Any] = {}


class MLflowConfig(BaseModel):
    """MLflow tracking configuration."""

    enabled: bool = False
    tracking_uri: str = "sqlite:///mlruns.db"
    experiment_name: str = "vineyard-productivity"
    log_artifacts: bool = True


class TrainingConfig(BaseModel):
    """Top-level training config."""

    features_csv: str
    targets: List[str] = ["yield_T_ha", "alcohol_degree"]
    log_transform_target: bool = False  # train on log(target), report metrics in original scale
    split_column: str | None = "split"
    test_size: float = 0.2
    random_state: int = 42
    experiment_directory: str = "experiments"
    dry_run: bool = False
    models: Dict[str, ModelConfig] = {}
    mlflow: MLflowConfig = MLflowConfig()

    @classmethod
    def from_yaml(cls, path: Path | str) -> TrainingConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
