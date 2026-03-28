"""Feature engineering CLI entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.export import export
from src.features.config import FeaturesConfig
from src.features.pipeline import FeaturePipeline


def main(config_path: str | Path | None = None) -> Path:
    """Run the feature pipeline from a YAML config and export results.

    Returns the path to the output CSV.
    """
    config_path = Path(
        config_path or Path(__file__).parent.parent / "config" / "features.yml"
    )
    config = FeaturesConfig.from_yaml(config_path)
    pipeline = FeaturePipeline(config)
    result = pipeline.execute()

    print(f"Feature matrix: {result.shape[0]} rows x {result.shape[1]} columns")

    output_dir = Path(config.dataset_csv).parent
    return export(round(result,4), output_dir=output_dir, name="features", config=config)


if __name__ == "__main__":
    main()
