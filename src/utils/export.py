import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

_BAND_UNITS: Dict[str, str] = {
    # Sentinel-1 (after pipeline preprocessing)
    "VV": "dB",
    "VH": "dB",
    "RVI": "dimensionless",
    "VH_VV": "dimensionless",
    "DpRVI": "dimensionless",
    # Sentinel-2 reflectance bands
    "B02": "reflectance",
    "B03": "reflectance",
    "B04": "reflectance",
    "B08": "reflectance",
    "B11": "reflectance",
    "B12": "reflectance",
    # Sentinel-2 indices
    "NDVI": "dimensionless",
    "EVI": "dimensionless",
    "SAVI": "dimensionless",
    "NBR2": "dimensionless",
    "NDWI": "dimensionless",
    # Sentinel-3
    "LST": "K",
    "LST_C": "°C",
}

_META_COLUMNS: Dict[str, str] = {
    "time": "date",
    "parcel_key": "identifier",
    "sensor": "identifier",
    "n_samples": "count",
}


def _unit_for_column(col: str) -> str:
    """Resolve the unit for a stat column like 'VV_mean' or a meta column."""
    if col in _META_COLUMNS:
        return _META_COLUMNS[col]
    # Strip stat suffix (e.g. 'VV_mean' → 'VV')
    band = col.rsplit("_", 1)[0]
    return _BAND_UNITS.get(band, "unknown")


def export(
    df: pd.DataFrame,
    output_dir: Path | str = "data/datasets/processed",
    name: str = "dataset",
) -> Path:
    """Export a stats DataFrame as parquet with a companion JSON metadata file.

    Returns the path to the parquet file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{name}.csv"
    meta_path = output_dir / f"{name}_metadata.json"

    df.to_csv(parquet_path, index=False)

    metadata: Dict[str, Any] = {"columns": {}}
    for col in df.columns:
        info: Dict[str, Any] = {"unit": _unit_for_column(col)}
        if pd.api.types.is_numeric_dtype(df[col]):
            info["min"] = float(np.nanmin(df[col]))
            info["max"] = float(np.nanmax(df[col]))
        else:
            info["n_unique"] = int(df[col].nunique())
        metadata["columns"][col] = info
    metadata["n_rows"] = len(df)

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4, default=str)

    return parquet_path
