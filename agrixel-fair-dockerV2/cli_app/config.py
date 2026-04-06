"""YAML config parsing and validation for CLI runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class S2Config(BaseModel):
    bands: list[str] = Field(default=["B04", "B08", "NDVI"])
    cloud_cover: int = Field(default=20, ge=0, le=100)


class S1Config(BaseModel):
    polarizations: list[str] = Field(default=["VV", "VH"])


class S3Config(BaseModel):
    products: list[str] = Field(default=["lst-in"])


class ERA5Config(BaseModel):
    variable: str = "tp"
    daily_agg: Literal["sum", "mean", "min", "max"] = "sum"


class MODISConfig(BaseModel):
    products: list[str] = Field(default=["ET_500m", "PET_500m"])


class SensorsConfig(BaseModel):
    S1: S1Config | None = None
    S2: S2Config | None = None
    S3: S3Config | None = None
    ERA5: ERA5Config | None = None
    MODIS: MODISConfig | None = None


class RunConfig(BaseModel):
    equidistant: bool = True
    max_parallel: int = Field(default=3, ge=1)

    buffer_m: int = Field(default=100, ge=0)
    target_res_m: int | None = None

    sensors: SensorsConfig

    def enabled_sensors(self) -> dict[str, BaseModel]:
        """Return dict of sensor_key -> config for sensors that are not None."""
        result = {}
        for key in ("S1", "S2", "S3", "ERA5", "MODIS"):
            cfg = getattr(self.sensors, key)
            if cfg is not None:
                result[key] = cfg
        return result


def load_config(path: Path) -> RunConfig:
    """Load and validate a YAML config file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunConfig(**raw)


def load_env(path: Path) -> dict[str, str]:
    """Load a .env file into a dict. Returns empty dict if file doesn't exist."""
    if not path.is_file():
        return {}
    from dotenv import dotenv_values

    return {k: v for k, v in dotenv_values(path).items() if v is not None}
