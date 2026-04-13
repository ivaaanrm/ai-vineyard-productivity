import ast
import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.config import DatasetConfig
from src.dataset.pipeline import DatasetPipeline, make_composite_pipeline
from src.utils.export import export

IMAGES = ROOT / "images/dataset"
BASE = str(ROOT / "data/output/files")
ERA5_PATH = str(ROOT / "data/output/files/ERA5_TIMESERIES")
DEFAULT_CONFIG = str(ROOT / "src/config/dataset.yml")


class DatasetProcessor:
    def __init__(self, pipeline: DatasetPipeline, df: pd.DataFrame, sensors: List[str]) -> None:
        self.pipeline = pipeline
        self.df = df
        self.sensors = sensors

    def run(self) -> pd.DataFrame:
        parcel_ids = self.df["parcel_id"].tolist()
        geometries = (
            dict(zip(self.df["parcel_id"], self.df["parcel_geometry"]))
            if "parcel_geometry" in self.df.columns
            else None
        )
        df = self.pipeline.execute(
            parcel_keys=parcel_ids,
            sensors=self.sensors,
            geometries=geometries,
        )
        return self._filter_valid_years(df)

    def _filter_valid_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows whose year is not in the parcel's declared years (aoi_table)."""
        if "years" not in self.df.columns or df.empty:
            return df
        valid = (
            self.df[["parcel_id", "years"]]
            .copy()
            .assign(years=lambda d: d["years"].apply(
                lambda v: ast.literal_eval(v) if isinstance(v, str) else v
            ))
            .explode("years")
            .rename(columns={"years": "year"})
        )
        before = len(df)
        df = df.merge(valid, on=["parcel_id", "year"], how="inner")
        dropped = before - len(df)
        if dropped:
            print(f"Filtered {dropped} rows outside declared parcel years.")
        return df

    def _available_sensors(self, base_path: str, parcel_id: str) -> List[str]:
        return [
            s
            for s in self.sensors
            if (Path(base_path) / s / parcel_id / "cube.zarr").exists()
        ]


def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main(config_path: str | None = None):
    config_path = config_path or DEFAULT_CONFIG
    config = DatasetConfig.from_yaml(config_path)
    pipeline = make_composite_pipeline(BASE, ERA5_PATH, config=config)
    df = load_df(config.paths["aoi_table"])

    sensors = config.enabled_sensors()
    print("Enabled sensors:", sensors)
    dataset = DatasetProcessor(pipeline, df, sensors)
    df_processed = dataset.run()
    print(round(df_processed, 2))

    output_dir = config.output_dir or str(ROOT / "experiments" / "data")
    
    dataset_file = export(
        round(df_processed, 4),
        output_dir=output_dir,
        name="dataset",
        config=config,
    )


if __name__ == "__main__":
    main()
