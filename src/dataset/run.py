import sys
import pandas as pd
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.config import DatasetConfig
from src.dataset.pipeline import DatasetPipeline, make_pipeline_from_config
from src.utils.export import export

IMAGES = ROOT / "images/dataset"
BASE = str(ROOT / "agrixel-fair-dockerV2/data/output/files")
CONFIG = str(ROOT / "src/config/dataset.yml")
SENSORS = [
    "SENTINEL-2", 
    "SENTINEL-1", 
    "SENTINEL-3", 
    # "ERA5", 
    # "MODIS"
]


class DatasetProcessor:
    def __init__(self, pipeline: DatasetPipeline, df: pd.DataFrame) -> None:
        self.pipeline = pipeline
        self.df = df

    def run(self) -> pd.DataFrame:
        parcel_ids = self.df["parcel_id"].tolist()
        geometries = (
            dict(zip(self.df["parcel_id"], self.df["parcel_geometry"]))
            if "parcel_geometry" in self.df.columns
            else None
        )
        df = self.pipeline.execute(
            parcel_keys=parcel_ids,
            sensors=SENSORS,
            geometries=geometries,
        )
        return df

    def _available_sensors(self, base_path: str, parcel_id: str) -> List[str]:
        return [
            s
            for s in SENSORS
            if (Path(base_path) / s / parcel_id / "cube.zarr").exists()
        ]


def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    config = DatasetConfig.from_yaml(CONFIG)
    pipeline = make_pipeline_from_config(BASE, CONFIG)
    df = load_df(config.paths["aoi_table"])

    dataset = DatasetProcessor(pipeline, df)
    df_processed = dataset.run()
    print(df_processed)
    
    export(
        df_processed,
        output_dir="/Users/ivanr/Developer/ai-vineyard-productivity/experiments/PRUEBA00/data",
        name="train_df_small_2020",
        config=config,
    )


if __name__ == "__main__":
    main()
