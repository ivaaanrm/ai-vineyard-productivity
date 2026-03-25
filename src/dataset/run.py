import sys
import json
import pandas as pd
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.config import DatasetConfig
from src.dataset.pipeline import DatasetPipeline, make_pipeline_from_config
from src.utils.export import export
from src.visual.visualizer import plot_snapshot

IMAGES = ROOT / "images/dataset"
BASE = str(ROOT / "agrixel-fair-dockerV2/data/output/files")
CONFIG = str(ROOT / "src/config/dataset.yml")
SENSORS = ["SENTINEL-2", "SENTINEL-1"]


class DatasetProcessor:
    def __init__(self, pipeline: DatasetPipeline, df: pd.DataFrame) -> None:
        self.pipeline = pipeline
        self.df = df
    
    def run(self) -> pd.DataFrame:
        parcel_ids = self.df['parcel_id'].tolist()
        df = self.pipeline.execute(
            parcel_keys=parcel_ids,
            sensors=SENSORS
        )
        return df

    def _available_sensors(self, base_path: str, parcel_id: str) -> List[str]:
        return [
            s for s in SENSORS
            if (Path(base_path) / s / parcel_id / "cube.zarr").exists()
        ]
    
def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def main():
    config = DatasetConfig.from_yaml(CONFIG)
    pipeline = make_pipeline_from_config(BASE, CONFIG)
    df = load_df(config.paths['aoi_table'])
    
    dataset = DatasetProcessor(pipeline, df)
    df_processed = dataset.run()
    export(
        df_processed, 
        output_dir="/Users/ivanr/Developer/ai-vineyard-productivity/data/datasets/processed",
        name="train_df_prueba"
    )
    
    
            


if __name__ == "__main__":
    main()