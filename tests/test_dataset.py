import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.pipeline import make_pipeline_from_config
from src.utils.export import export
from src.visual.visualizer import plot_snapshot

IMAGES = ROOT / "images/dataset"
BASE = str(ROOT / "agrixel-fair-dockerV2/data/output/files")
CONFIG = str(ROOT / "src/config/dataset.yml")
PARCEL = "L62196,L62198"

pipeline = make_pipeline_from_config(BASE, CONFIG)

# ---------------------------------------------------------------------------
# 1. Load cubes — indices from config are computed and appended automatically
# ---------------------------------------------------------------------------

cube_s2 = pipeline.load(PARCEL, "SENTINEL-2")
cube_s1 = pipeline.load(PARCEL, "SENTINEL-1")
cube_s3 = pipeline.load(PARCEL, "SENTINEL-3")

# Print the underlying xarray Dataset (shows all variables including computed ones)
print("=== SENTINEL-2 cube ===")
print(cube_s2)

print("\n=== SENTINEL-1 cube ===")
print(cube_s1)

print("\n=== SENTINEL-3 cube ===")
print(cube_s3)

# ---------------------------------------------------------------------------
# 2. Tabular stats — one row per timestamp, one column per band/index/stat
# ---------------------------------------------------------------------------

df = pipeline.execute(
    parcel_keys=[PARCEL],
    sensors=["SENTINEL-2", "SENTINEL-1"],
)

print("\n=== Tabular stats ===")
print(df)

export(df, output_dir=ROOT / "data/datasets/processed", name="test_train")

