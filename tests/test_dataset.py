import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import make_pipeline_from_config, plot_snapshot

BASE = str(ROOT / "agrixel-fair-dockerV2/data/output/files")
CONFIG = str(ROOT / "src/config/dataset.yml")
PARCEL = "L62196,L62198"

pipeline = make_pipeline_from_config(BASE, CONFIG)

cube_sample = pipeline.load(PARCEL, "SENTINEL-2")

print(cube_sample.ds)


# --- Tabular stats ---
df = pipeline.process_many(
    parcel_keys=[PARCEL],
    # sensors=["SENTINEL-2", "SENTINEL-1"],
    sensors=["SENTINEL-2"],
)
print(df)

# Sentinel-2: RGB + NDVI + EVI
# cube_s2 = pipeline.load(PARCEL, "SENTINEL-2")
# print(cube_s2)


