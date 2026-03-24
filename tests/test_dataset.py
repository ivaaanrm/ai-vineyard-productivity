import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import make_pipeline_from_config
from src.visual import plot_snapshot

IMAGES = ROOT / "data/images"
BASE = str(ROOT / "agrixel-fair-dockerV2/data/output/files")
CONFIG = str(ROOT / "src/config/dataset.yml")
PARCEL = "L62196,L62198"

pipeline = make_pipeline_from_config(BASE, CONFIG)

# ---------------------------------------------------------------------------
# 1. Load cubes — indices from config are computed and appended automatically
# ---------------------------------------------------------------------------

cube_s2 = pipeline.load(PARCEL, "SENTINEL-2")
cube_s1 = pipeline.load(PARCEL, "SENTINEL-1")

# Print the underlying xarray Dataset (shows all variables including computed ones)
print("=== SENTINEL-2 cube ===")
print(cube_s2)

print("\n=== SENTINEL-1 cube ===")
print(cube_s1)

# ---------------------------------------------------------------------------
# 2. Tabular stats — one row per timestamp, one column per band/index/stat
# ---------------------------------------------------------------------------

df = pipeline.process_many(
    parcel_keys=[PARCEL],
    sensors=["SENTINEL-2", "SENTINEL-1"],
)

print("\n=== Tabular stats ===")
print(df)

# ---------------------------------------------------------------------------
# 3. Snapshot plots — default panels (RGB/SAR-RGB + NDVI) + extra indices
# ---------------------------------------------------------------------------

# Sentinel-2: RGB composite + NDVI (default) + EVI and SAVI as extras
plot_snapshot(
    cube_s2,
    time="2021-07-29",
    extra_bands=["EVI", "NDWI"],
    save_path=IMAGES / "output_s2_snapshot.png",
)

# Sentinel-1: SAR-RGB composite + NDVI skipped (not available) + RVI as extra
plot_snapshot(
    cube_s1,
    time="2021-07-26",
    extra_bands=["VV", "RVI", "VH_VV"],
    save_path=IMAGES / "output_s1_snapshot.png",
)
