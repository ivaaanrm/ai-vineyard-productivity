import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.pipeline import make_pipeline_from_config
from src.visual.visualizer import plot_mosaic, plot_snapshot

IMAGES = ROOT / "images/dataset"
BASE = str(ROOT / "agrixel-fair-dockerV2/data/output/files")
CONFIG = str(ROOT / "src/config/dataset.yml")
PARCEL = "L62196,L62198"

pipeline = make_pipeline_from_config(BASE, CONFIG)

cube_s2 = pipeline.load(PARCEL, "SENTINEL-2")
cube_s1 = pipeline.load(PARCEL, "SENTINEL-1")

plot_snapshot(
    cube_s2,
    time="2021-02-20",
    extra_bands=["EVI"],
    save_path=IMAGES / "s2_20210220.png",
)
plot_snapshot(
    cube_s2,
    time="2021-07-26",
    extra_bands=["EVI"],
    save_path=IMAGES / "s2_20210726.png",
)

# Sentinel-1: SAR-RGB composite + NDVI skipped (not available) + RVI as extra
plot_snapshot(
    cube_s1,
    time="2021-07-26",
    extra_bands=["VV", "VH", "RVI"],
    save_path=IMAGES / "s1_20210726.png",
)

plot_mosaic(
    rows=[
        (cube_s2, "2021-07-26", ["EVI"]),
        (cube_s1, "2021-07-26", ["VV", "VH", "RVI"]),
    ],
    save_path=IMAGES / "mosaic_20210726.png",
)
