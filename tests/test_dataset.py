from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import make_pipeline_from_config, plot_snapshot

BASE = str(ROOT / "agrixel-fair-dockerV2/data/output/files")
CONFIG = str(ROOT / "src/config/dataset.yml")
PARCEL = "L62196,L62198"

pipeline = make_pipeline_from_config(BASE, CONFIG)

# --- Tabular stats ---
df = pipeline.process_many(
    parcel_keys=[PARCEL],
    sensors=["SENTINEL-2", "SENTINEL-1"],
)
print(df)