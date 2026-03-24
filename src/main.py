from dataset import make_pipeline_from_config, plot_snapshot

BASE = "/Users/ivanr/Developer/ai-vineyard-productivity/agrixel-fair-dockerV2/data/output/files"
CONFIG = "/Users/ivanr/Developer/ai-vineyard-productivity/src/config/dataset.yml"
PARCEL = "L62196,L62198"

pipeline = make_pipeline_from_config(BASE, CONFIG)

# --- Tabular stats ---
df = pipeline.process_many(
    parcel_keys=[PARCEL],
    sensors=["SENTINEL-2", "SENTINEL-1"],
)
print(df)

# --- Snapshot visualizations (indices already computed in the cube) ---

# Sentinel-2: RGB + NDVI + EVI
cube_s2 = pipeline.load(PARCEL, "SENTINEL-2")
print(cube_s2)
plot_snapshot(
    cube_s2,
    time="2021-07-29",
    extra_bands=["EVI"],
    save_path="output_s2_snapshot.png",
)

# Sentinel-1: SAR RGB + RVI
cube_s1 = pipeline.load(PARCEL, "SENTINEL-1")
print(cube_s1)
plot_snapshot(
    cube_s1,
    time="2021-07-26",
    extra_bands=["RVI"],
    save_path="output_s1_snapshot.png",
)