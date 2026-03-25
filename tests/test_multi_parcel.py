import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely import wkt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.pipeline import make_pipeline_from_config
from src.visual.panels import SARRGBComposite, _make_s2_rgb

IMAGES = ROOT / "images/dataset"
BASE = str(ROOT / "agrixel-fair-dockerV2/data/output/files")
CONFIG = str(ROOT / "src/config/dataset.yml")
CSV_PATH = ROOT / "data/datasets/raw/aoi_small_train.csv"
SENSORS = ["SENTINEL-2", "SENTINEL-1"]

# Representative summer snapshot date — script picks the closest available timestep
TARGET_DATE = pd.Timestamp("2021-07-26")

RESAMPLE_PERIOD = "2W"
ROLLING_WINDOW = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def available_sensors(base_path: str, parcel_id: str) -> list[str]:
    return [
        s for s in SENSORS
        if (Path(base_path) / s / parcel_id / "cube.zarr").exists()
    ]


def closest_time_index(cube, target: pd.Timestamp) -> int:
    return int(np.argmin(np.abs(pd.DatetimeIndex(cube.times) - target)))


def poly_to_pixel(geometry, x_coords: np.ndarray, y_coords: np.ndarray):
    """Map shapely polygon exterior to image pixel coordinates."""
    coords = np.array(geometry.exterior.coords)
    px = np.interp(coords[:, 0], x_coords, np.arange(len(x_coords)))
    py = np.interp(coords[:, 1], y_coords[::-1], np.arange(len(y_coords))[::-1])
    return px, py


def smooth(da, resample_period: str = RESAMPLE_PERIOD, window: int = ROLLING_WINDOW):
    return da.resample(time=resample_period).median().rolling(time=window, min_periods=1).mean()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df_csv = pd.read_csv(CSV_PATH)
pipeline = make_pipeline_from_config(BASE, CONFIG)

parcels = []
for _, row in df_csv.iterrows():
    parcel_id = row["parcel_id"]
    geometry = wkt.loads(row["parcel_geometry"])
    cubes = {}
    for sensor in available_sensors(BASE, parcel_id):
        try:
            cubes[sensor] = pipeline.load(parcel_id, sensor)
            print(f"  Loaded {parcel_id} / {sensor}: {len(cubes[sensor].times)} timesteps")
        except Exception as e:
            print(f"  Failed {parcel_id} / {sensor}: {e}")
    if cubes:
        parcels.append({"parcel_id": parcel_id, "geometry": geometry, "cubes": cubes})

# ---------------------------------------------------------------------------
# Figure 1: RGB snapshot multiplot — one panel per parcel
# ---------------------------------------------------------------------------

n = len(parcels)
ncols = min(n, 3)
nrows = int(np.ceil(n / ncols))

fig1, axes = plt.subplots(
    nrows, ncols,
    figsize=(5 * ncols, 4.5 * nrows),
    squeeze=False,
)

for i, p in enumerate(parcels):
    ax = axes[i // ncols][i % ncols]

    if "SENTINEL-2" in p["cubes"]:
        cube = p["cubes"]["SENTINEL-2"]
        t_idx = closest_time_index(cube, TARGET_DATE)
        rgb = _make_s2_rgb(
            cube.ds["B04"].isel(time=t_idx).values,
            cube.ds["B03"].isel(time=t_idx).values,
            cube.ds["B02"].isel(time=t_idx).values,
        )
        sensor_label = "S2 RGB"
    else:
        cube = next(iter(p["cubes"].values()))
        t_idx = closest_time_index(cube, TARGET_DATE)
        rgb = SARRGBComposite().render(cube, t_idx)
        sensor_label = "SAR RGB"

    timestamp = pd.Timestamp(cube.times[t_idx]).date()
    ax.imshow(rgb)

    px, py = poly_to_pixel(
        p["geometry"],
        cube.ds.coords["x"].values,
        cube.ds.coords["y"].values,
    )
    ax.add_collection(PatchCollection(
        [MplPolygon(np.column_stack([px, py]), closed=True)],
        facecolor="white", edgecolor="yellow", linewidth=1.5, alpha=0.2,
    ))
    ax.set_title(f"{p['parcel_id']}\n{sensor_label} · {timestamp}", fontsize=9)
    ax.axis("off")

for j in range(n, nrows * ncols):
    axes[j // ncols][j % ncols].set_visible(False)

fig1.suptitle("RGB Snapshots — Multi-Parcel Overview", fontsize=13, fontweight="bold")
fig1.tight_layout()

save1 = IMAGES / "multi_parcel_rgb.png"
fig1.savefig(save1, dpi=150, bbox_inches="tight")
print(f"Saved: {save1}")

# ---------------------------------------------------------------------------
# Figure 2: NDVI time series per parcel — raw dots + rolling-average line
# ---------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(14, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(parcels)))

for p, color in zip(parcels, colors):
    if "SENTINEL-2" not in p["cubes"]:
        continue
    cube = p["cubes"]["SENTINEL-2"]
    cube_masked = cube.mask(p["geometry"])

    times = pd.DatetimeIndex(cube_masked.times)
    ndvi_raw = cube_masked.ds.NDVI.mean(["x", "y"])
    ndvi_smooth = smooth(ndvi_raw)

    ax2.plot(times, ndvi_raw.values, "o", ms=2, color=color, alpha=0.25)
    ax2.plot(
        ndvi_smooth.time.values, ndvi_smooth.values,
        "-", lw=2, color=color, label=p["parcel_id"],
    )

ax2.set_title(
    f"NDVI time series — all parcels"
    f" (masked · {RESAMPLE_PERIOD} resample · {ROLLING_WINDOW}-step rolling avg)",
    fontsize=11,
)
ax2.set_ylabel("NDVI (spatial mean)")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax2.tick_params(axis="x", rotation=45)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper left", fontsize=8, ncol=2)
fig2.tight_layout()

save2 = IMAGES / "multi_parcel_timeseries.png"
fig2.savefig(save2, dpi=150, bbox_inches="tight")
print(f"Saved: {save2}")

plt.show()
