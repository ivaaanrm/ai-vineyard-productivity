import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from shapely import wkt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.pipeline import make_pipeline_from_config
from src.visual.panels import _make_s2_rgb

IMAGES = ROOT / "images/dataset"
BASE = str(ROOT / "agrixel-fair-dockerV2/data/output/files")
CONFIG = str(ROOT / "src/config/dataset.yml")
PARCEL = "L62196,L62198"

pipeline = make_pipeline_from_config(BASE, CONFIG)
cube_s2 = pipeline.load(PARCEL, "SENTINEL-2")
cube_s1 = pipeline.load(PARCEL, "SENTINEL-1")

# ---------------------------------------------------------------------------
# Build mask polygon 
# ---------------------------------------------------------------------------
geometry_wkt = ( 
    """POLYGON ((33613.334949522614 5031722.596604069, 33593.068685444116 5031760.739927518, 33590.67069747535 5031764.30780794, 33582.12387646556 5031780.152379262, 33553.59052553799 5031809.822338418, 33552.564242671506 5031824.787427334, 33588.15822190977 5031841.696239057, 33595.89635180184 5031845.36461716, 33603.09031570816 5031831.167560591, 33606.33086520837 5031823.671257508, 33615.80898263514 5031801.745421871, 33625.147046367696 5031784.127834359, 33637.36185590046 5031764.727762703, 33651.41014650662 5031739.2080491325, 33659.04451909826 5031728.130445795, 33660.843383302534 5031725.089440129, 33661.34752061753 5031724.237146061, 33661.98713958741 5031723.155798068, 33621.58328167979 5031706.987374397, 33614.15605046134 5031720.7630574675, 33613.334949522614 5031722.596604069))"""
)
geometry = wkt.loads(geometry_wkt)  # for pixel-coordinate overlay

cube_s2_mask = cube_s2.mask(geometry_wkt)

# ---------------------------------------------------------------------------
# Spatial means (raw)
# ---------------------------------------------------------------------------

ndvi_full   = cube_s2.ds.NDVI.mean(["x", "y"])
ndvi_masked = cube_s2_mask.ds.NDVI.mean(["x", "y"])
evi_full    = cube_s2.ds.EVI.mean(["x", "y"])
evi_masked  = cube_s2_mask.ds.EVI.mean(["x", "y"])
times = pd.DatetimeIndex(cube_s2.times)

# ---------------------------------------------------------------------------
# Smoothed series (resample → rolling)
# ---------------------------------------------------------------------------

resample_period = "4W"
window = 2

def smooth(da):
    return da.resample(time=resample_period).median().rolling(time=window, min_periods=1).mean()

ndvi_full_s   = smooth(ndvi_full)
ndvi_masked_s = smooth(ndvi_masked)
evi_full_s    = smooth(evi_full)
evi_masked_s  = smooth(evi_masked)

# ---------------------------------------------------------------------------
# RGB snapshot — pick a mid-summer date
# ---------------------------------------------------------------------------

t_idx = int(np.argmin(np.abs(times - pd.Timestamp("2021-07-26"))))

rgb_img = _make_s2_rgb(
    cube_s2.ds["B04"].isel(time=t_idx).values,
    cube_s2.ds["B03"].isel(time=t_idx).values,
    cube_s2.ds["B02"].isel(time=t_idx).values,
)

ndvi_img = cube_s2.ds.NDVI.isel(time=t_idx).values

# Polygon in pixel coordinates for overlay
x = cube_s2.ds.coords["x"].values
y = cube_s2.ds.coords["y"].values
poly_coords = np.array(geometry.exterior.coords)
px = np.interp(poly_coords[:, 0], x, np.arange(len(x)))
py = np.interp(poly_coords[:, 1], y[::-1], np.arange(len(y))[::-1])

def _fmt_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_tick_params(rotation=45)

def _mask_patch():
    return MplPolygon(np.column_stack([px, py]), closed=True)

# ---------------------------------------------------------------------------
# Figure: 3 rows × 2 cols
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(16, 9))

# --- Row 1: RGB | NDVI snapshot ---

axes[0, 0].imshow(rgb_img)
axes[0, 0].add_collection(PatchCollection(
    [_mask_patch()], facecolor="white", edgecolor="yellow", linewidth=2, alpha=0.25,
))
axes[0, 0].set_title(f"RGB · {times[t_idx].date()}")
axes[0, 0].axis("off")

im = axes[0, 1].imshow(ndvi_img, cmap="RdYlGn", vmin=-1, vmax=1)
axes[0, 1].add_collection(PatchCollection(
    [_mask_patch()], facecolor="none", edgecolor="yellow", linewidth=2,
))
fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
axes[0, 1].set_title(f"NDVI · {times[t_idx].date()}")
axes[0, 1].axis("off")

# --- Row 2: raw NDVI | raw EVI ---

axes[1, 0].plot(times, ndvi_full.values,   "g-o",  ms=3, lw=1.2, label="Full extent")
axes[1, 0].plot(times, ndvi_masked.values, "b--^", ms=3, lw=1.2, label="Masked area")
axes[1, 0].set_title("NDVI — raw")
axes[1, 0].set_ylabel("Spatial mean")
_fmt_time_axis(axes[1, 0])
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

axes[1, 1].plot(times, evi_full.values,   "g-o",  ms=3, lw=1.2, label="Full extent")
axes[1, 1].plot(times, evi_masked.values, "b--^", ms=3, lw=1.2, label="Masked area")
axes[1, 1].set_title("EVI — raw")
axes[1, 1].set_ylim(bottom=0)
axes[1, 1].set_ylabel("Spatial mean")
_fmt_time_axis(axes[1, 1])
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

# --- Row 3: smoothed NDVI | smoothed EVI ---

axes[2, 0].plot(ndvi_full_s.time.values,   ndvi_full_s.values,   "g-",  ms=4, lw=1.5, label="Full extent")
axes[2, 0].plot(ndvi_masked_s.time.values, ndvi_masked_s.values, "b-", ms=4, lw=1.5, label="Masked area")
axes[2, 0].set_title(f"NDVI — smoothed ({resample_period} resample, {window}-step rolling)")
axes[2, 0].set_ylabel("Spatial mean")
_fmt_time_axis(axes[2, 0])
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].legend()

axes[2, 1].plot(evi_full_s.time.values,   evi_full_s.values,   "g-",  ms=4, lw=1.5, label="Full extent")
axes[2, 1].plot(evi_masked_s.time.values, evi_masked_s.values, "b-", ms=4, lw=1.5, label="Masked area")
axes[2, 1].set_ylim(bottom=0)
axes[2, 1].set_title(f"EVI — smoothed ({resample_period} resample, {window}-step rolling)")
axes[2, 1].set_ylabel("Spatial mean")
_fmt_time_axis(axes[2, 1])
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].legend()

fig.suptitle(f"Parcel {PARCEL} — mask effect on vegetation indices", fontsize=13, fontweight="bold")
fig.tight_layout()

save_path = IMAGES / "mask_ndvi_rgb.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Saved: {save_path}")
plt.show()
