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
from src.visual.panels import _make_s2_rgb, SARRGBComposite

IMAGES = ROOT / "images/dataset"
BASE = str(ROOT / "agrixel-fair-dockerV2/data/output/files")
CONFIG = str(ROOT / "src/config/dataset.yml")
PARCEL = "L62196,L62198"

# ---------------------------------------------------------------------------
# Harvest metadata
# ---------------------------------------------------------------------------

harvest = pd.DataFrame([
    {"campaign": 2021, "harvest_date": pd.Timestamp("2021-09-07"), "yield_kg_ha": 7034.20, "alcohol_degree": 12.67},
    {"campaign": 2022, "harvest_date": pd.Timestamp("2022-08-24"), "yield_kg_ha": 2991.79, "alcohol_degree": 11.40},
    {"campaign": 2023, "harvest_date": pd.Timestamp("2023-09-11"), "yield_kg_ha": 1400.84, "alcohol_degree": 12.00},
    {"campaign": 2025, "harvest_date": pd.Timestamp("2025-08-23"), "yield_kg_ha": 2623.23, "alcohol_degree": 12.23},
])

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
cube_s1_mask = cube_s1.mask(geometry_wkt)

# ---------------------------------------------------------------------------
# Spatial means (raw)
# ---------------------------------------------------------------------------

ndvi_full   = cube_s2.ds.NDVI.mean(["x", "y"])
ndvi_masked = cube_s2_mask.ds.NDVI.mean(["x", "y"])
times_s2    = pd.DatetimeIndex(cube_s2.times)

rvi_full    = cube_s1.ds.RVI.mean(["x", "y"])
rvi_masked  = cube_s1_mask.ds.RVI.mean(["x", "y"])
times_s1    = pd.DatetimeIndex(cube_s1.times)

# ---------------------------------------------------------------------------
# Smoothed series (resample → rolling)
# ---------------------------------------------------------------------------

resample_period = "2W"
window = 4

def smooth(da):
    return da.resample(time=resample_period).median().rolling(time=window, min_periods=1).mean()

ndvi_full_s   = smooth(ndvi_full)
ndvi_masked_s = smooth(ndvi_masked)
rvi_full_s    = smooth(rvi_full)
rvi_masked_s  = smooth(rvi_masked)

# ---------------------------------------------------------------------------
# Snapshots — pick a mid-summer date
# ---------------------------------------------------------------------------

t_s2 = int(np.argmin(np.abs(times_s2 - pd.Timestamp("2021-07-26"))))
t_s1 = int(np.argmin(np.abs(times_s1 - pd.Timestamp("2021-07-26"))))

rgb_img  = _make_s2_rgb(
    cube_s2.ds["B04"].isel(time=t_s2).values,
    cube_s2.ds["B03"].isel(time=t_s2).values,
    cube_s2.ds["B02"].isel(time=t_s2).values,
)
ndvi_img = cube_s2.ds.NDVI.isel(time=t_s2).values
sar_img  = SARRGBComposite().render(cube_s1, t_s1)

# Polygon pixel coords — computed separately per cube (different grids)
poly_coords = np.array(geometry.exterior.coords)

x_s2 = cube_s2.ds.coords["x"].values
y_s2 = cube_s2.ds.coords["y"].values
px_s2 = np.interp(poly_coords[:, 0], x_s2, np.arange(len(x_s2)))
py_s2 = np.interp(poly_coords[:, 1], y_s2[::-1], np.arange(len(y_s2))[::-1])

x_s1 = cube_s1.ds.coords["x"].values
y_s1 = cube_s1.ds.coords["y"].values
px_s1 = np.interp(poly_coords[:, 0], x_s1, np.arange(len(x_s1)))
py_s1 = np.interp(poly_coords[:, 1], y_s1[::-1], np.arange(len(y_s1))[::-1])

def _fmt_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_tick_params(rotation=45)

def _add_harvest_lines(ax, times):
    """Draw a vertical dashed line for each harvest date that falls within the axis time range."""
    t_min, t_max = pd.Timestamp(times.min()), pd.Timestamp(times.max())
    for _, row in harvest.iterrows():
        if t_min <= row.harvest_date <= t_max:
            ax.axvline(row.harvest_date, color="red", lw=1.2, ls="--", alpha=0.7)
            ax.text(row.harvest_date, ax.get_ylim()[1], f" {row.campaign}",
                    color="red", fontsize=7, va="top", rotation=90)

def _patch(px, py):
    return MplPolygon(np.column_stack([px, py]), closed=True)

# ---------------------------------------------------------------------------
# Figure: 4 rows, 6-col GridSpec
#   Row 1: 3 images       (each spans 2 cols)
#   Rows 2-3: 2 plots     (each spans 3 cols)
#   Row 4: yield | alcohol (each spans 3 cols)
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 14))
gs  = fig.add_gridspec(4, 6, hspace=0.45, wspace=0.35)

ax_rgb      = fig.add_subplot(gs[0, 0:2])
ax_ndvi_img = fig.add_subplot(gs[0, 2:4])
ax_sar      = fig.add_subplot(gs[0, 4:6])

ax_ndvi_raw = fig.add_subplot(gs[1, 0:3])
ax_rvi_raw  = fig.add_subplot(gs[1, 3:6])

ax_ndvi_smo = fig.add_subplot(gs[2, 0:3])
ax_rvi_smo  = fig.add_subplot(gs[2, 3:6])

ax_yield    = fig.add_subplot(gs[3, 0:3])
ax_alcohol  = fig.add_subplot(gs[3, 3:6])

# --- Row 1: RGB | NDVI map | SAR RGB ---

ax_rgb.imshow(rgb_img)
ax_rgb.add_collection(PatchCollection(
    [_patch(px_s2, py_s2)], facecolor="white", edgecolor="yellow", linewidth=2, alpha=0.25,
))
ax_rgb.set_title(f"S2 RGB · {times_s2[t_s2].date()}")
ax_rgb.axis("off")

im = ax_ndvi_img.imshow(ndvi_img, cmap="RdYlGn", vmin=-1, vmax=1)
ax_ndvi_img.add_collection(PatchCollection(
    [_patch(px_s2, py_s2)], facecolor="none", edgecolor="yellow", linewidth=2,
))
fig.colorbar(im, ax=ax_ndvi_img, fraction=0.046, pad=0.04)
ax_ndvi_img.set_title(f"NDVI · {times_s2[t_s2].date()}")
ax_ndvi_img.axis("off")

ax_sar.imshow(sar_img)
ax_sar.add_collection(PatchCollection(
    [_patch(px_s1, py_s1)], facecolor="white", edgecolor="yellow", linewidth=2, alpha=0.25,
))
ax_sar.set_title(f"S1 SAR RGB · {times_s1[t_s1].date()}")
ax_sar.axis("off")

# --- Row 2: raw NDVI | raw RVI ---

ax_ndvi_raw.plot(times_s2, ndvi_full.values,   "g-o",  ms=3, lw=1., label="Full extent")
ax_ndvi_raw.plot(times_s2, ndvi_masked.values, "b--^", ms=3, lw=1.2, label="Parcel Area")
ax_ndvi_raw.set_title("NDVI — raw")
ax_ndvi_raw.set_ylabel("Spatial mean")
_fmt_time_axis(ax_ndvi_raw)
ax_ndvi_raw.grid(True, alpha=0.3)
ax_ndvi_raw.legend()
_add_harvest_lines(ax_ndvi_raw, times_s2)

ax_rvi_raw.plot(times_s1, rvi_full.values,   "g-o",  ms=3, lw=1., label="Full extent")
ax_rvi_raw.plot(times_s1, rvi_masked.values, "b--^", ms=3, lw=1.2, label="Parcel Area")
ax_rvi_raw.set_title("RVI (S1) — raw")
ax_rvi_raw.set_ylabel("Spatial mean")
_fmt_time_axis(ax_rvi_raw)
ax_rvi_raw.grid(True, alpha=0.3)
ax_rvi_raw.legend()
_add_harvest_lines(ax_rvi_raw, times_s1)

# --- Row 3: smoothed NDVI | smoothed RVI ---

ax_ndvi_smo.plot(ndvi_full_s.time.values,   ndvi_full_s.values,   "g--", lw=1., label="Full extent")
ax_ndvi_smo.plot(ndvi_masked_s.time.values, ndvi_masked_s.values, "b-", lw=1.5, label="Parcel Area")
ax_ndvi_smo.set_title(f"NDVI — smoothed ({resample_period} resample, {window}-step rolling)")
ax_ndvi_smo.set_ylabel("Spatial mean")
_fmt_time_axis(ax_ndvi_smo)
ax_ndvi_smo.grid(True, alpha=0.3)
ax_ndvi_smo.legend()
_add_harvest_lines(ax_ndvi_smo, ndvi_full_s.time.values)

ax_rvi_smo.plot(rvi_full_s.time.values,   rvi_full_s.values,   "g--", lw=1., label="Full extent")
ax_rvi_smo.plot(rvi_masked_s.time.values, rvi_masked_s.values, "b-", lw=1.5, label="Parcel Area")
ax_rvi_smo.set_title(f"RVI (S1) — smoothed ({resample_period} resample, {window}-step rolling)")
ax_rvi_smo.set_ylabel("Spatial mean")
_fmt_time_axis(ax_rvi_smo)
ax_rvi_smo.grid(True, alpha=0.3)
ax_rvi_smo.legend()
_add_harvest_lines(ax_rvi_smo, rvi_full_s.time.values)

# --- Row 4: yield | alcohol degree per campaign ---

bar_color = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db"]

ax_yield.bar(harvest["campaign"].astype(str), harvest["yield_kg_ha"], color=bar_color, edgecolor="black", linewidth=0.6)
for _, row in harvest.iterrows():
    ax_yield.text(str(row.campaign), row.yield_kg_ha + 80, f"{row.yield_kg_ha:,.0f}",
                  ha="center", va="bottom", fontsize=8)
ax_yield.set_title("Yield per campaign")
ax_yield.set_xlabel("Campaign (year)")
ax_yield.set_ylabel("Yield (kg/ha)")
ax_yield.grid(axis="y", alpha=0.3)

ax_alcohol.bar(harvest["campaign"].astype(str), harvest["alcohol_degree"], color=bar_color, edgecolor="black", linewidth=0.6)
for _, row in harvest.iterrows():
    ax_alcohol.text(str(row.campaign), row.alcohol_degree + 0.05, f"{row.alcohol_degree:.2f}°",
                    ha="center", va="bottom", fontsize=8)
ax_alcohol.set_title("Alcohol degree per campaign")
ax_alcohol.set_xlabel("Campaign (year)")
ax_alcohol.set_ylabel("Alcohol (°)")
ax_alcohol.set_ylim(bottom=10)
ax_alcohol.grid(axis="y", alpha=0.3)

fig.suptitle(f"Parcel {PARCEL} — vegetation indices & harvest data", fontsize=14, fontweight="bold")

save_path = IMAGES / "mask_ndvi_rgb.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Saved: {save_path}")
plt.show()
