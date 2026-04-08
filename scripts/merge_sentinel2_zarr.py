"""
Merge Sentinel-2 cube.zarr files from two sources into a single zarr per parcel.

Source A (original bands): agrixel-fair-dockerV2/data/output/files/SENTINEL-2/
    Bands: B02, B03, B04, B08, B11, B12

Source B (additional bands): agrixel-fair-dockerV2/data/output/files-sentinel-restantes/SENTINEL-2/
    Bands: B05, B06, B07

Output: agrixel-fair-dockerV2/data/output/files-merged/SENTINEL-2/<parcel>/cube.zarr
    Bands: all of the above, outer join on time (NaN where a band has no observation)

Parcels present only in source A are copied as-is (no new bands available).
Parcels present only in source B are skipped (shouldn't happen, but guarded).
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1] / "agrixel-fair-dockerV2" / "data" / "output"
SRC_A = ROOT / "files" / "SENTINEL-2"
SRC_B = ROOT / "files-sentinel-restantes" / "SENTINEL-2"
DST = ROOT / "files-merged" / "SENTINEL-2"

ZARR_OPEN_KWARGS = {"consolidated": False}


def open_cube(path: Path) -> xr.Dataset:
    """Open a cube.zarr and split the stacked `data` variable into per-band DataArrays."""
    ds = xr.open_zarr(path, **ZARR_OPEN_KWARGS)
    # Deduplicate time: some parcels were downloaded from overlapping tiles,
    # producing identical timestamps. Keep the first occurrence.
    _, first_idx = np.unique(ds.time.values, return_index=True)
    if len(first_idx) < len(ds.time):
        n_dups = len(ds.time) - len(first_idx)
        log.debug("%s: dropping %d duplicate time steps", path.parent.name, n_dups)
        ds = ds.isel(time=first_idx)
    # ds has: data(time, variable, y, x) where variable is a string coordinate
    # Unstack variable → one DataArray per band
    bands = {}
    for band in ds.variable.values:
        band_name = band.decode() if isinstance(band, bytes) else str(band)
        bands[band_name] = ds["data"].sel(variable=band).drop_vars("variable")
    return xr.Dataset(bands, attrs=ds.attrs)


def merge_cubes(ds_a: xr.Dataset, ds_b: xr.Dataset) -> xr.Dataset:
    """Merge two per-band datasets with an outer join on the time dimension."""
    merged = xr.merge([ds_a, ds_b], join="outer")
    return merged


def restack(ds: xr.Dataset, attrs: dict) -> xr.Dataset:
    """Re-stack per-band DataArrays back into data(time, variable, y, x)."""
    band_names = sorted(ds.data_vars)
    arrays = [ds[b].assign_coords(variable=b) for b in band_names]
    data = xr.concat(arrays, dim="variable").transpose("time", "variable", "y", "x")
    # Rebuild attrs
    new_attrs = {k: v for k, v in attrs.items() if k not in ("variables", "time_iso")}
    new_attrs["variables"] = band_names
    new_attrs["time_iso"] = [
        str(t)[:19] + "Z"
        for t in data.time.values.astype("datetime64[s]").astype(str)
    ]
    return xr.Dataset({"data": data}, attrs=new_attrs)


def write_zarr(ds: xr.Dataset, out_path: Path) -> None:
    """Write dataset to zarr v2 format (matching source files), removing destination first."""
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit encoding to avoid _FillValue=None errors on string/object coords
    encoding = {}
    for name, var in ds.data_vars.items():
        if np.issubdtype(var.dtype, np.floating):
            encoding[name] = {"dtype": "float32", "_FillValue": float("nan")}
    for name, coord in ds.coords.items():
        if coord.dtype == object:
            encoding[name] = {"dtype": str, "_FillValue": ""}
    # zarr_format=2 matches the source files (created with zarr v2)
    ds.to_zarr(str(out_path), mode="w", encoding=encoding, zarr_format=2)
    zarr.consolidate_metadata(str(out_path))


def process_parcel(parcel: str, dry_run: bool = False) -> str:
    """
    Merge or copy one parcel. Returns a status string.

    Returns:
        "merged"  – both sources found, bands merged
        "copied"  – only source A found, copied as-is
        "skipped" – source A missing (shouldn't happen)
        "error"   – exception during processing
    """
    src_a_zarr = SRC_A / parcel / "cube.zarr"
    src_b_zarr = SRC_B / parcel / "cube.zarr"
    dst_zarr = DST / parcel / "cube.zarr"

    has_a = src_a_zarr.exists()
    has_b = src_b_zarr.exists()

    if not has_a:
        log.warning("Parcel %s: source A missing — skipped", parcel)
        return "skipped"

    if dry_run:
        action = "merge" if has_b else "copy"
        log.info("DRY-RUN parcel %s → %s", parcel, action)
        return action

    try:
        if has_b:
            ds_a = open_cube(src_a_zarr)
            ds_b = open_cube(src_b_zarr)
            merged = merge_cubes(ds_a, ds_b)
            # Prefer attrs from source A, update variables list after merge
            combined_attrs = {**dict(ds_a.attrs)}
            out_ds = restack(merged, combined_attrs)
            write_zarr(out_ds, dst_zarr)
            log.info("Parcel %s: merged %s bands, %d time steps", parcel,
                     sorted(merged.data_vars), len(merged.time))
            return "merged"
        else:
            # No new bands for this parcel — copy source A verbatim
            if dst_zarr.exists():
                shutil.rmtree(dst_zarr)
            dst_zarr.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_a_zarr, dst_zarr)
            log.info("Parcel %s: copied (no new bands)", parcel)
            return "copied"

    except Exception:
        log.exception("Parcel %s: error during processing", parcel)
        return "error"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parcels",
        nargs="*",
        help="Specific parcel names to process. Defaults to all parcels in source A.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1).",
    )
    args = parser.parse_args()

    parcels = args.parcels or sorted(p.name for p in SRC_A.iterdir() if p.is_dir())
    log.info("Processing %d parcels → %s", len(parcels), DST)

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import functools

        fn = functools.partial(process_parcel, dry_run=args.dry_run)
        counts: dict[str, int] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fn, p): p for p in parcels}
            for fut in as_completed(futures):
                status = fut.result()
                counts[status] = counts.get(status, 0) + 1
    else:
        counts: dict[str, int] = {}
        for parcel in parcels:
            status = process_parcel(parcel, dry_run=args.dry_run)
            counts[status] = counts.get(status, 0) + 1

    log.info("Done. Summary: %s", counts)


if __name__ == "__main__":
    main()
