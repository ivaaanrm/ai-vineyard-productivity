"""
Lee aoi_processed.csv y genera aoi_bbox.csv con una sola fila
que contiene el bounding box rectangular de todos los poligonos.
"""

from pathlib import Path

import pandas as pd
from shapely import wkt
from shapely.geometry import box
from shapely.ops import unary_union

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
INPUT_CSV = DATA_DIR / "aoi_processed.csv"
OUTPUT_CSV = DATA_DIR / "aoi_bbox.csv"


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV.name}")

    geoms = [wkt.loads(g) for g in df["Geometry"]]
    merged = unary_union(geoms)
    bbox = box(*merged.bounds)

    minx, miny, maxx, maxy = merged.bounds
    print(f"Bounding box: minx={minx:.2f}, miny={miny:.2f}, maxx={maxx:.2f}, maxy={maxy:.2f}")

    result = pd.DataFrame([{"Geometry": bbox.wkt}])
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV.name}")


if __name__ == "__main__":
    main()
