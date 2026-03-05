"""
Agrupa los recintos de full_aoi.csv segun las agrupaciones
id_lugar_raw definidas en vinedos_modelo_limpio.csv, une sus
geometrias y genera aoi_processed.csv.
"""

from pathlib import Path

import pandas as pd
from shapely import wkt
from shapely.ops import unary_union

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
AOI_CSV = DATA_DIR / "full_aoi.csv"
MODELO_CSV = DATA_DIR / "vinedos_modelo_limpio.csv"
OUTPUT_CSV = DATA_DIR / "aoi_processed.csv"


def main() -> None:
    aoi = pd.read_csv(AOI_CSV)
    modelo = pd.read_csv(MODELO_CSV)
    print(f"Loaded {len(aoi)} rows from {AOI_CSV.name}")

    # Build lookup: ID_Lugar -> row in aoi
    aoi_lookup = aoi.set_index("ID_Lugar")

    # Get unique id_lugar_raw groups from modelo
    unique_groups = modelo["id_lugar_raw"].dropna().unique()
    print(f"Found {len(unique_groups)} unique id_lugar_raw groups in {MODELO_CSV.name}")

    rows = []
    skipped = 0
    for raw in unique_groups:
        ids = [id_.strip() for id_ in str(raw).split(",")]
        # Filter to IDs that exist in aoi
        matched = [id_ for id_ in ids if id_ in aoi_lookup.index]
        if not matched:
            skipped += 1
            continue

        subset = aoi_lookup.loc[matched]
        geoms = [wkt.loads(g) for g in subset["Geometry"]]
        merged_geom = unary_union(geoms)

        # Take SIGPAC keys from the first matched row
        first = subset.iloc[0]
        rows.append(
            {
                "id_lugar_raw": raw,
                "Provincia": first["Provincia"],
                "Municipio": first["Municipio"],
                "Agregado": first["Agregado"],
                "Zona": first["Zona"],
                "Poligono": first["Poligono"],
                "Parcela": first["Parcela"],
                "n_recintos": len(matched),
                "Geometry": merged_geom.wkt,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(result)} groups to {OUTPUT_CSV.name}")
    if skipped:
        print(f"Skipped {skipped} groups (no matching IDs in aoi)")


if __name__ == "__main__":
    main()
