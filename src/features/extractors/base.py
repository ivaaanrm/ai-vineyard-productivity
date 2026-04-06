"""Shared protocol and default phase definitions for all feature extractors."""

from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable

import pandas as pd

# ── Default Mediterranean vineyard phenology phases ──────────────────────────

DEFAULT_PHASES: Dict[str, List[int]] = {
    "dormancy": [12, 1, 2],
    "budbreak": [3, 4],
    "flowering": [5, 6],
    "veraison": [7, 8],     # envero
    "harvest": [8, 9],      # vendimia temprana y principal
    "postharvest": [10, 11],
}


# ── Protocol ─────────────────────────────────────────────────────────────────


@runtime_checkable
class FeatureExtractor(Protocol):
    """Contract for all feature extractors.

    Each extractor receives a DataFrame filtered to a single (parcel_id, year)
    with a ``month`` column (int 1–12) and value columns.  It returns a flat
    dict of ``{feature_name: value}``.
    """

    name: str

    def extract(self, group: pd.DataFrame, columns: List[str]) -> Dict[str, float]: ...
