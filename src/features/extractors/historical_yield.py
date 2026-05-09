"""Historical yield features processor.

Operates on the post-merge DataFrame (one row per parcel_id × year) and adds
cross-year expanding statistics derived from ``yield_T_ha``.

Features produced:
    historical_parcel_yield      — expanding mean of previous campaigns (shift 1)
    historical_parcel_yield_std  — expanding std of previous campaigns (shift 1)
    n_campaigns_observed         — number of previous campaigns available

NaN imputation (first campaign of a parcel):
    historical_parcel_yield     → variety group mean computed on train rows only
    historical_parcel_yield_std → median std across train parcels with ≥2 campaigns
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class HistoricalFeaturesProcessor:
    """Add expanding historical yield features to a (parcel_id × year) DataFrame.

    Parameters
    ----------
    target_column:
        Column containing the yield values (default: ``yield_T_ha``).
    variety_column:
        Column used for NaN imputation (default: ``variety``).
    split_column:
        Column marking the temporal split; imputation statistics are computed
        exclusively from rows where ``split == train_label`` (default: ``split``).
    train_label:
        Value in ``split_column`` that identifies training rows (default: ``train``).
    """

    name = "historical_features"

    def __init__(
        self,
        target_column: str = "yield_T_ha",
        variety_column: str = "variety",
        split_column: str = "split",
        train_label: str = "train",
    ) -> None:
        self.target_column = target_column
        self.variety_column = variety_column
        self.split_column = split_column
        self.train_label = train_label

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with three new columns appended."""
        if self.target_column not in df.columns:
            return df

        df = df.copy()
        df = df.sort_values(["parcel_id", "year"]).reset_index(drop=True)

        grp = df.groupby("parcel_id")[self.target_column]

        # transform keeps results aligned to the original index AND shifts within
        # each group — avoids cross-group leakage from a flat shift on MultiIndex.
        df["historical_parcel_yield"] = grp.transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df["historical_parcel_yield_std"] = grp.transform(
            lambda x: x.expanding().std().shift(1)
        )
        df["n_campaigns_observed"] = (
            grp.transform(lambda x: x.expanding().count().shift(1))
            .fillna(0)
            .astype(int)
        )

        df = self._impute_nans(df)
        return df

    # ── internals ────────────────────────────────────────────────────────

    def _impute_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaNs that arise for a parcel's first observed campaign."""
        train_mask = df.get(self.split_column, pd.Series(dtype=str)) == self.train_label
        train_rows = df[train_mask]

        # variety mean from train rows only
        if self.variety_column in df.columns and not train_rows.empty:
            variety_mean = (
                train_rows.groupby(self.variety_column)[self.target_column].mean()
            )
            global_mean = train_rows[self.target_column].mean()

            def _fill_yield(row: pd.Series) -> float:
                if not np.isnan(row["historical_parcel_yield"]):
                    return row["historical_parcel_yield"]
                variety = row.get(self.variety_column)
                return variety_mean.get(variety, global_mean)

            nan_mask = df["historical_parcel_yield"].isna()
            if nan_mask.any():
                df.loc[nan_mask, "historical_parcel_yield"] = (
                    df[nan_mask].apply(_fill_yield, axis=1)
                )
        else:
            global_mean = df[self.target_column].mean()
            df["historical_parcel_yield"] = df["historical_parcel_yield"].fillna(global_mean)

        # std: fill with median std of train parcels that have ≥2 campaigns
        nan_std_mask = df["historical_parcel_yield_std"].isna()
        if nan_std_mask.any():
            train_std_vals = train_rows["historical_parcel_yield_std"].dropna()
            fallback_std = float(train_std_vals.median()) if not train_std_vals.empty else 0.0
            df.loc[nan_std_mask, "historical_parcel_yield_std"] = fallback_std

        return df
