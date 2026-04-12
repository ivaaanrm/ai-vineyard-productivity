"""Static parcel feature encoder."""

from __future__ import annotations

import pandas as pd


class StaticFeaturesProcessor:
    """Encode static parcel columns into numeric features.

    Transformations:
    - ``certification``, ``variety`` (and any extra ``categorical_columns``):
      label-encoded to ``{col}_enc`` (float, NaN-safe); original dropped.
      NaN rows get NaN in the encoded column.
    - ``planting_date`` → ``vine_age = year - planting_date`` (float);
      original dropped.  NaN where planting_date is missing.
    - ``rainfed``: bool/nullable cast to int (True→1, False/NaN→0).
    """

    name = "static_features"

    def __init__(
        self,
        categorical_columns: list[str] | None = None,
        planting_date_column: str | None = "planting_date",
        rainfed_column: str | None = "rainfed",
    ) -> None:
        self.categorical_columns = categorical_columns or ["certification", "variety"]
        self.planting_date_column = planting_date_column
        self.rainfed_column = rainfed_column

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # vine_age first — original categorical strings still present for group imputation
        if self.planting_date_column and self.planting_date_column in df.columns:
            df["vine_age"] = df["year"] - df[self.planting_date_column]
            df = df.drop(columns=[self.planting_date_column])

            # if df["vine_age"].isna().any():
            #     group_cols = ["variety"]
            #     group_means = df.groupby(group_cols, dropna=True)["vine_age"].transform("mean")
            #     df["vine_age"] = df["vine_age"].fillna(group_means)
            # # fallback: any remaining nulls (whole variety unknown) → global mean
            # if df["vine_age"].isna().any():
            #     df["vine_age"] = df["vine_age"].fillna(df["vine_age"].mean())

        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            codes = pd.Categorical(df[col]).codes.astype(float)
            codes[codes == -1] = float("nan")  # keep NaN rather than -1 sentinel
            df[f"{col}_enc"] = codes
            # original string column is kept — it's non-numeric so it won't be
            # picked up as a feature, but remains available for grouping/analysis

        if self.rainfed_column and self.rainfed_column in df.columns:
            df[self.rainfed_column] = df[self.rainfed_column].map({True: 1, False: 0}).fillna(0).astype(int)

        return df
