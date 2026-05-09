"""Static parcel feature encoder."""

from __future__ import annotations

import pandas as pd


class StaticFeaturesProcessor:
    """Encode static parcel columns into numeric features.

    Transformations:
    - Columns in ``one_hot_columns``: binary ``{col}_{value}`` columns.
      String columns keep the original; numeric columns have the original
      dropped (it would otherwise carry an implicit ordinal meaning).
    - Remaining ``categorical_columns`` not in ``one_hot_columns``:
      label-encoded to ``{col}_enc`` (float, NaN-safe).
    - ``planting_date`` → ``vine_age = year - planting_date`` (float).
    - ``rainfed``: bool/nullable cast to int (True→1, False/NaN→0).
    """

    name = "static_features"

    def __init__(
        self,
        categorical_columns: list[str] | None = None,
        one_hot_columns: list[str] | None = None,
        planting_date_column: str | None = "planting_date",
        rainfed_column: str | None = "rainfed",
    ) -> None:
        self.categorical_columns = categorical_columns or ["certification", "variety"]
        self.one_hot_columns = one_hot_columns or []
        self.planting_date_column = planting_date_column
        self.rainfed_column = rainfed_column

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.planting_date_column and self.planting_date_column in df.columns:
            df["vine_age"] = df["year"] - df[self.planting_date_column]
            df = df.drop(columns=[self.planting_date_column])

        # One-hot encoding — all categories retained (no drop_first) so that
        # each SHAP value maps to exactly one category.
        for col in self.one_hot_columns:
            if col not in df.columns:
                continue
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            dummies = pd.get_dummies(
                df[col].astype(str), prefix=col, prefix_sep="_", dtype=float
            )
            df = pd.concat([df, dummies], axis=1)
            if is_numeric:
                # Drop the original so it isn't also used as an ordinal feature.
                df = df.drop(columns=[col])

        # Label encoding for remaining categorical columns
        one_hot_set = set(self.one_hot_columns)
        for col in self.categorical_columns:
            if col not in df.columns or col in one_hot_set:
                continue
            codes = pd.Categorical(df[col]).codes.astype(float)
            codes[codes == -1] = float("nan")
            df[f"{col}_enc"] = codes

        if self.rainfed_column and self.rainfed_column in df.columns:
            df[self.rainfed_column] = (
                df[self.rainfed_column].map({True: 1, False: 0}).fillna(0).astype(int)
            )

        return df
