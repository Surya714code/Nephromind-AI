"""Leakage-safe sklearn preprocessing definitions."""
from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(config: dict[str, Any]) -> ColumnTransformer:
    """Return an unfitted transformer; fitting occurs inside each model pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config["numeric_features"]),
            ("categorical", categorical_pipeline, config["categorical_features"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
