"""
Meta-Feature Extractor for Tabular Datasets.
Extracts normalized structural and statistical meta-features to form a DatasetFingerprint.
"""
from __future__ import annotations

import math
from typing import Any
import pandas as pd

from automl.domain.datasets.profile import Dataset, DatasetProfile
from automl.domain.meta_learning.fingerprint import DatasetFingerprint


def extract_fingerprint(
    dataset: Dataset,
    profile: DatasetProfile,
    df: pd.DataFrame | None = None,
) -> DatasetFingerprint:
    """Extracts a structured DatasetFingerprint from a Dataset and its DatasetProfile."""
    row_count = profile.row_count
    valid_features = [
        c for c in profile.columns
        if c.name != dataset.target_column and not c.is_identifier
    ]
    feature_count = len(valid_features)

    num_cols = len([
        c for c in valid_features
        if str(c.dtype).lower().startswith(("int", "float")) or c.mean is not None
    ])
    cat_cols = max(0, feature_count - num_cols)

    total_cells = max(1, row_count * feature_count)
    missing_cells = sum(c.null_count for c in valid_features)
    missing_ratio = round(missing_cells / total_cells, 4)

    target_entropy = 0.5  # default balanced prior
    if df is not None and dataset.target_column in df.columns:
        target_s = df[dataset.target_column].dropna()
        if len(target_s) > 0:
            if dataset.task_type in {"binary_classification", "multiclass_classification"}:
                counts = target_s.value_counts(normalize=True).values
                n_classes = len(counts)
                if n_classes > 1:
                    raw_ent = -sum(float(p) * math.log2(float(p)) for p in counts if p > 0)
                    max_ent = math.log2(n_classes)
                    target_entropy = round(raw_ent / max_ent, 3) if max_ent > 0 else 0.0
                else:
                    target_entropy = 0.0
            else:
                # Regression: target variance / range ratio normalized
                std = float(target_s.std())
                mean = abs(float(target_s.mean()))
                if mean > 1e-6:
                    cv = std / mean
                    target_entropy = min(1.0, round(cv / 2.0, 3))
                else:
                    target_entropy = 0.5

    return DatasetFingerprint(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        task_type=dataset.task_type or "binary_classification",
        row_count=row_count,
        feature_count=feature_count,
        numerical_feature_count=num_cols,
        categorical_feature_count=cat_cols,
        missing_cells_ratio=missing_ratio,
        target_entropy=target_entropy,
    )
