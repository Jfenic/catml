"""Bridge between Data Preparation pipeline-scoped steps and training column_transforms.

Pure domain module: no pandas, scikit-learn, or SQLite dependencies.
"""
from __future__ import annotations

from typing import Any

from automl.domain.preparation.models import (
    PreparationOperation,
    PreparationScope,
    PreparationStep,
)


def pipeline_steps_to_column_transforms(
    pipeline_steps: list[dict[str, Any]] | tuple[PreparationStep, ...] | list[PreparationStep],
) -> list[dict[str, Any]]:
    """
    Converts pipeline-scoped preparation steps into trainer-compatible column_transforms.

    Guarantees fold-safe execution: these transformations are deferred to cross-validation
    folds (or fit-time pipeline) and are never computed on the full dataset before split.
    """
    column_transforms: list[dict[str, Any]] = []

    for raw_step in pipeline_steps:
        if isinstance(raw_step, PreparationStep):
            step = raw_step
        elif isinstance(raw_step, dict):
            step = PreparationStep.from_dict(raw_step)
        else:
            continue

        if step.scope != PreparationScope.TRAINING_PIPELINE:
            continue

        op = step.operation
        cols = list(step.columns)
        params = dict(step.params or {})

        if op == PreparationOperation.MARK_IMPUTE:
            strategy = params.get("strategy") or "median"
            for col in cols:
                column_transforms.append(
                    {
                        "column": col,
                        "kind": "impute",
                        "strategy": strategy,
                        "params": {"strategy": strategy, **params},
                    }
                )
        elif op == PreparationOperation.MARK_SCALE:
            scaler = params.get("scaler") or params.get("method") or "standard_scaler"
            for col in cols:
                column_transforms.append(
                    {
                        "column": col,
                        "kind": scaler,
                        "params": params,
                    }
                )

    return column_transforms
