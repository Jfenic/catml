"""Immutable execution specification built before train / optimize / export."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from automl.domain.experiments.validation import require_known_validation_strategy
from automl.domain.metrics.spec import MetricSpec, resolve_metric_spec


@dataclass(frozen=True)
class ExecutionSpec:
    """Validated, immutable inputs for a single trial execution path."""

    dataset_id: str
    target: str
    feature_names: tuple[str, ...]
    model_id: str
    metric: str
    validation_strategy: str
    cv_folds: int
    random_seed: int
    test_size: float
    task_type: str
    dataset_content_hash: str | None = None
    group_column: str | None = None
    column_transforms: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    derived_features: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    metric_spec: MetricSpec | None = None
    params: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def resolved_metric_spec(self) -> MetricSpec:
        if self.metric_spec is not None:
            return self.metric_spec
        return resolve_metric_spec(self.metric)


def build_execution_spec(
    run: Any,
    experiment: Any,
    dataset: Any,
    *,
    model_id: str,
    params: dict[str, Any] | None = None,
    greater_is_better: bool | None = None,
    dataset_content_hash: str | None = None,
) -> ExecutionSpec:
    """Pure factory: normalize CV + resolve metric; does not mutate run/experiment."""
    metric_name = getattr(experiment, "metric", None) or getattr(run.config, "metric", None)
    if not metric_name:
        raise ValueError("Cannot build ExecutionSpec without a metric on experiment or run")

    raw_strategy = getattr(experiment, "validation_strategy", None) or getattr(
        run.config, "validation_strategy", None
    )
    strategy = require_known_validation_strategy(raw_strategy, default="holdout")

    raw_metric = str(metric_name)
    metric_spec = resolve_metric_spec(raw_metric, greater_is_better=greater_is_better)

    feature_names = tuple(getattr(experiment, "feature_names", None) or ())
    transforms = tuple(getattr(experiment, "column_transforms", None) or ())
    derived = tuple(getattr(experiment, "derived_features", None) or ())
    param_items = tuple(sorted((params or {}).items(), key=lambda kv: kv[0]))

    ds_id = getattr(dataset, "id", None) or getattr(run, "dataset_id", "")
    target = getattr(dataset, "target_column", None) or getattr(run.config, "target", "")
    task_type = (
        getattr(dataset, "task_type", None)
        or getattr(run.config, "task_type", None)
        or ""
    )

    if dataset_content_hash is None and dataset is not None:
        dataset_content_hash = getattr(dataset, "content_hash", None)
        if not dataset_content_hash and hasattr(dataset, "path") and dataset.path:
            from automl.domain.datasets.content_hash import try_compute_file_content_hash

            dataset_content_hash = try_compute_file_content_hash(dataset.path)

    return ExecutionSpec(
        dataset_id=str(ds_id),
        dataset_content_hash=dataset_content_hash,
        target=str(target),
        feature_names=feature_names,
        model_id=str(model_id),
        metric=raw_metric,
        metric_spec=metric_spec,
        validation_strategy=strategy,
        cv_folds=int(getattr(run.config, "cv_folds", 5) or 5),
        random_seed=int(getattr(run.config, "random_seed", 42)),
        test_size=float(getattr(run.config, "test_size", 0.2)),
        group_column=getattr(experiment, "group_column", None),
        column_transforms=transforms,
        derived_features=derived,
        task_type=str(task_type),
        params=param_items,
    )


def trial_execution_from_spec(
    *,
    trial: Any,
    experiment: Any,
    run: Any,
    dataset: Any,
    spec: ExecutionSpec,
) -> Any:
    """Build a ``TrialExecution`` DTO from a validated ``ExecutionSpec``."""
    from automl.domain.ports import TrialExecution

    return TrialExecution(
        trial=trial,
        experiment=experiment,
        run=run,
        feature_names=list(spec.feature_names),
        dataset_path=dataset.path,
        target_column=spec.target,
        task_type=spec.task_type,
        metric=spec.metric,
        validation_strategy=spec.validation_strategy,
        test_size=spec.test_size,
        cv_folds=spec.cv_folds,
        random_seed=spec.random_seed,
        group_column=spec.group_column,
        column_transforms=[dict(t) for t in spec.column_transforms],
        derived_features=[dict(d) for d in spec.derived_features],
        dataset_content_hash=spec.dataset_content_hash,
    )
