from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from automl.domain.experiments.trial import TrialResult, TrialStatus
from automl.domain.experiments.validation import require_known_validation_strategy
from automl.domain.runs.states import RunStatus
from automl.domain.ports import TrialExecution, TrainerPort
from automl.engine.profiling.dataset_profiler import load_dataframe
from automl.plugins.models.sklearn_models import build_sklearn_model
from automl.artifacts.model_artifact import collect_environment_dependencies
from automl.domain.datasets.content_hash import (
    compute_config_hash,
    compute_fold_assignment_hash,
    try_compute_file_content_hash,
)


class SklearnTrainer(TrainerPort):
    def __init__(self, plugin_registry: Any = None) -> None:
        self.plugin_registry = plugin_registry

    def run(self, execution: TrialExecution) -> TrialResult:
        import time

        trial = execution.trial
        if execution.run and getattr(execution.run, "status", None) == RunStatus.CANCELLED:
            trial.status = TrialStatus.FAILED
            trial.error_message = "Execution cancelled."
            return TrialResult(
                trial_id=trial.id,
                experiment_id=execution.experiment.id,
                model_id=trial.model_id,
                primary_metric=execution.metric,
                primary_score=0.0,
                secondary_metrics={},
                training_time_seconds=0.0,
                failure_reason="Execution cancelled.",
            )

        trial.status = TrialStatus.RUNNING
        started = time.perf_counter()

        try:
            df = load_dataframe(execution.dataset_path)
            derived_features = getattr(execution, "derived_features", None) or []
            if derived_features:
                from automl.domain.features.derived_feature import (
                    DerivedFeatureDefinition,
                    DerivedFeatureType,
                )
                from automl.engine.features.generation.derived_feature_engine import (
                    DerivedFeatureEngine,
                    assert_derived_specs_safe_for_cross_validation,
                )

                # Fail closed: no arbitrary python_code and no full-frame aggregates before CV.
                assert_derived_specs_safe_for_cross_validation(list(derived_features))

                derived_engine = DerivedFeatureEngine()
                for df_spec in derived_features:
                    defn = DerivedFeatureDefinition.from_dict(df_spec) if isinstance(df_spec, dict) else df_spec
                    if defn.name not in df.columns:
                        df, eval_res = derived_engine.apply(
                            df,
                            defn,
                            allow_python_code=False,
                            allow_stateful_aggregates=False,
                        )
                        if not eval_res.is_valid:
                            raise ValueError(f"Failed to compute derived feature {defn.name}: {eval_res.error_message}")

            strategy_early = (execution.validation_strategy or "holdout").lower()
            if strategy_early in {"time_series", "time_series_split"}:
                time_column = None
                if execution.run and execution.run.config and isinstance(execution.run.config.extra, dict):
                    time_column = execution.run.config.extra.get("time_column")
                if not time_column or time_column not in df.columns:
                    raise ValueError(
                        "validation_strategy 'time_series' requires run.config.extra['time_column'] "
                        "naming a real column. Rows are sorted ascending by that column before TimeSeriesSplit; "
                        "duplicate timestamps should be resolved upstream."
                    )
                df = df.sort_values(time_column, kind="mergesort").reset_index(drop=True)
                dup_count = int(df[time_column].duplicated().sum())
                if dup_count > 0:
                    import warnings as _warnings
                    _warnings.warn(
                        f"time_column '{time_column}' has {dup_count} duplicate values; "
                        "TimeSeriesSplit order may be ambiguous.",
                        UserWarning,
                        stacklevel=2,
                    )

            X = df[execution.feature_names]

            if execution.task_type == "clustering":
                return self._run_clustering(execution, X, started)

            y = df[execution.target_column]
            from automl.engine.training.target_adapter import TargetAdapter

            adapter = TargetAdapter()
            y_adapted = adapter.fit_transform(y, execution.task_type)

            model_params = {
                k: v for k, v in (execution.trial.parameters or {}).items()
                if k not in ("text_columns", "image_columns", "image_model", "time_budget")
            }

            model_plugin = None
            used_fallback = False
            if self.plugin_registry and self.plugin_registry.has(execution.trial.model_id):
                model_plugin = self.plugin_registry.get_model_plugin(execution.trial.model_id)
                if model_plugin:
                    self.plugin_registry.validate_plugin_for_task(
                        execution.trial.model_id,
                        execution.task_type,
                    )
                    used_fallback = not bool(
                        getattr(model_plugin, "is_available", getattr(model_plugin, "is_native", True))
                    )
                    model = model_plugin.build_estimator(
                        parameters=model_params,
                        task_type=execution.task_type,
                    )
                else:
                    model = build_sklearn_model(
                        execution.trial.model_id,
                        execution.task_type,
                        parameters=model_params,
                    )
            else:
                model = build_sklearn_model(
                    execution.trial.model_id,
                    execution.task_type,
                    parameters=model_params,
                )
            text_cols = execution.trial.parameters.get("text_columns") if execution.trial.parameters else None
            if text_cols is None and execution.run and execution.run.config and execution.run.config.extra:
                text_cols = execution.run.config.extra.get("text_columns")

            image_cols = execution.trial.parameters.get("image_columns") if execution.trial.parameters else None
            if image_cols is None and execution.run and execution.run.config and execution.run.config.extra:
                image_cols = execution.run.config.extra.get("image_columns")

            image_model = execution.trial.parameters.get("image_model") if execution.trial.parameters else None
            if image_model is None and execution.run and execution.run.config and execution.run.config.extra:
                image_model = execution.run.config.extra.get("image_model")

            pipeline = _build_pipeline(
                X,
                model,
                text_columns=text_cols,
                image_columns=image_cols,
                image_model=image_model,
                column_transforms=getattr(execution, "column_transforms", None),
                task_type=getattr(execution, "task_type", None),
            )

            metric_name = execution.metric
            strategy = require_known_validation_strategy(
                execution.validation_strategy,
                default="holdout",
            )

            dataset_content_hash = execution.dataset_content_hash or try_compute_file_content_hash(execution.dataset_path)
            config_dict = {
                "model_id": trial.model_id,
                "parameters": trial.parameters or {},
                "feature_names": sorted(execution.feature_names),
                "target_column": execution.target_column,
                "task_type": execution.task_type,
                "metric": metric_name,
                "validation_strategy": strategy,
                "random_seed": execution.random_seed,
                "trial_seed": getattr(trial, "seed", None),
                "column_transforms": execution.column_transforms or [],
                "derived_features": execution.derived_features or [],
            }
            config_hash = compute_config_hash(config_dict)
            dep_versions = collect_environment_dependencies()

            if strategy in {
                "cross_validation",
                "kfold",
                "stratified_kfold",
                "time_series",
                "time_series_split",
                "group_kfold",
                "group_cv",
                "grouped_kfold",
            }:
                from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold
                n_splits = max(2, execution.cv_folds or 5)
                if len(X) >= 4:
                    if metric_name.lower() == "r2":
                        n_splits = min(n_splits, max(2, len(X) // 2))
                    else:
                        n_splits = min(n_splits, len(X))
                else:
                    n_splits = min(n_splits, max(2, len(X)))

                groups = None
                if strategy in {"group_kfold", "group_cv", "grouped_kfold"}:
                    if not execution.group_column or execution.group_column not in df.columns:
                        raise ValueError(f"group_column '{execution.group_column}' required for {strategy} validation")
                    groups = df[execution.group_column].values
                    n_unique_groups = len(set(groups))
                    n_splits = min(n_splits, n_unique_groups)
                    cv_splitter = GroupKFold(n_splits=n_splits)
                elif strategy in {"time_series", "time_series_split"}:
                    cv_splitter = TimeSeriesSplit(n_splits=n_splits)
                elif strategy == "stratified_kfold" and execution.task_type != "regression":
                    min_class_count = int(pd.Series(y_adapted).value_counts().min()) if len(y_adapted) > 0 else 0
                    if min_class_count < 2:
                        cv_splitter = KFold(n_splits=min(n_splits, max(2, len(X))), shuffle=True, random_state=execution.random_seed)
                    else:
                        cv_splitter = StratifiedKFold(n_splits=min(n_splits, min_class_count), shuffle=True, random_state=execution.random_seed)
                else:
                    cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=execution.random_seed)

                fold_assignment = np.full(len(X), -1, dtype=int)
                scoring = _sklearn_scoring(metric_name, execution.task_type, self.plugin_registry)
                fold_scores, pruned_at = _cross_val_scores_with_callback(
                    pipeline,
                    X,
                    y_adapted,
                    cv_splitter=cv_splitter,
                    groups=groups,
                    scoring=scoring,
                    metric_name=metric_name,
                    plugin_registry=self.plugin_registry,
                    fold_callback=getattr(execution, "fold_callback", None),
                    fold_assignment=fold_assignment,
                )
                cv_folds = len(fold_scores)
                cv_fold_ids = [f"fold_{i}" for i in range(len(fold_scores))]
                cv_fold_assignment_hash = compute_fold_assignment_hash(fold_assignment)

                if pruned_at is not None:
                    elapsed = time.perf_counter() - started
                    trial.status = TrialStatus.FAILED
                    partial = float(np.mean(fold_scores)) if fold_scores else 0.0
                    return TrialResult(
                        trial_id=trial.id,
                        experiment_id=execution.experiment.id,
                        model_id=trial.model_id,
                        primary_metric=metric_name,
                        primary_score=partial,
                        secondary_metrics={
                            "cv_scores": [float(s) for s in fold_scores],
                            "pruned_at_fold": int(pruned_at),
                            "validation_strategy": strategy,
                            "pruned": True,
                        },
                        training_time_seconds=elapsed,
                        failure_reason="pruned",
                        artifacts={
                            "seed": getattr(trial, "seed", None) or execution.random_seed,
                            "validation_strategy": strategy,
                            "cv_folds": len(fold_scores),
                            "cv_fold_ids": cv_fold_ids,
                            "cv_fold_assignment_hash": cv_fold_assignment_hash,
                            "dataset_content_hash": dataset_content_hash,
                            "config_hash": config_hash,
                            "library_versions": dep_versions,
                        },
                    )

                scores = np.asarray(fold_scores, dtype=float)
                primary_score = float(np.mean(scores))
                if np.isnan(primary_score):
                    raise ValueError(f"Cross-validation returned NaN scores: {scores}")
                secondary = {
                    "cv_std": float(np.std(scores)),
                    "cv_scores": [float(s) for s in scores],
                    "validation_strategy": strategy,
                }
                if groups is not None:
                    secondary["group_column"] = execution.group_column
                    secondary["n_groups"] = len(set(groups))
                pipeline.fit(X, y_adapted)
            else:
                cv_folds = 1
                cv_fold_ids = ["holdout_split"]
                if execution.group_column and execution.group_column in df.columns:
                    from sklearn.model_selection import GroupShuffleSplit
                    gss = GroupShuffleSplit(n_splits=1, test_size=execution.test_size, random_state=execution.random_seed)
                    train_idx, test_idx = next(gss.split(X, y_adapted, groups=df[execution.group_column].values))
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train = y_adapted.iloc[train_idx] if hasattr(y_adapted, "iloc") else y_adapted[train_idx]
                    y_test = y_adapted.iloc[test_idx] if hasattr(y_adapted, "iloc") else y_adapted[test_idx]
                else:
                    stratify_target = y_adapted if execution.task_type != "regression" else None
                    effective_test_size = execution.test_size
                    if stratify_target is not None:
                        n_samples = len(y_adapted)
                        n_classes = len(np.unique(y_adapted))
                        if n_classes >= 2 and n_samples >= 2 * n_classes:
                            computed_test = int(n_samples * execution.test_size) if execution.test_size < 1.0 else int(execution.test_size)
                            if computed_test < n_classes:
                                effective_test_size = n_classes

                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y_adapted,
                            test_size=effective_test_size,
                            random_state=execution.random_seed,
                            stratify=stratify_target,
                        )
                    except ValueError:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y_adapted,
                            test_size=execution.test_size,
                            random_state=execution.random_seed,
                            stratify=None,
                        )

                holdout_assignment = np.zeros(len(X), dtype=int)
                if hasattr(X_test, "index"):
                    holdout_assignment[X.index.isin(X_test.index)] = 1
                cv_fold_assignment_hash = compute_fold_assignment_hash(holdout_assignment)

                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)

                if self.plugin_registry and self.plugin_registry.has(metric_name):
                    metric_plugin = self.plugin_registry.get_metric_plugin(metric_name)
                    if metric_plugin:
                        prob = None
                        if hasattr(pipeline, "predict_proba"):
                            try:
                                prob = pipeline.predict_proba(X_test)
                            except Exception:
                                prob = None
                        primary_score = float(metric_plugin.compute(y_test, predictions, prob))
                        secondary = {metric_name: primary_score}
                    else:
                        secondary = _compute_metrics(
                            y_test,
                            predictions,
                            pipeline,
                            X_test,
                            execution.task_type,
                        )
                        if metric_name not in secondary:
                            raise ValueError(f"Metric '{metric_name}' is not supported for task '{execution.task_type}'.")
                        primary_score = secondary[metric_name]
                else:
                    secondary = _compute_metrics(
                        y_test,
                        predictions,
                        pipeline,
                        X_test,
                        execution.task_type,
                    )
                    if metric_name not in secondary:
                        raise ValueError(f"Metric '{metric_name}' is not supported for task '{execution.task_type}'.")
                    primary_score = secondary[metric_name]

                if np.isnan(primary_score):
                    raise ValueError(f"Evaluation returned NaN score for metric '{metric_name}'")

            elapsed = time.perf_counter() - started
            trial.status = TrialStatus.COMPLETED
            artifacts = {
                "estimator_class": type(model).__name__,
                "estimator_module": type(model).__module__,
                "backend": "fallback" if used_fallback else "native",
                "backend_class": type(model).__name__,
                "backend_module": type(model).__module__,
                "seed": getattr(trial, "seed", None) or execution.random_seed,
                "validation_strategy": strategy,
                "cv_folds": cv_folds,
                "cv_fold_ids": cv_fold_ids,
                "cv_fold_assignment_hash": cv_fold_assignment_hash,
                "dataset_content_hash": dataset_content_hash,
                "config_hash": config_hash,
                "library_versions": dep_versions,
            }
            if used_fallback and model_plugin is not None:
                fb = getattr(model_plugin, "fallback_backend", None)
                if fb:
                    artifacts["fallback_backend"] = str(fb)
            if execution.group_column:
                artifacts["group_column"] = execution.group_column
            if "cv_scores" in secondary:
                artifacts["cv_scores"] = secondary["cv_scores"]
            return TrialResult(
                trial_id=trial.id,
                experiment_id=execution.experiment.id,
                model_id=trial.model_id,
                primary_metric=metric_name,
                primary_score=float(primary_score),
                secondary_metrics={k: float(v) for k, v in secondary.items() if isinstance(v, (int, float)) and not np.isnan(v)},
                training_time_seconds=elapsed,
                artifacts=artifacts,
            )
        except Exception as exc:  # noqa: BLE001 — surface failure in TrialResult
            elapsed = time.perf_counter() - started
            trial.status = TrialStatus.FAILED
            return TrialResult(
                trial_id=trial.id,
                experiment_id=execution.experiment.id,
                model_id=trial.model_id,
                primary_metric=execution.metric,
                primary_score=0.0,
                training_time_seconds=elapsed,
                failure_reason=str(exc),
            )

    def _run_clustering(
        self,
        execution: TrialExecution,
        X: pd.DataFrame,
        started: float,
    ) -> TrialResult:
        import time

        trial = execution.trial
        numeric_X = X.select_dtypes(include=["number"])
        if numeric_X.empty:
            raise ValueError("Clustering requires at least one numeric feature.")

        preprocessor = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        X_scaled = preprocessor.fit_transform(numeric_X)
        model = build_sklearn_model(
            execution.trial.model_id,
            execution.task_type,
            parameters=execution.trial.parameters,
        )

        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(X_scaled)
        else:
            model.fit(X_scaled)
            labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_scaled)

        unique_labels = set(labels)
        unique_labels.discard(-1)
        n_clusters = len(unique_labels)

        secondary: dict[str, float] = {"n_clusters": float(n_clusters)}
        metric_name = execution.metric

        if n_clusters >= 2 and len(labels) > n_clusters:
            secondary["silhouette"] = float(silhouette_score(X_scaled, labels))
            secondary["calinski_harabasz"] = float(calinski_harabasz_score(X_scaled, labels))
        else:
            secondary["silhouette"] = 0.0
            secondary["calinski_harabasz"] = 0.0

        primary_score = secondary.get(metric_name, secondary.get("silhouette", 0.0))
        elapsed = time.perf_counter() - started
        trial.status = TrialStatus.COMPLETED
        return TrialResult(
            trial_id=trial.id,
            experiment_id=execution.experiment.id,
            model_id=trial.model_id,
            primary_metric=metric_name,
            primary_score=float(primary_score),
            secondary_metrics=secondary,
            training_time_seconds=elapsed,
        )

    def fit_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        model_id: str,
        task_type: str = "binary_classification",
        parameters: dict[str, Any] | None = None,
    ) -> tuple[Pipeline, Any]:
        model_params = {
            k: v for k, v in (parameters or {}).items()
            if k not in (
                "text_columns",
                "image_columns",
                "image_model",
                "time_budget",
                "column_transforms",
                "derived_features",
            )
        }

        if self.plugin_registry and self.plugin_registry.has(model_id):
            model_plugin = self.plugin_registry.get_model_plugin(model_id)
            if model_plugin:
                self.plugin_registry.validate_plugin_for_task(model_id, task_type)
                model = model_plugin.build_estimator(parameters=model_params, task_type=task_type)
            else:
                model = build_sklearn_model(model_id, task_type, parameters=model_params)
        else:
            model = build_sklearn_model(model_id, task_type, parameters=model_params)

        from automl.engine.training.target_adapter import TargetAdapter

        adapter = TargetAdapter()
        y_train_adapted = adapter.fit_transform(y_train, task_type)

        text_cols = parameters.get("text_columns") if parameters else None
        image_cols = parameters.get("image_columns") if parameters else None
        image_model = parameters.get("image_model") if parameters else None
        column_transforms = parameters.get("column_transforms") if parameters else None
        pipeline = _build_pipeline(
            X_train,
            model,
            text_columns=text_cols,
            image_columns=image_cols,
            image_model=image_model,
            column_transforms=column_transforms,
            task_type=task_type,
        )
        pipeline.fit(X_train, y_train_adapted)
        return pipeline, adapter

    def fit_and_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_test: pd.DataFrame,
        model_id: str,
        task_type: str = "binary_classification",
        parameters: dict[str, Any] | None = None,
        predict_proba: bool = False,
    ) -> np.ndarray:
        pipeline, adapter = self.fit_pipeline(
            X_train=X_train,
            y_train=y_train,
            model_id=model_id,
            task_type=task_type,
            parameters=parameters,
        )

        if predict_proba and hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_test)
            if task_type == "binary_classification" and probs.ndim == 2 and probs.shape[1] == 2:
                return probs[:, 1]
            return probs

        preds = pipeline.predict(X_test)
        return adapter.inverse_transform(preds)


def _build_supervised_select_step(
    params: dict[str, Any],
    *,
    task_type: str | None,
) -> Any:
    """Fold-safe supervised selector fitted inside the Pipeline (train fold only)."""
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.feature_selection import (
        SelectFromModel,
        SelectKBest,
        f_classif,
        f_regression,
        mutual_info_classif,
        mutual_info_regression,
    )

    method = str(params.get("method", "mutual_information")).lower()
    k = int(params.get("k", 10))
    task = str(task_type or "binary_classification").lower()
    is_classif = "classif" in task or task in {
        "binary_classification",
        "multiclass_classification",
        "multiclass",
    }

    class _AdaptiveSelectKBest(BaseEstimator, TransformerMixin):
        def __init__(self, score_func: Any, k: int) -> None:
            self.score_func = score_func
            self.k = k
            self._inner: Any = None

        def fit(self, X: Any, y: Any = None):
            n_features = int(getattr(X, "shape", [0, 0])[1])
            effective_k = max(1, min(int(self.k), n_features)) if n_features else int(self.k)
            self._inner = SelectKBest(score_func=self.score_func, k=effective_k)
            self._inner.fit(X, y)
            return self

        def transform(self, X: Any):
            return self._inner.transform(X)

    if method in {"mutual_information", "mi"}:
        score_func = mutual_info_classif if is_classif else mutual_info_regression
        return _AdaptiveSelectKBest(score_func=score_func, k=k)
    if method in {"tree_importance", "importance"}:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        est = (
            RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=1)
            if is_classif
            else RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=1)
        )
        return SelectFromModel(est, max_features=k, threshold=-np.inf)
    score_func = f_classif if is_classif else f_regression
    return _AdaptiveSelectKBest(score_func=score_func, k=k)


def _build_pipeline(
    X: pd.DataFrame,
    model: Any,
    text_columns: list[str] | None = None,
    image_columns: list[str] | None = None,
    image_model: str | None = None,
    column_transforms: list[dict[str, Any]] | None = None,
    task_type: str | None = None,
) -> Pipeline:
    from automl.engine.features.text import LightweightTextExtractor, is_text_column
    from automl.plugins.modalities.image_plugin import is_image_column, ImageModalityPlugin
    from automl.engine.vision.image_encoder import ImageEncoderNode

    all_cols = list(X.columns)

    image_cols: list[str] = []
    if image_columns:
        image_cols = [c for c in image_columns if c in all_cols]
    else:
        for c in all_cols:
            if not pd.api.types.is_numeric_dtype(X[c]):
                if is_image_column(X[c]):
                    image_cols.append(c)

    text_cols: list[str] = []
    if text_columns:
        text_cols = [c for c in text_columns if c in all_cols and c not in image_cols]
    else:
        for c in all_cols:
            if c not in image_cols and not pd.api.types.is_numeric_dtype(X[c]):
                if is_text_column(X[c]):
                    text_cols.append(c)

    non_transformed_cols = [c for c in all_cols if c not in text_cols and c not in image_cols]
    numeric_cols = [c for c in non_transformed_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in non_transformed_cols if c not in numeric_cols]

    # Map custom column transforms
    col_transform_map: dict[str, dict[str, Any]] = {}
    if column_transforms:
        for ct in column_transforms:
            if isinstance(ct, dict) and "column" in ct:
                col = ct["column"]
                if col not in col_transform_map:
                    col_transform_map[col] = dict(ct)
                else:
                    existing = col_transform_map[col]
                    new_kind = str(ct.get("kind", "")).lower()
                    existing_kind = str(existing.get("kind", "")).lower()
                    if new_kind in ("impute", "imputation"):
                        if "strategy" in ct:
                            existing["strategy"] = ct["strategy"]
                        if "params" in ct and isinstance(ct["params"], dict):
                            existing_params = dict(existing.get("params", {}))
                            existing_params.update(ct["params"])
                            existing["params"] = existing_params
                    elif existing_kind in ("impute", "imputation"):
                        strat = existing.get("strategy")
                        imputer_params = dict(existing.get("params", {}))
                        merged = dict(ct)
                        if strat:
                            merged["strategy"] = strat
                        merged_params = dict(merged.get("params", {}))
                        merged_params.update(imputer_params)
                        merged["params"] = merged_params
                        col_transform_map[col] = merged
                    else:
                        col_transform_map[col] = dict(ct)

    transformers = []
    if numeric_cols:
        standard_num_cols = []
        for col in numeric_cols:
            if col in col_transform_map:
                ct = col_transform_map[col]
                kind = str(ct.get("kind", "")).lower()
                params = dict(ct.get("params", {}))

                scaler_params = {
                    k: v
                    for k, v in params.items()
                    if k not in ("strategy", "fill_value", "missing_values", "kind", "column", "scaler", "method")
                }

                if kind in ("robust_scaler", "robust"):
                    scaler_step = ("scaler", RobustScaler(**scaler_params))
                elif kind in ("yeo_johnson", "power_transform", "power", "yeojohnson"):
                    scaler_step = ("power", PowerTransformer(method="yeo-johnson", **scaler_params))
                elif kind in ("quantile_discretizer", "quantile", "kbins"):
                    n_bins = int(scaler_params.get("n_bins", 5))
                    scaler_step = ("discretize", KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile"))
                elif kind in ("standard_scaler", "standard"):
                    scaler_step = ("scaler", StandardScaler(**scaler_params))
                elif kind in ("passthrough", "none", "identity", "no_scaler"):
                    scaler_step = ("scaler", "passthrough")
                elif kind in ("impute", "imputation"):
                    scaler_step = ("scaler", StandardScaler(**scaler_params))
                else:
                    scaler_step = ("scaler", StandardScaler(**scaler_params))

                imputer_strategy = ct.get("strategy") or params.get("strategy") or "median"
                imputer_kwargs: dict[str, Any] = {"strategy": imputer_strategy}
                if "fill_value" in params:
                    imputer_kwargs["fill_value"] = params["fill_value"]

                transformers.append(
                    (
                        f"num_{kind}_{col}",
                        Pipeline([("imputer", SimpleImputer(**imputer_kwargs)), scaler_step]),
                        [col],
                    )
                )
            else:
                standard_num_cols.append(col)

        if standard_num_cols:
            transformers.append(
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                    standard_num_cols,
                )
            )
    if categorical_cols:
        standard_cat_cols = []
        for col in categorical_cols:
            if col in col_transform_map:
                ct = col_transform_map[col]
                kind = str(ct.get("kind", "")).lower()
                params = dict(ct.get("params", {}))
                cat_strat = ct.get("strategy") or params.get("strategy") or "most_frequent"
                if cat_strat in ("median", "mean"):
                    cat_strat = "most_frequent"
                cat_imputer_kwargs: dict[str, Any] = {"strategy": cat_strat}
                if "fill_value" in params:
                    cat_imputer_kwargs["fill_value"] = params["fill_value"]
                transformers.append(
                    (
                        f"cat_{kind}_{col}",
                        Pipeline([
                            ("imputer", SimpleImputer(**cat_imputer_kwargs)),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]),
                        [col],
                    )
                )
            else:
                standard_cat_cols.append(col)

        if standard_cat_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    standard_cat_cols,
                )
            )
    if text_cols:
        for col in text_cols:
            transformers.append(
                (
                    f"txt_{col}",
                    LightweightTextExtractor(max_features=50, column_prefix=col),
                    col,
                )
            )
    if image_cols:
        vision_available = ImageModalityPlugin().available()
        chosen_image_model = image_model

        if chosen_image_model is None:
            if vision_available:
                chosen_image_model = "resnet18"
            else:
                raise RuntimeError(
                    f"Dataset contains image columns {image_cols}, but deep learning vision "
                    "dependencies (PyTorch, torchvision, Pillow) are not installed. "
                    "Install vision dependencies via `pip install 'catml[vision]'` or "
                    "specify image_model='deterministic' explicitly for testing/benchmarks."
                )

        for col in image_cols:
            transformers.append(
                (
                    f"img_{col}",
                    ImageEncoderNode(
                        node_id=f"img_{col}",
                        model_name=chosen_image_model,
                        output_dim=None if chosen_image_model not in ("deterministic", "hash") else 128,
                        handle_missing="zero",
                        allow_fallback=False,
                    ),
                    col,
                )
            )

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.0)
    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]

    # Optional fold-safe reductions / selectors (column='*', kind=pca|supervised_select).
    pca_spec = None
    supervised_select_spec = None
    if column_transforms:
        for ct in column_transforms:
            if not isinstance(ct, dict):
                continue
            kind = str(ct.get("kind", "")).lower()
            col = str(ct.get("column", ""))
            if col not in {"*", "__all__", "__dataset__"}:
                continue
            if kind == "pca" and pca_spec is None:
                pca_spec = ct
            elif kind in {"supervised_select", "select_kbest", "nested_select"} and supervised_select_spec is None:
                supervised_select_spec = ct
    if supervised_select_spec is not None:
        params = dict(supervised_select_spec.get("params") or {})
        steps.append(
            (
                "supervised_select",
                _build_supervised_select_step(params, task_type=task_type),
            )
        )
    if pca_spec is not None:
        from sklearn.decomposition import PCA

        params = dict(pca_spec.get("params") or {})
        n_components = params.get("n_components", 0.95)
        steps.append(
            (
                "pca",
                PCA(n_components=n_components, random_state=params.get("random_state")),
            )
        )

    steps.append(("model", model))
    return Pipeline(steps)


def _is_minimizing_metric(metric_name: str, plugin_registry: Any = None) -> bool:
    from automl.domain.metrics.exceptions import InvalidMetric
    from automl.domain.metrics.spec import resolve_metric_spec

    greater: bool | None = None
    if plugin_registry and hasattr(plugin_registry, "has") and plugin_registry.has(metric_name):
        plugin = plugin_registry.get_metric_plugin(metric_name)
        if plugin is not None and hasattr(plugin, "greater_is_better"):
            greater = bool(plugin.greater_is_better)
    try:
        return resolve_metric_spec(metric_name, greater_is_better=greater).is_minimizing
    except InvalidMetric:
        return False


def _cross_val_scores_with_callback(
    pipeline: Pipeline,
    X: Any,
    y: Any,
    *,
    cv_splitter: Any,
    groups: Any,
    scoring: Any,
    metric_name: str,
    plugin_registry: Any,
    fold_callback: Any | None,
    fold_assignment: Any | None = None,
) -> tuple[list[float], int | None]:
    """Run CV fold-by-fold so Optuna MedianPruner can receive intermediate scores.

    Returns (fold_scores, pruned_at_fold_index). pruned_at is None when the trial
    completed all folds.
    """
    from sklearn.base import clone
    from sklearn.metrics import check_scoring

    scorer = check_scoring(pipeline, scoring=scoring)
    is_neg = (
        (isinstance(scoring, str) and scoring.startswith("neg_"))
        or _is_minimizing_metric(metric_name, plugin_registry)
        or getattr(scoring, "_sign", 1) == -1
    )

    split_kwargs = {"groups": groups} if groups is not None else {}
    fold_scores: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, **split_kwargs)):
        if fold_assignment is not None:
            try:
                fold_assignment[test_idx] = fold_idx
            except Exception:
                pass

        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_test = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
        y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        y_test = y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx]

        est = clone(pipeline)
        est.fit(X_train, y_train)
        raw = float(scorer(est, X_test, y_test))
        score = -raw if is_neg else raw
        fold_scores.append(score)

        if fold_callback is not None:
            should_prune = bool(fold_callback(fold_idx, score))
            if should_prune:
                return fold_scores, fold_idx

    return fold_scores, None


def _sklearn_scoring(metric_name: str, task_type: str, plugin_registry: Any = None) -> Any:
    # Binary ROC-AUC vs multiclass OvR (macro) — sklearn's plain "roc_auc" is binary-only.
    if metric_name == "roc_auc" and task_type == "multiclass_classification":
        return "roc_auc_ovr"
    mapping = {
        "accuracy": "accuracy",
        "f1": "f1_weighted",
        "roc_auc": "roc_auc",
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "mse": "neg_mean_squared_error",
        "log_loss": "neg_log_loss",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
    }
    if metric_name in mapping:
        return mapping[metric_name]

    if plugin_registry and plugin_registry.has(metric_name):
        plugin = plugin_registry.get_metric_plugin(metric_name)
        if plugin:
            from sklearn.metrics import make_scorer

            req_proba = getattr(plugin, "requires_probabilities", False)
            greater = getattr(plugin, "greater_is_better", True)
            try:
                return make_scorer(
                    lambda y_true, y_pred, **kwargs: plugin.compute(y_true, y_pred),
                    greater_is_better=greater,
                    response_method="predict" if not req_proba else "predict_proba",
                )
            except TypeError:
                return make_scorer(
                    lambda y_true, y_pred, **kwargs: plugin.compute(y_true, y_pred),
                    greater_is_better=greater,
                    needs_proba=req_proba,
                )

    raise ValueError(f"Unsupported metric '{metric_name}' for task '{task_type}'.")


def _compute_metrics(
    y_true: Any,
    predictions: Any,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    task_type: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if task_type == "regression":
        metrics["r2"] = float(r2_score(y_true, predictions))
        metrics["mae"] = float(mean_absolute_error(y_true, predictions))
        mse_val = float(mean_squared_error(y_true, predictions))
        metrics["mse"] = mse_val
        metrics["rmse"] = float(np.sqrt(mse_val))
        return metrics

    metrics["accuracy"] = float(accuracy_score(y_true, predictions))
    metrics["f1"] = float(f1_score(y_true, predictions, average="weighted"))
    if task_type in ("binary_classification", "multiclass_classification"):
        try:
            from sklearn.metrics import balanced_accuracy_score
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, predictions))
        except Exception:
            pass
    if task_type == "binary_classification":
        try:
            proba = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        except Exception:
            pass
        try:
            from sklearn.metrics import log_loss
            proba_all = pipeline.predict_proba(X_test)
            metrics["log_loss"] = float(log_loss(y_true, proba_all))
        except Exception:
            pass
    elif task_type == "multiclass_classification":
        try:
            proba_all = pipeline.predict_proba(X_test)
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, proba_all, multi_class="ovr", average="weighted")
            )
        except Exception:
            pass
        try:
            from sklearn.metrics import log_loss
            proba_all = pipeline.predict_proba(X_test)
            metrics["log_loss"] = float(log_loss(y_true, proba_all))
        except Exception:
            pass
    return metrics
