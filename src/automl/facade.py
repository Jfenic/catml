from __future__ import annotations

import contextlib
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    OptimizeExperimentCommand,
    RunExperimentCommand,
)
from automl.artifacts.model_artifact import ModelArtifact
from automl.domain.tasks.task_type import TaskType
from automl.engine.planning.task_planner import infer_task_type


@dataclass
class AutoMLResult:
    """
    Result container returned by AutoML.fit().

    Provides direct access to the winning ModelArtifact, experiment leaderboard,
    scoring metrics, and high-level prediction methods.
    """

    best_model: ModelArtifact
    best_score: float
    best_model_id: str
    task_type: str
    metric: str
    run_id: str
    _leaderboard_data: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _workspace: Any = field(default=None, repr=False)

    def leaderboard(self) -> pd.DataFrame:
        """
        Return the experiment leaderboard as a formatted pandas DataFrame.
        """
        if not self._leaderboard_data:
            return pd.DataFrame()
        df = pd.DataFrame(self._leaderboard_data)
        if "score" in df.columns:
            is_minimize = str(self.metric).lower() in {"mae", "rmse", "mse", "loss", "log_loss"}
            df = df.sort_values(by="score", ascending=is_minimize).reset_index(drop=True)
            df["rank"] = range(1, len(df) + 1)
        return df

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Generate predictions using the winning model artifact.
        """
        return self.best_model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Generate class probabilities using the winning model artifact.
        """
        return self.best_model.predict_proba(X)

    def summary(self) -> dict[str, Any]:
        """
        Return an overview dictionary of the execution results.
        """
        return {
            "run_id": self.run_id,
            "task_type": self.task_type,
            "best_model_id": self.best_model_id,
            "best_score": self.best_score,
            "metric": self.metric,
            "total_models_evaluated": len(self._leaderboard_data),
        }

    def save_model(self, path: str | Path) -> Path:
        """
        Save the winning model artifact to disk.
        """
        return self.best_model.save(path)


class AutoML:
    """
    High-level ergonomic facade for CATML.

    Enables simple, scikit-learn compatible model fitting and automated experimentation:
    >>> automl = AutoML()
    >>> result = automl.fit(df, target="churn")
    >>> result.leaderboard()
    >>> result.best_model.save("model.pkl")
    """

    def __init__(
        self,
        task: str | None = None,
        task_type: str | None = None,
        metric: str | None = None,
        cv_folds: int = 5,
        time_budget: int | None = None,
        models: list[str] | None = None,
        random_state: int = 42,
        workspace_dir: str | Path | None = None,
        group_column: str | None = None,
        image_columns: list[str] | None = None,
        image_model: str | None = None,
    ) -> None:
        """
        Initialize the AutoML runner.

        Parameters
        ----------
        task / task_type: str | None
            Type of ML task ('binary_classification', 'multiclass_classification', 'regression', 'clustering').
        metric: str | None
            Evaluation metric (e.g. 'roc_auc', 'f1', 'r2', 'mae', 'rmse', 'mse', 'log_loss').
        cv_folds: int
            Number of cross-validation splits (default: 5).
        time_budget: int | None
            Cooperative time budget in seconds across experiments and trials. Checked before
            starting each candidate trial. Running model fits are allowed to complete gracefully
            without mid-training interruption; subsequent trials are halted once the deadline expires.
        models: list[str] | None
            Specific model identifiers to evaluate.
        random_state: int
            Seed for reproducible dataset splits and model initialization.
        workspace_dir: str | Path | None
            Custom directory path for workspace artifacts and experiment tracking.
            If omitted, a temporary directory is created and cleaned up when this
            ``AutoML`` instance is garbage-collected (or when ``close()`` is called).
            Pass an explicit path to retain datasets/SQLite across process lifetime.
        group_column: str | None
            Column name representing entity or cluster groups to enforce non-overlapping splits.
        image_columns: list[str] | None
            Column names containing paths to image files for multimodal representation learning.
        image_model: str | None
            Image encoder backbone ('deterministic', 'resnet18', etc.).
        """
        self.task = task or task_type
        self.metric = metric
        self.cv_folds = cv_folds
        self.time_budget = time_budget
        self.models = list(models) if models is not None else None
        self.random_state = random_state
        self.workspace_dir = Path(workspace_dir) if workspace_dir is not None else None
        self.group_column = group_column
        self.image_columns = list(image_columns) if image_columns is not None else None
        self.image_model = image_model
        self._result: AutoMLResult | None = None
        self._temp_workspace: tempfile.TemporaryDirectory[str] | None = None

    def fit(
        self,
        data: pd.DataFrame | np.ndarray | str | Path,
        target: str | pd.Series | np.ndarray | None = None,
        text_columns: list[str] | None = None,
        image_columns: list[str] | None = None,
        group_column: str | None = None,
        image_model: str | None = None,
    ) -> AutoMLResult:
        """
        Fit multiple candidate models on the dataset, evaluate their performance,
        and construct the winning standalone ModelArtifact.

        Args:
            data: Feature matrix as pandas DataFrame, 2D numpy array, or file path (CSV/Parquet).
            target: Target column name (str) if data is DataFrame/path, or target values
                    (Series/ndarray).
            text_columns: Optional list of column names containing freeform natural language
                          text to be transformed via n-gram TF-IDF representations. If omitted,
                          text columns are discovered automatically via heuristics.
            image_columns: Optional list of column names referencing image files or paths to
                           be transformed via ImageEncoder embeddings. If omitted, image columns
                           are discovered automatically via heuristics.
            group_column: Optional entity or group column name (e.g. 'patient_id', 'user_id')
                          used to enforce GroupKFold validation and eliminate group data leakage.
                          If omitted and group leakage is detected, it is enforced automatically.

        Returns:
            AutoMLResult containing the winning model, leaderboard, and predictions.
        """
        fit_started = time.perf_counter()
        ws, cmd, _qry = self._get_application()

        dataset = None
        if isinstance(data, str):
            is_path_like = False
            try:
                is_path_like = Path(data).exists() or any(
                    data.lower().endswith(ext)
                    for ext in (".csv", ".parquet", ".pq", ".json", ".jsonl", ".tsv")
                )
            except (OSError, ValueError):
                is_path_like = False

            if not is_path_like:
                try:
                    ds_obj = ws.get_dataset(data)
                    if ds_obj is not None:
                        from automl.engine.profiling.dataset_profiler import load_dataframe

                        dataset = ds_obj
                        df = load_dataframe(dataset.path)
                        target_col = target if isinstance(target, str) else ds_obj.target_column
                except Exception:
                    dataset = None

        if dataset is None:
            df, target_col = self._standardize_input(data, target)
            resolved_task_type = self._resolve_task_type(df[target_col])
            dataset = ws.register_dataset(
                name="fit_dataset",
                path=df,
                target=target_col,
                task_type=resolved_task_type.value,
            )
        else:
            if not target_col or target_col not in df.columns:
                raise ValueError(
                    f"Target column '{target_col}' not found in registered dataset '{data}'."
                    if target_col
                    else f"Target column must be specified or present on registered dataset '{data}'."
                )
            resolved_task_type = self._resolve_task_type(df[target_col])

        resolved_group_col = group_column or self.group_column
        resolved_image_cols = list(image_columns) if image_columns is not None else (list(self.image_columns) if self.image_columns is not None else None)
        resolved_text_cols = list(text_columns) if text_columns is not None else None
        resolved_image_model = image_model or self.image_model

        extra_cfg: dict[str, Any] = {}
        if resolved_image_cols:
            extra_cfg["image_columns"] = resolved_image_cols
        if resolved_image_model:
            extra_cfg["image_model"] = resolved_image_model
        if resolved_text_cols:
            extra_cfg["text_columns"] = resolved_text_cols
        if self.time_budget:
            extra_cfg["time_budget"] = self.time_budget

        profile = ws.get_dataset_profile(dataset.id)
        if not resolved_group_col and profile and profile.has_group_leakage and profile.group_candidates:
            resolved_group_col = profile.group_candidates[0]

        if resolved_group_col:
            val_strategy = "group_kfold"
        elif self.cv_folds and self.cv_folds > 1:
            val_strategy = "stratified_kfold" if resolved_task_type.value != "regression" else "kfold"
        else:
            val_strategy = "holdout"

        run = ws.create_run(
            dataset,
            metric=self.metric,
            validation_strategy=val_strategy,
            group_column=resolved_group_col,
            cv_folds=self.cv_folds if (self.cv_folds and self.cv_folds > 1) else 5,
            random_seed=self.random_state,
            time_budget_seconds=self.time_budget,
            extra=extra_cfg,
        )

        # Strict anti-leakage resolution: automatically excludes leakage columns and non-predictive IDs
        exclude_cols = [resolved_group_col] if resolved_group_col else None
        if profile and hasattr(profile, "resolve_safe_feature_names"):
            feature_names = profile.resolve_safe_feature_names(
                exclude_columns=exclude_cols,
            )
        else:
            feature_names = ws.get_feature_registry(dataset.id).active_feature_names(target_col)
            if exclude_cols:
                feature_names = [f for f in feature_names if f not in exclude_cols]
            if not feature_names:
                raise ValueError(f"No safe feature candidates available for training on dataset '{dataset.name}'.")

        # Plan and run candidate models
        if self.models:
            experiment = cmd.dispatch(
                CreateExperimentCommand(
                    run_id=run.id,
                    name="custom_selection",
                    feature_names=feature_names,
                    model_ids=self.models,
                    validation_strategy=val_strategy,
                    group_column=resolved_group_col,
                )
            )
            cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=experiment.id))
        else:
            candidates = ws.plan_experiments(run_id=run.id, auto_enqueue=True)
            if candidates:
                ws.run_scheduled_experiments(run_id=run.id, max_experiments=5)
            else:
                # Fallback to registered models for task
                from automl.domain.tasks.task_type import models_for_task

                compatible = [
                    m for m in models_for_task(resolved_task_type)
                    if ws.model_registry.has(m) and m != "voting_ensemble"
                ]
                experiment = ws.create_experiment(
                    run_id=run.id,
                    name="baseline_models",
                    feature_names=feature_names,
                    model_ids=compatible[:4] if compatible else ["logistic_regression"],
                    validation_strategy=val_strategy,
                    group_column=resolved_group_col,
                )
                ws.run_experiment(run_id=run.id, experiment_id=experiment.id)

        # Extract leaderboard (excluding failed trials)
        raw_leaderboard = ws.get_leaderboard_results(run.id, include_failed=False)
        valid_leaderboard = [r for r in raw_leaderboard if r.succeeded]
        if not valid_leaderboard:
            refreshed_run = ws.get_run(run.id)
            if refreshed_run and refreshed_run.config.extra.get("time_budget_exhausted"):
                raise RuntimeError(
                    f"Time budget of {self.time_budget}s expired before any candidate model completed training."
                )
            raise RuntimeError("No models were successfully evaluated during fit().")

        # Spend remaining time_budget on Optuna HPO for the current best model
        # (warm-start priors are applied inside OptimizationService when available).
        if self.time_budget:
            elapsed = time.perf_counter() - fit_started
            remaining = float(self.time_budget) - elapsed
            if remaining >= 20.0:
                best_so_far = valid_leaderboard[0]
                trial_obj = ws.get_trial(best_so_far.trial_id)
                if trial_obj is not None:
                    n_trials = max(3, min(40, int(remaining // 8)))
                    with contextlib.suppress(Exception):
                        cmd.dispatch(
                            OptimizeExperimentCommand(
                                run_id=run.id,
                                experiment_id=trial_obj.experiment_id,
                                model_id=best_so_far.model_id,
                                optimizer="optuna",
                                n_trials=n_trials,
                                timeout_seconds=remaining,
                            )
                        )
                        # Baseline fit already succeeded; HPO is best-effort under budget.
                    raw_leaderboard = ws.get_leaderboard_results(run.id, include_failed=False)
                    valid_leaderboard = [r for r in raw_leaderboard if r.succeeded] or valid_leaderboard

        leaderboard_rows = []
        for rank, res in enumerate(valid_leaderboard, start=1):
            t_obj = ws.get_trial(res.trial_id)
            leaderboard_rows.append({
                "rank": rank,
                "model_id": res.model_id,
                "score": res.primary_score,
                "metric": res.primary_metric,
                "training_time_s": round(res.training_time_seconds, 3),
                "cv_std": res.secondary_metrics.get("cv_std") if res.secondary_metrics else None,
                "parameters": t_obj.parameters if t_obj else {},
            })

        # Export self-contained winning artifact
        best_artifact = ws.export_model_artifact(run_id=run.id)

        self._result = AutoMLResult(
            best_model=best_artifact,
            best_score=best_artifact.score,
            best_model_id=best_artifact.model_id,
            task_type=resolved_task_type.value,
            metric=best_artifact.metric,
            run_id=run.id,
            _leaderboard_data=leaderboard_rows,
            _workspace=ws,
        )
        return self._result

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Generate predictions using the fitted model.
        """
        if self._result is None:
            raise RuntimeError("AutoML instance is not fitted yet. Call fit() first.")
        return self._result.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Generate class probabilities using the fitted model.
        """
        if self._result is None:
            raise RuntimeError("AutoML instance is not fitted yet. Call fit() first.")
        return self._result.predict_proba(X)

    def leaderboard(self) -> pd.DataFrame:
        """
        Return the leaderboard DataFrame of the last fit() call.
        """
        if self._result is None:
            raise RuntimeError("AutoML instance is not fitted yet. Call fit() first.")
        return self._result.leaderboard()

    @classmethod
    def load_model(cls, path: str | Path) -> ModelArtifact:
        """
        Load a standalone saved ModelArtifact from disk.
        """
        return ModelArtifact.load(path)

    def _standardize_input(
        self,
        data: pd.DataFrame | np.ndarray | str | Path,
        target: str | pd.Series | np.ndarray | None,
    ) -> tuple[pd.DataFrame, str]:
        if isinstance(data, Path) or (
            isinstance(data, str)
            and (
                Path(data).exists()
                or any(data.lower().endswith(ext) for ext in (".csv", ".parquet", ".pq", ".json", ".jsonl", ".tsv"))
            )
        ):
            from automl.engine.profiling.dataset_profiler import load_dataframe
            data = load_dataframe(data)

        if isinstance(data, pd.DataFrame):
            if isinstance(target, str):
                if target not in data.columns:
                    raise ValueError(f"Target column '{target}' not found in DataFrame.")
                return data.copy(), target
            elif target is not None:
                target_series = pd.Series(target)
                target_col = target_series.name or "__catml_target__"
                df = data.copy()
                df[target_col] = target_series.values
                return df, target_col
            else:
                raise ValueError(
                    "Target must be specified as a column name (str) or target values (Series/array)."
                )
        elif isinstance(data, np.ndarray):
            if target is None:
                raise ValueError("Target array must be provided when data is a numpy array.")
            feature_cols = [f"feature_{i}" for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=feature_cols)
            target_series = pd.Series(target)
            target_col = "__catml_target__"
            df[target_col] = target_series.values
            return df, target_col
        else:
            raise TypeError(f"Unsupported input data type: {type(data)}")

    def _resolve_task_type(self, target_series: pd.Series) -> TaskType:
        if self.task:
            t = self.task.lower().strip()
            if t in {"classification", "clf"}:
                return (
                    TaskType.BINARY_CLASSIFICATION
                    if target_series.nunique(dropna=True) == 2
                    else TaskType.MULTICLASS_CLASSIFICATION
                )
            elif t in {"binary", "binary_classification"}:
                return TaskType.BINARY_CLASSIFICATION
            elif t in {"multiclass", "multiclass_classification"}:
                return TaskType.MULTICLASS_CLASSIFICATION
            elif t in {"regression", "reg"}:
                return TaskType.REGRESSION
            return TaskType.parse(t)
        return infer_task_type(target_series)

    def _get_application(self) -> tuple[Any, Any, Any]:
        """Return (AutoMLWorkspace, CommandBus, QueryBus) for this AutoML instance."""
        if self.workspace_dir:
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            ws_root = self.workspace_dir
        else:
            if self._temp_workspace is None:
                self._temp_workspace = tempfile.TemporaryDirectory(prefix="catml_fit_")
            ws_root = Path(self._temp_workspace.name)

        return build_application(root_dir=str(ws_root))

    @property
    def prepare(self) -> AutoMLPreparationFacade:
        """
        Data Preparation facade for quality inspection and reproducible preparation plans.
        """
        return AutoMLPreparationFacade(self)

    def close(self) -> None:
        """Remove a temporary fit workspace created when ``workspace_dir`` was omitted."""
        if self._temp_workspace is not None:
            self._temp_workspace.cleanup()
            self._temp_workspace = None

    def __enter__(self) -> AutoML:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


class AutoMLPreparationFacade:
    """
    Thin, scikit-learn ergonomic facade for data quality analysis and reproducible preparation plans.
    Operates strictly via CommandBus and QueryBus (Rule 3: Parity Principle).

    Can be accessed either as a namespace:
        >>> automl = AutoML()
        >>> report = automl.prepare.analyze(df)
        >>> plan = automl.prepare.suggest(df)
        >>> preview = automl.prepare.preview(plan["id"])
        >>> applied = automl.prepare.apply(plan["id"], approved=True)

    Or invoked directly as a convenience method:
        >>> applied = automl.prepare(df, target="churn", approve=True)
    """

    def __init__(self, automl: AutoML) -> None:
        self._automl = automl

    def _get_app(self) -> tuple[Any, Any, Any]:
        return self._automl._get_application()

    def _resolve_dataset_id(
        self,
        data: pd.DataFrame | np.ndarray | str | Path | None = None,
        target: str | pd.Series | np.ndarray | None = None,
        dataset_id: str | None = None,
    ) -> str:
        if dataset_id:
            return dataset_id
        if data is None:
            raise ValueError("Either `data` or `dataset_id` must be provided.")

        ws, _cmd, _qry = self._get_app()

        if isinstance(data, str):
            is_path_like = False
            try:
                is_path_like = Path(data).exists() or any(
                    data.lower().endswith(ext)
                    for ext in (".csv", ".parquet", ".pq", ".json", ".jsonl", ".tsv")
                )
            except (OSError, ValueError):
                is_path_like = False

            if not is_path_like:
                try:
                    existing = ws.get_dataset(data)
                    if existing:
                        return existing.id
                except Exception:
                    pass
                raise ValueError(f"Dataset '{data}' not found in workspace and is not an existing file path.")

        if target is not None:
            df, target_col = self._automl._standardize_input(data, target)
        else:
            if isinstance(data, (Path, str)) and (Path(data).exists() or any(str(data).lower().endswith(ext) for ext in (".csv", ".parquet", ".pq", ".json", ".jsonl", ".tsv"))):
                from automl.engine.profiling.dataset_profiler import load_dataframe
                df = load_dataframe(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            elif isinstance(data, np.ndarray):
                feature_cols = [f"feature_{i}" for i in range(data.shape[1])]
                df = pd.DataFrame(data, columns=feature_cols)
            else:
                raise TypeError(f"Unsupported input data type: {type(data)}")
            target_col = None

        task_type = None
        if target_col and target_col in df.columns:
            try:
                task_type = self._automl._resolve_task_type(df[target_col]).value
            except Exception:
                task_type = None

        ds = ws.register_dataset(
            name=f"prep_ds_{uuid.uuid4().hex[:8]}",
            path=df,
            target=target_col,
            task_type=task_type,
        )
        return ds.id

    def analyze(
        self,
        data: pd.DataFrame | np.ndarray | str | Path | None = None,
        target: str | pd.Series | np.ndarray | None = None,
        dataset_id: str | None = None,
    ) -> dict[str, Any]:
        """Analyze dataset quality and return actionable issues."""
        from automl.application.preparation.commands import AnalyzeDataQualityCommand

        ds_id = self._resolve_dataset_id(data, target=target, dataset_id=dataset_id)
        _ws, cmd, _qry = self._get_app()
        return cmd.dispatch(AnalyzeDataQualityCommand(dataset_id=ds_id))

    def suggest(
        self,
        data: pd.DataFrame | np.ndarray | str | Path | None = None,
        target: str | pd.Series | np.ndarray | None = None,
        dataset_id: str | None = None,
        name: str | None = None,
        include_pipeline_marks: bool = True,
    ) -> dict[str, Any]:
        """Suggest a typed preparation plan based on quality diagnostics."""
        from automl.application.preparation.commands import SuggestPreparationPlanCommand

        ds_id = self._resolve_dataset_id(data, target=target, dataset_id=dataset_id)
        _ws, cmd, _qry = self._get_app()
        return cmd.dispatch(
            SuggestPreparationPlanCommand(
                dataset_id=ds_id,
                name=name,
                include_pipeline_marks=include_pipeline_marks,
            )
        )

    def preview(self, plan_id: str) -> dict[str, Any]:
        """Preview plan effect without mutating stored dataset (purity guaranteed)."""
        from automl.application.preparation.commands import PreviewPreparationPlanCommand

        _ws, cmd, _qry = self._get_app()
        return cmd.dispatch(PreviewPreparationPlanCommand(plan_id=plan_id))

    def validate(self, plan_id: str) -> dict[str, Any]:
        """Validate preparation plan scopes and constraints."""
        from automl.application.preparation.commands import ValidatePreparationPlanCommand

        _ws, cmd, _qry = self._get_app()
        return cmd.dispatch(ValidatePreparationPlanCommand(plan_id=plan_id))

    def apply(
        self,
        plan_id: str,
        approved: bool = False,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Apply approved dataset-scoped operations to produce a new immutable dataset version."""
        from automl.application.preparation.commands import ApplyPreparationPlanCommand

        _ws, cmd, _qry = self._get_app()
        return cmd.dispatch(ApplyPreparationPlanCommand(plan_id=plan_id, approved=approved, notes=notes))

    def create(
        self,
        dataset_id: str,
        steps: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        name: str | None = None,
        notes: str | None = None,
    ) -> str:
        """Create an explicit preparation plan from step specifications."""
        from automl.application.preparation.commands import CreatePreparationPlanCommand

        _ws, cmd, _qry = self._get_app()
        return cmd.dispatch(
            CreatePreparationPlanCommand(
                dataset_id=dataset_id,
                steps=tuple(steps),
                name=name,
                notes=notes,
            )
        )

    def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        """Retrieve preparation plan details by ID."""
        from automl.application.preparation.queries import GetPreparationPlanQuery

        _ws, _cmd, qry = self._get_app()
        return qry.dispatch(GetPreparationPlanQuery(plan_id=plan_id))

    def list_plans(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        """List preparation plans, optionally filtered by dataset ID."""
        from automl.application.preparation.queries import ListPreparationPlansQuery

        _ws, _cmd, qry = self._get_app()
        return qry.dispatch(ListPreparationPlansQuery(dataset_id=dataset_id))

    def list_versions(self, parent_dataset_id: str) -> list[dict[str, Any]]:
        """List immutable dataset versions produced from parent dataset."""
        from automl.application.preparation.queries import ListDatasetVersionsQuery

        _ws, _cmd, qry = self._get_app()
        return qry.dispatch(ListDatasetVersionsQuery(parent_dataset_id=parent_dataset_id))

    def get_version(self, version_id: str) -> dict[str, Any] | None:
        """Retrieve dataset version details by ID."""
        from automl.application.preparation.queries import GetDatasetVersionQuery

        _ws, _cmd, qry = self._get_app()
        return qry.dispatch(GetDatasetVersionQuery(version_id=version_id))

    def __call__(
        self,
        data: pd.DataFrame | np.ndarray | str | Path,
        target: str | pd.Series | np.ndarray | None = None,
        approve: bool = True,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Convenience execution: analyze → suggest → validate → apply plan.
        Returns the applied result dictionary with version_id and path.
        """
        ds_id = self._resolve_dataset_id(data, target=target)
        suggested = self.suggest(dataset_id=ds_id)
        plan_id = suggested["id"]
        val = self.validate(plan_id)
        if not val.get("ok"):
            raise ValueError(f"Plan validation failed: {val.get('errors')}")
        return self.apply(plan_id, approved=approve, notes=notes)

