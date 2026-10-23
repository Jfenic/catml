"""Application service for experiment entity creation, validation, trials, and leaderboard evaluation."""
from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Any

from automl.domain.experiments.priority import ExperimentPriority
from automl.domain.experiments.trial import (
    Experiment,
    Trial,
    TrialResult,
)
from automl.domain.experiments.validation import require_known_validation_strategy
from automl.domain.metrics.exceptions import InvalidMetric
from automl.domain.metrics.spec import MetricSpec, resolve_metric_spec
from automl.domain.runs.run import AutoMLRun

if TYPE_CHECKING:
    from automl.application.services.workspace import AutoMLWorkspace


class ExperimentService:
    """Application service managing experiment definitions, safety validations, trials, and leaderboard queries."""

    def __init__(self, workspace: AutoMLWorkspace) -> None:
        self.workspace = workspace

    def get_trial(self, trial_id: str) -> Trial | None:
        return self.workspace.repository.get_trial(trial_id)

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        return self.workspace.repository.get_experiment(experiment_id)

    def list_experiments(self, run_id: str) -> list[Experiment]:
        return self.workspace.repository.list_experiments(run_id)

    def list_trial_results(self, experiment_id: str) -> list[TrialResult]:
        return self.workspace.repository.list_trial_results(experiment_id)

    def get_leaderboard_results(self, run_id: str, include_failed: bool = False) -> list[TrialResult]:
        return self.workspace.repository.get_leaderboard(run_id, include_failed=include_failed)

    def _metric_greater_is_better(self, metric_name: str) -> bool | None:
        """Plugin override for metric direction, or None to use builtin registry."""
        plugin = self.workspace.plugin_registry.get_metric_plugin(metric_name)
        if plugin is not None and hasattr(plugin, "greater_is_better"):
            return bool(plugin.greater_is_better)
        return None

    def _resolve_metric_spec(self, metric_name: str) -> MetricSpec:
        """Resolve MetricSpec via plugin direction when available, else builtins."""
        try:
            return resolve_metric_spec(
                metric_name,
                greater_is_better=self._metric_greater_is_better(metric_name),
            )
        except InvalidMetric:
            # Unknown plugin-less names: treat as maximize (legacy default).
            return resolve_metric_spec(metric_name, greater_is_better=True)

    def _is_minimizing_metric(self, metric_name: str) -> bool:
        """Prefer plugin-declared direction; fall back to builtin MetricSpec registry."""
        return self._resolve_metric_spec(metric_name).is_minimizing

    def create_experiment(
        self,
        run: AutoMLRun,
        name: str,
        feature_names: list[str] | None = None,
        feature_set_id: str | None = None,
        model_ids: list[str] | None = None,
        hypothesis: str = "",
        priority: str = "normal",
        validation_strategy: str | None = None,
        group_column: str | None = None,
        allow_leakage: bool = False,
        column_transforms: list[dict[str, Any]] | None = None,
        derived_features: list[dict[str, Any]] | None = None,
        recipe_id: str | None = None,
    ) -> Experiment:
        ws = self.workspace
        dataset = ws._get_dataset(run.dataset_id)
        priority_value = ExperimentPriority.parse(priority).value

        resolved_features = feature_names
        resolved_column_transforms = list(column_transforms or [])
        if feature_set_id:
            feature_set = ws.repository.get_feature_set(feature_set_id)
            if feature_set is None:
                raise KeyError(f"FeatureSet not found: {feature_set_id}")
            if feature_set.dataset_id != run.dataset_id:
                raise ValueError(
                    f"FeatureSet {feature_set_id} belongs to dataset {feature_set.dataset_id}, "
                    f"not run dataset {run.dataset_id}"
                )
            resolved_features = feature_set.feature_names
            if not resolved_column_transforms and feature_set.lineage:
                import json

                for line in str(feature_set.lineage).splitlines():
                    if line.startswith("catml_column_transforms:"):
                        try:
                            decoded = json.loads(line[len("catml_column_transforms:") :])
                            if isinstance(decoded, list):
                                resolved_column_transforms = decoded
                        except json.JSONDecodeError:
                            pass
                        break

        if not resolved_column_transforms:
            try:
                from automl.domain.preparation.pipeline_bridge import (
                    pipeline_steps_to_column_transforms,
                )
                from automl.infrastructure.preparation.json_store import JsonPreparationStore

                prep_store = JsonPreparationStore(ws.root_dir)
                version = prep_store.find_version_for_dataset(run.dataset_id)
                if not version and dataset and hasattr(dataset, "path"):
                    version = prep_store.find_version_by_path(str(dataset.path))
                if version and version.lineage and version.lineage.get("pipeline_steps"):
                    resolved_column_transforms = pipeline_steps_to_column_transforms(
                        version.lineage["pipeline_steps"]
                    )
            except Exception:
                pass

        profile = ws.repository.get_dataset_profile(run.dataset_id)
        effective_group = group_column or run.config.group_column
        exclude_cols = [effective_group] if effective_group else None

        if resolved_features is None:
            if profile and hasattr(profile, "resolve_safe_feature_names"):
                resolved_features = profile.resolve_safe_feature_names(
                    exclude_columns=exclude_cols,
                    allow_leakage=allow_leakage,
                )
            else:
                registry = ws.get_feature_registry(run.dataset_id)
                resolved_features = registry.active_feature_names(dataset.target_column)
        else:
            if profile and hasattr(profile, "resolve_safe_feature_names"):
                resolved_features = profile.resolve_safe_feature_names(
                    requested_features=resolved_features,
                    exclude_columns=exclude_cols,
                    allow_leakage=allow_leakage,
                )

        if not resolved_features:
            raise ValueError("An experiment requires at least one feature")

        registry = ws.get_feature_registry(run.dataset_id)
        known_features = {feature.name for feature in registry.list()}
        if derived_features:
            derived_names = {
                (df_item["name"] if isinstance(df_item, dict) else df_item.name)
                for df_item in derived_features
            }
            known_features = known_features | derived_names
        unknown_features = sorted(set(resolved_features) - known_features)
        if unknown_features:
            pc_like = [f for f in unknown_features if re.fullmatch(r"PC\d+", str(f))]
            if pc_like and set(pc_like) == set(unknown_features):
                raise ValueError(
                    "PCA component names "
                    f"({', '.join(pc_like)}) are not materializable dataset columns. "
                    "Use source features with a PCA reduction transform "
                    "(column='*', kind='pca') instead."
                )
            raise ValueError(f"Unknown features for dataset {run.dataset_id}: {', '.join(unknown_features)}")

        if model_ids:
            ws.model_registry.validate_for_task(model_ids, run.config.task_type)

        active_models = model_ids or ws.model_registry.resolve_active(
            run.config.models_include,
            run.config.models_exclude,
            task_type=run.config.task_type,
        )
        if not active_models:
            raise ValueError("An experiment requires at least one compatible model")

        resolved_group_col = group_column or run.config.group_column
        raw_val_strat = validation_strategy or (
            "group_kfold" if (resolved_group_col and run.config.validation_strategy == "holdout")
            else run.config.validation_strategy
        )
        resolved_val_strat = require_known_validation_strategy(raw_val_strat, default="holdout")

        experiment = Experiment(
            id=f"exp_{uuid.uuid4().hex[:8]}",
            run_id=run.id,
            name=name,
            hypothesis=hypothesis or f"Compare models on {', '.join(resolved_features[:5])}",
            feature_names=resolved_features,
            model_ids=active_models,
            metric=run.config.metric,
            validation_strategy=resolved_val_strat,
            priority=priority_value,
            feature_set_id=feature_set_id,
            group_column=resolved_group_col,
            column_transforms=list(resolved_column_transforms),
            derived_features=list(derived_features or []),
            recipe_id=recipe_id,
        )
        ws.repository.save_experiment(experiment)
        ws._emit(
            "ExperimentCreated",
            {"experiment_id": experiment.id, "name": name, "priority": priority_value},
            run_id=run.id,
        )
        return experiment

    def leaderboard(self, run: AutoMLRun, include_failed: bool = False) -> list[dict]:
        rows = self.workspace.repository.get_leaderboard(run.id, include_failed=include_failed)
        out = []
        for r in rows:
            arts = r.artifacts or {}
            out.append({
                "trial_id": r.trial_id,
                "experiment_id": r.experiment_id,
                "model_id": r.model_id,
                "metric": r.primary_metric,
                "score": round(r.primary_score, 4),
                "training_time_s": round(r.training_time_seconds, 3),
                "failed": not r.succeeded,
                "estimator_class": arts.get("estimator_class"),
                "backend": arts.get("backend"),
                "fallback_backend": arts.get("fallback_backend"),
                "seed": arts.get("seed"),
                "validation_strategy": arts.get("validation_strategy"),
                "cv_folds": arts.get("cv_folds"),
                "cv_fold_ids": arts.get("cv_fold_ids"),
                "cv_fold_assignment_hash": arts.get("cv_fold_assignment_hash"),
                "dataset_content_hash": arts.get("dataset_content_hash"),
                "config_hash": arts.get("config_hash"),
                "provenance": {
                    "seed": arts.get("seed"),
                    "validation_strategy": arts.get("validation_strategy"),
                    "cv_folds": arts.get("cv_folds"),
                    "cv_fold_ids": arts.get("cv_fold_ids"),
                    "cv_fold_assignment_hash": arts.get("cv_fold_assignment_hash"),
                    "dataset_content_hash": arts.get("dataset_content_hash"),
                    "config_hash": arts.get("config_hash"),
                    "estimator_class": arts.get("estimator_class"),
                    "estimator_module": arts.get("estimator_module"),
                    "backend_class": arts.get("backend_class") or arts.get("estimator_class"),
                    "backend_module": arts.get("backend_module") or arts.get("estimator_module"),
                    "library_versions": arts.get("library_versions"),
                    "backend": arts.get("backend"),
                },
            })
        return out

    def get_leaderboard(self, run_or_id: str | AutoMLRun, include_failed: bool = False) -> list[dict]:
        run = run_or_id if isinstance(run_or_id, AutoMLRun) else self.workspace._get_run(run_or_id)
        return self.leaderboard(run, include_failed=include_failed)

    def compare_experiments(self, experiment_ids: list[str]) -> list[dict]:
        rows = self.workspace.repository.compare_experiments(experiment_ids)
        grouped: dict[str, dict] = {}
        for row in rows:
            exp_id = row["experiment_id"]
            if exp_id not in grouped:
                grouped[exp_id] = {
                    "experiment_id": exp_id,
                    "name": row["name"],
                    "priority": row["priority"],
                    "status": row["status"],
                    "trials": [],
                }
            if row["trial_id"]:
                grouped[exp_id]["trials"].append(
                    {
                        "trial_id": row["trial_id"],
                        "model_id": row["model_id"],
                        "metric": row["primary_metric"],
                        "score": round(float(row["primary_score"]), 4),
                        "training_time_s": round(float(row["training_time_seconds"] or 0), 3),
                        "failed": row["failure_reason"] is not None,
                    }
                )
        return list(grouped.values())

    def get_best_trial(self, experiment_id: str) -> dict | None:
        results = self.workspace.repository.list_trial_results(experiment_id)
        if not results:
            return None
        valid = [r for r in results if r.succeeded]
        if not valid:
            return None

        metric_spec = self._resolve_metric_spec(valid[0].primary_metric)
        best_idx = metric_spec.select_best_index([r.primary_score for r in valid])
        best = valid[best_idx]

        trial = self.workspace.repository.get_trial(best.trial_id)
        params = trial.parameters if trial else {}

        return {
            "trial_id": best.trial_id,
            "experiment_id": best.experiment_id,
            "model_id": best.model_id,
            "metric": best.primary_metric,
            "best_score": round(best.primary_score, 4),
            "parameters": params,
            "training_time_seconds": round(best.training_time_seconds, 3),
        }

    def get_experiment_trials(self, experiment_id: str) -> list[dict]:
        results = self.workspace.repository.list_trial_results(experiment_id)
        trials = {r.trial_id: self.workspace.repository.get_trial(r.trial_id) for r in results}
        return [
            {
                "trial_id": r.trial_id,
                "experiment_id": r.experiment_id,
                "model_id": r.model_id,
                "metric": r.primary_metric,
                "score": round(r.primary_score, 4),
                "training_time_s": round(r.training_time_seconds, 3),
                "succeeded": r.succeeded,
                "failure_reason": r.failure_reason,
                "parameters": dict(trials[r.trial_id].parameters) if trials.get(r.trial_id) else {},
            }
            for r in results
        ]
