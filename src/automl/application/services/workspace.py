from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from automl import __version__
from automl.application.plugins.registry import PluginRegistry
from automl.domain.datasets.profile import Dataset, DatasetProfile
from automl.domain.experiments.candidate import ExperimentCandidate
from automl.domain.experiments.priority import ExperimentPriority, Priority
from automl.domain.experiments.trial import (
    Experiment,
    ExperimentStatus,
    Trial,
    TrialResult,
    TrialStatus,
)
from automl.domain.experiments.validation import require_known_validation_strategy
from automl.domain.features.evidence import FeatureEvidence
from automl.domain.features.feature import Feature
from automl.domain.features.feature_set import FeatureSet
from automl.domain.features.registry import FeatureRegistry
from automl.domain.features.selection_strategy import (
    FeatureRank,
    FeatureSelectionStrategy,
    FeatureSetCandidate,
)
from automl.domain.metrics.exceptions import InvalidMetric
from automl.domain.metrics.spec import MetricSpec, resolve_metric_spec
from automl.domain.models.registry import ModelRegistry
from automl.domain.pipelines.graph import PipelineGraph, PipelineNode
from automl.domain.policies.budget import BudgetPolicy
from automl.domain.ports import (
    ExperimentPlannerPort,
    PriorityScorerPort,
)
from automl.domain.runs.run import AutoMLRun, RunConfig
from automl.domain.runs.states import RunPhase, RunStatus
from automl.domain.tasks.problem_definition import ProblemDefinition
from automl.domain.tasks.task_type import TASK_CATALOG, TaskType
from automl.engine.planning.experiment_planner import RuleBasedExperimentPlanner
from automl.engine.planning.task_planner import plan_from_dataframe
from automl.engine.priority.scheduler import ExperimentQueue, Scheduler
from automl.engine.priority.scorer import RuleBasedPriorityScorer
from automl.engine.profiling.dataset_profiler import (
    infer_task_type,
    load_dataframe,
    profile_dataset,
)
from automl.infrastructure.database.sqlite_repository import SQLiteExperimentRepository
from automl.plugins.metrics.business_metrics import (
    CostSensitiveMetricPlugin,
    WeightedF1MetricPlugin,
)
from automl.plugins.modalities.image_plugin import ImageModalityPlugin
from automl.plugins.models.catboost_plugin import CatBoostPlugin
from automl.plugins.models.ensemble import VotingEnsemblePlugin
from automl.plugins.models.gradient_boosting import LightGBMPlugin, XGBoostPlugin
from automl.plugins.models.sklearn_models import (
    ExtraTreesPlugin,
    MLPPlugin,
    default_model_specs,
)
from automl.plugins.models.sklearn_plugin import create_default_sklearn_plugins
from automl.plugins.models.vision_plugin import TimmVisionPlugin

PLATFORM_VERSION = __version__


def _init_default_plugins(workspace: AutoMLWorkspace) -> None:
    for p in create_default_sklearn_plugins():
        workspace.plugin_registry.register(p)
    workspace.plugin_registry.register(LightGBMPlugin())
    workspace.plugin_registry.register(XGBoostPlugin())
    workspace.plugin_registry.register(CatBoostPlugin())
    workspace.plugin_registry.register(ExtraTreesPlugin())
    workspace.plugin_registry.register(MLPPlugin())
    workspace.plugin_registry.register(VotingEnsemblePlugin())
    workspace.plugin_registry.register(CostSensitiveMetricPlugin())
    workspace.plugin_registry.register(WeightedF1MetricPlugin())
    workspace.plugin_registry.register(ImageModalityPlugin())
    workspace.plugin_registry.register(TimmVisionPlugin())


@dataclass
class AutoMLWorkspace:
    id: str
    root_dir: Path
    repository: SQLiteExperimentRepository
    model_registry: ModelRegistry = field(default_factory=ModelRegistry)
    planner: ExperimentPlannerPort = field(default_factory=RuleBasedExperimentPlanner)
    scorer: PriorityScorerPort = field(default_factory=RuleBasedPriorityScorer)
    scheduler: Scheduler = field(default_factory=Scheduler)
    plugin_registry: PluginRegistry = field(default_factory=PluginRegistry)
    execution_check: Callable[[], None] | None = field(default=None, repr=False)
    execution_progress: Callable[[int, int, str], None] | None = field(default=None, repr=False)
    _runs: dict[str, AutoMLRun] = field(default_factory=dict)
    _datasets: dict[str, Dataset] = field(default_factory=dict)
    _feature_registries: dict[str, FeatureRegistry] = field(default_factory=dict)
    _queues: dict[str, ExperimentQueue] = field(default_factory=dict)
    _candidate_feature_sets: dict[str, list[FeatureSetCandidate]] = field(default_factory=dict)
    _feature_ranks: dict[str, list[FeatureRank]] = field(default_factory=dict)
    _run_deadlines: dict[str, float] = field(default_factory=dict)
    _pipeline_service: Any = field(default=None, repr=False)
    _inference_service: Any = field(default=None, repr=False)
    _feature_service: Any = field(default=None, repr=False)
    _experiment_execution_service: Any = field(default=None, repr=False)
    _optimization_service: Any = field(default=None, repr=False)
    _cached_study_repo: Any = field(default=None, repr=False)
    _cached_recipe_repo: Any = field(default=None, repr=False)

    @property
    def pipeline_service(self):
        if self._pipeline_service is None:
            from automl.application.services.pipeline_service import (
                PipelineExecutionService,
            )
            self._pipeline_service = PipelineExecutionService(self)
        return self._pipeline_service

    @property
    def inference_service(self):
        if self._inference_service is None:
            from automl.application.services.inference_service import InferenceService
            self._inference_service = InferenceService(self)
        return self._inference_service

    @property
    def feature_service(self):
        if self._feature_service is None:
            from automl.application.services.feature_service import (
                FeatureEngineeringService,
            )
            self._feature_service = FeatureEngineeringService(self)
        return self._feature_service

    @property
    def experiment_execution_service(self):
        if self._experiment_execution_service is None:
            from automl.application.services.experiment_execution_service import (
                ExperimentExecutionService,
            )
            self._experiment_execution_service = ExperimentExecutionService(self)
        return self._experiment_execution_service

    @property
    def optimization_service(self):
        if self._optimization_service is None:
            from automl.application.services.optimization_service import (
                OptimizationService,
            )
            self._optimization_service = OptimizationService(self)
        return self._optimization_service

    def _get_or_create_run_deadline(self, run: AutoMLRun) -> float | None:
        budget = run.config.effective_time_budget
        if budget is None or budget <= 0:
            return None
        if run.id not in self._run_deadlines:
            import time
            self._run_deadlines[run.id] = time.monotonic() + budget
        return self._run_deadlines[run.id]

    def _is_run_budget_exhausted(self, run: AutoMLRun) -> bool:
        deadline = self._get_or_create_run_deadline(run)
        if deadline is None:
            return False
        import time
        return time.monotonic() >= deadline

    def __post_init__(self) -> None:
        if self.repository is not None and hasattr(self.repository, "plugin_registry"):
            self.repository.plugin_registry = self.plugin_registry

    @classmethod
    def create(cls, name: str, root_dir: str | Path | None = None) -> AutoMLWorkspace:
        base = Path(root_dir or Path.cwd() / ".automl" / name)
        base.mkdir(parents=True, exist_ok=True)
        workspace_id = f"ws_{uuid.uuid4().hex[:8]}"
        repo = SQLiteExperimentRepository(base / "automl.db")
        workspace = cls(id=workspace_id, root_dir=base, repository=repo)
        for spec in default_model_specs():
            workspace.model_registry.register(spec)
        _init_default_plugins(workspace)
        return workspace

    @classmethod
    def load(cls, root_dir: str | Path) -> AutoMLWorkspace:
        base = Path(root_dir)
        repo = SQLiteExperimentRepository(base / "automl.db")
        datasets = repo.list_datasets()
        runs = repo.list_runs()
        workspace_id = (
            datasets[0].workspace_id
            if datasets
            else (runs[0].workspace_id if runs else f"ws_{uuid.uuid4().hex[:8]}")
        )
        workspace = cls(id=workspace_id, root_dir=base, repository=repo)
        for spec in default_model_specs():
            workspace.model_registry.register(spec)
        _init_default_plugins(workspace)
        for dataset in repo.list_datasets(workspace_id):
            workspace._datasets[dataset.id] = dataset
            workspace._hydrate_feature_registry(dataset.id)
        for run in repo.list_runs(workspace_id):
            workspace._runs[run.id] = run
        return workspace

    @classmethod
    def load_or_create(cls, name: str, root_dir: str | Path | None = None) -> AutoMLWorkspace:
        base = Path(root_dir or Path.cwd() / ".automl" / name)
        db_path = base / "automl.db"
        if db_path.exists():
            return cls.load(base)
        return cls.create(name, root_dir=base)

    def _hydrate_feature_registry(self, dataset_id: str) -> None:
        features = list(self.repository.list_features(dataset_id))
        profile = self.repository.get_dataset_profile(dataset_id)
        if profile is None:
            try:
                dataset = self._get_dataset(dataset_id)
                if dataset and Path(dataset.path).exists():
                    from automl.engine.profiling.dataset_profiler import profile_dataset
                    profile = profile_dataset(dataset)
                    self.repository.save_dataset_profile(profile)
            except Exception:
                pass

        registry = FeatureRegistry(features)
        known_names = {f.name for f in registry.list()}

        if profile:
            id_names = set(profile.identifier_column_names)
            for f in registry.list():
                if f.semantic_type == "unknown" and f.name in id_names:
                    f.semantic_type = "identifier"
                    self.repository.save_feature(f)

            target = profile.target_column
            for column in profile.columns:
                if target is not None and column.name == target:
                    continue
                if column.name not in known_names:
                    feature = Feature(
                        id=f"feat_{column.name}",
                        dataset_id=dataset_id,
                        name=column.name,
                        physical_dtype=column.dtype,
                        semantic_type="identifier" if column.is_identifier else "unknown",
                    )
                    registry.register(feature)
                    self.repository.save_feature(feature)
                    known_names.add(column.name)

        self._feature_registries[dataset_id] = registry

    def _emit(self, event_type: str, payload: dict, run_id: str | None = None) -> None:
        self.repository.append_event(event_type, payload, run_id=run_id)

    def _get_run(self, run_id: str) -> AutoMLRun:
        run = self._runs.get(run_id) or self.repository.get_run(run_id)
        if run is None:
            raise KeyError(f"Run not found: {run_id}")
        self._runs[run_id] = run
        return run

    def get_run(self, run_id: str) -> AutoMLRun:
        return self._get_run(run_id)

    def _get_dataset(self, dataset_id: str) -> Dataset:
        dataset = self._datasets.get(dataset_id) or self.repository.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(f"Dataset not found: {dataset_id}")
        self._datasets[dataset_id] = dataset
        return dataset

    def get_dataset(self, dataset_id: str) -> Dataset:
        return self._get_dataset(dataset_id)

    def list_datasets(self) -> list[Dataset]:
        return (
            self.repository.list_datasets(workspace_id=self.id)
            if hasattr(self.repository, "list_datasets")
            else list(self._datasets.values())
        )

    def get_dataset_profile(self, dataset_id: str) -> DatasetProfile | None:
        return self.repository.get_dataset_profile(dataset_id)

    def save_dataset_profile(self, profile: DatasetProfile) -> None:
        self.repository.save_dataset_profile(profile)

    def save_run(self, run: AutoMLRun) -> None:
        self.repository.save_run(run)

    def get_trial(self, trial_id: str) -> Trial | None:
        return self.repository.get_trial(trial_id)

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        return self.repository.get_experiment(experiment_id)

    def list_experiments(self, run_id: str) -> list[Experiment]:
        return self.repository.list_experiments(run_id)

    def list_trial_results(self, experiment_id: str) -> list[TrialResult]:
        return self.repository.list_trial_results(experiment_id)

    def get_leaderboard_results(self, run_id: str, include_failed: bool = False) -> list[TrialResult]:
        return self.repository.get_leaderboard(run_id, include_failed=include_failed)

    def list_feature_sets(self, dataset_id: str) -> list[Any]:
        return self.repository.list_feature_sets(dataset_id)

    def register_dataset(
        self,
        name: str,
        path: str | Path | Any,
        target: str | None = None,
        task_type: str | None = None,
    ) -> Dataset:
        dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
        if isinstance(path, pd.DataFrame):
            df = path.copy()
            ds_dir = self.root_dir / "datasets"
            ds_dir.mkdir(parents=True, exist_ok=True)
            try:
                save_path = ds_dir / f"{dataset_id}.parquet"
                df.to_parquet(save_path)
            except Exception:
                save_path = ds_dir / f"{dataset_id}.csv"
                df.to_csv(save_path, index=False)
            resolved = str(save_path.resolve())
        else:
            resolved = str(Path(path).resolve())
            df = load_dataframe(resolved)
        if task_type:
            resolved_task = TaskType.parse(task_type)
        elif target is not None and target in df.columns:
            resolved_task = TaskType.parse(infer_task_type(df[target]))
        else:
            resolved_task = None

        dataset = Dataset(
            id=dataset_id,
            workspace_id=self.id,
            name=name,
            path=resolved,
            target_column=target,
            task_type=resolved_task.value if resolved_task else None,
        )
        profile = profile_dataset(dataset)
        self._datasets[dataset_id] = dataset
        self.repository.save_dataset(dataset)
        self.repository.save_dataset_profile(profile)

        problem = plan_from_dataframe(dataset, df, task_type=resolved_task)
        self.repository.save_problem_definition(problem)

        registry = FeatureRegistry()
        for column in profile.columns:
            if target is not None and column.name == target:
                continue
            feature = Feature(
                id=f"feat_{column.name}",
                dataset_id=dataset_id,
                name=column.name,
                physical_dtype=column.dtype,
                semantic_type="identifier" if column.is_identifier else "unknown",
            )
            registry.register(feature)
            self.repository.save_feature(feature)
        self._feature_registries[dataset_id] = registry
        self._emit("DatasetRegistered", {"dataset_id": dataset_id, "name": name})
        return dataset

    def create_run(
        self,
        dataset: Dataset,
        metric: str | None = None,
        validation_strategy: str | None = None,
        group_column: str | None = None,
        cv_folds: int = 5,
        random_seed: int = 42,
        time_budget_seconds: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AutoMLRun:
        problem = self.repository.get_problem_definition(dataset.id)
        if problem is None:
            df = load_dataframe(dataset.path)
            problem = plan_from_dataframe(dataset, df)
            self.repository.save_problem_definition(problem)

        run_metric = metric or problem.default_metric
        strat = validation_strategy or ("group_kfold" if group_column else "holdout")
        effective_time_budget = time_budget_seconds
        if effective_time_budget is None and extra and "time_budget" in extra:
            try:
                effective_time_budget = float(extra["time_budget"])
            except (ValueError, TypeError):
                pass

        config = RunConfig(
            task_type=problem.task_type.value,
            target=dataset.target_column,
            metric=run_metric,
            validation_strategy=strat,
            group_column=group_column,
            cv_folds=cv_folds,
            random_seed=random_seed,
            time_budget_seconds=effective_time_budget,
            extra=dict(extra or {}),
        )
        run = AutoMLRun(
            id=f"run_{uuid.uuid4().hex[:8]}",
            workspace_id=self.id,
            dataset_id=dataset.id,
            config=config,
            status=RunStatus.CREATED,
            current_phase=RunPhase.DATASET_PROFILING,
        )
        self._runs[run.id] = run
        self.repository.save_run(run)
        self._emit("RunCreated", {"run_id": run.id, "dataset_id": dataset.id}, run_id=run.id)
        return run

    def list_runs(self, dataset_id: str | None = None) -> list[AutoMLRun]:
        runs = self.repository.list_runs(workspace_id=self.id)
        if dataset_id:
            return [r for r in runs if r.dataset_id == dataset_id]
        return runs

    def get_study_repository(self):
        """Retrieve SQLiteStudyRepository for this workspace."""
        if self._cached_study_repo is None:
            from automl.infrastructure.database.sqlite_studies import (
                SQLiteStudyRepository,
            )
            self._cached_study_repo = SQLiteStudyRepository(self.root_dir / "automl.db")
        return self._cached_study_repo

    def get_analysis_service(self):
        """Retrieve AnalysisStudyService for this workspace."""
        from automl.application.analysis.study_service import AnalysisStudyService
        from automl.engine.analysis.statistical_analyzer import StatisticalAnalyzer
        return AnalysisStudyService(
            repository=self.get_study_repository(),
            dataset_resolver=self.get_dataset,
            statistical_engine=StatisticalAnalyzer(),
            profile_resolver=self.get_dataset_profile,
        )

    def get_recipe_repository(self):
        """Retrieve SQLiteRecipeRepository for this workspace."""
        if self._cached_recipe_repo is None:
            from automl.infrastructure.database.sqlite_recipes import (
                SQLiteRecipeRepository,
            )
            self._cached_recipe_repo = SQLiteRecipeRepository(self.root_dir / "automl.db")
        return self._cached_recipe_repo

    def get_recipe_service(self):
        """Retrieve RecipeService for this workspace."""
        from automl.application.recipes.recipe_service import RecipeService
        return RecipeService(
            recipe_repository=self.get_recipe_repository(),
            study_repository=self.get_study_repository(),
            workspace=self,
        )

    def dispose(self) -> None:
        """Close and release any cached SQLite repositories or long-lived resources."""
        if self._cached_study_repo is not None:
            try:
                self._cached_study_repo.close()
            except Exception:
                pass
            self._cached_study_repo = None
        if self._cached_recipe_repo is not None:
            try:
                self._cached_recipe_repo.close()
            except Exception:
                pass
            self._cached_recipe_repo = None
        if hasattr(self, "repository") and hasattr(self.repository, "close"):
            try:
                self.repository.close()
            except Exception:
                pass

    def close(self) -> None:
        """Alias for dispose(). Close and release any cached repositories."""
        self.dispose()

    def __enter__(self) -> AutoMLWorkspace:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dispose()

    def __del__(self) -> None:
        try:
            self.dispose()
        except Exception:
            pass

    def get_feature_registry(self, dataset_id: str) -> FeatureRegistry:
        if dataset_id not in self._feature_registries:
            self._hydrate_feature_registry(dataset_id)
        registry = self._feature_registries.get(dataset_id)
        if registry is None:
            raise KeyError(f"No feature registry for dataset {dataset_id}")
        return registry

    def create_feature_set(
        self,
        dataset_id: str,
        name: str,
        feature_names: list[str],
        lineage: str = "",
    ) -> FeatureSet:
        feature_set = FeatureSet(
            id=f"fset_{uuid.uuid4().hex[:8]}",
            dataset_id=dataset_id,
            name=name,
            feature_names=feature_names,
            lineage=lineage,
        )
        self.repository.save_feature_set(feature_set)
        self._emit("FeatureSetCreated", feature_set.to_dict())
        return feature_set

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
        dataset = self._get_dataset(run.dataset_id)
        priority_value = ExperimentPriority.parse(priority).value

        resolved_features = feature_names
        resolved_column_transforms = list(column_transforms or [])
        if feature_set_id:
            feature_set = self.repository.get_feature_set(feature_set_id)
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

                prep_store = JsonPreparationStore(self.root_dir)
                version = prep_store.find_version_for_dataset(run.dataset_id)
                if not version and dataset and hasattr(dataset, "path"):
                    version = prep_store.find_version_by_path(str(dataset.path))
                if version and version.lineage and version.lineage.get("pipeline_steps"):
                    resolved_column_transforms = pipeline_steps_to_column_transforms(
                        version.lineage["pipeline_steps"]
                    )
            except Exception:
                pass

        profile = self.repository.get_dataset_profile(run.dataset_id)
        effective_group = group_column or run.config.group_column
        exclude_cols = [effective_group] if effective_group else None

        if resolved_features is None:
            if profile and hasattr(profile, "resolve_safe_feature_names"):
                resolved_features = profile.resolve_safe_feature_names(
                    exclude_columns=exclude_cols,
                    allow_leakage=allow_leakage,
                )
            else:
                registry = self.get_feature_registry(run.dataset_id)
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

        registry = self.get_feature_registry(run.dataset_id)
        known_features = {feature.name for feature in registry.list()}
        if derived_features:
            derived_names = {
                (df_item["name"] if isinstance(df_item, dict) else df_item.name)
                for df_item in derived_features
            }
            known_features = known_features | derived_names
        unknown_features = sorted(set(resolved_features) - known_features)
        if unknown_features:
            import re

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
            self.model_registry.validate_for_task(model_ids, run.config.task_type)

        active_models = model_ids or self.model_registry.resolve_active(
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
        self.repository.save_experiment(experiment)
        self._emit(
            "ExperimentCreated",
            {"experiment_id": experiment.id, "name": name, "priority": priority_value},
            run_id=run.id,
        )
        return experiment

    def run_experiment(
        self,
        run: AutoMLRun,
        experiment: Experiment,
        start_index: int = 0,
        skip_model_ids: set[str] | None = None,
    ) -> list[TrialResult]:
        """Facade: delegates to ExperimentExecutionService (Clean Architecture PR2)."""
        return self.experiment_execution_service.run_experiment(
            run,
            experiment,
            start_index=start_index,
            skip_model_ids=skip_model_ids,
        )

    def resume_run(self, run_id: str) -> list[TrialResult]:
        run = self._get_run(run_id)
        if run.status not in {RunStatus.PAUSED, RunStatus.EXPERIMENTING}:
            raise RuntimeError(f"Run {run_id} is not resumable (status={run.status.value})")

        checkpoint = self.repository.get_checkpoint(run_id)
        if checkpoint is None:
            raise RuntimeError(f"No checkpoint found for run {run_id}")

        experiment_id, start_index = checkpoint
        experiment = self.repository.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(f"Experiment not found: {experiment_id}")

        run.transition_to(RunStatus.EXPERIMENTING, RunPhase.EXPERIMENT_EXECUTION)
        self.repository.save_run(run)
        self._emit("RunResumed", {"experiment_id": experiment_id, "start_index": start_index}, run_id=run_id)
        return self.run_experiment(run, experiment, start_index=start_index)

    def pause_run(self, run_id: str) -> AutoMLRun:
        run = self._get_run(run_id)
        run.transition_to(RunStatus.PAUSED, RunPhase.EXPERIMENT_EXECUTION)
        self.repository.save_run(run)
        self._emit("RunPauseRequested", {}, run_id=run_id)
        return run

    def cancel_run(self, run_id: str) -> AutoMLRun:
        run = self._get_run(run_id)
        run.transition_to(RunStatus.CANCELLED, run.current_phase)
        self.repository.save_run(run)
        self.repository.clear_checkpoint(run_id)
        self._emit("RunCancelled", {}, run_id=run_id)
        return run

    def clone_run(self, run_id: str, new_name: str | None = None) -> AutoMLRun:
        source = self._get_run(run_id)
        self._get_dataset(source.dataset_id)  # ensure dataset still exists
        cloned_config = RunConfig.from_dict(source.config.to_dict())
        cloned = AutoMLRun(
            id=f"run_{uuid.uuid4().hex[:8]}",
            workspace_id=self.id,
            dataset_id=source.dataset_id,
            config=cloned_config,
            status=RunStatus.CREATED,
            current_phase=RunPhase.EXPERIMENT_PLANNING,
        )
        self._runs[cloned.id] = cloned
        self.repository.save_run(cloned)
        self._emit(
            "RunCloned",
            {"source_run_id": run_id, "new_run_id": cloned.id, "name": new_name or cloned.id},
            run_id=cloned.id,
        )
        return cloned

    def leaderboard(self, run: AutoMLRun, include_failed: bool = False) -> list[dict]:
        rows = self.repository.get_leaderboard(run.id, include_failed=include_failed)
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
        run = run_or_id if isinstance(run_or_id, AutoMLRun) else self._get_run(run_or_id)
        return self.leaderboard(run, include_failed=include_failed)

    def compare_experiments(self, experiment_ids: list[str]) -> list[dict]:
        rows = self.repository.compare_experiments(experiment_ids)
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

    def list_models(self, run_id: str | None = None, task_type: str | None = None) -> list[dict]:
        if run_id:
            run = self._get_run(run_id)
            task_type = task_type or run.config.task_type
            active = self.model_registry.resolve_active(
                run.config.models_include,
                run.config.models_exclude,
                task_type=task_type,
            )
            return [
                {
                    "id": spec.id,
                    "name": spec.name,
                    "active": spec.id in active,
                    "excluded": spec.id in run.config.models_exclude,
                    "compatible": task_type in spec.task_types,
                    "task_types": spec.task_types,
                }
                for spec in self.model_registry.list(task_type)
            ]
        return [
            {
                "id": spec.id,
                "name": spec.name,
                "task_types": spec.task_types,
                "compatible": task_type in spec.task_types if task_type else True,
            }
            for spec in self.model_registry.list(task_type)
        ]

    def get_task_plan(self, dataset_id: str) -> ProblemDefinition:
        problem = self.repository.get_problem_definition(dataset_id)
        if problem is not None:
            return problem
        dataset = self._get_dataset(dataset_id)
        df = load_dataframe(dataset.path)
        problem = plan_from_dataframe(dataset, df)
        self.repository.save_problem_definition(problem)
        return problem

    def get_dataset_questionnaire(
        self,
        dataset_id: str,
        auto_generate: bool = True,
        llm_provider: Any = None,
    ) -> Any | None:
        """Retrieve dataset framing questionnaire, optionally generating it if not found."""
        q = self.repository.get_dataset_questionnaire(dataset_id)
        if q is not None:
            return q
        if not auto_generate:
            return None
        return self.generate_dataset_questionnaire(dataset_id, llm_provider=llm_provider)

    def generate_dataset_questionnaire(
        self,
        dataset_id: str,
        llm_provider: Any = None,
    ) -> Any:
        """Synthesize and persist a problem context questionnaire for the dataset."""
        dataset = self._get_dataset(dataset_id)
        profile = self.repository.get_dataset_profile(dataset_id)
        profile_summary = profile.to_dict() if profile else None
        columns = [c.name for c in profile.columns] if profile else []
        target_col = getattr(dataset, "target_column", getattr(dataset, "target", ""))

        from automl.engine.planning.questionnaire_advisor import QuestionnaireAdvisor
        advisor = QuestionnaireAdvisor(llm_provider=llm_provider)
        q = advisor.infer(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            column_names=columns,
            target_column=target_col,
            profile_summary=profile_summary,
        )
        self.repository.save_dataset_questionnaire(q)
        return q

    def save_dataset_questionnaire(self, questionnaire: Any) -> None:
        """Explicitly persist an edited or reviewed questionnaire."""
        self.repository.save_dataset_questionnaire(questionnaire)

    def validate_derived_feature(
        self,
        dataset_id: str,
        name: str,
        expression: str,
        expression_type: str = "formula",
    ) -> Any:
        """Validate and evaluate a derived feature definition on dataset without modifying it."""
        return self.feature_service.validate_derived_feature(
            dataset_id=dataset_id,
            name=name,
            expression=expression,
            expression_type=expression_type,
        )

    def apply_derived_feature(
        self,
        dataset_id: str,
        name: str,
        expression: str,
        expression_type: str = "formula",
        description: str = "",
    ) -> tuple[Any, Any]:
        """Evaluate and apply a derived feature to the dataset, re-profiling the dataset and saving changes."""
        return self.feature_service.apply_derived_feature(
            dataset_id=dataset_id,
            name=name,
            expression=expression,
            expression_type=expression_type,
            description=description,
        )

    def suggest_derived_features(
        self,
        dataset_id: str,
        llm_provider: Any = None,
    ) -> list[dict[str, Any]]:
        """Suggest candidate derived features tailored to dataset profile and validate them."""
        return self.feature_service.suggest_derived_features(
            dataset_id=dataset_id,
            llm_provider=llm_provider,
        )

    def detect_temporal_structure(self, dataset_id: str) -> dict[str, Any]:
        """Inspects dataset and returns detected temporal periodicities and sequential columns."""
        return self.feature_service.detect_temporal_structure(dataset_id=dataset_id)

    def generate_temporal_features(
        self,
        run_id: str,
        dataset_id: str | None = None,
        max_lags: int = 1,
        include_lags: bool = True,
        include_deltas: bool = True,
        include_cyclical: bool = True,
        include_rolling: bool = False,
    ) -> list[FeatureSet]:
        """
        Discovers temporal dynamics (lags, deltas, cyclical periodicities) and registers
        candidate FeatureSet instances obeying 'Propose != Accept'.
        """
        return self.feature_service.generate_temporal_features(
            run_id=run_id,
            dataset_id=dataset_id,
            max_lags=max_lags,
            include_lags=include_lags,
            include_deltas=include_deltas,
            include_cyclical=include_cyclical,
            include_rolling=include_rolling,
        )

    def list_task_types(self) -> list[dict]:
        rows = []
        for task in TaskType:
            catalog = TASK_CATALOG[task]
            rows.append(
                {
                    "task_type": task.value,
                    "label": catalog["label"],
                    "description": catalog["description"],
                    "default_metric": catalog["default_metric"],
                    "metrics": catalog["metrics"],
                    "models": catalog["model_ids"],
                }
            )
        return rows

    def exclude_feature(self, dataset_id: str, name: str, run_id: str | None = None) -> None:
        feature = self.get_feature_registry(dataset_id).get(name)
        if feature is None:
            raise KeyError(f"Feature not found: {name}")
        feature.exclude()
        self.repository.save_feature(feature)
        if run_id:
            run = self._get_run(run_id)
            if name not in run.config.features_excluded:
                run.config.features_excluded.append(name)
            self.repository.save_run(run)
        self._emit("FeatureExcluded", {"dataset_id": dataset_id, "feature": name}, run_id=run_id)

    def exclude_identifiers(self, dataset_id: str, run_id: str | None = None) -> list[str]:
        profile = self.repository.get_dataset_profile(dataset_id)
        if profile is None:
            dataset = self._get_dataset(dataset_id)
            profile = profile_dataset(dataset)
            self.repository.save_dataset_profile(profile)

        excluded: list[str] = []
        for name in profile.identifier_column_names:
            if name != profile.target_column:
                self.exclude_feature(dataset_id, name, run_id=run_id)
                excluded.append(name)
        return excluded

    def prioritize_feature(
        self,
        dataset_id: str,
        name: str,
        score: float = 1.0,
        run_id: str | None = None,
    ) -> None:
        feature = self.get_feature_registry(dataset_id).get(name)
        if feature is None:
            raise KeyError(f"Feature not found: {name}")
        feature.prioritize(score)
        self.repository.save_feature(feature)
        if run_id:
            run = self._get_run(run_id)
            if name not in run.config.features_priority:
                run.config.features_priority.append(name)
            self.repository.save_run(run)
        self._emit(
            "FeaturePrioritized",
            {"dataset_id": dataset_id, "feature": name, "score": score},
            run_id=run_id,
        )

    def exclude_model(self, run: AutoMLRun, model_id: str) -> None:
        self.model_registry.validate_for_task([model_id], run.config.task_type)
        if model_id not in run.config.models_exclude:
            run.config.models_exclude.append(model_id)
        self.repository.save_run(run)
        self._emit("ModelExcluded", {"model_id": model_id}, run_id=run.id)

    def include_model(self, run: AutoMLRun, model_id: str) -> None:
        self.model_registry.validate_for_task([model_id], run.config.task_type)
        if model_id not in run.config.models_include:
            run.config.models_include.append(model_id)
        if model_id in run.config.models_exclude:
            run.config.models_exclude.remove(model_id)
        self.repository.save_run(run)
        self._emit("ModelAdded", {"model_id": model_id}, run_id=run.id)

    # --- V0.3 Experiment Planner & Priority Engine ---

    def get_experiment_queue(self, run_id: str) -> ExperimentQueue:
        if run_id not in self._queues:
            self._queues[run_id] = ExperimentQueue()
        return self._queues[run_id]

    def plan_experiments(
        self,
        run_id: str,
        auto_enqueue: bool = True,
        user_priorities: dict[str, str] | None = None,
    ) -> list[ExperimentCandidate]:
        run = self._get_run(run_id)
        if run.status == RunStatus.CANCELLED:
            raise RuntimeError(f"Run {run_id} is cancelled")

        run.transition_to(RunStatus.PLANNING, RunPhase.EXPERIMENT_PLANNING)
        self.repository.save_run(run)

        profile = self.repository.get_dataset_profile(run.dataset_id)
        if profile is None:
            dataset = self._get_dataset(run.dataset_id)
            profile = profile_dataset(dataset)
            self.repository.save_dataset_profile(profile)

        feature_registry = self.get_feature_registry(run.dataset_id)
        history = self.repository.get_leaderboard(run.id)

        candidates = self.planner.propose(
            run=run,
            profile=profile,
            feature_registry=feature_registry,
            model_registry=self.model_registry,
            history=history,
        )

        priorities_map = user_priorities or {}
        for candidate in candidates:
            user_prio = (
                priorities_map.get(candidate.id)
                or priorities_map.get(candidate.name)
                or None
            )
            score = self.scorer.score(
                candidate=candidate,
                run=run,
                profile=profile,
                user_priority=user_prio,
            )
            candidate.priority = score

        if auto_enqueue:
            queue = self.get_experiment_queue(run.id)
            queue.enqueue_all(candidates)

        self._emit(
            "ExperimentsPlanned",
            {
                "run_id": run_id,
                "candidate_count": len(candidates),
                "candidates": [c.name for c in candidates],
            },
            run_id=run.id,
        )
        return candidates

    def prioritize_candidate(
        self,
        run_id: str,
        candidate_id: str,
        priority: str,
    ) -> Priority:
        run = self._get_run(run_id)
        queue = self.get_experiment_queue(run_id)
        candidate = queue.get(candidate_id)
        if candidate is None:
            raise KeyError(f"Candidate not found in queue: {candidate_id}")

        profile = self.repository.get_dataset_profile(run.dataset_id)
        if profile is None:
            dataset = self._get_dataset(run.dataset_id)
            profile = profile_dataset(dataset)

        new_priority = self.scorer.score(
            candidate=candidate,
            run=run,
            profile=profile,
            user_priority=priority,
        )
        queue.reprioritize(candidate_id, new_priority)
        self._emit(
            "CandidateReprioritized",
            {
                "candidate_id": candidate_id,
                "priority": priority,
                "effective_score": new_priority.effective_score,
                "is_pinned": new_priority.is_pinned,
            },
            run_id=run.id,
        )
        return new_priority

    def list_candidates(self, run_id: str) -> list[dict]:
        queue = self.get_experiment_queue(run_id)
        return [c.to_dict() for c in queue.list_queued()]

    def execute_next_experiment(
        self,
        run_id: str,
        budget: BudgetPolicy | None = None,
    ) -> tuple[Experiment | None, list[TrialResult]]:
        run = self._get_run(run_id)
        if run.status in {RunStatus.PAUSED, RunStatus.CANCELLED}:
            raise RuntimeError(f"Cannot execute next experiment: Run {run_id} is {run.status.value}")

        queue = self.get_experiment_queue(run_id)
        effective_budget = budget or BudgetPolicy()

        existing_experiments = self.repository.list_experiments(run_id)
        candidate = self.scheduler.select_next(
            queue=queue,
            budget=effective_budget,
            executed_experiments=len(existing_experiments),
        )
        if candidate is None:
            return None, []

        prio_value = candidate.priority.level.value if candidate.priority else "normal"
        experiment = self.create_experiment(
            run=run,
            name=candidate.name,
            feature_names=candidate.feature_names,
            feature_set_id=candidate.feature_set_id,
            model_ids=candidate.model_ids,
            hypothesis=candidate.hypothesis,
            priority=prio_value,
        )

        results = self.run_experiment(run, experiment)
        return experiment, results

    def run_scheduled_experiments(
        self,
        run_id: str,
        max_experiments: int | None = None,
        max_trials: int | None = None,
        budget: BudgetPolicy | None = None,
    ) -> list[dict]:
        run = self._get_run(run_id)
        if budget is not None:
            effective_budget = budget
        else:
            existing = len(self.repository.list_experiments(run_id))
            effective_budget = BudgetPolicy(
                max_experiments=existing + max_experiments if max_experiments is not None else None,
                max_trials=max_trials,
            )
        executed: list[dict] = []


        while len(self.get_experiment_queue(run_id)) > 0:
            refreshed = self._get_run(run_id)
            if refreshed.status in {RunStatus.PAUSED, RunStatus.CANCELLED}:
                break

            if self._is_run_budget_exhausted(refreshed):
                refreshed.config.extra["time_budget_exhausted"] = True
                self.repository.save_run(refreshed)
                self._emit("TimeBudgetExhausted", {"run_id": run_id}, run_id=run_id)
                break

            exp, results = self.execute_next_experiment(run_id, budget=effective_budget)
            if exp is None:
                break

            succeeded = [r for r in results if r.succeeded]
            if succeeded:
                metric_name = getattr(exp, "metric", None) or run.config.metric
                best_score = self._resolve_metric_spec(metric_name).select_best_score(
                    r.primary_score for r in succeeded
                )
            else:
                best_score = 0.0
            executed.append(
                {
                    "experiment_id": exp.id,
                    "name": exp.name,
                    "priority": exp.priority,
                    "trials_count": len(results),
                    "best_score": round(best_score, 4),
                    "succeeded": bool(succeeded),
                }
            )

        return executed

    # --- V0.4 Model & Hyperparameter Optimization ---

    def optimize_experiment(
        self,
        run_id: str,
        experiment_id: str,
        model_id: str | None = None,
        optimizer: str = "optuna",
        n_trials: int = 10,
        timeout_seconds: float | None = None,
        patience: int = 5,
        min_delta: float = 0.0001,
    ) -> dict:
        """Facade: delegates to OptimizationService (Clean Architecture PR2)."""
        return self.optimization_service.optimize_experiment(
            run_id=run_id,
            experiment_id=experiment_id,
            model_id=model_id,
            optimizer=optimizer,
            n_trials=n_trials,
            timeout_seconds=timeout_seconds,
            patience=patience,
            min_delta=min_delta,
        )

    def _metric_greater_is_better(self, metric_name: str) -> bool | None:
        """Plugin override for metric direction, or None to use builtin registry."""
        plugin = self.plugin_registry.get_metric_plugin(metric_name)
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

    def get_best_trial(self, experiment_id: str) -> dict | None:
        results = self.repository.list_trial_results(experiment_id)
        if not results:
            return None
        valid = [r for r in results if r.succeeded]
        if not valid:
            return None

        metric_spec = self._resolve_metric_spec(valid[0].primary_metric)
        best_idx = metric_spec.select_best_index([r.primary_score for r in valid])
        best = valid[best_idx]

        trial = self.repository.get_trial(best.trial_id)
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
        results = self.repository.list_trial_results(experiment_id)
        trials = {r.trial_id: self.repository.get_trial(r.trial_id) for r in results}
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
                "parameters": dict(trials[r.trial_id].parameters) if trials[r.trial_id] else {},
            }
            for r in results
        ]

    # --- V0.5 Feature Discovery & Selection ---

    def select_features(
        self,
        run_id: str,
        strategy: FeatureSelectionStrategy | None = None,
    ) -> list[FeatureSetCandidate]:
        return self.feature_service.select_features(run_id=run_id, strategy=strategy)

    def plan_ablation_experiments(
        self,
        run_id: str,
        base_feature_names: list[str] | None = None,
        model_ids: list[str] | None = None,
        max_features: int = 5,
        auto_enqueue: bool = True,
    ) -> list[ExperimentCandidate]:
        return self.feature_service.plan_ablation_experiments(
            run_id=run_id,
            base_feature_names=base_feature_names,
            model_ids=model_ids,
            max_features=max_features,
            auto_enqueue=auto_enqueue,
        )

    def promote_candidate_feature_set(
        self,
        run_id: str,
        candidate_id: str,
        new_name: str | None = None,
        *,
        allow_unverified_promotion: bool = False,
    ) -> FeatureSet:
        return self.feature_service.promote_candidate_feature_set(
            run_id=run_id,
            candidate_id=candidate_id,
            new_name=new_name,
            allow_unverified_promotion=allow_unverified_promotion,
        )

    def get_feature_evidence(
        self,
        run_id: str,
        feature_id: str | None = None,
    ) -> FeatureEvidence | list[FeatureEvidence] | None:
        return self.feature_service.get_feature_evidence(run_id=run_id, feature_id=feature_id)

    def list_candidate_feature_sets(self, run_id: str) -> list[dict]:
        return self.feature_service.list_candidate_feature_sets(run_id=run_id)

    def get_feature_ranking(self, run_id: str, method: str | None = None) -> list[dict]:
        return self.feature_service.get_feature_ranking(run_id=run_id, method=method)

    # --- V0.6 Plugin System ---

    def list_plugins(
        self,
        plugin_type: str | None = None,
        task_type: str | None = None,
    ) -> list[dict]:
        plugins = self.plugin_registry.list(plugin_type=plugin_type, task_type=task_type)
        items = []
        for p in plugins:
            is_native = getattr(p, "is_native", getattr(p, "is_available", True))
            fallback = getattr(p, "fallback_backend", None)
            if is_native:
                backend_status = "native"
            elif fallback:
                backend_status = f"fallback ({fallback})"
            else:
                backend_status = "unavailable"

            items.append({
                "plugin_id": p.plugin_id,
                "name": p.name,
                "version": p.version,
                "plugin_type": p.plugin_type.value if hasattr(p.plugin_type, "value") else str(p.plugin_type),
                "capabilities": p.capabilities.to_dict(),
                "is_native": bool(is_native),
                "backend_status": backend_status,
                "fallback_backend": fallback,
            })
        return items

    def register_plugin(self, plugin: Any) -> None:
        from automl.domain.plugins.plugin import PluginType
        self.plugin_registry.register(plugin)
        if getattr(plugin, "plugin_type", None) == PluginType.MODEL:
            from automl.domain.models.registry import ModelSpec
            self.model_registry.register(
                ModelSpec(
                    id=plugin.plugin_id,
                    name=plugin.name,
                    task_types=list(plugin.capabilities.supported_tasks),
                )
            )
        self._emit(
            "PluginRegistered",
            {
                "plugin_id": plugin.plugin_id,
                "plugin_type": str(getattr(plugin, "plugin_type", "unknown")),
            },
        )

    def predict(
        self,
        run_id: str,
        test_dataset_path: str | Path,
        experiment_id: str | None = None,
        trial_id: str | None = None,
        predict_proba: bool = False,
        template_path: str | Path | None = None,
        id_column: str | None = None,
    ) -> list[Any]:
        return self.inference_service.predict(
            run_id=run_id,
            test_dataset_path=test_dataset_path,
            experiment_id=experiment_id,
            trial_id=trial_id,
            predict_proba=predict_proba,
            template_path=template_path,
            id_column=id_column,
        )

    def generate_submission(
        self,
        run_id: str,
        test_dataset_path: str | Path,
        output_path: str | Path,
        id_column: str | None = None,
        template_path: str | Path | None = None,
        experiment_id: str | None = None,
        trial_id: str | None = None,
        predict_proba: bool = False,
    ) -> dict[str, Any]:
        return self.inference_service.generate_submission(
            run_id=run_id,
            test_dataset_path=test_dataset_path,
            output_path=output_path,
            id_column=id_column,
            template_path=template_path,
            experiment_id=experiment_id,
            trial_id=trial_id,
            predict_proba=predict_proba,
        )

    def export_model_artifact(
        self,
        run_id: str,
        experiment_id: str | None = None,
        trial_id: str | None = None,
    ) -> Any:
        return self.inference_service.export_model_artifact(
            run_id=run_id,
            experiment_id=experiment_id,
            trial_id=trial_id,
        )

    def _write_submission(
        self,
        run_id: str,
        test_dataset_path: str | Path,
        output_path: str | Path,
        preds: list[Any],
        id_column: str | None = None,
        template_path: str | Path | None = None,
        predict_proba: bool = False,
    ) -> dict[str, Any]:
        return self.inference_service._write_submission(
            run_id=run_id,
            test_dataset_path=test_dataset_path,
            output_path=output_path,
            preds=preds,
            id_column=id_column,
            template_path=template_path,
            predict_proba=predict_proba,
        )

    def generate_oof_submission(self, command) -> str:
        from automl.application.services.oof_submission import generate_oof_submission
        return generate_oof_submission(self, command)

    def get_oof_result(self, run_id: str, experiment_id: str) -> dict:
        from automl.application.services.oof_submission import get_oof_report
        return get_oof_report(self, run_id, experiment_id)

    def validate_pipeline_graph(self, graph: PipelineGraph) -> None:
        """Validates that a pipeline graph is well-formed, acyclic, and modality-consistent."""
        self.pipeline_service.validate_pipeline_graph(graph)

    def get_pipeline_execution_order(self, graph: PipelineGraph) -> list[PipelineNode]:
        """Returns the topologically sorted execution order of nodes in the pipeline graph."""
        return self.pipeline_service.get_pipeline_execution_order(graph)

    def build_multimodal_pipeline(
        self,
        tabular_source_id: str = "tabular_input",
        image_source_id: str = "image_input",
        model_id: str = "random_forest",
        embedding_dim: int = 16,
        pipeline_id: str = "multimodal_pipeline",
        pipeline_name: str = "Multimodal Tabular + Vision Pipeline",
    ) -> PipelineGraph:
        """Constructs a validated standard multimodal DAG with tabular and vision branches."""
        return self.pipeline_service.build_multimodal_pipeline(
            tabular_source_id=tabular_source_id,
            image_source_id=image_source_id,
            model_id=model_id,
            embedding_dim=embedding_dim,
            pipeline_id=pipeline_id,
            pipeline_name=pipeline_name,
        )

    def execute_pipeline(self, graph: PipelineGraph, inputs: dict[str, Any]) -> Any:
        """Executes a pipeline DAG given modal inputs, returning the terminal node's output."""
        return self.pipeline_service.execute_pipeline(graph, inputs)

    def fit_predict_multimodal(
        self,
        graph: PipelineGraph,
        inputs: dict[str, Any],
        target: Any,
        model_id: str = "random_forest",
        task_type: str = "binary_classification",
    ) -> dict[str, Any]:
        """Trains a model on the fused representations produced by a multimodal DAG pipeline."""
        return self.pipeline_service.fit_predict_multimodal(
            graph=graph,
            inputs=inputs,
            target=target,
            model_id=model_id,
            task_type=task_type,
        )

    # --- Meta-Learning & Warm Starts ---

    def get_meta_knowledge(
        self,
        dataset_id: str,
        run_id: str | None = None,
    ) -> Any:
        from automl.engine.meta_learning.extractor import extract_fingerprint
        from automl.engine.meta_learning.knowledge_base import MetaKnowledgeBase

        dataset = self._get_dataset(dataset_id)
        profile = self.repository.get_dataset_profile(dataset_id)
        if profile is None:
            profile = profile_dataset(dataset)
            self.repository.save_dataset_profile(profile)

        df = None
        try:
            df = load_dataframe(dataset.path)
        except Exception:
            pass

        fingerprint = extract_fingerprint(dataset, profile, df=df)

        # Retrieve any empirical trials in the workspace for this dataset
        trials = []
        try:
            runs = self.repository.list_runs()
            for r in runs:
                if r.dataset_id == dataset_id:
                    results = self.repository.list_trial_results(r.id)
                    trials.extend(results)
        except Exception:
            pass

        kb = MetaKnowledgeBase()
        return kb.synthesize(fingerprint, workspace_trials=trials if trials else None)

    # --- Multi-Model Ensemble Builder ---

    def build_ensemble(
        self,
        run_id: str,
        model_ids: list[str],
        method: str = "average",
        meta_model: str = "ridge",
        folds: int = 5,
        name: str | None = None,
    ) -> Experiment:
        run = self._get_run(run_id)
        dataset = self._get_dataset(run.dataset_id)

        valid_methods = {"average", "rank", "simplex", "stacked"}
        method_norm = method.lower()
        if method_norm not in valid_methods:
            raise ValueError(f"Unknown ensemble method '{method}'. Valid: {sorted(valid_methods)}")

        unique_models: list[str] = []
        for m in model_ids:
            if m not in unique_models and m not in {"voting_ensemble", "oof_blend"}:
                unique_models.append(m)

        if len(unique_models) < 2:
            raise ValueError("Building an ensemble requires at least 2 distinct base models.")

        self.model_registry.validate_for_task(unique_models, dataset.task_type)

        registry = self.get_feature_registry(run.dataset_id)
        features = registry.active_feature_names(dataset.target_column)
        if not features:
            profile = self.repository.get_dataset_profile(run.dataset_id)
            if profile:
                features = [c.name for c in profile.columns if not c.is_identifier and c.name != dataset.target_column]
            if not features:
                raise ValueError("No features available to train ensemble.")

        exp_name = name or f"Ensemble ({method_norm.title()} - {len(unique_models)} models)"
        experiment = self.create_experiment(
            run=run,
            name=exp_name,
            feature_names=features,
            model_ids=["voting_ensemble"],
            hypothesis=f"Ensemble {method_norm.upper()} blend of [{', '.join(unique_models)}] with {folds} folds",
            validation_strategy="oof" if dataset.task_type == "binary_classification" else "kfold",
        )
        experiment.transition_to(ExperimentStatus.RUNNING)
        self.repository.save_experiment(experiment)

        df = load_dataframe(dataset.path)
        X = df[features]
        y = df[dataset.target_column]

        def _get_factory(mid: str):
            plugin = self.plugin_registry.get_model_plugin(mid)
            if plugin:
                return lambda: plugin.build_estimator(task_type=dataset.task_type)
            from automl.plugins.models.sklearn_models import build_sklearn_model
            return lambda: build_sklearn_model(mid, dataset.task_type)

        factories = {m: _get_factory(m) for m in unique_models}

        if dataset.task_type == "binary_classification":
            from automl.engine.ensemble.oof import evaluate_oof
            test_slice = X.iloc[:min(len(X), 10)].copy()
            oof_res = evaluate_oof(
                X=X,
                y=y,
                X_test=test_slice,
                factories=factories,
                folds=folds,
                seed=run.config.random_seed,
                method=method_norm,
                meta_model=meta_model,
            )
            best_score = float(oof_res.scores["oof_blend"])
            model_scores = oof_res.scores
            weights = oof_res.weights or {m: 1.0 / len(unique_models) for m in unique_models}
            cv_std = float(np.std(oof_res.fold_scores["oof_blend"]))
        else:
            from sklearn.model_selection import KFold, cross_val_score

            from automl.engine.ensemble.voting import VotingEnsembleEstimator
            from automl.engine.training.sklearn_trainer import (
                _build_pipeline,
                _sklearn_scoring,
            )

            estimators = [(m, factories[m]()) for m in unique_models]
            voting_mode = "soft"
            opt_w = (method_norm == "simplex")
            if method_norm == "rank" and "classification" in dataset.task_type:
                voting_mode = "rank"
            estimator = VotingEnsembleEstimator(
                estimators=estimators,
                task_type=dataset.task_type,
                voting=voting_mode,
                optimize_weights=opt_w,
            )
            pipeline = _build_pipeline(X, estimator)
            cv_splitter = KFold(n_splits=folds, shuffle=True, random_state=run.config.random_seed)
            from automl.engine.training.sklearn_trainer import _is_minimizing_metric

            scoring = _sklearn_scoring(
                run.config.metric, dataset.task_type, self.plugin_registry
            )
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=1,
            )
            # sklearn neg_* scorers return negative values; persist the natural metric scale.
            is_neg = (
                (isinstance(scoring, str) and scoring.startswith("neg_"))
                or _is_minimizing_metric(run.config.metric, self.plugin_registry)
                or getattr(scoring, "_sign", 1) == -1
            )
            if is_neg:
                scores = -scores
            best_score = float(np.mean(scores))
            cv_std = float(np.std(scores))
            model_scores = {"ensemble": best_score}
            weights = {m: 1.0 / len(unique_models) for m in unique_models}

        trial = Trial(
            id=f"trial_{uuid.uuid4().hex[:8]}",
            experiment_id=experiment.id,
            model_id="voting_ensemble",
            seed=run.config.random_seed,
            parameters={
                "models": unique_models,
                "method": method_norm,
                "meta_model": meta_model,
                "folds": folds,
                "weights": weights,
            },
            status=TrialStatus.COMPLETED,
        )
        self.repository.save_trial(trial)

        trial_result = TrialResult(
            trial_id=trial.id,
            experiment_id=experiment.id,
            model_id="voting_ensemble",
            primary_metric=run.config.metric,
            primary_score=best_score,
            secondary_metrics={
                "method": method_norm,
                "meta_model": meta_model,
                "folds": folds,
                "weights": weights,
                "model_scores": model_scores,
                "cv_std": cv_std,
            },
        )
        self.repository.save_trial_result(trial_result)

        experiment.transition_to(ExperimentStatus.COMPLETED)
        self.repository.save_experiment(experiment)

        self._emit(
            "EnsembleBuilt",
            {
                "experiment_id": experiment.id,
                "score": best_score,
                "models": unique_models,
                "method": method_norm,
            },
            run_id=run.id,
        )
        return experiment




