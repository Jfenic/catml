from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from automl import __version__
from automl.application.plugins.registry import PluginRegistry
from automl.domain.datasets.profile import Dataset, DatasetProfile
from automl.domain.experiments.candidate import ExperimentCandidate
from automl.domain.experiments.priority import Priority
from automl.domain.experiments.trial import (
    Experiment,
    Trial,
    TrialResult,
)
from automl.domain.features.evidence import FeatureEvidence
from automl.domain.features.feature_set import FeatureSet
from automl.domain.features.registry import FeatureRegistry
from automl.domain.features.selection_strategy import (
    FeatureRank,
    FeatureSelectionStrategy,
    FeatureSetCandidate,
)
from automl.domain.metrics.spec import MetricSpec
from automl.domain.models.registry import ModelRegistry
from automl.domain.pipelines.graph import PipelineGraph, PipelineNode
from automl.domain.policies.budget import BudgetPolicy
from automl.domain.ports import (
    ExperimentPlannerPort,
    PriorityScorerPort,
)
from automl.domain.runs.run import AutoMLRun
from automl.domain.tasks.problem_definition import ProblemDefinition
from automl.domain.tasks.task_type import TASK_CATALOG, TaskType
from automl.engine.planning.experiment_planner import RuleBasedExperimentPlanner
from automl.engine.priority.scheduler import ExperimentQueue, Scheduler
from automl.engine.priority.scorer import RuleBasedPriorityScorer
from automl.engine.profiling.dataset_profiler import profile_dataset
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
    _dataset_service: Any = field(default=None, repr=False)
    _run_service: Any = field(default=None, repr=False)
    _experiment_service: Any = field(default=None, repr=False)
    _planning_service: Any = field(default=None, repr=False)
    _ensemble_service: Any = field(default=None, repr=False)
    _cached_study_repo: Any = field(default=None, repr=False)
    _cached_recipe_repo: Any = field(default=None, repr=False)

    @property
    def dataset_service(self):
        if self._dataset_service is None:
            from automl.application.services.dataset_service import DatasetService

            self._dataset_service = DatasetService(self)
        return self._dataset_service

    @property
    def run_service(self):
        if self._run_service is None:
            from automl.application.services.run_service import RunService

            self._run_service = RunService(self)
        return self._run_service

    @property
    def experiment_service(self):
        if self._experiment_service is None:
            from automl.application.services.experiment_service import ExperimentService

            self._experiment_service = ExperimentService(self)
        return self._experiment_service

    @property
    def planning_service(self):
        if self._planning_service is None:
            from automl.application.services.experiment_planning_service import (
                ExperimentPlanningService,
            )

            self._planning_service = ExperimentPlanningService(self)
        return self._planning_service

    @property
    def ensemble_service(self):
        if self._ensemble_service is None:
            from automl.application.services.ensemble_service import EnsembleService

            self._ensemble_service = EnsembleService(self)
        return self._ensemble_service

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
        return self.run_service._get_or_create_run_deadline(run)

    def _is_run_budget_exhausted(self, run: AutoMLRun) -> bool:
        return self.run_service._is_run_budget_exhausted(run)

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
        self.dataset_service._hydrate_feature_registry(dataset_id)

    def _emit(self, event_type: str, payload: dict, run_id: str | None = None) -> None:
        self.repository.append_event(event_type, payload, run_id=run_id)

    def _get_run(self, run_id: str) -> AutoMLRun:
        return self.run_service._get_run(run_id)

    def get_run(self, run_id: str) -> AutoMLRun:
        return self.run_service.get_run(run_id)

    def _get_dataset(self, dataset_id: str) -> Dataset:
        return self.dataset_service._get_dataset(dataset_id)

    def get_dataset(self, dataset_id: str) -> Dataset:
        return self.dataset_service.get_dataset(dataset_id)

    def list_datasets(self) -> list[Dataset]:
        return self.dataset_service.list_datasets()

    def get_dataset_profile(self, dataset_id: str) -> DatasetProfile | None:
        return self.dataset_service.get_dataset_profile(dataset_id)

    def save_dataset_profile(self, profile: DatasetProfile) -> None:
        self.dataset_service.save_dataset_profile(profile)

    def save_run(self, run: AutoMLRun) -> None:
        self.run_service.save_run(run)

    def get_trial(self, trial_id: str) -> Trial | None:
        return self.experiment_service.get_trial(trial_id)

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        return self.experiment_service.get_experiment(experiment_id)

    def list_experiments(self, run_id: str) -> list[Experiment]:
        return self.experiment_service.list_experiments(run_id)

    def list_trial_results(self, experiment_id: str) -> list[TrialResult]:
        return self.experiment_service.list_trial_results(experiment_id)

    def get_leaderboard_results(self, run_id: str, include_failed: bool = False) -> list[TrialResult]:
        return self.experiment_service.get_leaderboard_results(run_id, include_failed=include_failed)

    def list_feature_sets(self, dataset_id: str) -> list[Any]:
        return self.repository.list_feature_sets(dataset_id)

    def register_dataset(
        self,
        name: str,
        path: str | Path | Any,
        target: str | None = None,
        task_type: str | None = None,
    ) -> Dataset:
        return self.dataset_service.register_dataset(
            name=name, path=path, target=target, task_type=task_type
        )

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
        return self.run_service.create_run(
            dataset=dataset,
            metric=metric,
            validation_strategy=validation_strategy,
            group_column=group_column,
            cv_folds=cv_folds,
            random_seed=random_seed,
            time_budget_seconds=time_budget_seconds,
            extra=extra,
        )

    def list_runs(self, dataset_id: str | None = None) -> list[AutoMLRun]:
        return self.run_service.list_runs(dataset_id=dataset_id)

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
        return self.experiment_service.create_experiment(
            run=run,
            name=name,
            feature_names=feature_names,
            feature_set_id=feature_set_id,
            model_ids=model_ids,
            hypothesis=hypothesis,
            priority=priority,
            validation_strategy=validation_strategy,
            group_column=group_column,
            allow_leakage=allow_leakage,
            column_transforms=column_transforms,
            derived_features=derived_features,
            recipe_id=recipe_id,
        )

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
        return self.run_service.resume_run(run_id)

    def pause_run(self, run_id: str) -> AutoMLRun:
        return self.run_service.pause_run(run_id)

    def cancel_run(self, run_id: str) -> AutoMLRun:
        return self.run_service.cancel_run(run_id)

    def clone_run(self, run_id: str, new_name: str | None = None) -> AutoMLRun:
        return self.run_service.clone_run(run_id, new_name=new_name)

    def leaderboard(self, run: AutoMLRun, include_failed: bool = False) -> list[dict]:
        return self.experiment_service.leaderboard(run, include_failed=include_failed)

    def get_leaderboard(self, run_or_id: str | AutoMLRun, include_failed: bool = False) -> list[dict]:
        return self.experiment_service.get_leaderboard(run_or_id, include_failed=include_failed)

    def compare_experiments(self, experiment_ids: list[str]) -> list[dict]:
        return self.experiment_service.compare_experiments(experiment_ids)

    def list_models(self, run_id: str | None = None, task_type: str | None = None) -> list[dict]:
        return self.run_service.list_models(run_id=run_id, task_type=task_type)

    def get_task_plan(self, dataset_id: str) -> ProblemDefinition:
        return self.dataset_service.get_task_plan(dataset_id)

    def get_dataset_questionnaire(
        self,
        dataset_id: str,
        auto_generate: bool = True,
        llm_provider: Any = None,
    ) -> Any | None:
        """Retrieve dataset framing questionnaire, optionally generating it if not found."""
        return self.dataset_service.get_dataset_questionnaire(
            dataset_id=dataset_id, auto_generate=auto_generate, llm_provider=llm_provider
        )

    def generate_dataset_questionnaire(
        self,
        dataset_id: str,
        llm_provider: Any = None,
    ) -> Any:
        """Synthesize and persist a problem context questionnaire for the dataset."""
        return self.dataset_service.generate_dataset_questionnaire(
            dataset_id=dataset_id, llm_provider=llm_provider
        )

    def save_dataset_questionnaire(self, questionnaire: Any) -> None:
        """Explicitly persist an edited or reviewed questionnaire."""
        self.dataset_service.save_dataset_questionnaire(questionnaire)

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
        self.run_service.exclude_model(run, model_id)

    def include_model(self, run: AutoMLRun, model_id: str) -> None:
        self.run_service.include_model(run, model_id)

    # --- V0.3 Experiment Planner & Priority Engine ---

    def get_experiment_queue(self, run_id: str) -> ExperimentQueue:
        return self.planning_service.get_experiment_queue(run_id)

    def plan_experiments(
        self,
        run_id: str,
        auto_enqueue: bool = True,
        user_priorities: dict[str, str] | None = None,
    ) -> list[ExperimentCandidate]:
        return self.planning_service.plan_experiments(
            run_id=run_id, auto_enqueue=auto_enqueue, user_priorities=user_priorities
        )

    def prioritize_candidate(
        self,
        run_id: str,
        candidate_id: str,
        priority: str,
    ) -> Priority:
        return self.planning_service.prioritize_candidate(
            run_id=run_id, candidate_id=candidate_id, priority=priority
        )

    def list_candidates(self, run_id: str) -> list[dict]:
        return self.planning_service.list_candidates(run_id)

    def execute_next_experiment(
        self,
        run_id: str,
        budget: BudgetPolicy | None = None,
    ) -> tuple[Experiment | None, list[TrialResult]]:
        return self.planning_service.execute_next_experiment(run_id, budget=budget)

    def run_scheduled_experiments(
        self,
        run_id: str,
        max_experiments: int | None = None,
        max_trials: int | None = None,
        budget: BudgetPolicy | None = None,
    ) -> list[dict]:
        return self.planning_service.run_scheduled_experiments(
            run_id=run_id,
            max_experiments=max_experiments,
            max_trials=max_trials,
            budget=budget,
        )

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
        return self.experiment_service._metric_greater_is_better(metric_name)

    def _resolve_metric_spec(self, metric_name: str) -> MetricSpec:
        """Resolve MetricSpec via plugin direction when available, else builtins."""
        return self.experiment_service._resolve_metric_spec(metric_name)

    def _is_minimizing_metric(self, metric_name: str) -> bool:
        """Prefer plugin-declared direction; fall back to builtin MetricSpec registry."""
        return self.experiment_service._is_minimizing_metric(metric_name)

    def get_best_trial(self, experiment_id: str) -> dict | None:
        return self.experiment_service.get_best_trial(experiment_id)

    def get_experiment_trials(self, experiment_id: str) -> list[dict]:
        return self.experiment_service.get_experiment_trials(experiment_id)

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
        return self.dataset_service.get_meta_knowledge(dataset_id=dataset_id, run_id=run_id)

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
        return self.ensemble_service.build_ensemble(
            run_id=run_id,
            model_ids=model_ids,
            method=method,
            meta_model=meta_model,
            folds=folds,
            name=name,
        )
