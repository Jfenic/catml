from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from automl.domain.datasets.profile import DatasetProfile
from automl.domain.experiments.candidate import ExperimentCandidate
from automl.domain.experiments.priority import ExperimentPriority, Priority
from automl.domain.experiments.trial import Experiment, Trial, TrialResult
from automl.domain.features.evidence import FeatureEvidence, FeatureInteractionEvidence
from automl.domain.features.registry import FeatureRegistry
from automl.domain.features.selection_strategy import (
    FeatureRank,
    FeatureSetCandidate,
)
from automl.domain.jobs.job import Job, JobStatus
from automl.domain.modalities.modality import DataSource, Modality
from automl.domain.models.registry import ModelRegistry
from automl.domain.optimization.search_space import SearchSpace
from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.domain.runs.run import AutoMLRun


@dataclass
class TrialExecution:
    trial: Trial
    experiment: Experiment
    run: AutoMLRun
    feature_names: list[str]
    dataset_path: str
    target_column: str
    task_type: str
    metric: str
    validation_strategy: str
    test_size: float
    cv_folds: int
    random_seed: int
    group_column: str | None = None
    column_transforms: list[dict[str, Any]] = field(default_factory=list)
    derived_features: list[dict[str, Any]] = field(default_factory=list)
    dataset_content_hash: str | None = None
    # Optional HPO hook: (fold_index, fold_score) -> True to prune early (MedianPruner telemetry).
    # Kept as Any so domain stays free of Optuna/sklearn imports.
    fold_callback: Any | None = None


class TrainerPort(Protocol):
    def run(self, execution: TrialExecution) -> TrialResult:
        ...

    def fit_and_predict(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        model_id: str,
        task_type: str = "binary_classification",
        parameters: dict[str, Any] | None = None,
        predict_proba: bool = False,
    ) -> Any:
        ...


class EvaluatorPort(Protocol):
    def evaluate(self, execution: TrialExecution, predictions: Any, y_true: Any) -> dict[str, float]:
        ...


class ExperimentRepositoryPort(Protocol):
    def save_run(self, run: AutoMLRun) -> None:
        ...

    def get_run(self, run_id: str) -> AutoMLRun | None:
        ...

    def save_dataset_profile(self, profile: DatasetProfile) -> None:
        ...

    def get_dataset_profile(self, dataset_id: str) -> DatasetProfile | None:
        ...

    def save_experiment(self, experiment: Experiment) -> None:
        ...

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        ...

    def list_experiments(self, run_id: str) -> list[Experiment]:
        ...

    def save_trial(self, trial: Trial) -> None:
        ...

    def get_trial(self, trial_id: str) -> Trial | None:
        ...

    def save_trial_result(self, result: TrialResult) -> None:

        ...

    def list_trial_results(self, experiment_id: str) -> list[TrialResult]:
        ...

    def get_leaderboard(self, run_id: str, include_failed: bool = False) -> list[TrialResult]:
        ...


class ExperimentPlannerPort(Protocol):
    def propose(
        self,
        run: AutoMLRun,
        profile: DatasetProfile,
        feature_registry: FeatureRegistry,
        model_registry: ModelRegistry,
        history: list[TrialResult] | None = None,
    ) -> list[ExperimentCandidate]:
        ...


class PriorityScorerPort(Protocol):
    def score(
        self,
        candidate: ExperimentCandidate,
        run: AutoMLRun,
        profile: DatasetProfile,
        user_priority: ExperimentPriority | str | None = None,
    ) -> Priority:
        ...


class OptimizerPort(Protocol):
    def suggest(self, trial_number: int, search_space: SearchSpace) -> dict[str, Any]:
        ...

    def observe(
        self,
        trial_number: int,
        parameters: dict[str, Any],
        score: float,
        succeeded: bool = True,
        pruned: bool = False,
    ) -> None:
        ...

    def should_stop(self) -> bool:
        ...

    def best_parameters(self) -> dict[str, Any]:
        ...

    def best_score(self) -> float:
        ...


@dataclass
class FeatureAnalysisContext:
    dataset_path: str
    target_column: str
    task_type: str
    active_feature_names: list[str]
    random_seed: int = 42


class FeatureSelectorPort(Protocol):
    method_id: str
    selector_type: str  # "filter" | "wrapper" | "embedded"

    def fit(self, context: FeatureAnalysisContext) -> None:
        ...

    def rank_features(self) -> list[FeatureRank]:
        ...

    def select(self, k: int) -> FeatureSetCandidate:
        ...


class DimensionalityReducerPort(Protocol):
    method_id: str

    def fit(self, context: FeatureAnalysisContext, n_components: float | None = None) -> None:
        ...

    def transform(self, data: Any) -> Any:
        ...

    def n_components(self) -> int:
        ...

    def explained_variance_ratio(self) -> list[float]:
        ...


class FeatureEvidenceRepositoryPort(Protocol):
    def save_feature_evidence(self, run_id: str, evidence: FeatureEvidence) -> None:
        ...

    def get_feature_evidence(self, run_id: str, feature_id: str) -> FeatureEvidence | None:
        ...

    def list_feature_evidence(self, run_id: str) -> list[FeatureEvidence]:
        ...

    def save_interaction_evidence(self, run_id: str, evidence: FeatureInteractionEvidence) -> None:
        ...

    def list_interaction_evidence(self, run_id: str) -> list[FeatureInteractionEvidence]:
        ...


@runtime_checkable
class PluginPort(Protocol):
    plugin_id: str
    name: str
    version: str
    plugin_type: PluginType
    capabilities: PluginCapability


@runtime_checkable
class ModelPluginPort(PluginPort, Protocol):
    def build_estimator(self, parameters: dict[str, Any] | None = None) -> Any:
        ...

    def get_search_space(self, task_type: str) -> SearchSpace:
        ...


@runtime_checkable
class MetricPluginPort(PluginPort, Protocol):
    greater_is_better: bool

    def compute(self, y_true: Any, y_pred: Any, y_prob: Any | None = None) -> float:
        ...


@runtime_checkable
class PreprocessorPluginPort(PluginPort, Protocol):
    def fit_transform(self, X: Any, y: Any | None = None) -> Any:
        ...

    def transform(self, X: Any) -> Any:
        ...


@runtime_checkable
class ModalityPluginPort(PluginPort, Protocol):
    modality: Modality

    def validate_source(self, source: DataSource) -> bool:
        ...

    def load_data(self, source: DataSource) -> Any:
        ...







class JobRepositoryPort(Protocol):
    """Atomic state transitions and claims for durable local jobs."""
    def enqueue(self, job: Job) -> str: ...
    def get(self, job_id: str) -> Job: ...
    def list(self, run_id: str | None = None) -> list[Job]: ...
    def change(self, job_id: str, allowed: set[JobStatus], **changes: Any) -> Job: ...
    def finish(self, job_id: str, status: JobStatus, **changes: Any) -> Job: ...
    def claim(self) -> Job | None: ...
    def recover(self) -> None: ...


class ExperimentRecipeRepositoryPort(Protocol):
    """Persistence port for ExperimentRecipe entities."""
    def save(self, recipe: Any) -> None: ...
    def get(self, recipe_id: str) -> Any | None: ...
    def list(self, dataset_id: str | None = None, status: str | None = None) -> list[Any]: ...
    def delete(self, recipe_id: str) -> None: ...

