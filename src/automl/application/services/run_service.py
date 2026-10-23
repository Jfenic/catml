"""Application service for AutoML runs lifecycle, budgets, and model scoping."""
from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from automl.domain.datasets.profile import Dataset
from automl.domain.experiments.trial import TrialResult
from automl.domain.runs.run import AutoMLRun, RunConfig
from automl.domain.runs.states import RunPhase, RunStatus
from automl.engine.planning.task_planner import plan_from_dataframe
from automl.engine.profiling.dataset_profiler import load_dataframe

if TYPE_CHECKING:
    from automl.application.services.workspace import AutoMLWorkspace


class RunService:
    """Application service for managing AutoML runs lifecycle, budgets, and state transitions."""

    def __init__(self, workspace: AutoMLWorkspace) -> None:
        self.workspace = workspace

    def _get_or_create_run_deadline(self, run: AutoMLRun) -> float | None:
        budget = run.config.effective_time_budget
        if budget is None or budget <= 0:
            return None
        if run.id not in self.workspace._run_deadlines:
            self.workspace._run_deadlines[run.id] = time.monotonic() + budget
        return self.workspace._run_deadlines[run.id]

    def _is_run_budget_exhausted(self, run: AutoMLRun) -> bool:
        deadline = self._get_or_create_run_deadline(run)
        if deadline is None:
            return False
        return time.monotonic() >= deadline

    def _get_run(self, run_id: str) -> AutoMLRun:
        ws = self.workspace
        run = ws._runs.get(run_id) or ws.repository.get_run(run_id)
        if run is None:
            raise KeyError(f"Run not found: {run_id}")
        ws._runs[run_id] = run
        return run

    def get_run(self, run_id: str) -> AutoMLRun:
        return self._get_run(run_id)

    def list_runs(self, dataset_id: str | None = None) -> list[AutoMLRun]:
        runs = self.workspace.repository.list_runs(workspace_id=self.workspace.id)
        if dataset_id:
            return [r for r in runs if r.dataset_id == dataset_id]
        return runs

    def save_run(self, run: AutoMLRun) -> None:
        self.workspace.repository.save_run(run)

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
        ws = self.workspace
        problem = ws.repository.get_problem_definition(dataset.id)
        if problem is None:
            df = load_dataframe(dataset.path)
            problem = plan_from_dataframe(dataset, df)
            ws.repository.save_problem_definition(problem)

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
            target=dataset.target_column or "",
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
            workspace_id=ws.id,
            dataset_id=dataset.id,
            config=config,
            status=RunStatus.CREATED,
            current_phase=RunPhase.DATASET_PROFILING,
        )
        ws._runs[run.id] = run
        ws.repository.save_run(run)
        ws._emit("RunCreated", {"run_id": run.id, "dataset_id": dataset.id}, run_id=run.id)
        return run

    def resume_run(self, run_id: str) -> list[TrialResult]:
        ws = self.workspace
        run = self._get_run(run_id)
        if run.status not in {RunStatus.PAUSED, RunStatus.EXPERIMENTING}:
            raise RuntimeError(f"Run {run_id} is not resumable (status={run.status.value})")

        checkpoint = ws.repository.get_checkpoint(run_id)
        if checkpoint is None:
            raise RuntimeError(f"No checkpoint found for run {run_id}")

        experiment_id, start_index = checkpoint
        experiment = ws.repository.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(f"Experiment not found: {experiment_id}")

        run.transition_to(RunStatus.EXPERIMENTING, RunPhase.EXPERIMENT_EXECUTION)
        ws.repository.save_run(run)
        ws._emit("RunResumed", {"experiment_id": experiment_id, "start_index": start_index}, run_id=run_id)
        return ws.run_experiment(run, experiment, start_index=start_index)

    def pause_run(self, run_id: str) -> AutoMLRun:
        ws = self.workspace
        run = self._get_run(run_id)
        run.transition_to(RunStatus.PAUSED, RunPhase.EXPERIMENT_EXECUTION)
        ws.repository.save_run(run)
        ws._emit("RunPauseRequested", {}, run_id=run_id)
        return run

    def cancel_run(self, run_id: str) -> AutoMLRun:
        ws = self.workspace
        run = self._get_run(run_id)
        run.transition_to(RunStatus.CANCELLED, run.current_phase)
        ws.repository.save_run(run)
        ws.repository.clear_checkpoint(run_id)
        ws._emit("RunCancelled", {}, run_id=run_id)
        return run

    def clone_run(self, run_id: str, new_name: str | None = None) -> AutoMLRun:
        ws = self.workspace
        source = self._get_run(run_id)
        ws._get_dataset(source.dataset_id)  # ensure dataset still exists
        cloned_config = RunConfig.from_dict(source.config.to_dict())
        cloned = AutoMLRun(
            id=f"run_{uuid.uuid4().hex[:8]}",
            workspace_id=ws.id,
            dataset_id=source.dataset_id,
            config=cloned_config,
            status=RunStatus.CREATED,
            current_phase=RunPhase.EXPERIMENT_PLANNING,
        )
        ws._runs[cloned.id] = cloned
        ws.repository.save_run(cloned)
        ws._emit(
            "RunCloned",
            {"source_run_id": run_id, "new_run_id": cloned.id, "name": new_name or cloned.id},
            run_id=cloned.id,
        )
        return cloned

    def exclude_model(self, run: AutoMLRun, model_id: str) -> None:
        self.workspace.model_registry.validate_for_task([model_id], run.config.task_type)
        if model_id not in run.config.models_exclude:
            run.config.models_exclude.append(model_id)
        self.workspace.repository.save_run(run)
        self.workspace._emit("ModelExcluded", {"model_id": model_id}, run_id=run.id)

    def include_model(self, run: AutoMLRun, model_id: str) -> None:
        self.workspace.model_registry.validate_for_task([model_id], run.config.task_type)
        if model_id not in run.config.models_include:
            run.config.models_include.append(model_id)
        if model_id in run.config.models_exclude:
            run.config.models_exclude.remove(model_id)
        self.workspace.repository.save_run(run)
        self.workspace._emit("ModelAdded", {"model_id": model_id}, run_id=run.id)

    def list_models(self, run_id: str | None = None, task_type: str | None = None) -> list[dict]:
        ws = self.workspace
        if run_id:
            run = self._get_run(run_id)
            resolved_task = task_type or run.config.task_type
            active = ws.model_registry.resolve_active(
                run.config.models_include,
                run.config.models_exclude,
                task_type=resolved_task,
            )
            return [
                {
                    "id": spec.id,
                    "name": spec.name,
                    "active": spec.id in active,
                    "excluded": spec.id in run.config.models_exclude,
                    "compatible": resolved_task in spec.task_types,
                    "task_types": spec.task_types,
                }
                for spec in ws.model_registry.list(resolved_task)
            ]
        return [
            {
                "id": spec.id,
                "name": spec.name,
                "task_types": spec.task_types,
                "compatible": task_type in spec.task_types if task_type else True,
            }
            for spec in ws.model_registry.list(task_type)
        ]
