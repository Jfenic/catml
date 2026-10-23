"""Application service for experiment planning, candidate prioritization, and scheduling."""
from __future__ import annotations

from typing import TYPE_CHECKING

from automl.domain.experiments.candidate import ExperimentCandidate
from automl.domain.experiments.priority import Priority
from automl.domain.experiments.trial import Experiment, TrialResult
from automl.domain.policies.budget import BudgetPolicy
from automl.domain.runs.states import RunPhase, RunStatus
from automl.engine.priority.scheduler import ExperimentQueue
from automl.engine.profiling.dataset_profiler import profile_dataset

if TYPE_CHECKING:
    from automl.application.services.workspace import AutoMLWorkspace


class ExperimentPlanningService:
    """Application service coordinating experiment planning, priority queues, and scheduled execution."""

    def __init__(self, workspace: AutoMLWorkspace) -> None:
        self.workspace = workspace

    def get_experiment_queue(self, run_id: str) -> ExperimentQueue:
        if run_id not in self.workspace._queues:
            self.workspace._queues[run_id] = ExperimentQueue()
        return self.workspace._queues[run_id]

    def plan_experiments(
        self,
        run_id: str,
        auto_enqueue: bool = True,
        user_priorities: dict[str, str] | None = None,
    ) -> list[ExperimentCandidate]:
        ws = self.workspace
        run = ws._get_run(run_id)
        if run.status == RunStatus.CANCELLED:
            raise RuntimeError(f"Run {run_id} is cancelled")

        run.transition_to(RunStatus.PLANNING, RunPhase.EXPERIMENT_PLANNING)
        ws.repository.save_run(run)

        profile = ws.repository.get_dataset_profile(run.dataset_id)
        if profile is None:
            dataset = ws._get_dataset(run.dataset_id)
            profile = profile_dataset(dataset)
            ws.repository.save_dataset_profile(profile)

        feature_registry = ws.get_feature_registry(run.dataset_id)
        history = ws.repository.get_leaderboard(run.id)

        candidates = ws.planner.propose(
            run=run,
            profile=profile,
            feature_registry=feature_registry,
            model_registry=ws.model_registry,
            history=history,
        )

        priorities_map = user_priorities or {}
        for candidate in candidates:
            user_prio = (
                priorities_map.get(candidate.id)
                or priorities_map.get(candidate.name)
                or None
            )
            score = ws.scorer.score(
                candidate=candidate,
                run=run,
                profile=profile,
                user_priority=user_prio,
            )
            candidate.priority = score

        if auto_enqueue:
            queue = self.get_experiment_queue(run.id)
            queue.enqueue_all(candidates)

        ws._emit(
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
        ws = self.workspace
        run = ws._get_run(run_id)
        queue = self.get_experiment_queue(run_id)
        candidate = queue.get(candidate_id)
        if candidate is None:
            raise KeyError(f"Candidate not found in queue: {candidate_id}")

        profile = ws.repository.get_dataset_profile(run.dataset_id)
        if profile is None:
            dataset = ws._get_dataset(run.dataset_id)
            profile = profile_dataset(dataset)

        new_priority = ws.scorer.score(
            candidate=candidate,
            run=run,
            profile=profile,
            user_priority=priority,
        )
        queue.reprioritize(candidate_id, new_priority)
        ws._emit(
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
        ws = self.workspace
        run = ws._get_run(run_id)
        if run.status in {RunStatus.PAUSED, RunStatus.CANCELLED}:
            raise RuntimeError(f"Cannot execute next experiment: Run {run_id} is {run.status.value}")

        queue = ws.get_experiment_queue(run_id)
        effective_budget = budget or BudgetPolicy()

        existing_experiments = ws.repository.list_experiments(run_id)
        candidate = ws.scheduler.select_next(
            queue=queue,
            budget=effective_budget,
            executed_experiments=len(existing_experiments),
        )
        if candidate is None:
            return None, []

        prio_value = candidate.priority.level.value if candidate.priority else "normal"
        experiment = ws.create_experiment(
            run=run,
            name=candidate.name,
            feature_names=candidate.feature_names,
            feature_set_id=candidate.feature_set_id,
            model_ids=candidate.model_ids,
            hypothesis=candidate.hypothesis,
            priority=prio_value,
        )

        results = ws.run_experiment(run, experiment)
        return experiment, results

    def run_scheduled_experiments(
        self,
        run_id: str,
        max_experiments: int | None = None,
        max_trials: int | None = None,
        budget: BudgetPolicy | None = None,
    ) -> list[dict]:
        ws = self.workspace
        run = ws._get_run(run_id)
        if budget is not None:
            effective_budget = budget
        else:
            existing = len(ws.repository.list_experiments(run_id))
            effective_budget = BudgetPolicy(
                max_experiments=existing + max_experiments if max_experiments is not None else None,
                max_trials=max_trials,
            )
        executed: list[dict] = []

        while len(ws.get_experiment_queue(run_id)) > 0:
            refreshed = ws._get_run(run_id)
            if refreshed.status in {RunStatus.PAUSED, RunStatus.CANCELLED}:
                break

            if ws._is_run_budget_exhausted(refreshed):
                refreshed.config.extra["time_budget_exhausted"] = True
                ws.repository.save_run(refreshed)
                ws._emit("TimeBudgetExhausted", {"run_id": run_id}, run_id=run_id)
                break

            exp, results = ws.execute_next_experiment(run_id, budget=effective_budget)
            if exp is None:
                break

            succeeded = [r for r in results if r.succeeded]
            if succeeded:
                metric_name = getattr(exp, "metric", None) or run.config.metric
                best_score = ws._resolve_metric_spec(metric_name).select_best_score(
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
