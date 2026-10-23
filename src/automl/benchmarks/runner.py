from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    CreateFeatureSetCommand,
    ExcludeFeatureCommand,
    ExcludeModelCommand,
    OptimizeExperimentCommand,
    PlanExperimentsCommand,
    PrioritizeFeatureCommand,
    PromoteCandidateFeatureSetCommand,
    RunExperimentCommand,
    SelectFeaturesCommand,
)
from automl.application.services.workspace import PLATFORM_VERSION, AutoMLWorkspace

from automl.domain.experiments.trial import TrialResult


@dataclass
class BenchmarkScenario:
    id: str
    version: str
    description: str
    setup: Callable[[AutoMLWorkspace, object, object], tuple[str, str]]
    metric: str = "roc_auc"
    executor: Callable[[AutoMLWorkspace, object, object, str, str], list[TrialResult]] | None = None


def is_minimizing_metric(metric: str, ws: AutoMLWorkspace | None = None) -> bool:
    """Returns True if the metric is minimizing (lower is better, e.g. loss/error metrics)."""
    from automl.domain.metrics.exceptions import InvalidMetric
    from automl.domain.metrics.spec import resolve_metric_spec

    greater: bool | None = None
    if ws is not None and hasattr(ws, "plugin_registry") and ws.plugin_registry is not None:
        plugin = ws.plugin_registry.get_metric_plugin(metric)
        if plugin is not None and hasattr(plugin, "greater_is_better"):
            greater = bool(plugin.greater_is_better)
    try:
        return resolve_metric_spec(metric, greater_is_better=greater).is_minimizing
    except InvalidMetric:
        return False


def _best_result(
    results: list[TrialResult],
    metric: str = "roc_auc",
    ws: AutoMLWorkspace | None = None,
) -> tuple[float | None, str | None]:
    """Return best score among successful trials, or (None, None) if all failed.

    Failed evaluations must not be reported as score 0.0 (ambiguous with a real zero).
    """
    ok = [r for r in results if r.succeeded]
    if not ok:
        return None, None
    if is_minimizing_metric(metric, ws=ws):
        best = min(ok, key=lambda r: r.primary_score)
    else:
        best = max(ok, key=lambda r: r.primary_score)
    return best.primary_score, best.model_id


@dataclass
class BenchmarkRunner:
    dataset_path: str
    target: str = "churn"
    workspace_root: str | None = None

    def scenarios(self) -> list[BenchmarkScenario]:
        return [
            BenchmarkScenario(
                id="baseline_all_features",
                version="0.1",
                description="All features, all models — V0.1 baseline",
                setup=self._setup_baseline,
            ),
            BenchmarkScenario(
                id="exclude_leaky_id",
                version="0.2",
                description="Exclude customer_id (non-predictive identifier)",
                setup=self._setup_exclude_id,
            ),
            BenchmarkScenario(
                id="financial_feature_set",
                version="0.2",
                description="Curated FeatureSet: salary, debt, account_balance, tenure, age",
                setup=self._setup_financial_feature_set,
            ),
            BenchmarkScenario(
                id="curated_models",
                version="0.2",
                description="Exclude slow SVC; keep logistic_regression + random_forest",
                setup=self._setup_curated_models,
            ),
            BenchmarkScenario(
                id="full_v02_pipeline",
                version="0.2",
                description="Exclude ID + financial FeatureSet + curated models + priorities",
                setup=self._setup_full_pipeline,
            ),
            BenchmarkScenario(
                id="automated_planning_v03",
                version="0.3",
                description="RuleBasedExperimentPlanner + Priority Engine + Scheduler — V0.3",
                setup=self._setup_automated_v03,
            ),
            BenchmarkScenario(
                id="optuna_optimization_v04",
                version="0.4",
                description="Bayesian Hyperparameter Optimization (Optuna TPE) — V0.4",
                setup=self._setup_optuna_v04,
                executor=self._execute_optuna_v04,
            ),
            BenchmarkScenario(
                id="feature_selection_v05",
                version="0.5",
                description="Feature Discovery & Selection (mutual information & tree ranking) — V0.5",
                setup=self._setup_feature_selection_v05,
            ),
        ]


    def _register_base(self, ws: AutoMLWorkspace):
        dataset = ws.register_dataset(
            name="customers_churn",
            path=self.dataset_path,
            target=self.target,
            task_type="binary_classification",
        )
        run = ws.create_run(dataset, metric="roc_auc")
        return dataset, run

    def _setup_baseline(self, ws, _cmd, _qry):
        dataset, run = self._register_base(ws)
        exp = ws.create_experiment(
            run,
            name="baseline_all",
            model_ids=["logistic_regression", "random_forest", "svc"],
        )
        return run.id, exp.id

    def _setup_exclude_id(self, ws, cmd, _qry):
        dataset, run = self._register_base(ws)
        cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
        exp = ws.create_experiment(
            run,
            name="exclude_id",
            model_ids=["logistic_regression", "random_forest", "svc"],
        )
        return run.id, exp.id

    def _setup_financial_feature_set(self, ws, cmd, _qry):
        dataset, run = self._register_base(ws)
        cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
        fset = cmd.dispatch(
            CreateFeatureSetCommand(
                dataset_id=dataset.id,
                name="financial_core",
                feature_names=["salary", "debt", "account_balance", "tenure", "age"],
                lineage="benchmark_v02",
            )
        )
        exp = ws.create_experiment(
            run,
            name="financial_set",
            feature_set_id=fset.id,
            model_ids=["logistic_regression", "random_forest"],
        )
        return run.id, exp.id

    def _setup_curated_models(self, ws, cmd, _qry):
        dataset, run = self._register_base(ws)
        cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
        cmd.dispatch(ExcludeModelCommand(run.id, "svc"))
        exp = ws.create_experiment(
            run,
            name="curated_models",
            model_ids=["logistic_regression", "random_forest"],
        )
        return run.id, exp.id

    def _setup_full_pipeline(self, ws, cmd, _qry):
        dataset, run = self._register_base(ws)
        cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
        cmd.dispatch(PrioritizeFeatureCommand(dataset.id, "salary", score=1.0, run_id=run.id))
        cmd.dispatch(PrioritizeFeatureCommand(dataset.id, "debt", score=0.9, run_id=run.id))
        cmd.dispatch(ExcludeModelCommand(run.id, "svc"))
        fset = cmd.dispatch(
            CreateFeatureSetCommand(
                dataset_id=dataset.id,
                name="financial_plus_country",
                feature_names=["salary", "debt", "account_balance", "tenure", "age", "country"],
                lineage="benchmark_v02_full",
            )
        )
        exp = ws.create_experiment(
            run,
            name="full_v02",
            feature_set_id=fset.id,
            model_ids=["logistic_regression", "random_forest"],
            priority="high",
        )
        return run.id, exp.id

    def _setup_automated_v03(self, ws, cmd, qry):
        dataset, run = self._register_base(ws)
        cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
        cmd.dispatch(PrioritizeFeatureCommand(dataset.id, "salary", score=1.0, run_id=run.id))
        cmd.dispatch(PrioritizeFeatureCommand(dataset.id, "debt", score=0.9, run_id=run.id))
        cmd.dispatch(ExcludeModelCommand(run.id, "svc"))
        cmd.dispatch(PlanExperimentsCommand(run_id=run.id))
        queue = ws.get_experiment_queue(run.id)
        best_cand = queue.pop_next()
        exp = ws.create_experiment(
            run=run,
            name=best_cand.name,
            feature_names=best_cand.feature_names,
            feature_set_id=best_cand.feature_set_id,
            model_ids=best_cand.model_ids,
            hypothesis=best_cand.hypothesis,
            priority=best_cand.priority.level.value if best_cand.priority else "normal",
        )
        return run.id, exp.id

    def _setup_optuna_v04(self, ws, cmd, qry):
        dataset, run = self._register_base(ws)
        cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
        cmd.dispatch(PrioritizeFeatureCommand(dataset.id, "salary", score=1.0, run_id=run.id))
        cmd.dispatch(PrioritizeFeatureCommand(dataset.id, "debt", score=0.9, run_id=run.id))
        exp = ws.create_experiment(
            run=run,
            name="optuna_lr_tune",
            feature_names=["salary", "debt", "account_balance", "tenure", "age"],
            model_ids=["logistic_regression"],
            priority="high",
        )
        return run.id, exp.id

    def _execute_optuna_v04(self, ws, cmd, qry, run_id, experiment_id):
        cmd.dispatch(
            OptimizeExperimentCommand(
                run_id=run_id,
                experiment_id=experiment_id,
                model_id="logistic_regression",
                optimizer="optuna",
                n_trials=5,
            )
        )
        return ws.repository.list_trial_results(experiment_id)

    def _setup_feature_selection_v05(self, ws, cmd, qry):
        dataset, run = self._register_base(ws)
        cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
        candidates = cmd.dispatch(SelectFeaturesCommand(run.id))
        top_cand = candidates[0] if candidates else None
        if top_cand:
            promoted = cmd.dispatch(
                PromoteCandidateFeatureSetCommand(
                    run_id=run.id,
                    candidate_id=top_cand.id,
                    new_name="v05_selected_features",
                    allow_unverified_promotion=True,
                )
            )
            feature_names = promoted.feature_names
            feature_set_id = promoted.id
        else:
            feature_names = ["salary", "debt", "account_balance", "tenure", "age"]
            feature_set_id = None
        exp = ws.create_experiment(
            run=run,
            name="feature_selection_v05",
            feature_names=feature_names,
            feature_set_id=feature_set_id,
            model_ids=["logistic_regression", "random_forest"],
            priority="high",
        )
        return run.id, exp.id

    def run_all(self, scenario_filter: str | None = None) -> list[dict]:
        import time

        results: list[dict] = []

        for scenario in self.scenarios():
            if scenario_filter and scenario.id != scenario_filter:
                continue

            scenario_root = None
            if self.workspace_root:
                scenario_root = str(Path(self.workspace_root) / scenario.id)
            ws, cmd, qry = build_application(root_dir=scenario_root)
            started = time.perf_counter()
            run_id, experiment_id = scenario.setup(ws, cmd, qry)
            if scenario.executor:
                trial_results = scenario.executor(ws, cmd, qry, run_id, experiment_id)
            else:
                trial_results = cmd.dispatch(RunExperimentCommand(run_id, experiment_id))
            elapsed = time.perf_counter() - started
            best_score, best_model = _best_result(trial_results, metric=scenario.metric, ws=ws)
            failed_count = sum(1 for r in trial_results if not r.succeeded)
            status = "ok" if best_score is not None else "failed"

            details = {
                "description": scenario.description,
                "scenario_version": scenario.version,
                "run_id": run_id,
                "experiment_id": experiment_id,
                "status": status,
                "failed_trials": failed_count,
                "trials": [
                    {
                        "model_id": r.model_id,
                        "score": None if not r.succeeded else round(r.primary_score, 4),
                        "failed": not r.succeeded,
                    }
                    for r in trial_results
                ],
            }
            ws.repository.save_benchmark_result(
                platform_version=PLATFORM_VERSION,
                scenario_id=scenario.id,
                metric=scenario.metric,
                best_score=best_score,
                best_model=best_model,
                total_time_s=elapsed,
                details=details,
            )
            results.append(
                {
                    "scenario_id": scenario.id,
                    "version": scenario.version,
                    "description": scenario.description,
                    "metric": scenario.metric,
                    "status": status,
                    "best_score": None if best_score is None else round(best_score, 4),
                    "best_model": best_model,
                    "total_time_s": round(elapsed, 3),
                    "delta_vs_baseline": None,
                    "failed_trials": failed_count,
                }
            )

        baseline = next((r for r in results if r["scenario_id"] == "baseline_all_features"), None)
        if baseline and baseline["best_score"] is not None:
            for row in results:
                if row.get("best_score") is None:
                    continue
                # If metric is minimizing (lower is better, e.g. RMSE, loss, MAE),
                # an improvement over baseline represents error reduction (baseline - score > 0).
                # If metric is maximizing (higher is better, e.g. ROC-AUC, accuracy),
                # an improvement represents score gain (score - baseline > 0).
                if is_minimizing_metric(row.get("metric", "")):
                    row["delta_vs_baseline"] = round(baseline["best_score"] - row["best_score"], 4)
                else:
                    row["delta_vs_baseline"] = round(row["best_score"] - baseline["best_score"], 4)

        return results

    def compare_history(self) -> list[dict]:
        root = self.workspace_root or str(Path.cwd() / ".automl" / "benchmark")
        rows: list[dict] = []
        base = Path(root)
        if not base.exists():
            return rows
        for scenario_dir in sorted(base.iterdir()):
            if not scenario_dir.is_dir():
                continue
            db = scenario_dir / "automl.db"
            if not db.exists():
                continue
            ws, _, _ = build_application(root_dir=str(scenario_dir))
            rows.extend(ws.repository.list_benchmark_results(platform_version=PLATFORM_VERSION))
        rows.sort(key=lambda r: r["timestamp"], reverse=True)
        return rows
