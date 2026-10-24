"""Shared metric/status helpers for Workbench API controllers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from automl.application.queries.workspace_queries import GetLeaderboardQuery
from automl.domain.runs.states import is_active_run_status

def _status_value(status: Any) -> str:
    """Normalize Enum / str status to a plain uppercase string."""
    if status is None:
        return "CREATED"
    return str(getattr(status, "value", status)).upper()


def _effective_experiment_status(experiment: Any, trials: list[Any]) -> str:
    """Surface honest experiment status for Workbench (derive when persisted status lags)."""
    status = _status_value(getattr(experiment, "status", None))
    if status != "CREATED" or not trials:
        return status
    if any(getattr(t, "succeeded", getattr(t, "failure_reason", None) is None) for t in trials):
        return "COMPLETED"
    return "FAILED"


def _is_metric_minimizing(metric_name: str | None, ws: Any = None) -> bool:
    """Returns True if the metric is minimizing (lower is better, e.g. loss/error metrics)."""
    if not metric_name:
        return False
    from automl.domain.metrics.exceptions import InvalidMetric
    from automl.domain.metrics.spec import resolve_metric_spec

    metric_str = str(metric_name).lower().strip()
    greater: bool | None = None
    if ws is not None and hasattr(ws, "plugin_registry") and ws.plugin_registry is not None:
        try:
            if hasattr(ws.plugin_registry, "has") and ws.plugin_registry.has(metric_str):
                plugin = ws.plugin_registry.get_metric_plugin(metric_str)
                if plugin is not None and hasattr(plugin, "greater_is_better"):
                    greater = bool(plugin.greater_is_better)
        except Exception:
            pass
    try:
        return resolve_metric_spec(metric_str, greater_is_better=greater).is_minimizing
    except InvalidMetric:
        return False


def _get_best_trial_result(trials: list[Any], metric: str | None = None, ws: Any = None) -> Any | None:
    """Selects the best trial result respecting metric optimization direction and trial success."""
    if not trials:
        return None
    valid = [t for t in trials if getattr(t, "succeeded", False)]
    if not valid:
        return None
    target_metric = metric or getattr(valid[0], "primary_metric", None)
    is_min = _is_metric_minimizing(target_metric, ws=ws)
    return min(valid, key=lambda t: getattr(t, "primary_score", 0.0)) if is_min else max(valid, key=lambda t: getattr(t, "primary_score", 0.0))


def _build_real_activity_feed(
    ws: Any,
    qry: Any,
    runs: list[Any],
    all_trials: list[Any],
    recent_datasets: list[dict],
    workspace_dir: str,
) -> list[dict]:
    activity: list[dict] = []

    # 1. Best CV leaderboard result per run
    for r in runs:
        lb = qry.dispatch(GetLeaderboardQuery(r.id))
        if lb:
            top = lb[0]
            score = float(top["score"])
            model_id = str(top["model_id"])
            metric = str(top.get("metric") or getattr(r.config, "metric", "score") or "score")
            run_id = str(r.id)
            ds_name = None
            if hasattr(ws, "get_dataset") and getattr(r, "dataset_id", None):
                try:
                    ds_obj = ws.get_dataset(r.dataset_id)
                    if ds_obj and hasattr(ds_obj, "name"):
                        ds_name = ds_obj.name
                except Exception:
                    pass
            if not ds_name:
                for d in recent_datasets:
                    if d.get("id") == getattr(r, "dataset_id", None):
                        ds_name = d.get("name")
                        break
            ds_desc = f" on dataset {ds_name}" if ds_name else ""
            activity.append({
                "type": "ACCEPT",
                "title": f"Best CV Leaderboard: {model_id}",
                "description": f"Model {model_id} achieved {score:.5f} ({metric}){ds_desc} (run {run_id[:8]}).",
            })

    # 2. Agent ledger events (promoted / rejected hypotheses)
    ledger_file = Path(workspace_dir) / "agent_ledger.db"
    if ledger_file.exists():
        try:
            from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger

            ledger = SqliteAgentLedger(ledger_file)
            for r in runs:
                for h in ledger.list_hypotheses(r.id):
                    h_status = h.status.value.lower() if hasattr(h.status, "value") else str(h.status).lower()
                    h_id = getattr(h, "hypothesis_id", getattr(h, "id", "hyp"))
                    if h_status == "accepted":
                        activity.append({
                            "type": "ACCEPT",
                            "title": f"Agent Promoted: #{h_id[:8]}",
                            "description": f"{h.reasoning} (Verified gain over baseline).",
                        })
                    elif h_status == "rejected":
                        activity.append({
                            "type": "REJECT",
                            "title": f"Agent Rejected: #{h_id[:8]}",
                            "description": f"{h.reasoning} (Rejected by empirical critic).",
                        })
                    elif h_status in ("proposed", "pending"):
                        activity.append({
                            "type": "PLAN",
                            "title": f"Agent Proposed: #{h_id[:8]}",
                            "description": h.reasoning,
                        })
        except Exception:
            pass

    # 3. Failed trials (empirical verification rejects)
    for t in all_trials:
        if not t.succeeded:
            activity.append({
                "type": "REJECT",
                "title": f"Trial Rejected: {t.model_id}",
                "description": f"Failed during execution: {t.failure_reason or 'Validation error'}.",
            })

    # 4. Anti-leakage guardian exclusions on datasets
    for ds_info in recent_datasets:
        profile = ws.get_dataset_profile(ds_info["id"])
        if profile:
            if profile.leakage_column_names:
                leak_cols = ", ".join(sorted(profile.leakage_column_names))
                activity.append({
                    "type": "REJECT",
                    "title": f"Anti-Leakage Guardian: {ds_info['name']}",
                    "description": f"Automatically excluded contaminated columns ({leak_cols}) to protect generalization.",
                })
            if profile.identifier_column_names:
                id_cols = ", ".join(sorted(profile.identifier_column_names))
                activity.append({
                    "type": "REJECT",
                    "title": f"Pseudo-Identifier Filter: {ds_info['name']}",
                    "description": f"Excluded non-predictive identifiers ({id_cols}).",
                })

    # 5. Background Jobs (if available)
    if hasattr(ws, "job_service") and ws.job_service is not None:
        try:
            jobs = ws.job_service.list_jobs()
            for j in jobs:
                if j.status.value == "completed":
                    activity.append({
                        "type": "ACCEPT",
                        "title": f"Job Completed: {j.job_type}",
                        "description": f"Background job {j.id[:8]} finished successfully.",
                    })
                elif j.status.value == "failed":
                    activity.append({
                        "type": "REJECT",
                        "title": f"Job Failed: {j.job_type}",
                        "description": f"Job {j.id[:8]} failed: {j.error_message or 'Execution error'}.",
                    })
                elif j.status.value in ("running", "queued"):
                    activity.append({
                        "type": "PLAN",
                        "title": f"Job {j.status.value.capitalize()}: {j.job_type}",
                        "description": f"Job {j.id[:8]} progress: {j.progress_pct}%.",
                    })
        except Exception:
            pass

    # 6. Planned experiments
    for r in runs:
        exps = ws.list_experiments(r.id)
        for e in exps:
            models_str = ", ".join(e.model_ids) if hasattr(e, "model_ids") and e.model_ids else "models"
            activity.append({
                "type": "PLAN",
                "title": f"Experiment Planned: {e.name}",
                "description": f"Run {r.id[:8]} • models: {models_str} • priority: {e.priority}.",
            })

    # 7. Dataset registrations (if few other entries)
    if len(activity) < 5:
        for ds_info in recent_datasets:
            profile = ws.get_dataset_profile(ds_info["id"])
            rows_str = f"{profile.row_count} rows, {profile.column_count} features" if profile else "registered"
            target_str = f"Target: {ds_info.get('target')}" if ds_info.get("target") else "Unsupervised"
            activity.append({
                "type": "INFO",
                "title": f"Dataset Ingested: {ds_info['name']}",
                "description": f"{rows_str}. {target_str}.",
            })

    return activity[:15]


