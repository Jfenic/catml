"""Runs / experiments / models / inference API controller."""
from __future__ import annotations

import html
import json
from http import HTTPStatus
from pathlib import Path
from typing import Any

from automl import __version__
from automl.application.commands.job_commands import ControlJobCommand
from automl.application.commands.workspace_commands import (
    BuildEnsembleCommand,
    CancelRunCommand,
    CloneRunCommand,
    CreateExperimentCommand,
    GenerateOOFSubmissionCommand,
    GenerateSubmissionCommand,
    OptimizeExperimentCommand,
    PauseRunCommand,
    ResumeRunCommand,
    RunExperimentCommand,
)
from automl.application.queries.job_queries import ListJobsQuery
from automl.application.queries.workspace_queries import (
    CompareExperimentsQuery,
    GetExperimentTrialsQuery,
    GetLeaderboardQuery,
    GetOOFResultQuery,
    ListExperimentsQuery,
)
from automl.application.recipes.queries import (
    GetExperimentRecipeQuery,
    ListExperimentRecipesQuery,
)
from automl.domain.runs.states import is_active_run_status
from automl.interfaces.web.api._helpers import (
    _build_real_activity_feed,
    _effective_experiment_status,
    _get_best_trial_result,
    _is_metric_minimizing,
)


def try_get(handler: Any, path: str, query_params: dict, ws: Any, cmd: Any, qry: Any) -> bool:
    """Handle GET routes for this resource. Return True if the request was handled."""
    if path == "/api/overview":
        runs = ws.list_runs()
        all_trials = []
        best_by_dataset: dict[str, dict[str, dict]] = {}
        best_by_metric: dict[str, dict] = {}
        active_run_top: dict | None = None
        recent_datasets = []

        for r in runs:
            exps = ws.list_experiments(r.id)
            for e in exps:
                all_trials.extend(ws.list_trial_results(e.id))
            lb = qry.dispatch(GetLeaderboardQuery(r.id))
            if lb:
                top = lb[0]
                score = float(top["score"])
                model_id = str(top["model_id"])
                metric_name = str(top.get("metric") or r.config.metric or "score").lower()
                is_minimize = _is_metric_minimizing(metric_name, ws=ws)
                ds_id = str(r.dataset_id)

                # Track per-dataset metrics segregation
                if ds_id not in best_by_dataset:
                    best_by_dataset[ds_id] = {}
                if metric_name not in best_by_dataset[ds_id]:
                    best_by_dataset[ds_id][metric_name] = {
                        "score": score,
                        "model_id": model_id,
                        "run_id": r.id,
                        "dataset_id": ds_id,
                        "metric": metric_name,
                        "is_minimizing": is_minimize,
                        "metric_direction": "minimize" if is_minimize else "maximize",
                    }
                else:
                    cur_ds_score = best_by_dataset[ds_id][metric_name]["score"]
                    if (is_minimize and score < cur_ds_score) or (not is_minimize and score > cur_ds_score):
                        best_by_dataset[ds_id][metric_name] = {
                            "score": score,
                            "model_id": model_id,
                            "run_id": r.id,
                            "dataset_id": ds_id,
                            "metric": metric_name,
                            "is_minimizing": is_minimize,
                            "metric_direction": "minimize" if is_minimize else "maximize",
                        }

                # Maintain best_by_metric without incomparable cross-dataset numeric comparisons
                if metric_name not in best_by_metric:
                    best_by_metric[metric_name] = {
                        "score": score,
                        "model_id": model_id,
                        "run_id": r.id,
                        "dataset_id": ds_id,
                        "metric": metric_name,
                        "is_minimizing": is_minimize,
                        "metric_direction": "minimize" if is_minimize else "maximize",
                    }
                else:
                    cur_entry = best_by_metric[metric_name]
                    cur_score = cur_entry["score"]
                    cur_ds = cur_entry.get("dataset_id")
                    is_active_run = is_active_run_status(r.status)
                    # Only compare numerically if both runs belong to the exact same dataset
                    if cur_ds == ds_id:
                        if (is_minimize and score < cur_score) or (not is_minimize and score > cur_score):
                            best_by_metric[metric_name] = {
                                "score": score,
                                "model_id": model_id,
                                "run_id": r.id,
                                "dataset_id": ds_id,
                                "metric": metric_name,
                                "is_minimizing": is_minimize,
                                "metric_direction": "minimize" if is_minimize else "maximize",
                            }
                    elif is_active_run:
                        best_by_metric[metric_name] = {
                            "score": score,
                            "model_id": model_id,
                            "run_id": r.id,
                            "dataset_id": ds_id,
                            "metric": metric_name,
                            "is_minimizing": is_minimize,
                            "metric_direction": "minimize" if is_minimize else "maximize",
                        }

                # Track active or latest run for top-level scalar display
                is_active = is_active_run_status(r.status)
                if is_active and (active_run_top is None or not active_run_top.get("is_active")):
                    active_run_top = {
                        "score": score,
                        "model_id": model_id,
                        "metric": metric_name,
                        "status": r.status.value,
                        "is_active": True,
                    }
                elif active_run_top is None:
                    active_run_top = {
                        "score": score,
                        "model_id": model_id,
                        "metric": metric_name,
                        "status": r.status.value,
                        "is_active": is_active,
                    }
            else:
                is_active = is_active_run_status(r.status)
                if is_active and (active_run_top is None or not active_run_top.get("is_active")):
                    active_run_top = {
                        "score": None,
                        "model_id": "-",
                        "metric": str(getattr(r.config, "metric", "-") or "-").lower(),
                        "status": r.status.value,
                        "is_active": True,
                    }

            try:
                ds = ws.get_dataset(r.dataset_id)
            except KeyError:
                ds = None
            if ds and ds.name not in [d["name"] for d in recent_datasets]:
                profile = ws.get_dataset_profile(ds.id)
                recent_datasets.append({
                    "id": ds.id,
                    "name": ds.name,
                    "path": ds.path,
                    "target": ds.target_column,
                    "rows": profile.row_count if profile else None,
                    "features": profile.column_count if profile else None,
                })

        all_ds = ws.list_datasets()
        for ds in all_ds:
            if ds.name not in [d["name"] for d in recent_datasets]:
                profile = ws.get_dataset_profile(ds.id)
                recent_datasets.append({
                    "id": ds.id,
                    "name": ds.name,
                    "path": ds.path,
                    "target": ds.target_column,
                    "rows": profile.row_count if profile else None,
                    "features": profile.column_count if profile else None,
                })

        if active_run_top and active_run_top.get("score") is not None:
            best_score = active_run_top["score"]
            best_model = active_run_top["model_id"]
            best_metric = active_run_top["metric"]
        elif best_by_metric:
            first_metric = next(iter(best_by_metric.values()))
            best_score = first_metric["score"]
            best_model = first_metric["model_id"]
            best_metric = first_metric["metric"]
        else:
            best_score = None
            best_model = None
            best_metric = None

        activity_feed = _build_real_activity_feed(
            ws=ws,
            qry=qry,
            runs=runs,
            all_trials=all_trials,
            recent_datasets=recent_datasets,
            workspace_dir=handler.workspace_dir,
        )

        is_best_min = bool(best_metric and _is_metric_minimizing(best_metric, ws=ws))

        handler._send_json({
            "platform": "CATML AutoML Platform",
            "version": __version__,
            "workspace": handler.workspace_dir,
            "total_runs": len(runs),
            "total_trials": len(all_trials),
            "is_active": any(is_active_run_status(r.status) for r in runs),
            "best_score": best_score,
            "best_model": best_model,
            "best_metric": best_metric,
            "metric_direction": "minimize" if is_best_min else "maximize",
            "is_minimizing": is_best_min,
            "best_by_metric": best_by_metric,
            "best_by_dataset": best_by_dataset,
            "recent_datasets": recent_datasets,
            "activity_feed": activity_feed,
        })
        return True

    if path == "/api/runs":
        ds_filter = query_params.get("dataset_id", [None])[0]
        runs = ws.list_runs(dataset_id=ds_filter) if ds_filter else ws.list_runs()
        result = []
        for r in runs:
            try:
                dataset = ws.get_dataset(r.dataset_id)
            except KeyError:
                dataset = None
            lb = qry.dispatch(GetLeaderboardQuery(r.id))
            exps = ws.list_experiments(r.id)
            trials_cnt = sum(len(ws.list_trial_results(e.id)) for e in exps)
            is_run_min = _is_metric_minimizing(r.config.metric, ws=ws)
            result.append({
                "id": r.id,
                "dataset_id": r.dataset_id,
                "dataset_name": dataset.name if dataset else "-",
                "dataset_path": dataset.path if dataset else "-",
                "task_type": r.config.task_type,
                "target": r.config.target,
                "metric": r.config.metric,
                "metric_direction": "minimize" if is_run_min else "maximize",
                "is_minimizing": is_run_min,
                "validation_strategy": getattr(r.config, "validation_strategy", "kfold"),
                "cv_folds": getattr(r.config, "cv_folds", 5),
                "status": r.status.value,
                "is_active": is_active_run_status(r.status),
                "best_score": lb[0]["score"] if lb else None,
                "best_model": lb[0]["model_id"] if lb else None,
                "trials_count": trials_cnt,
                "experiments_count": len(exps),
            })
        handler._send_json(result)
        return True

    if path == "/api/experiments":
        run_id = query_params.get("run_id", [""])[0]
        if not run_id:
            handler._send_json({"error": "run_id parameter required"}, HTTPStatus.BAD_REQUEST)
            return True
        recipe_map = {}
        recipe_name_map = {}
        try:
            run_obj = ws.get_run(run_id)
            if run_obj:
                rcps = qry.dispatch(ListExperimentRecipesQuery(dataset_id=run_obj.dataset_id))
                for r in rcps:
                    for exp_id in (r.experiment_ids or []):
                        recipe_map[exp_id] = r.id
                        recipe_name_map[exp_id] = r.name
        except Exception:
            pass

        exps = ws.list_experiments(run_id)
        result = []
        for e in exps:
            trials = ws.list_trial_results(e.id)
            parameters = {t["trial_id"]: t["parameters"]
                          for t in qry.dispatch(GetExperimentTrialsQuery(e.id))}
            best_trial = _get_best_trial_result(trials, metric=e.metric, ws=ws)
            rec_id = getattr(e, "recipe_id", None) or recipe_map.get(e.id)
            rec_name = recipe_name_map.get(e.id)
            if not rec_name and rec_id:
                try:
                    rcp_obj = qry.dispatch(GetExperimentRecipeQuery(recipe_id=rec_id))
                    if rcp_obj:
                        rec_name = rcp_obj.name
                except Exception:
                    pass

            is_exp_min = _is_metric_minimizing(e.metric, ws=ws)
            result.append({
                "id": e.id,
                "name": e.name,
                "hypothesis": e.hypothesis,
                "status": _effective_experiment_status(e, trials),
                "priority": e.priority,
                "model_ids": e.model_ids,
                "features_count": len(e.feature_names),
                "feature_names": e.feature_names,
                "column_transforms": getattr(e, "column_transforms", []) or [],
                "derived_features": getattr(e, "derived_features", []) or [],
                "recipe_id": rec_id,
                "recipe_name": rec_name,
                "trials_count": len(trials),
                "best_score": best_trial.primary_score if best_trial else None,
                "best_model": best_trial.model_id if best_trial else None,
                "metric": e.metric,
                "metric_direction": "minimize" if is_exp_min else "maximize",
                "is_minimizing": is_exp_min,
                "created_by": e.created_by,
                "trials": [
                    {
                        "trial_id": t.trial_id,
                        "model_id": t.model_id,
                        "score": round(t.primary_score, 5),
                        "time_s": round(t.training_time_seconds, 2),
                        "succeeded": t.succeeded,
                        "params": parameters.get(t.trial_id, {}),
                    }
                    for t in trials
                ],
            })
        handler._send_json(result)
        return True

    if path == "/api/experiments/compare":
        exp_ids_str = query_params.get("ids", [""])[0]
        if not exp_ids_str:
            handler._send_json({"error": "ids parameter required"}, HTTPStatus.BAD_REQUEST)
            return True
        exp_ids = [i.strip() for i in exp_ids_str.split(",") if i.strip()]
        comparison = qry.dispatch(CompareExperimentsQuery(exp_ids))
        if isinstance(comparison, list):
            for item in comparison:
                trs = item.get("trials", [])
                valid_trs = [t for t in trs if not t.get("failed", False)]
                met = valid_trs[0].get("metric") if valid_trs else (trs[0].get("metric") if trs else None)
                is_min = _is_metric_minimizing(met, ws=ws)
                best_t = (min(valid_trs, key=lambda x: x.get("score", 0.0)) if is_min else max(valid_trs, key=lambda x: x.get("score", 0.0))) if valid_trs else None
                item["best_score"] = best_t.get("score") if best_t else None
                item["best_model"] = best_t.get("model_id") if best_t else None
                item["metric"] = met
                item["is_minimizing"] = is_min
                item["metric_direction"] = "minimize" if is_min else "maximize"
        handler._send_json(comparison)
        return True

    if path == "/api/leaderboard":
        run_id = query_params.get("run_id", [""])[0]
        if not run_id:
            handler._send_json({"error": "run_id parameter required"}, HTTPStatus.BAD_REQUEST)
            return True
        lb = qry.dispatch(GetLeaderboardQuery(run_id))
        handler._send_json(lb)
        return True

    if path == "/api/models/export":
        run_id = query_params.get("run_id", [""])[0]
        if not run_id:
            runs = ws.list_runs()
            if runs:
                run_id = runs[0].id
        if not run_id:
            handler._send_json({"error": "run_id parameter required or no runs available"}, HTTPStatus.BAD_REQUEST)
            return True

        exp_id = query_params.get("experiment_id", [None])[0]
        trial_id = query_params.get("trial_id", [None])[0]

        try:
            import io
            import tempfile
            import zipfile

            artifact = ws.export_model_artifact(run_id=run_id, experiment_id=exp_id, trial_id=trial_id)
            # Persist via ModelArtifact.save so the .sha256 sidecar is produced; ship both in a zip
            # so ModelArtifact.load(verify_checksum=True) works after download.
            with tempfile.TemporaryDirectory(prefix="catml_export_") as tmp:
                pkl_name = f"catml_{artifact.model_id}.pkl"
                pkl_path = Path(tmp) / pkl_name
                artifact.save(pkl_path, generate_checksum=True)
                sha_path = pkl_path.with_suffix(pkl_path.suffix + ".sha256")
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.write(pkl_path, arcname=pkl_name)
                    if sha_path.is_file():
                        zf.write(sha_path, arcname=sha_path.name)
                data = buf.getvalue()

            filename = f"catml_{artifact.model_id}.zip"
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", "application/zip")
            handler.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            handler.send_header("Content-Length", str(len(data)))
            handler._send_cors_headers()
            handler.end_headers()
            handler.wfile.write(data)
            return True
        except Exception as e:
            handler._send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            return True

    if path == "/api/models/export-info":
        run_id = query_params.get("run_id", [""])[0]
        if not run_id:
            runs = ws.list_runs()
            if runs:
                run_id = runs[0].id
        if not run_id:
            handler._send_json({"error": "run_id parameter required"}, HTTPStatus.BAD_REQUEST)
            return True
        exp_id = query_params.get("experiment_id", [None])[0]
        trial_id = query_params.get("trial_id", [None])[0]
        try:
            artifact = ws.export_model_artifact(run_id=run_id, experiment_id=exp_id, trial_id=trial_id)
            handler._send_json({
                "model_id": artifact.model_id,
                "task_type": artifact.task_type,
                "metric": artifact.metric,
                "score": artifact.score,
                "feature_names": artifact.feature_names,
                "target_name": artifact.target_name,
                "parameters": artifact.parameters,
                "download_url": f"/api/models/export?run_id={run_id}" + (f"&experiment_id={exp_id}" if exp_id else "") + (f"&trial_id={trial_id}" if trial_id else ""),
                "download_format": "zip",
                "contains": ["*.pkl", "*.pkl.sha256"],
                "load_hint": "Unzip, then ModelArtifact.load('catml_<model>.pkl') with the sidecar present.",
            })
            return True
        except Exception as e:
            handler._send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            return True

    if path == "/api/reports/executive":
        run_id = query_params.get("run_id", [""])[0]
        if not run_id:
            runs = ws.list_runs()
            if runs:
                run_id = runs[0].id
        if not run_id:
            handler._send_json({"error": "run_id parameter required"}, HTTPStatus.BAD_REQUEST)
            return True

        report_data = _generate_executive_report_data(ws, run_id)
        if not report_data:
            handler._send_json({"error": f"Run {run_id} not found"}, HTTPStatus.NOT_FOUND)
            return True

        fmt = query_params.get("format", ["html"])[0].lower()
        as_download = query_params.get("download", ["0"])[0].lower() in {"1", "true", "yes"}
        short_id = run_id[:8]

        if fmt == "json":
            payload = json.dumps(report_data, indent=2).encode("utf-8")
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            if as_download:
                handler.send_header("Content-Disposition", f'attachment; filename="catml_report_{short_id}.json"')
            handler.send_header("Content-Length", str(len(payload)))
            handler._send_cors_headers()
            handler.end_headers()
            handler.wfile.write(payload)
            return True

        if fmt in {"md", "markdown"}:
            md_content = _render_executive_report_markdown(report_data)
            payload = md_content.encode("utf-8")
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", "text/markdown; charset=utf-8")
            if as_download:
                handler.send_header("Content-Disposition", f'attachment; filename="catml_report_{short_id}.md"')
            handler.send_header("Content-Length", str(len(payload)))
            handler._send_cors_headers()
            handler.end_headers()
            handler.wfile.write(payload)
            return True

        # Default: HTML
        html_content = _render_executive_report_html(report_data)
        payload = html_content.encode("utf-8")
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        if as_download:
            handler.send_header("Content-Disposition", f'attachment; filename="catml_report_{short_id}.html"')
        handler.send_header("Content-Length", str(len(payload)))
        handler._send_cors_headers()
        handler.end_headers()
        handler.wfile.write(payload)
        return True

    if path == "/api/reports/executive-summary":
        run_id = query_params.get("run_id", [""])[0]
        if not run_id:
            runs = ws.list_runs()
            if runs:
                run_id = runs[0].id
        if not run_id:
            handler._send_json({"error": "run_id parameter required"}, HTTPStatus.BAD_REQUEST)
            return True

        report_data = _generate_executive_report_data(ws, run_id)
        if not report_data:
            handler._send_json({"error": f"Run {run_id} not found"}, HTTPStatus.NOT_FOUND)
            return True

        handler._send_json(report_data)
        return True
    return False


def _trial_duration_seconds(row: dict[str, Any]) -> float | None:
    """Extract training duration; None when missing (do not fake 0.0)."""
    for key in ("training_time_s", "training_time_seconds", "duration"):
        if key not in row or row[key] is None:
            continue
        try:
            return float(row[key])
        except (TypeError, ValueError):
            continue
    return None


def _format_duration_seconds(row: dict[str, Any], *, suffix: str = "") -> str:
    value = _trial_duration_seconds(row)
    if value is None:
        return "—"
    return f"{value:.2f}{suffix}"


def _generate_executive_report_data(ws: Any, run_id: str) -> dict[str, Any] | None:
    from datetime import datetime, timezone

    try:
        run = ws.get_run(run_id)
    except (KeyError, Exception):
        return None
    if not run:
        return None

    dataset = None
    if hasattr(ws, "get_dataset") and getattr(run, "dataset_id", None):
        try:
            dataset = ws.get_dataset(run.dataset_id)
        except Exception:
            dataset = None

    profile = None
    if hasattr(ws, "get_dataset_profile") and getattr(run, "dataset_id", None):
        try:
            profile = ws.get_dataset_profile(run.dataset_id)
        except Exception:
            profile = None

    leaderboard = ws.get_leaderboard(run_id, include_failed=True) if hasattr(ws, "get_leaderboard") else []
    experiments = ws.list_experiments(run_id) if hasattr(ws, "list_experiments") else []

    features = []
    if dataset:
        try:
            reg = ws.get_feature_registry(dataset.id)
            features = reg.active_feature_names(dataset.target_column)
        except Exception:
            features = [c.name for c in profile.columns] if profile and hasattr(profile, "columns") else []

    target_name = getattr(dataset, "target_column", None) or "target"
    task_type = getattr(dataset, "task_type", None) or "binary_classification"
    dataset_name = getattr(dataset, "name", None) or getattr(run, "dataset_name", None) or run.id
    cfg = getattr(run, "config", None)
    metric = getattr(run, "metric", None) or getattr(cfg, "metric", None) or "ROC-AUC"
    val_strategy = getattr(run, "validation_strategy", None) or getattr(cfg, "validation_strategy", "kfold")
    cv_folds = getattr(run, "cv_folds", None) or getattr(cfg, "cv_folds", 5)
    random_seed = getattr(run, "random_seed", None) or getattr(cfg, "random_seed", 42)

    champion_model = "Sin evaluar"
    champion_score: float | None = None
    champion_time: float | None = None
    # Δ vs worst succeeded on the same board — NOT a paired baseline experiment.
    delta_vs_worst_succeeded: float | None = None

    if leaderboard:
        succeeded_trials = [
            r for r in leaderboard
            if not r.get("failed") and r.get("succeeded", True)
        ]
        if succeeded_trials:
            champ = succeeded_trials[0]
            champion_model = champ.get("model_id") or champ.get("model") or "Desconocido"
            champ_sc = champ.get("score") if champ.get("score") is not None else champ.get("primary_score")
            try:
                champion_score = float(champ_sc) if champ_sc is not None else None
            except (ValueError, TypeError):
                champion_score = None

            champion_time = _trial_duration_seconds(champ)

            if len(succeeded_trials) > 1 and champion_score is not None:
                worst_ok = succeeded_trials[-1]
                worst_sc = worst_ok.get("score") if worst_ok.get("score") is not None else worst_ok.get("primary_score")
                try:
                    delta_vs_worst_succeeded = champion_score - float(worst_sc)
                except (ValueError, TypeError):
                    delta_vs_worst_succeeded = None
        else:
            champion_model = "Sin modelos exitosos"
            champion_time = _trial_duration_seconds(leaderboard[0])

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return {
        "metadata": {
            "title": "CATML — Informe Ejecutivo de Resultados",
            "version": __version__,
            "generated_at": timestamp,
            "workspace_id": ws.id,
            "run_id": run_id,
            "run_status": str(getattr(run, "status", "COMPLETED")),
        },
        "dataset": {
            "name": dataset_name,
            "id": getattr(dataset, "id", run.dataset_id),
            "target": target_name,
            "task_type": task_type,
            "rows": profile.row_count if profile else None,
            "columns": profile.column_count if profile else None,
            "features_count": len(features),
            "features_list": features,
        },
        "validation": {
            "strategy": val_strategy,
            "cv_folds": cv_folds,
            "random_seed": random_seed,
            "metric": metric,
            "in_fold_preprocessing": True,
            # Honest: in-fold intent is recorded; not a formal leakage certificate.
            "anti_leakage_guaranteed": False,
        },
        "champion": {
            "model_id": champion_model,
            "score": champion_score,
            "metric": metric,
            "delta_vs_worst_succeeded": delta_vs_worst_succeeded,
            # Legacy alias (same value); not a true paired baseline.
            "delta_vs_baseline": delta_vs_worst_succeeded,
            "training_time_seconds": champion_time,
        },
        "leaderboard": leaderboard,
        "experiments_count": len(experiments),
    }


def _render_executive_report_markdown(data: dict[str, Any]) -> str:
    meta = data["metadata"]
    ds = data["dataset"]
    val = data["validation"]
    champ = data["champion"]
    lb = data["leaderboard"]

    champ_score_str = f"{champ['score']:.4f}" if champ["score"] is not None else "—"
    delta_val = champ.get("delta_vs_worst_succeeded")
    if delta_val is None:
        delta_val = champ.get("delta_vs_baseline")
    delta_str = f"{delta_val:+.4f}" if delta_val is not None else "—"
    champ_time = champ.get("training_time_seconds")
    champ_time_str = f"{float(champ_time):.2f} s" if champ_time is not None else "—"

    lb_lines = []
    lb_lines.append(f"| Rango | Modelo | {val['metric']} | Tiempo (s) | Estado |")
    lb_lines.append("|---|---|---|---|---|")
    if not lb:
        lb_lines.append("| — | Sin ensayos evaluados | — | — | — |")
    else:
        for i, row in enumerate(lb, 1):
            m_id = row.get("model_id") or row.get("model") or "modelo"
            sc = row.get("score") if row.get("score") is not None else row.get("primary_score")
            try:
                sc_str = f"{float(sc):.4f}" if sc is not None else "—"
            except (ValueError, TypeError):
                sc_str = "—"
            dur_str = _format_duration_seconds(row)
            is_succ = (not row.get("failed")) if "failed" in row else row.get("succeeded", True)
            st = "Completado" if is_succ else "Fallido"
            lb_lines.append(f"| #{i} | {m_id} | {sc_str} | {dur_str} | {st} |")

    features_str = ", ".join(f"`{f}`" for f in ds["features_list"][:15])
    if len(ds["features_list"]) > 15:
        features_str += f" *(y {len(ds['features_list']) - 15} variables adicionales)*"

    if champ["score"] is not None:
        tech_conclusion = (
            f"El modelo **{champ['model_id']}** ha obtenido el mejor puntaje entre ensayos exitosos "
            f"bajo validación cruzada {val['strategy']} "
            f"({val['cv_folds']} pliegues, semilla={val['random_seed']}).\n"
            "El preprocesamiento del trainer está diseñado para ajustarse *in-fold*; "
            "esto reduce el riesgo de fuga, pero **no constituye un certificado** de ausencia total de leakage."
        )
    else:
        tech_conclusion = "No se registraron modelos completados exitosamente en esta ejecución."

    return f"""# {meta['title']}

- **Fecha de Generación:** {meta['generated_at']}
- **Plataforma:** CATML AutoML Engine v{meta['version']}
- **Espacio de Trabajo:** `{meta['workspace_id']}`
- **ID de Ejecución:** `{meta['run_id']}`

---

## 1. Resumen Ejecutivo

| Métrica / Dimensión | Valor Evaluado |
|---|---|
| **Dataset Evaluado** | **{ds['name']}** ({ds['rows'] or '—'} filas, {ds['columns'] or '—'} columnas) |
| **Tipo de Problema** | {ds['task_type']} |
| **Variable Objetivo** | `{ds['target']}` |
| **Métrica Principal** | **{val['metric']}** |
| **Modelo Campeón** | **{champ['model_id']}** |
| **Puntaje de Validación (CV)** | **{champ_score_str}** |
| **Δ vs peor ensayo OK** | **{delta_str}** |
| **Tiempo de Entrenamiento del Campeón** | {champ_time_str} |
| **Estrategia de Validación** | {val['strategy']} ({val['cv_folds']} pliegues, semilla={val['random_seed']}) |

### Conclusión Técnica
{tech_conclusion}

---

## 2. Leaderboard de Modelos Evaluados

{chr(10).join(lb_lines)}

---

## 3. Variables Predictivas del Pipeline

Variables activas ({ds['features_count']}):
{features_str or 'Sin variables registradas'}

---

## 4. Gobernanza, Verificación y Despliegue

- **Anti-fuga (honesto):** Preprocesamiento *in-fold* por diseño; sin fuga detectada ≠ dataset plenamente validado.
- **Exportación de Artefactos:** Modelo autónomo serializado disponible para producción mediante `ModelArtifact` (`.pkl`) con verificación de integridad SHA-256 (`.sha256`).
- **Reproducibilidad:** Semilla pseudoaleatoria fija `{val['random_seed']}`.
"""


def _render_executive_report_html(data: dict[str, Any]) -> str:
    meta = data["metadata"]
    ds = data["dataset"]
    val = data["validation"]
    champ = data["champion"]
    lb = data["leaderboard"]

    champ_score_str = f"{champ['score']:.4f}" if champ["score"] is not None else "—"
    delta_val = champ.get("delta_vs_worst_succeeded")
    if delta_val is None:
        delta_val = champ.get("delta_vs_baseline")
    delta_str = f"{delta_val:+.4f}" if delta_val is not None else "—"

    rows_html = ""
    for i, row in enumerate(lb, 1):
        m_id = html.escape(str(row.get("model_id") or row.get("model") or "modelo"))
        sc = row.get("score") if row.get("score") is not None else row.get("primary_score")
        try:
            sc_str = f"{float(sc):.4f}" if sc is not None else "—"
        except (ValueError, TypeError):
            sc_str = "—"
        dur_str = _format_duration_seconds(row, suffix="s")
        is_succ = (not row.get("failed")) if "failed" in row else row.get("succeeded", True)
        status_badge = '<span style="color:#22C55E">● OK</span>' if is_succ else '<span style="color:#EF4444">● ERROR</span>'
        rank_badge = f'<span style="display:inline-block;padding:2px 6px;border-radius:4px;background:#161B26;font-size:11px;font-weight:600;color:{"#4F67FF" if i == 1 else "#8B95A7"}">#{i}</span>'
        score_color = "#22C55E" if (i == 1 and is_succ) else "#F7F8FA"
        rows_html += f"""<tr>
          <td>{rank_badge}</td>
          <td style="font-weight:600;color:#FFFFFF">{m_id}</td>
          <td style="font-weight:700;color:{score_color}">{sc_str}</td>
          <td>{dur_str}</td>
          <td>{status_badge}</td>
        </tr>"""

    features_html = "".join(f'<span class="pill">{html.escape(f)}</span>' for f in ds["features_list"][:24])
    if len(ds["features_list"]) > 24:
        features_html += f'<span class="pill" style="opacity:0.7">+{len(ds["features_list"]) - 24} más</span>'

    if champ["score"] is not None:
        exec_conclusion_html = (
            f"<strong>Conclusión Ejecutiva:</strong> El modelo <strong>{html.escape(champ['model_id'])}</strong> obtuvo el mejor puntaje "
            f"entre ensayos exitosos bajo validación cruzada {html.escape(str(val['strategy']))} ({val['cv_folds']} folds). "
            "El preprocesamiento está diseñado <em>in-fold</em>; esto reduce el riesgo de fuga, "
            "pero <strong>no es un certificado</strong> de ausencia total de <em>data leakage</em>."
        )
    else:
        exec_conclusion_html = "<strong>Conclusión Ejecutiva:</strong> No se registraron modelos completados exitosamente en esta ejecución."

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>{html.escape(meta['title'])} — {html.escape(ds['name'])}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --catml-black: #0B0D12;
      --catml-graphite: #171A22;
      --catml-card: #11151E;
      --catml-border: #242A36;
      --catml-text: #F7F8FA;
      --catml-muted: #8B95A7;
      --catml-blue: #4F67FF;
      --catml-green: #22C55E;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--catml-black);
      color: var(--catml-text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      line-height: 1.5;
      padding: 36px 20px;
    }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    .header {{
      border-bottom: 1px solid var(--catml-border);
      padding-bottom: 24px;
      margin-bottom: 28px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
    }}
    .badge {{
      display: inline-block;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      padding: 4px 10px;
      border-radius: 6px;
      background: rgba(79, 103, 255, 0.15);
      color: var(--catml-blue);
      border: 1px solid rgba(79, 103, 255, 0.3);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    h1 {{ font-size: 24px; font-weight: 700; margin-top: 8px; color: #FFFFFF; letter-spacing: -0.02em; }}
    .meta-sub {{ color: var(--catml-muted); font-size: 12px; margin-top: 4px; font-family: ui-monospace, monospace; }}
    .card {{
      background: var(--catml-card);
      border: 1px solid var(--catml-border);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
    }}
    .card-title {{
      font-size: 14px;
      font-weight: 700;
      color: #FFFFFF;
      margin-bottom: 16px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      border-bottom: 1px solid rgba(36, 42, 54, 0.6);
      padding-bottom: 8px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .kpi {{
      background: #090C12;
      border: 1px solid var(--catml-border);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .kpi-label {{ font-size: 10px; color: var(--catml-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}
    .kpi-value {{ font-size: 20px; font-weight: 700; color: #FFFFFF; margin-top: 4px; font-family: ui-monospace, monospace; }}
    .kpi-value.gain {{ color: var(--catml-green); }}
    .kpi-value.accent {{ color: var(--catml-blue); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
    th {{ text-align: left; padding: 10px 12px; color: var(--catml-muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; border-bottom: 1px solid var(--catml-border); }}
    td {{ padding: 12px; border-bottom: 1px solid rgba(36, 42, 54, 0.5); font-family: ui-monospace, monospace; font-size: 12px; }}
    tr:last-child td {{ border-bottom: none; }}
    .pill {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; margin: 3px 2px; background: #161B26; border: 1px solid var(--catml-border); color: #E1E7F0; font-family: ui-monospace, monospace; }}
    .conclusion {{ font-size: 13px; color: #CBD5E1; line-height: 1.6; padding: 12px 14px; background: #0A0D14; border-radius: 8px; border-left: 3px solid var(--catml-blue); }}
    .footer {{ text-align: center; color: var(--catml-muted); font-size: 11px; margin-top: 40px; border-top: 1px solid var(--catml-border); padding-top: 16px; font-family: ui-monospace, monospace; }}
    @media print {{
      body {{ background: #FFFFFF !important; color: #111827 !important; padding: 0; }}
      .card {{ background: #FFFFFF !important; border: 1px solid #E5E7EB !important; color: #111827 !important; }}
      .kpi {{ background: #F9FAFB !important; border: 1px solid #E5E7EB !important; }}
      .kpi-value {{ color: #111827 !important; }}
      .kpi-value.gain {{ color: #059669 !important; }}
      .kpi-value.accent {{ color: #2563EB !important; }}
      .card-title, h1 {{ color: #111827 !important; }}
      th, td {{ color: #111827 !important; border-color: #E5E7EB !important; }}
      .pill {{ background: #F3F4F6 !important; border-color: #E5E7EB !important; color: #111827 !important; }}
      .conclusion {{ background: #F9FAFB !important; color: #374151 !important; border-color: #2563EB !important; }}
      .header {{ border-color: #E5E7EB !important; }}
      .badge {{ background: #EFF6FF !important; color: #1D4ED8 !important; border-color: #BFDBFE !important; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <span class="badge">Informe Ejecutivo de Resultados</span>
        <h1>{html.escape(ds['name'])}</h1>
        <div class="meta-sub">Run ID: {html.escape(meta['run_id'])} • Generado: {html.escape(meta['generated_at'])}</div>
      </div>
      <div style="text-align:right">
        <span class="badge" style="background:#161B26;color:#8B95A7;border-color:#242A36">v{html.escape(meta['version'])}</span>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Resumen del Rendimiento Óptimo</div>
      <div class="kpi-grid">
        <div class="kpi">
          <div class="kpi-label">Modelo Campeón</div>
          <div class="kpi-value accent">{html.escape(champ['model_id'])}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Puntaje CV ({html.escape(val['metric'])})</div>
          <div class="kpi-value gain">{champ_score_str}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Δ vs peor ensayo OK</div>
          <div class="kpi-value">{delta_str}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Volumen de Datos</div>
          <div class="kpi-value">{ds['rows'] or '—'} <span style="font-size:12px;color:var(--catml-muted)">filas</span></div>
        </div>
      </div>

      <div class="conclusion">
        {exec_conclusion_html}
      </div>
    </div>

    <div class="card">
      <div class="card-title">Leaderboard Completo de Ensayos</div>
      <table>
        <thead>
          <tr>
            <th>Rango</th>
            <th>Modelo</th>
            <th>{html.escape(val['metric'])}</th>
            <th>Duración</th>
            <th>Estado</th>
          </tr>
        </thead>
        <tbody>
          {rows_html or '<tr><td colspan="5" style="text-align:center;color:var(--catml-muted)">Sin ensayos evaluados</td></tr>'}
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-title">Variables Evaluadas ({ds['features_count']})</div>
      <div>
        {features_html or '<span style="color:var(--catml-muted);font-size:12px">Sin variables</span>'}
      </div>
    </div>

    <div class="card">
      <div class="card-title">Gobernanza y Preparación para Producción</div>
      <div style="font-size:12px;color:#CBD5E1;line-height:1.8">
        <div>✔ <strong>Validación honesta:</strong> Partición con semilla <code>{val['random_seed']}</code>; preprocesamiento <em>in-fold</em> por diseño (no certificado de leakage cero).</div>
        <div>✔ <strong>Artefactos serializados:</strong> Modelo disponible como <code>ModelArtifact</code> (.pkl) junto a su hash criptográfico SHA-256.</div>
        <div>✔ <strong>Plataforma:</strong> CATML AutoML Platform • Motor auditado e independiente.</div>
      </div>
    </div>

    <div class="footer">
      Generado por CATML Platform • Laboratorio Local • {html.escape(meta['generated_at'])}
    </div>
  </div>
</body>
</html>"""


def try_post(handler: Any, path: str, payload: dict, ws: Any, cmd: Any, qry: Any) -> bool:
    """Handle POST routes for this resource. Return True if the request was handled."""
    if path == "/api/run/create":
        dataset_id = payload.get("dataset_id")
        if not dataset_id:
            handler._send_json({"error": "dataset_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        dataset = ws.get_dataset(dataset_id)
        if not dataset:
            handler._send_json({"error": f"Dataset {dataset_id} not found"}, HTTPStatus.NOT_FOUND)
            return True
        run = ws.create_run(dataset)
        handler._send_json({
            "status": "success",
            "run_id": run.id,
            "dataset_id": dataset.id,
            "task_type": run.config.task_type,
            "target": run.config.target,
            "metric": run.config.metric,
        })
        return True

    if path == "/api/run/pause":
        run_id = payload.get("run_id")
        if not run_id:
            handler._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        jobs = [j for j in qry.dispatch(ListJobsQuery(run_id)) if j["status"] in {"queued", "running"}]
        if jobs:
            for job in jobs:
                cmd.dispatch(ControlJobCommand(job["id"], "pause"))
            handler._send_json({"status": "success", "run_id": run_id, "state": "PAUSE_REQUESTED"})
            return True
        cmd.dispatch(PauseRunCommand(run_id=run_id))
        handler._send_json({"status": "success", "run_id": run_id, "state": "PAUSED"})
        return True

    if path == "/api/run/resume":
        run_id = payload.get("run_id")
        if not run_id:
            handler._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        jobs = [j for j in qry.dispatch(ListJobsQuery(run_id)) if j["status"] == "paused"]
        if jobs:
            for job in jobs:
                cmd.dispatch(ControlJobCommand(job["id"], "resume"))
            handler._send_json({"status": "success", "run_id": run_id, "state": "QUEUED"})
            return True
        try:
            cmd.dispatch(ResumeRunCommand(run_id=run_id))
        except RuntimeError:
            from automl.domain.runs.states import RunPhase, RunStatus
            run = ws._get_run(run_id)
            run.transition_to(RunStatus.EXPERIMENTING, RunPhase.EXPERIMENT_EXECUTION)
            ws.save_run(run)
        handler._send_json({"status": "success", "run_id": run_id, "state": "RUNNING"})
        return True

    if path == "/api/run/cancel":
        run_id = payload.get("run_id")
        if not run_id:
            handler._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        jobs = [j for j in qry.dispatch(ListJobsQuery(run_id)) if j["status"] in {"queued", "running", "pause_requested", "paused", "interrupted", "failed"}]
        if jobs:
            for job in jobs:
                cmd.dispatch(ControlJobCommand(job["id"], "cancel"))
            handler._send_json({"status": "success", "run_id": run_id, "state": "CANCEL_REQUESTED"})
            return True
        cmd.dispatch(CancelRunCommand(run_id=run_id))
        handler._send_json({"status": "success", "run_id": run_id, "state": "CANCELLED"})
        return True

    if path == "/api/run/clone":
        run_id = payload.get("run_id")
        new_name = payload.get("new_name")
        if not run_id:
            handler._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        new_run = cmd.dispatch(CloneRunCommand(run_id=run_id, new_name=new_name))
        handler._send_json({"status": "success", "new_run_id": new_run.id if hasattr(new_run, "id") else str(new_run)})
        return True

    if path == "/api/experiment/run":
        run_id = payload.get("run_id")
        model_id = payload.get("model_id", "lightgbm")
        name = payload.get("name", f"exp_{model_id}")

        if not run_id:
            handler._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
            return True

        run = ws._get_run(run_id)
        dataset = ws._get_dataset(run.dataset_id)
        profile = ws.get_dataset_profile(dataset.id)
        feature_names = payload.get("feature_names") or [
            c.name for c in profile.columns if not c.is_identifier and c.name != dataset.target_column
        ]

        exp = cmd.dispatch(
            CreateExperimentCommand(
                run_id=run.id,
                name=name,
                feature_names=feature_names,
                model_ids=[model_id],
            )
        )

        results = cmd.dispatch(RunExperimentCommand(run.id, exp.id))
        res = results[0] if results else None

        handler._send_json({
            "status": "success",
            "experiment_id": exp.id,
            "model_id": model_id,
            "succeeded": res.succeeded if res else False,
            "primary_score": res.primary_score if res else 0.0,
            "primary_metric": res.primary_metric if res else "-",
            "training_time_seconds": res.training_time_seconds if res else 0.0,
        })
        return True

    if path == "/api/experiment/create_and_run":
        run_id = payload.get("run_id")
        mode = payload.get("mode", "auto")
        models = payload.get("models") or ["lightgbm"]
        budget = payload.get("budget", "balanced")
        name = payload.get("name", f"exp_{mode}_{models[0]}")
        recipe_id = payload.get("recipe_id")

        if not run_id:
            handler._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
            return True

        run = ws._get_run(run_id)
        dataset = ws._get_dataset(run.dataset_id)
        profile = ws.get_dataset_profile(dataset.id)

        # Time budget enforcement — same labels as JobExecutor / NewExperimentModal
        from automl.domain.experiments.budgets import resolve_time_budget_seconds

        time_budget_seconds = resolve_time_budget_seconds(
            budget,
            payload.get("time_budget_seconds"),
            default=300.0,
        )
        run.config.time_budget_seconds = time_budget_seconds
        ws.save_run(run)

        column_transforms = payload.get("column_transforms")
        derived_features = payload.get("derived_features")
        if recipe_id and (not column_transforms or not derived_features):
            try:
                rcp = qry.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
                if rcp:
                    column_transforms = column_transforms or rcp.column_transforms
                    derived_features = derived_features or rcp.derived_features
            except Exception:
                pass

        feature_names = payload.get("feature_names") or [
            c.name for c in profile.columns if not c.is_identifier and c.name != dataset.target_column
        ]

        val_strategy = payload.get("validation_strategy")

        exp = cmd.dispatch(
            CreateExperimentCommand(
                run_id=run.id,
                name=name,
                feature_names=feature_names,
                model_ids=models,
                hypothesis=f"AutoML Workbench {mode.upper()} mode with {budget} ({time_budget_seconds}s) compute budget.",
                validation_strategy=val_strategy,
                column_transforms=column_transforms,
                derived_features=derived_features,
                recipe_id=recipe_id,
                time_budget_seconds=time_budget_seconds,
            )
        )

        if recipe_id:
            try:
                rcp_repo = ws.get_recipe_repository()
                rcp_to_update = rcp_repo.get(recipe_id) if rcp_repo else None
                if rcp_to_update:
                    rcp_to_update.submit(exp.id)
                    rcp_repo.save(rcp_to_update)
            except Exception:
                pass

        results = cmd.dispatch(RunExperimentCommand(run.id, exp.id))
        res = results[0] if results else None

        handler._send_json({
            "status": "success",
            "experiment_id": exp.id,
            "primary_score": res.primary_score if res else 0.0,
            "primary_metric": res.primary_metric if res else "-",
        })
        return True

    if path == "/api/optimize":
        run_id = payload.get("run_id")
        model_id = payload.get("model_id", "lightgbm")
        n_trials = int(payload.get("n_trials", 5))

        if not run_id:
            handler._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
            return True

        run = ws._get_run(run_id)
        dataset = ws._get_dataset(run.dataset_id)
        profile = ws.get_dataset_profile(dataset.id)
        feature_names = [
            c.name for c in profile.columns if not c.is_identifier and c.name != dataset.target_column
        ]

        exp = cmd.dispatch(
            CreateExperimentCommand(
                run_id=run.id,
                name=f"opt_{model_id}",
                feature_names=feature_names,
                model_ids=[model_id],
            )
        )

        opt_result = cmd.dispatch(
            OptimizeExperimentCommand(
                run_id=run.id,
                experiment_id=exp.id,
                model_id=model_id,
                optimizer="optuna",
                n_trials=n_trials,
            )
        )

        # Only return importances computed from real HPO trials (never fabricated).
        param_importance = opt_result.get("param_importance")
        if param_importance is None:
            param_importance = []

        handler._send_json({
            "status": "success",
            "run_id": run.id,
            "experiment_id": exp.id,
            "best_score": opt_result.get("best_score"),
            "best_params": opt_result.get("best_params"),
            "trials_count": opt_result.get("trials_executed"),
            "trials": opt_result.get("trials", []),
            "param_importance": param_importance,
            "param_importance_available": bool(param_importance),
        })
        return True

    if path == "/api/predict":
        run_id = payload.get("run_id")
        test_dataset_path = payload.get("test_dataset_path")
        output_path = payload.get("output_path", "submission.csv")
        template_path = payload.get("template_path")
        predict_proba = bool(payload.get("predict_proba", True))

        if not run_id or not test_dataset_path:
            handler._send_json({"error": "run_id and test_dataset_path are required"}, HTTPStatus.BAD_REQUEST)
            return True

        # HTTP boundary: confine read/write paths to cwd ∪ workspace (CLI stays free).
        from automl.infrastructure.security.workspace_paths import (
            PathOutsideWorkspaceError,
            resolve_allowed_path,
        )

        try:
            test_dataset_path = str(
                resolve_allowed_path(test_dataset_path, workspace_dir=handler.workspace_dir)
            )
            # Relative outputs land under workspace/exports/ so predictions stay sandboxed.
            out_raw = Path(output_path)
            if not out_raw.is_absolute():
                export_dir = Path(handler.workspace_dir) / "exports"
                export_dir.mkdir(parents=True, exist_ok=True)
                out_raw = export_dir / out_raw.name
            output_path = str(
                resolve_allowed_path(out_raw, workspace_dir=handler.workspace_dir)
            )
            if template_path:
                template_path = str(
                    resolve_allowed_path(template_path, workspace_dir=handler.workspace_dir)
                )
        except PathOutsideWorkspaceError as pe:
            handler._send_json({"error": str(pe)}, HTTPStatus.FORBIDDEN)
            return True

        if payload.get("folds") is not None:
            experiment_id = cmd.dispatch(GenerateOOFSubmissionCommand(
                run_id=run_id, test_dataset_path=test_dataset_path,
                output_path=output_path, template_path=template_path,
                id_column=payload.get("id_column"), predict_proba=predict_proba,
                experiment_id=payload.get("experiment_id"), folds=payload["folds"],
                model_ids=payload.get("model_ids"),
                method=payload.get("method", "average"),
                meta_model=payload.get("meta_model", "ridge"),
                max_seconds=payload.get("max_seconds", 300.0),
            ))
            sub_res = qry.dispatch(GetOOFResultQuery(run_id=run_id, experiment_id=experiment_id))
        else:
            sub_res = cmd.dispatch(
                GenerateSubmissionCommand(
                    run_id=run_id,
                    test_dataset_path=test_dataset_path,
                    output_path=output_path,
                    template_path=template_path,
                    predict_proba=predict_proba,
                )
            )

        # Checklist only reports facts we actually know — no hardcoded True claims.
        id_col = sub_res.get("id_column")
        row_count = sub_res.get("row_count")
        handler._send_json({
            "status": "success",
            "output_path": sub_res.get("output_path"),
            "row_count": row_count,
            "id_column": id_col,
            "target_column": sub_res.get("target_column"),
            "oof": sub_res if payload.get("folds") is not None else None,
            "checklist": {
                "id_column_present": id_col is not None and str(id_col).strip() != "",
                "row_count": row_count,
                "predict_proba_requested": bool(predict_proba),
                "template_used": bool(template_path or sub_res.get("template_used")),
            },
        })
        return True

    if path == "/api/ensemble/build":
        run_id = payload.get("run_id")
        models = payload.get("models") or payload.get("model_ids") or []
        method = payload.get("method", "average")
        meta_model = payload.get("meta_model", "ridge")
        folds = int(payload.get("folds", 5))
        name = payload.get("name")

        if not run_id:
            handler._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        if not models or len(models) < 2:
            handler._send_json({"error": "At least 2 distinct models required for ensemble"}, HTTPStatus.BAD_REQUEST)
            return True

        exp_id = cmd.dispatch(
            BuildEnsembleCommand(
                run_id=run_id,
                model_ids=models,
                method=method,
                meta_model=meta_model,
                folds=folds,
                name=name,
            )
        )

        exp = ws.get_experiment(exp_id)
        trials = ws.list_trial_results(exp_id)
        res = trials[0] if trials else None
        score = res.primary_score if res else 0.0
        sec_metrics = res.secondary_metrics if res else {}

        handler._send_json({
            "status": "success",
            "experiment_id": exp.id,
            "name": exp.name,
            "score": score,
            "method": method,
            "models": models,
            "weights": sec_metrics.get("weights"),
            "model_scores": sec_metrics.get("model_scores"),
            "cv_std": sec_metrics.get("cv_std"),
        })
        return True
    return False
