"""OOF command orchestration and side-effect-free artifact queries."""

import hashlib
import json
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from automl import __version__
from automl.domain.experiments.trial import (
    ExperimentStatus,
    Trial,
    TrialResult,
    TrialStatus,
)
from automl.domain.runs.states import RunPhase, RunStatus
from automl.engine.ensemble.oof import evaluate_oof
from automl.engine.profiling.dataset_profiler import load_dataframe
from automl.plugins.models.sklearn_models import build_sklearn_model


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_oof_report(workspace, run_id: str, experiment_id: str) -> dict:
    experiment = workspace.repository.get_experiment(experiment_id)
    if experiment is None or experiment.run_id != run_id:
        raise ValueError("OOF experiment does not belong to the requested run")
    results = workspace.repository.list_trial_results(experiment_id)
    result = next((r for r in results if r.model_id == "oof_blend" and r.succeeded), None)
    if result is None or "report" not in result.artifacts:
        raise ValueError("OOF experiment has no completed report")
    return json.loads(Path(result.artifacts["report"]).read_text())


def cached_oof_predictions(workspace, run_id, experiment_id, test_path, predict_proba):
    report = get_oof_report(workspace, run_id, experiment_id)
    if file_hash(test_path) != report["test_hash"]:
        raise ValueError("OOF predictions belong to a different test dataset; rerun predict with --folds")
    if file_hash(report["test_predictions_path"]) != report["test_predictions_hash"]:
        raise ValueError("Stored OOF predictions changed; rerun predict with --folds")
    values = pd.read_csv(report["test_predictions_path"])["oof_blend"].to_numpy()
    if predict_proba:
        return values.tolist()
    classes = np.asarray(report["classes"])
    return classes[(values >= 0.5).astype(int)].tolist()


def generate_oof_submission(workspace, command) -> str:
    run = workspace._get_run(command.run_id)
    dataset = workspace._get_dataset(run.dataset_id)
    if dataset.task_type != "binary_classification":
        raise ValueError("OOF currently supports binary classification only")

    def check_allowed():
        if workspace.execution_check:
            workspace.execution_check()
        current = workspace.repository.get_run(run.id)
        if current.status in {RunStatus.CANCELLED, RunStatus.PAUSED}:
            raise RuntimeError(f"Run is {current.status.value.lower()}; restart OOF after resuming")

    check_allowed()
    source_id = command.experiment_id
    if source_id is None:
        results = workspace.repository.get_leaderboard(run.id)
        source_id = next((r.experiment_id for r in results if r.succeeded and
                         workspace.repository.get_experiment(r.experiment_id).validation_strategy != "oof"), None)
    source = workspace.repository.get_experiment(source_id) if source_id else None
    if source is None or source.run_id != run.id or source.validation_strategy == "oof":
        raise ValueError("Select a source experiment belonging to this run with --experiment-id")
    individual_models = [m for m in source.model_ids if m not in {"voting_ensemble", "oof_blend"}]
    models = command.model_ids if command.model_ids is not None else (individual_models or ["logistic_regression", "random_forest"])
    if not models or len(set(models)) != len(models):
        raise ValueError("OOF requires at least one distinct individual model ID")
    if any(m in {"voting_ensemble", "oof_blend"} for m in models):
        raise ValueError("Select individual models rather than an ensemble for OOF")
    workspace.model_registry.validate_for_task(models, dataset.task_type)
    train_hash = file_hash(dataset.path)
    test_hash = file_hash(command.test_dataset_path)
    train_df = load_dataframe(dataset.path)
    test_df = load_dataframe(command.test_dataset_path)
    features = list(source.feature_names)
    X, X_test = train_df[features], test_df[features]
    # Freeze and record parameters before evaluation. First model is the declared baseline.
    parameters = {}
    source_results = workspace.repository.list_trial_results(source.id)
    greater_is_better = True
    if hasattr(workspace, "experiment_service") and hasattr(workspace.experiment_service, "_metric_greater_is_better"):
        greater_val = workspace.experiment_service._metric_greater_is_better(source.metric)
        if greater_val is not None:
            greater_is_better = greater_val
    elif hasattr(workspace, "_metric_greater_is_better"):
        greater_val = workspace._metric_greater_is_better(source.metric)
        if greater_val is not None:
            greater_is_better = greater_val

    for model_id in models:
        candidates = [r for r in source_results if r.model_id == model_id and r.succeeded]
        if candidates:
            best = (
                max(candidates, key=lambda r: getattr(r, "primary_score", 0.0))
                if greater_is_better
                else min(candidates, key=lambda r: getattr(r, "primary_score", 0.0))
            )
            parameters[model_id] = dict(workspace.repository.get_trial(best.trial_id).parameters)
        else:
            parameters[model_id] = {}

    def factory(model_id):
        def build():
            plugin = workspace.plugin_registry.get_model_plugin(model_id)
            if plugin:
                workspace.plugin_registry.validate_plugin_for_task(model_id, dataset.task_type)
                model = plugin.build_estimator(parameters=parameters[model_id], task_type=dataset.task_type)
            else:
                model = build_sklearn_model(model_id, dataset.task_type, parameters[model_id])
            params = model.get_params(deep=False)
            if "random_state" in params and "random_state" not in parameters[model_id]:
                model.set_params(random_state=run.config.random_seed)
            return model
        return build

    method = getattr(command, "method", "average") or "average"
    meta_model = getattr(command, "meta_model", "ridge") or "ridge"
    method_title = method.replace("_", " ").title()

    experiment = workspace.create_experiment(
        run,
        name=f"OOF {method_title} blending",
        feature_names=features,
        model_ids=models,
        hypothesis=f"{method_title} OOF blend ({', '.join(models)}) vs predeclared baseline {models[0]} on identical folds; no automatic promotion.",
    )
    experiment.validation_strategy = "oof"
    experiment.metric = "roc_auc"
    experiment.transition_to(ExperimentStatus.RUNNING)
    workspace.repository.save_experiment(experiment)
    trial = Trial(
        id=f"trial_{uuid.uuid4().hex[:8]}",
        experiment_id=experiment.id,
        model_id="oof_blend",
        seed=run.config.random_seed,
        parameters={
            "models": models,
            "folds": command.folds,
            "method": method,
            "weights": [1 / len(models)] * len(models),
        },
        status=TrialStatus.RUNNING,
    )
    workspace.repository.save_trial(trial)
    run.transition_to(RunStatus.EXPERIMENTING, RunPhase.EXPERIMENT_EXECUTION)
    workspace.repository.save_run(run)
    workspace._emit(
        "OOFStarted",
        {"experiment_id": experiment.id, "models": models, "folds": command.folds, "method": method},
        run_id=run.id,
    )
    started = monotonic()
    try:
        holdout_dataset_path = getattr(command, "holdout_dataset_path", None)
        holdout_ratio = float(getattr(command, "holdout_ratio", 0.0) or 0.0)
        min_improvement = float(getattr(command, "min_improvement", 0.0) or 0.0)
        if min_improvement < 0.0 or not np.isfinite(min_improvement):
            raise ValueError("min_improvement must be a non-negative finite float")

        group_column = (
            getattr(command, "group_column", None)
            or getattr(source, "group_column", None)
            or getattr(run.config, "group_column", None)
        )

        X_train_eval = X
        y_train_eval = train_df[dataset.target_column]
        X_holdout = None
        y_holdout = None
        holdout_hash = None

        if holdout_dataset_path:
            # Audit #3: Reject identical file path
            if Path(holdout_dataset_path).resolve() == Path(dataset.path).resolve():
                raise ValueError("Holdout dataset cannot be the same file as the training dataset")
            holdout_hash = file_hash(holdout_dataset_path)
            # Audit #3: Reject identical file hash
            if holdout_hash == train_hash:
                raise ValueError("Holdout dataset content is identical to the training dataset")
            holdout_df = load_dataframe(holdout_dataset_path)
            X_holdout = holdout_df[features]
            y_holdout = holdout_df[dataset.target_column]

            # Audit #3: Detect duplicate / overlapping row leakage
            train_cols = [*features, dataset.target_column]
            if all(c in holdout_df.columns for c in train_cols):
                train_hashes = set(pd.util.hash_pandas_object(train_df[train_cols], index=False))
                holdout_hashes = set(pd.util.hash_pandas_object(holdout_df[train_cols], index=False))
                overlap_hashes = train_hashes.intersection(holdout_hashes)
                if overlap_hashes:
                    raise ValueError(
                        f"Holdout dataset contains {len(overlap_hashes)} overlapping rows with training dataset (data leakage detected)"
                    )

            # Audit #2/#3: Detect group overlap between train and holdout
            if group_column and group_column in train_df.columns and group_column in holdout_df.columns:
                train_groups = set(train_df[group_column].dropna().unique())
                holdout_groups = set(holdout_df[group_column].dropna().unique())
                shared_groups = train_groups.intersection(holdout_groups)
                if shared_groups:
                    raise ValueError(
                        f"Holdout dataset shares {len(shared_groups)} groups with training dataset: {shared_groups}"
                    )

        elif holdout_ratio > 0.0:
            if holdout_ratio >= 0.5 or holdout_ratio <= 0.0:
                raise ValueError("holdout_ratio must be strictly between 0.0 and 0.5")
            if group_column and group_column in train_df.columns:
                from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
                n_splits = max(2, round(1.0 / holdout_ratio))
                try:
                    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=run.config.random_seed)
                    split_gen = sgkf.split(X, train_df[dataset.target_column], groups=train_df[group_column].values)
                except Exception:
                    gkf = GroupKFold(n_splits=n_splits)
                    split_gen = gkf.split(X, train_df[dataset.target_column], groups=train_df[group_column].values)
                train_idx, holdout_idx = next(split_gen)
                X_train_eval = X.iloc[train_idx]
                X_holdout = X.iloc[holdout_idx]
                y_train_eval = train_df[dataset.target_column].iloc[train_idx]
                y_holdout = train_df[dataset.target_column].iloc[holdout_idx]
            else:
                from sklearn.model_selection import train_test_split
                stratify_col = train_df[dataset.target_column] if dataset.task_type == "binary_classification" else None
                X_train_eval, X_holdout, y_train_eval, y_holdout = train_test_split(
                    X,
                    train_df[dataset.target_column],
                    test_size=holdout_ratio,
                    random_state=run.config.random_seed,
                    stratify=stratify_col,
                )

        groups_eval = None
        if group_column and group_column in train_df.columns:
            groups_eval = train_df.loc[X_train_eval.index, group_column].values

        result = evaluate_oof(
            X_train_eval,
            y_train_eval,
            X_test,
            {m: factory(m) for m in models},
            folds=command.folds,
            seed=run.config.random_seed,
            method=method,
            meta_model=meta_model,
            max_seconds=command.max_seconds,
            check_allowed=check_allowed,
            progress=workspace.execution_progress,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            groups=groups_eval,
        )
        if file_hash(dataset.path) != train_hash or file_hash(command.test_dataset_path) != test_hash:
            raise ValueError("Dataset changed during OOF evaluation; discard this run and retry")
        check_allowed()
        artifact_dir = workspace.root_dir.resolve() / "oof" / experiment.id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        oof_path, test_path, report_path = (artifact_dir / name for name in ("oof.csv", "test_predictions.csv", "report.json"))
        pd.DataFrame({"source_row": range(len(X_train_eval)), "fold": result.fold_ids,
                      "target": result.target, **result.oof, "oof_blend": result.blended_oof}).to_csv(oof_path, index=False)
        pd.DataFrame({"source_row": range(len(X_test)), **result.test, "oof_blend": result.blended_test}).to_csv(test_path, index=False)
        predictions = result.blended_test.tolist() if command.predict_proba else np.asarray(result.classes)[(result.blended_test >= 0.5).astype(int)].tolist()
        submission = workspace._write_submission(run.id, command.test_dataset_path, command.output_path,
                                                  predictions, command.id_column, command.template_path,
                                                  command.predict_proba)
        versions = {"automl": __version__}
        for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost"):
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                pass

        if result.weights is not None:
            trial.parameters["weights"] = [float(result.weights.get(m, 0.0)) for m in models]

        is_promoted = False
        evaluation_msg = f"{result.method} OOF comparison; not independent holdout evidence"
        if result.holdout_scores is not None and result.holdout_delta is not None:
            if result.holdout_delta > 0.0 and result.holdout_delta > min_improvement:
                is_promoted = True
                evaluation_msg = f"{result.method} OOF blend verified on independent holdout; promoted candidate (delta: {result.holdout_delta:+.4f})"
            else:
                is_promoted = False
                evaluation_msg = f"{result.method} OOF holdout delta {result.holdout_delta:+.4f} did not exceed threshold {min_improvement:.4f}; not promoted"

        report = {**submission, "run_id": run.id, "experiment_id": experiment.id,
                  "source_experiment_id": source.id, "folds": command.folds,
                  "seed": run.config.random_seed, "models": models, "method": result.method,
                  "weights": trial.parameters["weights"], "weights_by_model": result.weights,
                  "metric": "roc_auc", "score": result.scores["oof_blend"],
                  "baseline_model": models[0], "baseline_score": result.scores[models[0]],
                  "delta": result.scores["oof_blend"] - result.scores[models[0]],
                  "model_scores": result.scores, "fold_scores": result.fold_scores,
                  "cv_std": float(np.std(result.fold_scores["oof_blend"])),
                  "classes": result.classes, "features": features,
                  "positive_class": result.classes[1],
                  "backends": result.backends, "parameters": result.parameters, "versions": versions,
                  "train_hash": train_hash, "test_hash": test_hash, "holdout_hash": holdout_hash,
                  "holdout_scores": result.holdout_scores,
                  "holdout_score": result.holdout_scores["oof_blend"] if result.holdout_scores else None,
                  "holdout_baseline_score": result.holdout_scores[models[0]] if result.holdout_scores else None,
                  "holdout_delta": result.holdout_delta,
                  "stacking_residuals": result.stacking_residuals,
                  "max_seconds": command.max_seconds, "training_seconds": result.training_seconds,
                  "oof_predictions_path": str(oof_path), "test_predictions_path": str(test_path),
                  "oof_predictions_hash": file_hash(oof_path), "test_predictions_hash": file_hash(test_path),
                  "promoted": is_promoted, "evaluation": evaluation_msg}
        report["plugin_versions"] = {m: workspace.plugin_registry.get_model_plugin(m).version
                                     for m in models if workspace.plugin_registry.get_model_plugin(m)}
        config = {key: report[key] for key in ("models", "folds", "seed", "features", "parameters", "weights", "method")}
        report["config_hash"] = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
        try:
            import resource
            import sys
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report["peak_process_memory_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
        except ImportError:
            report["peak_process_memory_bytes"] = None
        report_path.write_text(json.dumps(report, indent=2, default=str))
        trial.status = TrialStatus.COMPLETED
        workspace.repository.save_trial_result(TrialResult(
            trial_id=trial.id, experiment_id=experiment.id, model_id=trial.model_id,
            primary_metric="roc_auc", primary_score=report["score"],
            secondary_metrics={
                "cv_std": report["cv_std"],
                "baseline_score": report["baseline_score"],
                "delta": report["delta"],
                "holdout_score": report["holdout_score"],
                "holdout_delta": report["holdout_delta"],
            },
            training_time_seconds=result.training_seconds,
            artifacts={"report": str(report_path), "oof": str(oof_path), "test": str(test_path)},
        ))
        experiment.transition_to(ExperimentStatus.COMPLETED)
        workspace._emit(
            "OOFCompleted",
            {
                "experiment_id": experiment.id,
                "score": report["score"],
                "delta": report["delta"],
                "promoted": is_promoted,
                "holdout_delta": result.holdout_delta,
            },
            run_id=run.id,
        )
    except Exception as exc:
        trial.status = TrialStatus.FAILED
        experiment.transition_to(ExperimentStatus.FAILED)
        workspace.repository.save_trial_result(TrialResult(
            trial_id=trial.id, experiment_id=experiment.id, model_id=trial.model_id,
            primary_metric="roc_auc", primary_score=0.0, failure_reason=str(exc),
            training_time_seconds=monotonic() - started,
        ))
        workspace._emit("OOFFailed", {"experiment_id": experiment.id, "reason": str(exc)}, run_id=run.id)
        raise
    finally:
        workspace.repository.save_trial(trial)
        workspace.repository.save_experiment(experiment)
        current = workspace.repository.get_run(run.id)
        if current.status not in {RunStatus.PAUSED, RunStatus.CANCELLED}:
            current.transition_to(RunStatus.COMPLETED if trial.status == TrialStatus.COMPLETED else RunStatus.FAILED,
                                  RunPhase.EVALUATION)
            workspace.repository.save_run(current)
        workspace._runs[run.id] = current
    return experiment.id
