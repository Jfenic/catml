"""Application service for multi-model ensemble construction and out-of-fold blending."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np

from automl.domain.experiments.trial import (
    Experiment,
    ExperimentStatus,
    Trial,
    TrialResult,
    TrialStatus,
)
from automl.engine.profiling.dataset_profiler import load_dataframe

if TYPE_CHECKING:
    from automl.application.services.workspace import AutoMLWorkspace


class EnsembleService:
    """Application service for building voting ensembles and OOF blends across candidate models."""

    def __init__(self, workspace: AutoMLWorkspace) -> None:
        self.workspace = workspace

    def build_ensemble(
        self,
        run_id: str,
        model_ids: list[str],
        method: str = "average",
        meta_model: str = "ridge",
        folds: int = 5,
        name: str | None = None,
    ) -> Experiment:
        ws = self.workspace
        run = ws._get_run(run_id)
        dataset = ws._get_dataset(run.dataset_id)
        task_type = dataset.task_type or run.config.task_type

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

        ws.model_registry.validate_for_task(unique_models, task_type)

        registry = ws.get_feature_registry(run.dataset_id)
        features = registry.active_feature_names(dataset.target_column)
        if not features:
            profile = ws.repository.get_dataset_profile(run.dataset_id)
            if profile:
                features = [
                    c.name for c in profile.columns
                    if not c.is_identifier and c.name != dataset.target_column
                ]
            if not features:
                raise ValueError("No features available to train ensemble.")

        exp_name = name or f"Ensemble ({method_norm.title()} - {len(unique_models)} models)"
        experiment = ws.create_experiment(
            run=run,
            name=exp_name,
            feature_names=features,
            model_ids=["voting_ensemble"],
            hypothesis=f"Ensemble {method_norm.upper()} blend of [{', '.join(unique_models)}] with {folds} folds",
            validation_strategy="oof" if task_type == "binary_classification" else "kfold",
        )
        experiment.transition_to(ExperimentStatus.RUNNING)
        ws.repository.save_experiment(experiment)

        df = load_dataframe(dataset.path)
        X = df[features]
        target_col = dataset.target_column or run.config.target
        y = df[target_col]

        def _get_factory(mid: str):
            plugin = ws.plugin_registry.get_model_plugin(mid)
            if plugin:
                try:
                    return lambda: plugin.build_estimator(task_type=task_type)
                except (TypeError, ValueError):
                    return lambda: plugin.build_estimator()
            from automl.plugins.models.sklearn_models import build_sklearn_model
            return lambda: build_sklearn_model(mid, task_type)

        factories = {m: _get_factory(m) for m in unique_models}

        if task_type == "binary_classification":
            from automl.engine.ensemble.oof import evaluate_oof

            test_slice = X.iloc[:min(len(X), 10)].copy()
            group_col = getattr(run.config, "group_column", None)
            groups = df[group_col].values if (group_col and group_col in df.columns) else None
            oof_res = evaluate_oof(
                X=X,
                y=y,
                X_test=test_slice,
                factories=factories,
                folds=folds,
                seed=run.config.random_seed,
                method=method_norm,
                meta_model=meta_model,
                groups=groups,
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
                _is_minimizing_metric,
                _sklearn_scoring,
            )

            estimators = [(m, factories[m]()) for m in unique_models]
            voting_mode = "soft"
            opt_w = (method_norm == "simplex")
            if method_norm == "rank" and "classification" in task_type:
                voting_mode = "rank"
            estimator = VotingEnsembleEstimator(
                estimators=estimators,
                task_type=task_type,
                voting=voting_mode,
                optimize_weights=opt_w,
            )
            pipeline = _build_pipeline(X, estimator)
            cv_splitter = KFold(n_splits=folds, shuffle=True, random_state=run.config.random_seed)

            scoring = _sklearn_scoring(
                run.config.metric, task_type, ws.plugin_registry
            )
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=1,
            )
            is_neg = (
                (isinstance(scoring, str) and scoring.startswith("neg_"))
                or _is_minimizing_metric(run.config.metric, ws.plugin_registry)
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
        ws.repository.save_trial(trial)

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
        ws.repository.save_trial_result(trial_result)

        experiment.transition_to(ExperimentStatus.COMPLETED)
        ws.repository.save_experiment(experiment)

        ws._emit(
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
