from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from automl import __version__
from automl.artifacts.model_artifact import ModelArtifact
from automl.engine.profiling.dataset_profiler import load_dataframe
from automl.engine.training.sklearn_trainer import SklearnTrainer

if TYPE_CHECKING:
    from automl.application.services.workspace import AutoMLWorkspace


class InferenceService:
    """Application service for running inference, generating competition submissions, and exporting portable artifacts."""

    def __init__(self, workspace: AutoMLWorkspace) -> None:
        self.workspace = workspace

    def _resolve_validated_trial_chain(
        self,
        run_id: str,
        *,
        experiment_id: str | None = None,
        trial_id: str | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        """Strict membership: Run → Experiment → Trial → succeeded TrialResult.

        Returns ``(run, experiment, trial, trial_result)``.
        """
        run = self.workspace._get_run(run_id)

        if trial_id:
            trial = self.workspace.repository.get_trial(trial_id)
            if trial is None:
                raise KeyError(f"Trial '{trial_id}' not found.")
            experiment = self.workspace.repository.get_experiment(trial.experiment_id)
            if experiment is None:
                raise KeyError(f"Experiment '{trial.experiment_id}' not found for trial '{trial_id}'.")
            if experiment_id is not None and experiment.id != experiment_id:
                raise ValueError(
                    f"Trial '{trial_id}' belongs to experiment '{experiment.id}', "
                    f"not requested experiment '{experiment_id}'."
                )
            if experiment.run_id != run.id:
                raise ValueError(
                    f"Trial '{trial_id}' (experiment '{experiment.id}') does not belong to run '{run_id}'."
                )
            results = self.workspace.repository.list_trial_results(experiment.id)
            trial_result = next((r for r in results if r.trial_id == trial.id and r.succeeded), None)
            if trial_result is None:
                raise ValueError(
                    f"Trial '{trial_id}' has no successful evaluation result; "
                    "refusing inference/export without a validated trial."
                )
            return run, experiment, trial, trial_result

        if experiment_id:
            experiment = self.workspace.repository.get_experiment(experiment_id)
            if experiment is None:
                raise KeyError(f"Experiment '{experiment_id}' not found.")
            if experiment.run_id != run.id:
                raise ValueError(f"Experiment '{experiment_id}' does not belong to run '{run_id}'.")
            leaderboard = self.workspace.repository.get_leaderboard(run.id, include_failed=False)
            exp_results = [
                r for r in leaderboard
                if r.experiment_id == experiment_id and getattr(r, "succeeded", True)
            ]
            if not exp_results:
                raise ValueError(
                    f"Experiment '{experiment_id}' has no successful trials on the leaderboard; "
                    "refusing inference/export of an unevaluated model."
                )
            best = exp_results[0]
            trial = self.workspace.repository.get_trial(best.trial_id)
            if trial is None:
                raise KeyError(f"Trial '{best.trial_id}' not found for experiment '{experiment_id}'.")
            if trial.experiment_id != experiment.id:
                raise ValueError(
                    f"Leaderboard trial '{trial.id}' does not belong to experiment '{experiment.id}'."
                )
            return run, experiment, trial, best

        leaderboard = self.workspace.repository.get_leaderboard(run.id, include_failed=False)
        valid = [r for r in leaderboard if getattr(r, "succeeded", True)]
        if not valid:
            raise ValueError(
                f"Run '{run_id}' has no successful trials; refusing inference/export without validation."
            )
        best = valid[0]
        experiment = self.workspace.repository.get_experiment(best.experiment_id)
        if experiment is None:
            raise KeyError(f"Experiment '{best.experiment_id}' not found.")
        if experiment.run_id != run.id:
            raise ValueError(
                f"Leaderboard experiment '{experiment.id}' does not belong to run '{run_id}'."
            )
        trial = self.workspace.repository.get_trial(best.trial_id)
        if trial is None:
            raise KeyError(f"Trial '{best.trial_id}' not found.")
        if trial.experiment_id != experiment.id:
            raise ValueError(
                f"Trial '{trial.id}' does not belong to experiment '{experiment.id}'."
            )
        return run, experiment, trial, best

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
        run, experiment, trial, _trial_result = self._resolve_validated_trial_chain(
            run_id, experiment_id=experiment_id, trial_id=trial_id
        )
        dataset = self.workspace._get_dataset(run.dataset_id)
        target_model_id = trial.model_id
        parameters = dict(trial.parameters or {})

        test_df = load_dataframe(test_dataset_path)
        if experiment.validation_strategy == "oof":
            from automl.application.services.oof_submission import (
                cached_oof_predictions,
            )

            raw_preds = cached_oof_predictions(self.workspace, run_id, experiment.id, test_dataset_path, predict_proba)
        else:
            feature_names = list(experiment.feature_names)
            column_transforms = list(getattr(experiment, "column_transforms", None) or [])
            derived_features = list(getattr(experiment, "derived_features", None) or [])

            train_df = load_dataframe(dataset.path)
            # Parity with SklearnTrainer.run / export: materialize derived cols then apply transforms.
            if derived_features:
                from automl.domain.features.derived_feature import (
                    DerivedFeatureDefinition,
                )
                from automl.engine.features.generation.derived_feature_engine import (
                    DerivedFeatureEngine,
                    assert_derived_specs_safe_for_cross_validation,
                )

                assert_derived_specs_safe_for_cross_validation(derived_features)
                derived_engine = DerivedFeatureEngine()
                for df_spec in derived_features:
                    defn = (
                        DerivedFeatureDefinition.from_dict(df_spec)
                        if isinstance(df_spec, dict)
                        else df_spec
                    )
                    if defn.name not in train_df.columns:
                        train_df, eval_res = derived_engine.apply(
                            train_df,
                            defn,
                            allow_python_code=False,
                            allow_stateful_aggregates=False,
                        )
                        if not eval_res.is_valid:
                            raise ValueError(
                                f"Failed to compute derived feature {defn.name}: {eval_res.error_message}"
                            )
                    if defn.name not in test_df.columns:
                        test_df, eval_res = derived_engine.apply(
                            test_df,
                            defn,
                            allow_python_code=False,
                            allow_stateful_aggregates=False,
                        )
                        if not eval_res.is_valid:
                            raise ValueError(
                                f"Failed to compute derived feature {defn.name} on test: {eval_res.error_message}"
                            )

            X_train = train_df[feature_names]
            y_train = train_df[dataset.target_column]

            missing_features = [f for f in feature_names if f not in test_df.columns]
            if missing_features:
                raise ValueError(f"Test dataset is missing required features: {missing_features}")
            X_test = test_df[feature_names]

            fit_params = dict(parameters or {})
            if column_transforms:
                fit_params["column_transforms"] = column_transforms

            trainer = SklearnTrainer(plugin_registry=self.workspace.plugin_registry)
            preds = trainer.fit_and_predict(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                model_id=target_model_id,
                task_type=dataset.task_type,
                parameters=fit_params,
                predict_proba=predict_proba,
            )
            raw_preds = preds.tolist() if hasattr(preds, "tolist") else list(preds)

        if template_path is not None:
            try:
                template_df = load_dataframe(template_path)
            except Exception as e:
                raise ValueError(f"Template CSV is empty or invalid: {e}") from e
            if template_df.empty:
                raise ValueError("Template CSV is empty.")
            if id_column:
                if id_column not in template_df.columns:
                    raise ValueError(f"Specified ID column '{id_column}' not found in template CSV.")
                if id_column not in test_df.columns:
                    raise ValueError(f"Specified ID column '{id_column}' not found in test dataset.")
                actual_id_col = id_column
            else:
                candidate_id = next(
                    (
                        c
                        for c in template_df.columns
                        if c in test_df.columns
                        and (c.lower() in {"id", "passengerid", "customer_id", "guid"} or c.lower().endswith("_id"))
                    ),
                    None,
                )
                if candidate_id is None:
                    first_col = template_df.columns[0]
                    if first_col in test_df.columns:
                        candidate_id = first_col
                    else:
                        raise ValueError(
                            f"Could not automatically detect matching ID column between template {list(template_df.columns)} and test dataset {list(test_df.columns)}. Please specify id_column."
                        )
                actual_id_col = candidate_id

            test_preds_series = pd.Series(raw_preds, index=test_df[actual_id_col].values)
            aligned_preds = template_df[actual_id_col].map(test_preds_series)
            if aligned_preds.isna().any():
                missing_mask = aligned_preds.isna()
                missing_sample = template_df[actual_id_col][missing_mask].head(5).tolist()
                raise ValueError(
                    f"Template contains {missing_mask.sum()} IDs not found in the test dataset predictions (e.g. {missing_sample})."
                )
            return aligned_preds.tolist()

        return raw_preds

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
        preds = self.predict(
            run_id=run_id,
            test_dataset_path=test_dataset_path,
            experiment_id=experiment_id,
            trial_id=trial_id,
            predict_proba=predict_proba,
        )
        return self._write_submission(
            run_id,
            test_dataset_path,
            output_path,
            preds,
            id_column,
            template_path,
            predict_proba,
        )

    def export_model_artifact(
        self,
        run_id: str,
        experiment_id: str | None = None,
        trial_id: str | None = None,
    ) -> ModelArtifact:
        """Fits and exports a standalone, portable ModelArtifact for the specified or winning trial."""
        run, experiment, target_trial, trial_result = self._resolve_validated_trial_chain(
            run_id, experiment_id=experiment_id, trial_id=trial_id
        )
        dataset = self.workspace._get_dataset(run.dataset_id)
        target_exp_id = experiment.id

        train_df = load_dataframe(dataset.path)

        # Replay experiment recipe so export matches the validated trial pipeline.
        derived_features = list(getattr(experiment, "derived_features", []) or [])
        column_transforms = list(getattr(experiment, "column_transforms", []) or [])
        if derived_features:
            from automl.domain.features.derived_feature import DerivedFeatureDefinition
            from automl.engine.features.generation.derived_feature_engine import (
                DerivedFeatureEngine,
                assert_derived_specs_safe_for_cross_validation,
            )

            assert_derived_specs_safe_for_cross_validation(derived_features)
            derived_engine = DerivedFeatureEngine()
            for df_spec in derived_features:
                defn = (
                    DerivedFeatureDefinition.from_dict(df_spec)
                    if isinstance(df_spec, dict)
                    else df_spec
                )
                if defn.name not in train_df.columns:
                    train_df, eval_res = derived_engine.apply(
                        train_df,
                        defn,
                        allow_python_code=False,
                        allow_stateful_aggregates=False,
                    )
                    if not eval_res.is_valid:
                        raise ValueError(
                            f"Failed to materialize derived feature '{defn.name}' for export: "
                            f"{eval_res.error_message}"
                        )

        feature_names = experiment.feature_names
        missing = [c for c in feature_names if c not in train_df.columns]
        if missing:
            raise ValueError(
                f"Cannot export model: experiment features missing from dataset after recipe replay: {missing}"
            )
        X_train = train_df[feature_names]
        y_train = train_df[dataset.target_column]

        params = dict(target_trial.parameters or {})
        if column_transforms and "column_transforms" not in params:
            params["column_transforms"] = column_transforms
        if run.config and run.config.extra:
            if "text_columns" in run.config.extra and "text_columns" not in params:
                params["text_columns"] = run.config.extra["text_columns"]
            if "image_columns" in run.config.extra and "image_columns" not in params:
                params["image_columns"] = run.config.extra["image_columns"]
            if "image_model" in run.config.extra and "image_model" not in params:
                params["image_model"] = run.config.extra["image_model"]

        trainer = SklearnTrainer(plugin_registry=self.workspace.plugin_registry)
        pipeline, adapter = trainer.fit_pipeline(
            X_train=X_train,
            y_train=y_train,
            model_id=target_trial.model_id,
            task_type=dataset.task_type,
            parameters=params,
        )

        score = float(getattr(trial_result, "primary_score", 0.0) or 0.0)
        metric = str(getattr(trial_result, "primary_metric", None) or "score")

        dep_versions: dict[str, str] = {"catml": __version__}
        for name in (
            "numpy",
            "pandas",
            "scikit-learn",
            "joblib",
            "lightgbm",
            "xgboost",
            "catboost",
            "optuna",
            "torch",
            "torchvision",
            "timm",
            "pillow",
            "pyarrow",
        ):
            try:
                dep_versions[name] = version(name)
            except PackageNotFoundError:
                pass

        estimator_step = pipeline[-1] if hasattr(pipeline, "__getitem__") and hasattr(pipeline, "steps") else pipeline

        dataset_hash = None
        if hasattr(dataset, "path") and dataset.path:
            dp = Path(dataset.path)
            if dp.is_file():
                digest = hashlib.sha256()
                with dp.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                dataset_hash = digest.hexdigest()

        arts = getattr(trial_result, "artifacts", {}) or {}
        seed_val = getattr(target_trial, "seed", None) or getattr(getattr(run, "config", None), "random_seed", None)
        strategy_val = getattr(experiment, "validation_strategy", None) or getattr(getattr(run, "config", None), "validation_strategy", None)
        folds_val = arts.get("cv_folds") or getattr(getattr(run, "config", None), "cv_folds", None)

        provenance = {
            "catml_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "estimator_class": type(estimator_step).__name__,
            "estimator_module": type(estimator_step).__module__,
            "dependencies": dep_versions,
            "dataset_id": dataset.id,
            "dataset_path": str(dataset.path) if hasattr(dataset, "path") else None,
            "dataset_hash": dataset_hash,
            "content_hash": dataset_hash,
            "config_hash": arts.get("config_hash"),
            "seed": seed_val,
            "trial_seed": getattr(target_trial, "seed", None),
            "run_seed": getattr(getattr(run, "config", None), "random_seed", None),
            "validation_strategy": strategy_val,
            "cv_folds": folds_val,
            "cv_fold_ids": arts.get("cv_fold_ids"),
            "cv_fold_assignment_hash": arts.get("cv_fold_assignment_hash"),
            "backend": arts.get("backend", "native"),
        }

        return ModelArtifact(
            pipeline=pipeline,
            model_id=target_trial.model_id,
            task_type=dataset.task_type,
            feature_names=feature_names,
            target_name=dataset.target_column,
            target_adapter=adapter,
            metric=metric,
            score=score,
            parameters=target_trial.parameters or {},
            metadata={
                "run_id": run_id,
                "experiment_id": target_exp_id,
                "trial_id": target_trial.id,
                "column_transforms": column_transforms,
                "derived_features": derived_features,
            },
            provenance=provenance,
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
        if self.workspace.execution_check:
            self.workspace.execution_check()
        run = self.workspace._get_run(run_id)
        dataset = self.workspace._get_dataset(run.dataset_id)

        test_df = load_dataframe(test_dataset_path)

        template_used = None
        if template_path is not None:
            try:
                template_df = load_dataframe(template_path)
            except Exception as e:
                raise ValueError(f"Template CSV is empty or invalid: {e}") from e
            if template_df.empty:
                raise ValueError("Template CSV is empty.")

            if id_column:
                if id_column not in template_df.columns:
                    raise ValueError(f"Specified ID column '{id_column}' not found in template CSV.")
                if id_column not in test_df.columns:
                    raise ValueError(f"Specified ID column '{id_column}' not found in test dataset.")
                actual_id_col = id_column
            else:
                candidate_id = next(
                    (
                        c
                        for c in template_df.columns
                        if c in test_df.columns
                        and (c.lower() in {"id", "passengerid", "customer_id", "guid"} or c.lower().endswith("_id"))
                    ),
                    None,
                )
                if candidate_id is None:
                    first_col = template_df.columns[0]
                    if first_col in test_df.columns:
                        candidate_id = first_col
                    else:
                        raise ValueError(
                            f"Could not automatically detect matching ID column between template {list(template_df.columns)} and test dataset {list(test_df.columns)}. Please pass id_column explicitly."
                        )
                actual_id_col = candidate_id

            target_cols = [c for c in template_df.columns if c != actual_id_col]
            if not target_cols:
                raise ValueError("Template CSV must contain at least one target column in addition to the ID column.")

            test_preds_series = pd.Series(preds, index=test_df[actual_id_col].values)
            aligned_preds = template_df[actual_id_col].map(test_preds_series)
            if aligned_preds.isna().any():
                missing_mask = aligned_preds.isna()
                missing_sample = template_df[actual_id_col][missing_mask].head(5).tolist()
                raise ValueError(
                    f"Template contains {missing_mask.sum()} IDs not found in the test dataset predictions (e.g. {missing_sample})."
                )

            submission_df = template_df.copy()
            target_col = target_cols[0]
            submission_df[target_col] = aligned_preds.values
            submission_df = submission_df[template_df.columns]
            template_used = str(Path(template_path).resolve())
        else:
            if id_column:
                if id_column not in test_df.columns:
                    raise ValueError(f"ID column '{id_column}' not found in test dataset.")
                ids = test_df[id_column]
                actual_id_col = id_column
            else:
                candidate_id = next(
                    (c for c in ["id", "Id", "ID", "PassengerId", "customer_id"] if c in test_df.columns), None
                )
                if candidate_id:
                    ids = test_df[candidate_id]
                    actual_id_col = candidate_id
                else:
                    ids = pd.Series(range(len(test_df)), name="id")
                    actual_id_col = "id"

            target_col = dataset.target_column or "prediction"
            submission_df = pd.DataFrame({
                actual_id_col: ids,
                target_col: preds,
            })

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        submission_df.to_csv(out, index=False)

        self.workspace._emit(
            "SubmissionGenerated",
            {
                "run_id": run_id,
                "output_path": str(out.resolve()),
                "row_count": len(submission_df),
            },
            run_id=run_id,
        )

        return {
            "output_path": str(out.resolve()),
            "row_count": len(submission_df),
            "id_column": actual_id_col,
            "target_column": target_col,
            "predict_proba": predict_proba,
            "template_used": template_used,
        }
