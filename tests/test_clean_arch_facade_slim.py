"""Tests for progressive clean architecture facade slimming (AutoMLWorkspace decomposition)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.application.services.dataset_service import DatasetService
from automl.application.services.ensemble_service import EnsembleService
from automl.application.services.experiment_execution_service import (
    ExperimentExecutionService,
)
from automl.application.services.experiment_planning_service import (
    ExperimentPlanningService,
)
from automl.application.services.experiment_service import ExperimentService
from automl.application.services.feature_service import FeatureEngineeringService
from automl.application.services.inference_service import InferenceService
from automl.application.services.optimization_service import OptimizationService
from automl.application.services.pipeline_service import PipelineExecutionService
from automl.application.services.run_service import RunService
from automl.application.services.workspace import AutoMLWorkspace
from automl.domain.experiments.trial import ExperimentStatus
from automl.domain.runs.states import RunStatus


def _sample_binary_df(tmp_path: Path, n: int = 50) -> Path:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "feature1": rng.normal(size=n),
            "feature2": rng.normal(size=n),
            "target": [0, 1] * (n // 2),
        }
    )
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)
    return p


def test_workspace_lazy_application_services(tmp_path: Path):
    ws = AutoMLWorkspace.create("facade_slim_test", root_dir=tmp_path / "ws")

    assert isinstance(ws.dataset_service, DatasetService)
    assert isinstance(ws.run_service, RunService)
    assert isinstance(ws.experiment_service, ExperimentService)
    assert isinstance(ws.planning_service, ExperimentPlanningService)
    assert isinstance(ws.ensemble_service, EnsembleService)
    assert isinstance(ws.experiment_execution_service, ExperimentExecutionService)
    assert isinstance(ws.optimization_service, OptimizationService)
    assert isinstance(ws.inference_service, InferenceService)
    assert isinstance(ws.pipeline_service, PipelineExecutionService)
    assert isinstance(ws.feature_service, FeatureEngineeringService)

    # Verify identity caching
    assert ws.dataset_service is ws.dataset_service
    assert ws.run_service is ws.run_service
    assert ws.experiment_service is ws.experiment_service
    assert ws.planning_service is ws.planning_service
    assert ws.ensemble_service is ws.ensemble_service

    # Verify back-reference to workspace
    assert ws.dataset_service.workspace is ws
    assert ws.run_service.workspace is ws
    assert ws.experiment_service.workspace is ws
    assert ws.planning_service.workspace is ws
    assert ws.ensemble_service.workspace is ws


def test_facade_delegation_lifecycle(tmp_path: Path):
    ws = AutoMLWorkspace.create("facade_lifecycle", root_dir=tmp_path / "ws")
    csv_path = _sample_binary_df(tmp_path)

    # 1. Dataset delegation
    dataset = ws.register_dataset(
        name="test_data",
        path=csv_path,
        target="target",
        task_type="binary_classification",
    )
    assert dataset.id.startswith("ds_")
    assert ws.get_dataset(dataset.id).id == dataset.id
    assert len(ws.list_datasets()) == 1

    # 2. Run delegation
    run = ws.create_run(
        dataset=dataset,
        metric="roc_auc",
        validation_strategy="kfold",
        cv_folds=3,
        random_seed=42,
    )
    assert run.id.startswith("run_")
    assert ws.get_run(run.id).id == run.id
    assert len(ws.list_runs()) == 1

    # 3. Planning delegation
    candidates = ws.plan_experiments(run.id)
    assert len(candidates) > 0
    assert ws.list_candidates(run.id)

    # 4. Experiment creation delegation
    exp = ws.create_experiment(
        run=run,
        name="log_reg_exp",
        feature_names=["feature1", "feature2"],
        model_ids=["logistic_regression"],
    )
    assert exp.id.startswith("exp_")
    assert ws.get_experiment(exp.id).id == exp.id
    assert len(ws.list_experiments(run.id)) == 1

    # 5. Execution and leaderboard delegation
    results = ws.run_experiment(run, exp)
    assert results
    assert any(r.succeeded for r in results)

    lb = ws.leaderboard(run)
    assert len(lb) >= 1
    assert "provenance" in lb[0]
    assert lb[0]["provenance"]["seed"] == 42
    assert lb[0]["provenance"]["cv_folds"] == 3

    # 6. Ensemble building delegation
    ens_exp = ws.build_ensemble(
        run_id=run.id,
        model_ids=["logistic_regression", "random_forest"],
        method="average",
        folds=3,
    )
    assert ens_exp.id.startswith("exp_")
    assert ens_exp.status == ExperimentStatus.COMPLETED

    ens_lb = ws.get_leaderboard(run)
    assert any(row["model_id"] == "voting_ensemble" for row in ens_lb)

    # 7. Run control delegation
    paused = ws.pause_run(run.id)
    assert paused.status == RunStatus.PAUSED
    ws.repository.save_checkpoint(run.id, exp.id, 0)
    resumed_results = ws.resume_run(run.id)
    assert resumed_results
    cancelled = ws.cancel_run(run.id)
    assert cancelled.status == RunStatus.CANCELLED


def test_service_property_setters(tmp_path: Path):
    ws = AutoMLWorkspace.create("setter_test", root_dir=tmp_path / "ws_setters")
    sentinel = object()
    ws.dataset_service = sentinel
    assert ws.dataset_service is sentinel

    ws.run_service = sentinel
    assert ws.run_service is sentinel

    ws.feature_service = sentinel
    assert ws.feature_service is sentinel


def test_run_service_exclude_and_include_model_consistency(tmp_path: Path):
    ws = AutoMLWorkspace.create("model_scoping", root_dir=tmp_path / "ws_scope")
    csv_path = _sample_binary_df(tmp_path)
    ds = ws.register_dataset("scope_ds", csv_path, target="target", task_type="binary_classification")
    run = ws.create_run(ds)

    # 1. Include model
    ws.include_model(run, "random_forest")
    assert "random_forest" in run.config.models_include
    assert "random_forest" not in run.config.models_exclude

    # 2. Exclude the same model -> must be removed from models_include
    ws.exclude_model(run, "random_forest")
    assert "random_forest" in run.config.models_exclude
    assert "random_forest" not in run.config.models_include

    # 3. Re-include -> must be removed from models_exclude
    ws.include_model(run, "random_forest")
    assert "random_forest" in run.config.models_include
    assert "random_forest" not in run.config.models_exclude


def test_dataset_service_get_meta_knowledge_with_trials(tmp_path: Path):
    ws = AutoMLWorkspace.create("meta_trials", root_dir=tmp_path / "ws_meta")
    csv_path = _sample_binary_df(tmp_path, n=60)
    ds = ws.register_dataset("meta_ds", csv_path, target="target", task_type="binary_classification")
    run = ws.create_run(ds, metric="roc_auc", cv_folds=2, random_seed=42)

    exp = ws.create_experiment(
        run=run,
        name="meta_exp",
        feature_names=["feature1", "feature2"],
        model_ids=["logistic_regression", "random_forest"],
    )
    results = ws.run_experiment(run, exp)
    assert len(results) == 2

    # Now call get_meta_knowledge - it must correctly find trial results via get_leaderboard(r.id)
    knowledge = ws.get_meta_knowledge(ds.id)
    assert knowledge is not None
    assert knowledge.dataset_name == "meta_ds"
    assert len(knowledge.historical_rankings) > 0


def test_feature_engineering_service_direct_and_cqrs(tmp_path: Path):
    from automl.application.bootstrap import build_application
    from automl.application.commands.workspace_commands import (
        ExcludeFeatureCommand,
        PrioritizeFeatureCommand,
        PromoteCandidateFeatureSetCommand,
        SelectFeaturesCommand,
    )

    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws_feature"))
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "feature1": rng.normal(size=60),
            "feature2": rng.normal(size=60),
            "feature3": rng.normal(size=60),
            "feature4": rng.normal(size=60),
            "feature5": rng.normal(size=60),
            "target": [0, 1] * 30,
        }
    )
    csv_path = tmp_path / "feat_sample.csv"
    df.to_csv(csv_path, index=False)
    ds = ws.register_dataset("feat_ds", csv_path, target="target", task_type="binary_classification")
    run = ws.create_run(ds)

    # 1. Feature operations via FeatureEngineeringService
    reg = ws.feature_service.get_feature_registry(ds.id)
    assert reg.get("feature1") is not None

    fset = ws.feature_service.create_feature_set(ds.id, "my_features", ["feature1"])
    assert fset.name == "my_features"
    listed_fsets = ws.feature_service.list_feature_sets(ds.id)
    assert len(listed_fsets) == 1

    ws.feature_service.prioritize_feature(ds.id, "feature1", score=0.8, run_id=run.id)
    assert "feature1" in run.config.features_priority

    ws.feature_service.exclude_feature(ds.id, "feature2", run_id=run.id)
    assert "feature2" in run.config.features_excluded

    # 2. Task types catalog from DatasetService
    task_types = ws.list_task_types()
    assert len(task_types) >= 3

    # 3. CQRS Select and Promote with allow_unverified_promotion=True
    candidates = cmd.dispatch(SelectFeaturesCommand(run.id))
    assert len(candidates) > 0
    top_cand = candidates[0]

    promoted = cmd.dispatch(
        PromoteCandidateFeatureSetCommand(
            run_id=run.id,
            candidate_id=top_cand.id,
            new_name="cqrs_promoted",
            allow_unverified_promotion=True,
        )
    )
    assert promoted.name == "cqrs_promoted"
    assert len(promoted.feature_names) > 0
