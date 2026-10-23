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
