"""Tests for Provenance & Reproducibility metadata across trials, leaderboard, and model artifacts."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    RunExperimentCommand,
)
from automl.domain.datasets.content_hash import (
    compute_config_hash,
    compute_fold_assignment_hash,
    compute_file_content_hash,
    try_compute_file_content_hash,
)
from automl.facade import AutoML
from automl.artifacts.model_artifact import ModelArtifact


def test_content_hash_helpers_deterministic(tmp_path: Path):
    # Config hash determinism across dict key orderings
    cfg1 = {"model_id": "ridge", "metric": "roc_auc", "params": {"alpha": 1.0, "fit_intercept": True}}
    cfg2 = {"metric": "roc_auc", "params": {"fit_intercept": True, "alpha": 1.0}, "model_id": "ridge"}
    hash1 = compute_config_hash(cfg1)
    hash2 = compute_config_hash(cfg2)
    assert hash1 == hash2
    assert len(hash1) == 64

    # Fold assignment hash determinism
    arr1 = np.array([0, 1, 2, 0, 1, 2])
    arr2 = [0, 1, 2, 0, 1, 2]
    arr3 = [1, 0, 2, 0, 1, 2]
    assert compute_fold_assignment_hash(arr1) == compute_fold_assignment_hash(arr2)
    assert compute_fold_assignment_hash(arr1) != compute_fold_assignment_hash(arr3)

    # File hash
    sample_file = tmp_path / "sample.csv"
    sample_file.write_text("a,b,c\n1,2,3\n")
    f_hash = compute_file_content_hash(sample_file)
    assert len(f_hash) == 64
    assert try_compute_file_content_hash(sample_file) == f_hash
    assert try_compute_file_content_hash(tmp_path / "nonexistent.csv") is None


def test_trial_provenance_recorded_in_artifacts_and_database(tmp_path: Path):
    ws_dir = tmp_path / "ws_prov"
    csv_file = tmp_path / "train.csv"
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 3,
        "feat2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5] * 3,
        "label": [0, 1, 0, 1, 0, 1, 0, 1] * 3,
    })
    df.to_csv(csv_file, index=False)

    ws, cmd, qry = build_application(root_dir=str(ws_dir))
    dataset = ws.register_dataset(name="prov_data", path=csv_file, target="label")
    run = ws.create_run(dataset)

    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="exp_prov",
            hypothesis="Evaluate provenance recording",
            feature_names=["feat1", "feat2"],
            model_ids=["logistic_regression"],
            validation_strategy="stratified_kfold",
        )
    )

    results = cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=exp.id))
    assert len(results) == 1
    res = results[0]
    assert res.succeeded

    arts = res.artifacts
    assert arts is not None
    assert arts.get("seed") == 42
    assert arts.get("validation_strategy") == "stratified_kfold"
    assert arts.get("cv_folds") == 5
    assert arts.get("cv_fold_ids") == ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]
    assert len(arts.get("cv_fold_assignment_hash", "")) == 64
    assert len(arts.get("dataset_content_hash", "")) == 64
    assert len(arts.get("config_hash", "")) == 64
    assert arts.get("backend") == "native"
    assert "scikit-learn" in arts.get("library_versions", {})


def test_leaderboard_query_and_export_provenance_full_cycle(tmp_path: Path):
    ws_dir = tmp_path / "ws_full"
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
        "x2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] * 3,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3,
    })

    automl = AutoML(
        cv_folds=3,
        models=["logistic_regression"],
        workspace_dir=ws_dir,
        random_state=123,
    )
    result = automl.fit(df, target="target")
    assert result.best_model is not None

    # Check AutoMLResult.leaderboard() DataFrame has provenance columns
    lb_df = result.leaderboard()
    assert len(lb_df) >= 1
    row = lb_df.iloc[0]
    assert row["seed"] == 123
    assert row["cv_folds"] == 3
    assert row["cv_fold_ids"] == ["fold_0", "fold_1", "fold_2"]
    assert len(row["dataset_content_hash"]) == 64
    assert len(row["config_hash"]) == 64
    assert row["backend"] == "native"

    # Check exported ModelArtifact has comprehensive provenance
    artifact = result.best_model
    assert isinstance(artifact, ModelArtifact)
    prov = artifact.provenance
    assert prov["seed"] == 123
    assert prov["cv_folds"] == 3
    assert prov["cv_fold_ids"] == ["fold_0", "fold_1", "fold_2"]
    assert len(prov["cv_fold_assignment_hash"]) == 64
    assert len(prov["dataset_hash"]) == 64
    assert len(prov["content_hash"]) == 64
    assert len(prov["config_hash"]) == 64
    assert prov["backend"] == "native"
    assert "numpy" in prov["dependencies"]
    assert "scikit-learn" in prov["dependencies"]

    # Verify persistence & reload retains provenance exactly
    save_path = tmp_path / "exported_model.pkl"
    result.save_model(save_path)
    reloaded = AutoML.load_model(save_path)
    assert reloaded.provenance["seed"] == 123
    assert reloaded.provenance["cv_fold_ids"] == ["fold_0", "fold_1", "fold_2"]
    assert reloaded.provenance["dataset_hash"] == prov["dataset_hash"]
    assert reloaded.provenance["config_hash"] == prov["config_hash"]
