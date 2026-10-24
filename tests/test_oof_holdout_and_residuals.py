"""Tests for OOF holdout validation, candidate promotion gates, and stacking residuals."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import GenerateOOFSubmissionCommand
from automl.application.queries.workspace_queries import GetOOFResultQuery
from automl.engine.ensemble.oof import evaluate_oof


@pytest.fixture
def sample_binary_data():
    rng = np.random.default_rng(42)
    n_train = 100
    n_test = 30
    n_holdout = 40

    X_train = pd.DataFrame({
        "x1": rng.normal(size=n_train),
        "x2": rng.normal(size=n_train) * 2,
    })
    y_train = pd.Series((X_train["x1"] + X_train["x2"] > 0).astype(int))

    X_test = pd.DataFrame({
        "x1": rng.normal(size=n_test),
        "x2": rng.normal(size=n_test) * 2,
    })

    X_holdout = pd.DataFrame({
        "x1": rng.normal(size=n_holdout),
        "x2": rng.normal(size=n_holdout) * 2,
    })
    y_holdout = pd.Series((X_holdout["x1"] + X_holdout["x2"] > 0).astype(int))

    return X_train, y_train, X_test, X_holdout, y_holdout


def test_evaluate_oof_with_holdout_and_stacking_residuals(sample_binary_data):
    X_train, y_train, X_test, X_holdout, y_holdout = sample_binary_data

    factories = {
        "lr": lambda: LogisticRegression(random_state=42),
        "rf": lambda: RandomForestClassifier(n_estimators=10, random_state=42),
    }

    # Evaluate stacked blending with independent holdout
    res = evaluate_oof(
        X=X_train,
        y=y_train,
        X_test=X_test,
        factories=factories,
        folds=3,
        seed=42,
        method="stacked",
        meta_model="ridge",
        X_holdout=X_holdout,
        y_holdout=y_holdout,
    )

    assert res.holdout_scores is not None
    assert "oof_blend" in res.holdout_scores
    assert "lr" in res.holdout_scores
    assert "rf" in res.holdout_scores
    assert res.holdout_delta is not None
    assert isinstance(res.holdout_delta, float)

    # Check stacking residuals
    assert res.stacking_residuals is not None
    assert "mean_residual" in res.stacking_residuals
    assert "std_residual" in res.stacking_residuals
    assert "brier_score" in res.stacking_residuals
    assert "holdout_mean_residual" in res.stacking_residuals
    assert "holdout_std_residual" in res.stacking_residuals
    assert "holdout_brier_score" in res.stacking_residuals


def test_evaluate_oof_without_holdout_defaults(sample_binary_data):
    X_train, y_train, X_test, _, _ = sample_binary_data

    factories = {
        "lr": lambda: LogisticRegression(random_state=42),
        "rf": lambda: RandomForestClassifier(n_estimators=10, random_state=42),
    }

    res = evaluate_oof(
        X=X_train,
        y=y_train,
        X_test=X_test,
        factories=factories,
        folds=3,
        seed=42,
        method="average",
    )

    assert res.holdout_scores is None
    assert res.holdout_delta is None
    assert res.stacking_residuals is None


def test_oof_submission_candidate_promotion_gate(tmp_path: Path):
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws"))

    # Create synthetic train, test, and holdout datasets with non-linear circular boundary
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "target": ((x1**2 + x2**2) > 1.2).astype(int),
    })
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    holdout_path = tmp_path / "holdout.csv"

    df.iloc[:120].to_csv(train_path, index=False)
    df.iloc[120:140].drop(columns=["target"]).to_csv(test_path, index=False)
    # Ensure holdout has disjoint rows from train
    df.iloc[140:].to_csv(holdout_path, index=False)

    ds = ws.register_dataset(name="synth", path=train_path, target="target", task_type="binary_classification")
    run = ws.create_run(ds, metric="roc_auc")
    source_exp = ws.create_experiment(
        run=run,
        name="source",
        feature_names=["x1", "x2"],
        model_ids=["logistic_regression", "random_forest"],
    )
    ws.run_experiment(run, source_exp)

    # 1. Run without holdout -> not promoted
    out_path = tmp_path / "sub_default.csv"
    exp_id_1 = cmd.dispatch(GenerateOOFSubmissionCommand(
        run_id=run.id,
        test_dataset_path=str(test_path),
        output_path=str(out_path),
        experiment_id=source_exp.id,
        folds=3,
        method="average",
    ))
    report_1 = qry.dispatch(GetOOFResultQuery(run_id=run.id, experiment_id=exp_id_1))
    assert report_1["promoted"] is False
    assert "not independent holdout evidence" in report_1["evaluation"]

    # 2a. Negative min_improvement must be rejected
    with pytest.raises(ValueError, match="min_improvement must be a non-negative finite float"):
        cmd.dispatch(GenerateOOFSubmissionCommand(
            run_id=run.id,
            test_dataset_path=str(test_path),
            output_path=str(tmp_path / "sub_invalid.csv"),
            experiment_id=source_exp.id,
            folds=3,
            method="average",
            holdout_dataset_path=str(holdout_path),
            min_improvement=-1.0,
        ))

    # 2b. Run with holdout dataset and legitimate non-negative min_improvement -> promoted
    out_path_2 = tmp_path / "sub_holdout.csv"
    exp_id_2 = cmd.dispatch(GenerateOOFSubmissionCommand(
        run_id=run.id,
        test_dataset_path=str(test_path),
        output_path=str(out_path_2),
        experiment_id=source_exp.id,
        folds=3,
        method="average",
        holdout_dataset_path=str(holdout_path),
        min_improvement=0.01,
    ))
    report_2 = qry.dispatch(GetOOFResultQuery(run_id=run.id, experiment_id=exp_id_2))
    assert report_2["promoted"] is True
    assert report_2["holdout_score"] is not None
    assert report_2["holdout_delta"] is not None
    assert report_2["holdout_delta"] > 0.01
    assert "promoted candidate" in report_2["evaluation"]

    # 3. Run with holdout ratio and unattainable threshold -> not promoted
    out_path_3 = tmp_path / "sub_ratio.csv"
    exp_id_3 = cmd.dispatch(GenerateOOFSubmissionCommand(
        run_id=run.id,
        test_dataset_path=str(test_path),
        output_path=str(out_path_3),
        experiment_id=source_exp.id,
        folds=3,
        method="stacked",
        meta_model="ridge",
        holdout_ratio=0.2,
        min_improvement=0.99,  # Unattainable threshold -> not promoted
    ))
    report_3 = qry.dispatch(GetOOFResultQuery(run_id=run.id, experiment_id=exp_id_3))
    assert report_3["promoted"] is False
    assert report_3["stacking_residuals"] is not None
    assert "did not exceed threshold" in report_3["evaluation"]
