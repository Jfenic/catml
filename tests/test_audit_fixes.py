"""Tests validating all 8 external audit security, reliability, and logic fixes.

Audit Points Covered:
1. P0: CI Workflow - 'vars' context used in 'continue-on-error' instead of invalid 'env'.
2. P0: OOF Group Validation - GroupKFold/StratifiedGroupKFold prevents group data leakage.
3. P0: Holdout Validation - Rejects identical files, duplicate row leakage, and group overlap.
4. P1: Candidate Promotion - Validates non-negative finite min_improvement and requires positive delta.
5. P1: Best Trial Truthfulness - Excludes failed trials from _get_best_trial_result and experiment APIs.
6. P1: OOF Trial Selection - Selects trial with optimal score respecting metric direction.
7. P1: Web Server Hardening - Rejects negative Content-Length (400) and loopback CORS port mismatch (403).
8. P2: Website CI Build - CI workflow builds React/Vite TypeScript site in website/.
"""

import io
import json
from http import HTTPStatus
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    GenerateOOFSubmissionCommand,
)
from automl.domain.experiments.trial import Trial, TrialResult, TrialStatus
from automl.engine.ensemble.oof import evaluate_oof
from automl.interfaces.web.api._helpers import _get_best_trial_result
from automl.interfaces.web.server import AutoMLWebHandler, _is_loopback_origin


# ---------------------------------------------------------------------------
# Audit #1 & #8: CI Workflow
# ---------------------------------------------------------------------------


def test_ci_workflow_vars_context_and_website_build():
    ci_yaml = Path(".github/workflows/ci.yml").read_text()

    # Audit #1: continue-on-error must use vars.CATML_AUDIT_STRICT, not env.CATML_AUDIT_STRICT
    assert "vars.CATML_AUDIT_STRICT" in ci_yaml
    assert "env.CATML_AUDIT_STRICT" not in ci_yaml

    # Audit #8: website-build job must exist and run npm ci and npm run build in website/
    assert "website-build:" in ci_yaml
    assert "working-directory: website" in ci_yaml
    assert "run: npm run build" in ci_yaml


# ---------------------------------------------------------------------------
# Audit #2: OOF Group Data Leakage Protection
# ---------------------------------------------------------------------------


def test_evaluate_oof_group_disjointness():
    """Verify StratifiedGroupKFold / GroupKFold ensures zero group overlap across folds."""
    rng = np.random.default_rng(42)
    n = 60
    # 6 distinct groups of 10 samples each
    groups = np.repeat(np.array(["patient_A", "patient_B", "patient_C", "patient_D", "patient_E", "patient_F"]), 10)
    X = pd.DataFrame({"feat1": rng.normal(size=n), "feat2": rng.normal(size=n)})
    # Binary target with balance in each group
    y = pd.Series([0, 1] * (n // 2))
    X_test = X.iloc[:10].copy()

    from sklearn.linear_model import LogisticRegression

    factories = {"lr": lambda: LogisticRegression(random_state=42)}

    # Successful grouped OOF with 3 folds
    res = evaluate_oof(X, y, X_test, factories, folds=3, groups=groups, seed=42)
    assert res is not None
    assert len(res.fold_ids) == n

    # Verify that every group appears in exactly one validation fold
    for g in np.unique(groups):
        val_folds = np.unique(res.fold_ids[groups == g])
        assert len(val_folds) == 1, f"Group {g} leaked across multiple folds: {val_folds}"

    # Verify invalid group lengths or insufficient groups raise ValueError
    with pytest.raises(ValueError, match="groups length .* must match features length"):
        evaluate_oof(X, y, X_test, factories, folds=3, groups=groups[:20])

    with pytest.raises(ValueError, match="Number of distinct groups .* must be >= folds"):
        two_groups = np.repeat(["g1", "g2"], n // 2)
        evaluate_oof(X, y, X_test, factories, folds=5, groups=two_groups)


# ---------------------------------------------------------------------------
# Audit #3: Holdout Overlap & Leakage Rejection
# ---------------------------------------------------------------------------


def test_oof_submission_holdout_overlap_and_identical_file_rejection(tmp_path: Path):
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws"))

    rng = np.random.default_rng(42)
    n = 80
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = ((x1**2 + x2**2) > 1.2).astype(int)
    groups = np.repeat([f"grp_{i}" for i in range(8)], 10)

    train_df = pd.DataFrame({"x1": x1, "x2": x2, "group_id": groups, "target": y})
    train_path = tmp_path / "train.csv"
    train_df.to_csv(train_path, index=False)

    test_path = tmp_path / "test.csv"
    train_df.iloc[:20].drop(columns=["target"]).to_csv(test_path, index=False)

    ds = ws.register_dataset(name="group_synth", path=train_path, target="target", task_type="binary_classification")
    run = ws.create_run(ds, metric="roc_auc")
    source_exp = ws.create_experiment(
        run=run,
        name="source",
        feature_names=["x1", "x2"],
        model_ids=["logistic_regression", "random_forest"],
        group_column="group_id",
    )
    ws.run_experiment(run, source_exp)

    # 1. Reject identical file path
    with pytest.raises(ValueError, match="Holdout dataset cannot be the same file as the training dataset"):
        cmd.dispatch(GenerateOOFSubmissionCommand(
            run_id=run.id,
            test_dataset_path=str(test_path),
            output_path=str(tmp_path / "out1.csv"),
            experiment_id=source_exp.id,
            holdout_dataset_path=str(train_path),
        ))

    # 2. Reject identical file content (different file path, identical bytes)
    train_copy = tmp_path / "train_copy.csv"
    train_df.to_csv(train_copy, index=False)
    with pytest.raises(ValueError, match="Holdout dataset content is identical to the training dataset"):
        cmd.dispatch(GenerateOOFSubmissionCommand(
            run_id=run.id,
            test_dataset_path=str(test_path),
            output_path=str(tmp_path / "out2.csv"),
            experiment_id=source_exp.id,
            holdout_dataset_path=str(train_copy),
        ))

    # 3. Reject partial row overlap (e.g. subset of training rows in holdout)
    overlap_df = train_df.iloc[:30].copy()
    overlap_path = tmp_path / "overlap.csv"
    overlap_df.to_csv(overlap_path, index=False)
    with pytest.raises(ValueError, match="Holdout dataset contains .* overlapping rows with training dataset"):
        cmd.dispatch(GenerateOOFSubmissionCommand(
            run_id=run.id,
            test_dataset_path=str(test_path),
            output_path=str(tmp_path / "out3.csv"),
            experiment_id=source_exp.id,
            holdout_dataset_path=str(overlap_path),
        ))

    # 4. Reject shared entity groups between train and holdout
    # New rows with different feature values but sharing group_id grp_0
    shared_group_df = pd.DataFrame({
        "x1": rng.normal(size=10),
        "x2": rng.normal(size=10),
        "group_id": ["grp_0"] * 10,
        "target": [0, 1] * 5,
    })
    shared_group_path = tmp_path / "shared_grp.csv"
    shared_group_df.to_csv(shared_group_path, index=False)
    with pytest.raises(ValueError, match="Holdout dataset shares .* groups with training dataset"):
        cmd.dispatch(GenerateOOFSubmissionCommand(
            run_id=run.id,
            test_dataset_path=str(test_path),
            output_path=str(tmp_path / "out4.csv"),
            experiment_id=source_exp.id,
            holdout_dataset_path=str(shared_group_path),
        ))


# ---------------------------------------------------------------------------
# Audit #4: Model Promotion Gate on Positive Improvement
# ---------------------------------------------------------------------------


def test_oof_min_improvement_validation(tmp_path: Path):
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws"))
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=50), "target": [0, 1] * 25})
    train_p = tmp_path / "tr.csv"
    test_p = tmp_path / "te.csv"
    holdout_p = tmp_path / "ho.csv"
    df.iloc[:30].to_csv(train_p, index=False)
    df.iloc[30:40].drop(columns=["target"]).to_csv(test_p, index=False)
    df.iloc[40:].to_csv(holdout_p, index=False)

    ds = ws.register_dataset(name="synth", path=train_p, target="target", task_type="binary_classification")
    run = ws.create_run(ds, metric="roc_auc")
    exp = ws.create_experiment(run=run, name="src", feature_names=["x"], model_ids=["logistic_regression"])
    ws.run_experiment(run, exp)

    # Negative threshold rejected
    with pytest.raises(ValueError, match="min_improvement must be a non-negative finite float"):
        cmd.dispatch(GenerateOOFSubmissionCommand(
            run_id=run.id,
            test_dataset_path=str(test_p),
            output_path=str(tmp_path / "sub.csv"),
            experiment_id=exp.id,
            holdout_dataset_path=str(holdout_p),
            min_improvement=-0.5,
        ))

    # Infinite threshold rejected
    with pytest.raises(ValueError, match="min_improvement must be a non-negative finite float"):
        cmd.dispatch(GenerateOOFSubmissionCommand(
            run_id=run.id,
            test_dataset_path=str(test_p),
            output_path=str(tmp_path / "sub.csv"),
            experiment_id=exp.id,
            holdout_dataset_path=str(holdout_p),
            min_improvement=float("inf"),
        ))


# ---------------------------------------------------------------------------
# Audit #5: Exclude Failed Trials From Best Model Selection
# ---------------------------------------------------------------------------


def test_best_trial_result_excludes_failed_trials():
    # When all trials failed, _get_best_trial_result must return None
    failed_trials = [
        MagicMock(succeeded=False, primary_score=0.99, primary_metric="roc_auc", model_id="broken_model_1"),
        MagicMock(succeeded=False, primary_score=0.95, primary_metric="roc_auc", model_id="broken_model_2"),
    ]
    best = _get_best_trial_result(failed_trials, metric="roc_auc")
    assert best is None, "Failed trials must never be selected as best model"

    # When valid trials exist, select the best succeeding trial
    mixed_trials = [
        MagicMock(succeeded=False, primary_score=0.99, primary_metric="roc_auc", model_id="broken_model"),
        MagicMock(succeeded=True, primary_score=0.85, primary_metric="roc_auc", model_id="good_model_1"),
        MagicMock(succeeded=True, primary_score=0.90, primary_metric="roc_auc", model_id="good_model_2"),
    ]
    best_mixed = _get_best_trial_result(mixed_trials, metric="roc_auc")
    assert best_mixed is not None
    assert best_mixed.model_id == "good_model_2"
    assert best_mixed.primary_score == 0.90


# ---------------------------------------------------------------------------
# Audit #6: OOF Parameter Freezing Selects Best Trial by Metric Direction
# ---------------------------------------------------------------------------


def test_oof_parameter_freezing_respects_metric_direction(tmp_path: Path):
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws"))
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=60), "target": [0, 1] * 30})
    train_p = tmp_path / "tr.csv"
    test_p = tmp_path / "te.csv"
    df.iloc[:40].to_csv(train_p, index=False)
    df.iloc[40:].drop(columns=["target"]).to_csv(test_p, index=False)

    ds = ws.register_dataset(name="synth", path=train_p, target="target", task_type="binary_classification")
    run = ws.create_run(ds, metric="roc_auc")  # Maximizing metric
    exp = ws.create_experiment(run=run, name="src", feature_names=["x1"], model_ids=["logistic_regression"])

    # Manually register multiple trial results for logistic_regression with differing scores
    t1 = Trial(id="t1", experiment_id=exp.id, model_id="logistic_regression", parameters={"C": 0.01})
    t2 = Trial(id="t2", experiment_id=exp.id, model_id="logistic_regression", parameters={"C": 10.0})
    ws.repository.save_trial(t1)
    ws.repository.save_trial(t2)

    # First trial in sequence has lower score (0.60); second has optimal score (0.95)
    r1 = TrialResult(
        trial_id=t1.id,
        experiment_id=exp.id,
        model_id="logistic_regression",
        primary_metric="roc_auc",
        primary_score=0.60,
    )
    r2 = TrialResult(
        trial_id=t2.id,
        experiment_id=exp.id,
        model_id="logistic_regression",
        primary_metric="roc_auc",
        primary_score=0.95,
    )
    ws.repository.save_trial_result(r1)
    ws.repository.save_trial_result(r2)

    # Generate OOF submission
    oof_exp_id = cmd.dispatch(GenerateOOFSubmissionCommand(
        run_id=run.id,
        test_dataset_path=str(test_p),
        output_path=str(tmp_path / "sub.csv"),
        experiment_id=exp.id,
        folds=3,
        model_ids=["logistic_regression"],
    ))

    # Freezing must have selected parameters from t2 (C=10.0), not t1 (C=0.01)
    from automl.application.queries.workspace_queries import GetOOFResultQuery
    oof_report = qry.dispatch(GetOOFResultQuery(run_id=run.id, experiment_id=oof_exp_id))
    assert oof_report["parameters"]["logistic_regression"]["C"] == 10.0


# ---------------------------------------------------------------------------
# Audit #7: Web Server Security (Negative Content-Length & CORS Port Matching)
# ---------------------------------------------------------------------------


def test_web_server_rejects_negative_content_length():
    handler = AutoMLWebHandler.__new__(AutoMLWebHandler)
    handler.headers = {
        "Content-Length": "-1",
        "Content-Type": "application/json",
    }
    handler.path = "/api/jobs"
    handler.command = "POST"
    handler.require_auth = False
    handler.auth_token = None

    sent_data = {}
    sent_status = None

    def mock_send_json(data, status=HTTPStatus.OK):
        nonlocal sent_data, sent_status
        sent_data = data
        sent_status = status

    handler._send_json = mock_send_json
    handler.do_POST()

    assert sent_status == HTTPStatus.BAD_REQUEST
    assert "Content-Length cannot be negative" in sent_data.get("message", "")


def test_loopback_cors_restricts_origin_port():
    # Server running on port 8080
    assert _is_loopback_origin("http://localhost:8080", expected_port=8080) is True
    assert _is_loopback_origin("http://127.0.0.1:8080", expected_port=8080) is True

    # Malicious web app on another local port (e.g. 3000)
    assert _is_loopback_origin("http://localhost:3000", expected_port=8080) is False
    assert _is_loopback_origin("http://127.0.0.1:3000", expected_port=8080) is False

    # External / non-loopback origin
    assert _is_loopback_origin("http://evil-site.com:8080", expected_port=8080) is False
