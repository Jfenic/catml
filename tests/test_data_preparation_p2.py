"""Data Preparation P2: CLI, Python facade, fold-safe pipeline bridging, and recipe parity."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from automl.application.analysis.commands import CreateStudyCommand
from automl.application.bootstrap import build_application
from automl.application.recipes.commands import CreateExperimentRecipeFromEvidenceCommand
from automl.application.recipes.queries import GetExperimentRecipeQuery
from automl.domain.analysis.models import (
    AnalysisHypothesis,
    AnalysisRun,
    EvidenceLink,
    StatisticalFinding,
)
from automl.domain.preparation.models import (
    PreparationOperation,
    PreparationScope,
    PreparationStep,
)
from automl.domain.preparation.pipeline_bridge import pipeline_steps_to_column_transforms
from automl.domain.preparation.validation import PreparationValidationError
from automl.facade import AutoML
from automl.interfaces.cli.main import main


def _create_sample_df() -> pd.DataFrame:
    """Creates a sample dataset with missing values, constant column, whitespace, and a target."""
    return pd.DataFrame(
        {
            "num_feat": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, np.nan] * 4,
            "scale_feat": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0] * 4,
            "cat_feat": [" cat ", "dog", " cat ", "bird", "dog", "bird", "cat", "dog"] * 4,
            "const_col": [42.0] * 32,
            "target": [0, 1, 0, 1, 0, 1, 0, 1] * 4,
        }
    )


def _setup_workspace_and_dataset(tmp_path: Path, name: str = "sample.csv") -> tuple[Any, Any, Any, Any, Path]:
    ws_dir = tmp_path / "ws"
    ws, cmd, qry = build_application(root_dir=str(ws_dir))
    df = _create_sample_df()
    csv_path = tmp_path / name
    df.to_csv(csv_path, index=False)
    ds = ws.register_dataset(name="sample_ds", path=str(csv_path), target="target", task_type="binary_classification")
    return ws, cmd, qry, ds, csv_path


# =====================================================================
# 1. CLI Tests: automl data inspect
# =====================================================================


def test_cli_data_inspect_text_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _ws, _cmd, _qry, ds, _path = _setup_workspace_and_dataset(tmp_path)
    ws_dir = str(tmp_path / "ws")

    # Positional dataset_id with text output
    code = main(["data", "inspect", ds.id, "--workspace", ws_dir])
    assert code == 0
    captured = capsys.readouterr()
    assert "CATML Data Quality Inspection:" in captured.out
    assert "Rows:    32" in captured.out
    assert "Columns: 5" in captured.out
    assert "nulls" in captured.out
    assert "constant_column" in captured.out

    # Alias --dataset / -d with --json output
    code = main(["data", "inspect", "-d", ds.id, "--workspace", ws_dir, "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["dataset_id"] == ds.id
    assert data["row_count"] == 32
    assert data["column_count"] == 5
    issue_codes = [i["code"] for i in data["issues"]]
    assert "nulls" in issue_codes
    assert "constant_column" in issue_codes

    # Missing dataset ID
    code = main(["data", "inspect", "--workspace", ws_dir])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: Dataset ID is required" in captured.err


# =====================================================================
# 2. CLI Tests: automl data prepare (suggest, preview, validate, apply, create, plans, show-plan, versions)
# =====================================================================


def test_cli_data_prepare_subcommands_lifecycle(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _ws, _cmd, _qry, ds, orig_path = _setup_workspace_and_dataset(tmp_path)
    orig_bytes = orig_path.read_bytes()
    ws_dir = str(tmp_path / "ws")

    # A. suggest (text and JSON)
    code = main(["data", "prepare", "suggest", ds.id, "--workspace", ws_dir])
    assert code == 0
    captured = capsys.readouterr()
    assert "Suggested Preparation Plan:" in captured.out
    assert "Status:  SUGGESTED" in captured.out
    assert "[training_pipeline] mark_impute" in captured.out
    assert "[dataset] drop_columns" in captured.out

    code = main(["data", "prepare", "suggest", "-d", ds.id, "--name", "json_plan", "--workspace", ws_dir, "--json"])
    assert code == 0
    captured = capsys.readouterr()
    suggest_json = json.loads(captured.out)
    plan_id = suggest_json["id"]
    assert plan_id.startswith("pplan_")
    assert suggest_json["status"] == "suggested"
    assert len(suggest_json["steps"]) >= 2

    # B. preview (text and JSON) — verifies purity guarantee
    code = main(["data", "prepare", "preview", plan_id, "--workspace", ws_dir])
    assert code == 0
    captured = capsys.readouterr()
    assert "Preparation Plan Preview:" in captured.out
    assert "Mutated Stored Dataset:  False (purity guaranteed)" in captured.out
    assert orig_path.read_bytes() == orig_bytes

    code = main(["data", "prepare", "preview", plan_id, "--workspace", ws_dir, "--json"])
    assert code == 0
    captured = capsys.readouterr()
    preview_json = json.loads(captured.out)
    assert preview_json["plan_id"] == plan_id
    assert preview_json["mutated_stored_dataset"] is False
    assert preview_json["before_col_count"] == 5
    assert preview_json["after_col_count"] == 4  # const_col dropped
    assert "const_col" in preview_json["touched_columns"]

    # C. validate (text and JSON)
    code = main(["data", "prepare", "validate", plan_id, "--workspace", ws_dir])
    assert code == 0
    captured = capsys.readouterr()
    assert "Preparation Plan Validation: PASS" in captured.out

    code = main(["data", "prepare", "validate", plan_id, "--workspace", ws_dir, "--json"])
    assert code == 0
    captured = capsys.readouterr()
    val_json = json.loads(captured.out)
    assert val_json["ok"] is True

    # D. apply — without --approve must fail cleanly (Propose ≠ Accept)
    code = main(["data", "prepare", "apply", plan_id, "--workspace", ws_dir])
    assert code == 1
    captured = capsys.readouterr()
    assert "Validation error applying plan:" in captured.err or "approval" in captured.err

    # E. apply — with --approve must succeed (JSON and text modes)
    code = main(["data", "prepare", "apply", plan_id, "--approve", "--notes", "Audit approved", "--workspace", ws_dir, "--json"])
    assert code == 0
    captured = capsys.readouterr()
    apply_json = json.loads(captured.out)
    assert apply_json["plan_id"] == plan_id
    assert apply_json["version_id"].startswith("dsver_")
    assert apply_json["registered_dataset_id"].startswith("ds_")
    assert apply_json["original_path_intact"] is True
    assert orig_path.read_bytes() == orig_bytes

    # E2. apply text output check (create and apply a second plan)
    code = main([
        "data", "prepare", "suggest", ds.id, "--name", "plan_for_text_apply",
        "--workspace", ws_dir, "--json",
    ])
    assert code == 0
    captured = capsys.readouterr()
    p2_id = json.loads(captured.out)["id"]
    code = main(["data", "prepare", "apply", p2_id, "--approve", "--workspace", ws_dir])
    assert code == 0
    captured = capsys.readouterr()
    assert f"Successfully Applied Preparation Plan: {p2_id}" in captured.out
    assert "New Version ID:" in captured.out
    assert "Registered Dataset ID:" in captured.out

    # F. plans / list-plans (text and JSON, with flags and positional dataset_id)
    code = main(["data", "prepare", "plans", "--workspace", ws_dir, "--json"])
    assert code == 0
    captured = capsys.readouterr()
    plans_json = json.loads(captured.out)
    assert any(p["id"] == plan_id for p in plans_json)

    # Positional dataset argument
    code = main(["data", "prepare", "plans", ds.id, "--workspace", ws_dir])
    assert code == 0
    captured = capsys.readouterr()
    assert "Preparation Plans" in captured.out
    assert plan_id in captured.out

    # -d flag
    code = main(["data", "prepare", "plans", "-d", ds.id, "--workspace", ws_dir])
    assert code == 0
    captured = capsys.readouterr()
    assert "Preparation Plans" in captured.out
    assert plan_id in captured.out

    # G. show-plan (text and JSON)
    code = main(["data", "prepare", "show-plan", plan_id, "--workspace", ws_dir])
    assert code == 0
    captured = capsys.readouterr()
    assert f"Preparation Plan: {plan_id}" in captured.out

    code = main(["data", "prepare", "show-plan", plan_id, "--workspace", ws_dir, "--json"])
    assert code == 0
    captured = capsys.readouterr()
    show_json = json.loads(captured.out)
    assert show_json["id"] == plan_id

    # Nonexistent plan error
    code = main(["data", "prepare", "show-plan", "plan_nonexistent", "--workspace", ws_dir])
    assert code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err

    # H. versions (text, JSON, and --parent alias)
    code = main(["data", "prepare", "versions", ds.id, "--workspace", ws_dir, "--json"])
    assert code == 0
    captured = capsys.readouterr()
    versions_json = json.loads(captured.out)
    assert len(versions_json) >= 1
    assert versions_json[0]["parent_dataset_id"] == ds.id

    code = main(["data", "prepare", "versions", ds.id, "--workspace", ws_dir])
    assert code == 0
    captured = capsys.readouterr()
    assert f"Dataset Versions for {ds.id}" in captured.out
    assert apply_json["version_id"] in captured.out

    # Test --parent alias
    code = main(["data", "prepare", "versions", "--parent", ds.id, "--workspace", ws_dir])
    assert code == 0
    captured = capsys.readouterr()
    assert f"Dataset Versions for {ds.id}" in captured.out


def test_cli_data_prepare_create(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _ws, _cmd, _qry, ds, _path = _setup_workspace_and_dataset(tmp_path)
    ws_dir = str(tmp_path / "ws")

    # Create via inline JSON string
    steps = json.dumps([
        {"operation": "normalize_whitespace", "scope": "dataset", "columns": ["cat_feat"]},
        {"operation": "mark_scale", "scope": "training_pipeline", "columns": ["scale_feat"], "params": {"scaler": "robust_scaler"}},
    ])
    code = main([
        "data", "prepare", "create", ds.id,
        "--steps", steps,
        "--name", "custom_plan",
        "--workspace", ws_dir,
        "--json",
    ])
    assert code == 0
    captured = capsys.readouterr()
    res = json.loads(captured.out)
    plan_id = res["plan_id"]
    assert plan_id.startswith("pplan_")

    # Create via JSON file path
    steps_file = tmp_path / "steps.json"
    steps_file.write_text(steps, encoding="utf-8")
    code = main([
        "data", "prepare", "create", ds.id,
        "--steps", str(steps_file),
        "--name", "file_plan",
        "--workspace", ws_dir,
    ])
    assert code == 0
    captured = capsys.readouterr()
    # Create via long inline JSON string (> 255 chars, prevents OSError)
    long_steps = json.dumps([
        {
            "operation": "normalize_whitespace",
            "scope": "dataset",
            "columns": [f"extra_feature_column_for_length_testing_{i}" for i in range(12)],
        }
    ])
    assert len(long_steps) > 255
    code = main([
        "data", "prepare", "create", ds.id,
        "--steps", long_steps,
        "--name", "long_json_plan",
        "--workspace", ws_dir,
        "--json",
    ])
    assert code == 0
    captured = capsys.readouterr()
    res_long = json.loads(captured.out)
    assert res_long["plan_id"].startswith("pplan_")

    # Reject illegal scope (e.g. mark_impute on dataset scope)
    illegal_steps = json.dumps([
        {"operation": "mark_impute", "scope": "dataset", "columns": ["num_feat"]},
    ])
    code = main([
        "data", "prepare", "create", ds.id,
        "--steps", illegal_steps,
        "--workspace", ws_dir,
    ])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error creating preparation plan:" in captured.err


# =====================================================================
# 3. Python Facade Tests: AutoML.prepare
# =====================================================================


def test_facade_prepare_callable_end_to_end(tmp_path: Path):
    df = _create_sample_df()
    ws_dir = tmp_path / "ws_facade"
    automl = AutoML(workspace_dir=ws_dir)

    # Invoked directly as callable: automl.prepare(df, target="target", approve=True)
    applied = automl.prepare(df, target="target", approve=True, notes="Prepared via facade callable")
    assert applied is not None
    assert applied["version_id"].startswith("dsver_")
    assert applied["registered_dataset_id"].startswith("ds_")
    assert Path(applied["path"]).exists()
    assert applied["original_path_intact"] is True

    # Check that const_col was dropped in the prepared data
    prepared_df = pd.read_csv(applied["path"])
    assert "const_col" not in prepared_df.columns
    # Check that num_feat still has NaNs in the prepared CSV (fold safety!)
    assert prepared_df["num_feat"].isna().sum() > 0


def test_facade_prepare_methods_lifecycle(tmp_path: Path):
    df = _create_sample_df()
    ws_dir = tmp_path / "ws_facade_methods"
    automl = AutoML(workspace_dir=ws_dir)

    # 1. analyze
    report = automl.prepare.analyze(df)
    assert report["row_count"] == 32
    assert any(i["code"] == "nulls" for i in report["issues"])
    assert any(i["code"] == "constant_column" for i in report["issues"])

    # 2. suggest
    suggested = automl.prepare.suggest(df, name="facade_suggested_plan")
    plan_id = suggested["id"]
    assert suggested["status"] == "suggested"

    # 3. preview
    preview = automl.prepare.preview(plan_id)
    assert preview["plan_id"] == plan_id
    assert preview["mutated_stored_dataset"] is False

    # 4. validate
    val = automl.prepare.validate(plan_id)
    assert val["ok"] is True

    # 4b. unauthorized apply must raise PreparationValidationError (Propose != Accept)
    with pytest.raises(PreparationValidationError):
        automl.prepare.apply(plan_id, approved=False)

    # 5. apply with approval
    applied = automl.prepare.apply(plan_id, approved=True, notes="Applied via facade methods")
    assert applied["version_id"].startswith("dsver_")

    # 6. list_plans & get_plan
    plans = automl.prepare.list_plans()
    assert len(plans) >= 1
    fetched_plan = automl.prepare.get_plan(plan_id)
    assert fetched_plan is not None
    assert fetched_plan["id"] == plan_id
    assert fetched_plan["status"] == "applied"

    # 7. list_versions & get_version
    parent_ds_id = suggested["dataset_id"]
    versions = automl.prepare.list_versions(parent_ds_id)
    assert len(versions) == 1
    assert versions[0]["id"] == applied["version_id"]

    fetched_version = automl.prepare.get_version(applied["version_id"])
    assert fetched_version is not None
    assert fetched_version["id"] == applied["version_id"]
    assert fetched_version["parent_dataset_id"] == parent_ds_id

    # 7b. analyze and suggest using existing dataset_id
    re_analyzed = automl.prepare.analyze(dataset_id=parent_ds_id)
    assert re_analyzed["dataset_id"] == parent_ds_id
    re_suggested = automl.prepare.suggest(dataset_id=parent_ds_id)
    assert re_suggested["dataset_id"] == parent_ds_id

    # 7c. invalid dataset ID resolution raises clear ValueError
    with pytest.raises(ValueError, match="not found in workspace"):
        automl.prepare._resolve_dataset_id("ds_nonexistent_12345")

    # 8. create custom plan via facade
    custom_plan_id = automl.prepare.create(
        parent_ds_id,
        steps=[
            {"operation": "normalize_whitespace", "scope": "dataset", "columns": ["cat_feat"]},
        ],
        name="custom_facade_plan",
    )
    assert custom_plan_id.startswith("pplan_")


# =====================================================================
# 4. Fold-Safe Execution & Pipeline Bridge Tests
# =====================================================================


def test_pipeline_steps_to_column_transforms_bridge():
    steps = [
        PreparationStep(
            operation=PreparationOperation.DROP_COLUMNS,
            scope=PreparationScope.DATASET,
            columns=("const_col",),
        ),
        PreparationStep(
            operation=PreparationOperation.MARK_IMPUTE,
            scope=PreparationScope.TRAINING_PIPELINE,
            columns=("num_feat",),
            params={"strategy": "median"},
        ),
        PreparationStep(
            operation=PreparationOperation.MARK_SCALE,
            scope=PreparationScope.TRAINING_PIPELINE,
            columns=("scale_feat",),
            params={"scaler": "robust_scaler"},
        ),
    ]

    transforms = pipeline_steps_to_column_transforms(steps)
    assert len(transforms) == 2

    # Drop columns (DATASET scope) is excluded from column_transforms
    assert not any(t["column"] == "const_col" for t in transforms)

    # Impute is converted
    impute_t = next(t for t in transforms if t["column"] == "num_feat")
    assert impute_t["kind"] == "impute"
    assert impute_t["strategy"] == "median"

    # Scale is converted
    scale_t = next(t for t in transforms if t["column"] == "scale_feat")
    assert scale_t["kind"] == "robust_scaler"


def test_fold_safe_execution_in_workspace_create_experiment_and_fit(tmp_path: Path):
    """
    Ensures pipeline-scoped steps from a prepared dataset version:
    1. Are not applied full-frame before split (no data leakage).
    2. Are automatically resolved into experiment.column_transforms.
    3. Fit and predict cleanly inside cross-validation folds in sklearn_trainer.
    """
    ws_dir = tmp_path / "ws_fold_safety"
    automl = AutoML(workspace_dir=ws_dir, models=["logistic_regression"], random_state=42)

    df = _create_sample_df()
    applied = automl.prepare(df, target="target", approve=True)

    prepared_ds_id = applied["registered_dataset_id"]
    prepared_path = applied["path"]

    # 1. Purity / Fold-Safety check: NaNs are still present in prepared CSV
    prep_csv = pd.read_csv(prepared_path)
    assert prep_csv["num_feat"].isna().sum() > 0, "Pipeline imputation must NOT be applied to full frame!"

    # 2. create_experiment automatically wires column_transforms from version lineage
    ws, _cmd, _qry = automl._get_application()
    run = ws.create_run(ws.get_dataset(prepared_ds_id), metric="roc_auc")
    exp = ws.create_experiment(run=run, name="exp_prep_transforms", model_ids=["logistic_regression"])

    assert exp.column_transforms is not None
    assert len(exp.column_transforms) > 0
    impute_trans = [t for t in exp.column_transforms if t["column"] == "num_feat"]
    assert len(impute_trans) == 1
    assert impute_trans[0]["kind"] == "impute"

    # 3. AutoML.fit runs on prepared dataset ID without crash, learning imputation within folds
    result = automl.fit(prepared_ds_id, target="target")
    assert result is not None
    assert result.best_model is not None
    assert result.best_score is not None

    # Predictions succeed on data with NaNs
    preds = result.predict(prep_csv.drop(columns=["target"]))
    assert len(preds) == len(prep_csv)

    # 4. Fitting with implicit target resolves target_column from registered dataset
    result_implicit = automl.fit(prepared_ds_id)
    assert result_implicit is not None
    assert result_implicit.best_model is not None

    # 5. Nonexistent dataset ID in fit raises TypeError (unsupported input data type)
    with pytest.raises(TypeError, match="Unsupported input data type"):
        automl.fit("ds_nonexistent_99999")


# =====================================================================
# 5. Recipe Integration Tests
# =====================================================================


def test_recipe_captures_pipeline_column_transforms_from_prepared_dataset(tmp_path: Path):
    """
    Verifies that RecipeService creates recipes containing the fold-safe column transforms
    inheriting from prepared dataset versions.
    """
    ws_dir = tmp_path / "ws_recipes"
    automl = AutoML(workspace_dir=ws_dir, models=["logistic_regression"], random_state=42)
    df = _create_sample_df()
    applied = automl.prepare(df, target="target", approve=True)
    prep_ds_id = applied["registered_dataset_id"]

    ws, cmd_bus, qry_bus = automl._get_application()
    ds = ws.get_dataset(prep_ds_id)
    run = ws.create_run(ds)

    # Create candidate experiment on prepared dataset
    cand_exp = ws.create_experiment(
        run=run,
        name="Candidate with Prep Transforms",
        model_ids=["logistic_regression"],
        feature_names=[c for c in ds.path and pd.read_csv(ds.path).columns if c != "target"],
    )
    assert len(cand_exp.column_transforms) > 0

    # Set up explore study and hypothesis
    study_id = cmd_bus.dispatch(CreateStudyCommand(dataset_id=ds.id, name="Prep Study", target_column="target"))
    study_repo = ws.get_study_repository()
    analysis_run = AnalysisRun.create(study_id=study_id)
    study_repo.save_analysis_run(analysis_run)

    finding = StatisticalFinding.create(
        study_id=study_id,
        analysis_run_id=analysis_run.id,
        finding_type="missing_data",
        summary="Missing values in num_feat",
        column_name="num_feat",
    )
    study_repo.save_finding(finding)

    hypothesis = AnalysisHypothesis.create(
        study_id=study_id,
        finding_id=finding.id,
        description="Impute num_feat inside folds",
        proposed_action="impute_in_fold",
        experiment_delta={},
    )
    study_repo.save_hypothesis(hypothesis)

    evidence = EvidenceLink.create(
        hypothesis_id=hypothesis.id,
        experiment_id=cand_exp.id,
        baseline_score=0.75,
        candidate_score=0.85,
        metric="roc_auc",
        accepted=True,
    )
    study_repo.save_evidence_link(evidence)

    # Create Recipe from EvidenceLink
    recipe_id = cmd_bus.dispatch(
        CreateExperimentRecipeFromEvidenceCommand(
            evidence_link_id=evidence.id,
            recipe_name="Prepared Imputation Recipe",
        )
    )
    recipe_dto = qry_bus.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
    assert recipe_dto is not None
    assert recipe_dto.name == "Prepared Imputation Recipe"
    assert len(recipe_dto.column_transforms) > 0
    assert any(ct["column"] == "num_feat" for ct in recipe_dto.column_transforms)
