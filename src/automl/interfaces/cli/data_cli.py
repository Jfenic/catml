"""CLI commands for CATML Data Preparation subsystem."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from automl.application.bootstrap import build_application
from automl.application.preparation.commands import (
    AnalyzeDataQualityCommand,
    ApplyPreparationPlanCommand,
    CreatePreparationPlanCommand,
    PreviewPreparationPlanCommand,
    SuggestPreparationPlanCommand,
    ValidatePreparationPlanCommand,
)
from automl.application.preparation.queries import (
    GetPreparationPlanQuery,
    ListDatasetVersionsQuery,
    ListPreparationPlansQuery,
)
from automl.domain.preparation.validation import PreparationValidationError


def run_data_cli(args: argparse.Namespace) -> int:
    """Dispatches `automl data` and `automl data prepare` subcommands via CommandBus/QueryBus."""
    ws_dir = getattr(args, "workspace", None)
    _ws, cmd, qry = build_application(root_dir=ws_dir)

    action = getattr(args, "data_action", None)

    if action == "inspect":
        dataset_id = getattr(args, "dataset", None) or getattr(args, "dataset_id", None)
        if not dataset_id:
            print("Error: Dataset ID is required. Use `automl data inspect <dataset_id>`.", file=sys.stderr)
            return 1

        try:
            report: dict[str, Any] = cmd.dispatch(AnalyzeDataQualityCommand(dataset_id=dataset_id))
            if getattr(args, "json", False):
                print(json.dumps(report, indent=2, default=str))
            else:
                print("=" * 65)
                print(f"  CATML Data Quality Inspection: {dataset_id}")
                print("=" * 65)
                print(f"  Rows:    {report.get('row_count', 'N/A')}")
                print(f"  Columns: {report.get('column_count', 'N/A')}")
                issues = report.get("issues", [])
                print(f"  Issues:  {len(issues)}")
                print("-" * 65)
                if not issues:
                    print("  No data quality issues detected (clean).")
                else:
                    for iss in issues:
                        sev = str(iss.get("severity", "info")).upper()
                        code = iss.get("code", "")
                        msg = iss.get("message", "")
                        cols = ", ".join(iss.get("columns", []))
                        cols_str = f" (columns: {cols})" if cols else ""
                        print(f"  [{sev:<7}] {code}: {msg}{cols_str}")
            return 0
        except Exception as exc:
            print(f"Error inspecting data quality: {exc}", file=sys.stderr)
            return 1

    prep_action = getattr(args, "prepare_action", None)

    if prep_action == "suggest":
        dataset_id = getattr(args, "dataset", None) or getattr(args, "dataset_id", None)
        if not dataset_id:
            print("Error: Dataset ID is required. Use `automl data prepare suggest <dataset_id>`.", file=sys.stderr)
            return 1

        name = getattr(args, "name", None)
        include_pipeline_marks = not getattr(args, "no_pipeline_marks", False)

        try:
            plan: dict[str, Any] = cmd.dispatch(
                SuggestPreparationPlanCommand(
                    dataset_id=dataset_id,
                    name=name,
                    include_pipeline_marks=include_pipeline_marks,
                )
            )
            if getattr(args, "json", False):
                print(json.dumps(plan, indent=2, default=str))
            else:
                print(f"Suggested Preparation Plan: {plan.get('id')}")
                print(f"  Dataset: {plan.get('dataset_id')}")
                print(f"  Status:  {str(plan.get('status', '')).upper()}")
                steps = plan.get("steps", [])
                print(f"  Steps ({len(steps)}):")
                for idx, s in enumerate(steps, 1):
                    scope = s.get("scope", "dataset")
                    op = s.get("operation", "")
                    cols = ", ".join(s.get("columns", [])) or "all"
                    req = " (requires approval)" if s.get("requires_approval") else ""
                    print(f"    {idx}. [{scope}] {op} on ({cols}){req}")
            return 0
        except Exception as exc:
            print(f"Error suggesting preparation plan: {exc}", file=sys.stderr)
            return 1

    elif prep_action == "preview":
        plan_id = getattr(args, "plan_id", None)
        if not plan_id:
            print("Error: Plan ID is required. Use `automl data prepare preview <plan_id>`.", file=sys.stderr)
            return 1

        try:
            preview: dict[str, Any] = cmd.dispatch(PreviewPreparationPlanCommand(plan_id=plan_id))
            if getattr(args, "json", False):
                print(json.dumps(preview, indent=2, default=str))
            else:
                print(f"Preparation Plan Preview: {preview.get('plan_id')}")
                print(f"  Dataset ID:              {preview.get('dataset_id')}")
                print(f"  Before:                  {preview.get('before_row_count')} rows x {preview.get('before_col_count')} cols")
                print(f"  After (dataset scope):   {preview.get('after_row_count')} rows x {preview.get('after_col_count')} cols")
                touched = preview.get("touched_columns", [])
                print(f"  Touched Columns:         {', '.join(touched) if touched else 'None'}")
                print(f"  Rows Touched:            {preview.get('rows_touched', 0)}")
                warnings = preview.get("warnings", [])
                print(f"  Warnings:                {len(warnings)}")
                print(f"  Mutated Stored Dataset:  {preview.get('mutated_stored_dataset')} (purity guaranteed)")
            return 0
        except Exception as exc:
            print(f"Error previewing preparation plan: {exc}", file=sys.stderr)
            return 1

    elif prep_action == "validate":
        plan_id = getattr(args, "plan_id", None)
        if not plan_id:
            print("Error: Plan ID is required. Use `automl data prepare validate <plan_id>`.", file=sys.stderr)
            return 1

        try:
            validated: dict[str, Any] = cmd.dispatch(ValidatePreparationPlanCommand(plan_id=plan_id))
            if getattr(args, "json", False):
                print(json.dumps(validated, indent=2, default=str))
            else:
                ok = validated.get("ok", False)
                status_str = "PASS" if ok else "FAIL"
                print(f"Preparation Plan Validation: {status_str}")
                print(f"  Plan ID:  {validated.get('plan_id')}")
                print(f"  Status:   {validated.get('status')}")
                errs = validated.get("errors", [])
                if errs:
                    print(f"  Errors ({len(errs)}):")
                    for err in errs:
                        print(f"    - {err}")
                warns = validated.get("warnings", [])
                if warns:
                    print(f"  Warnings ({len(warns)}):")
                    for warn in warns:
                        print(f"    - {warn}")
            return 0 if validated.get("ok") else 1
        except Exception as exc:
            print(f"Error validating preparation plan: {exc}", file=sys.stderr)
            return 1

    elif prep_action == "apply":
        plan_id = getattr(args, "plan_id", None)
        if not plan_id:
            print("Error: Plan ID is required. Use `automl data prepare apply <plan_id>`.", file=sys.stderr)
            return 1

        approved = getattr(args, "approve", False) or getattr(args, "approved", False) or getattr(args, "yes", False)
        notes = getattr(args, "notes", None)

        try:
            applied: dict[str, Any] = cmd.dispatch(
                ApplyPreparationPlanCommand(
                    plan_id=plan_id,
                    approved=approved,
                    notes=notes,
                )
            )
            if getattr(args, "json", False):
                print(json.dumps(applied, indent=2, default=str))
            else:
                print(f"Successfully Applied Preparation Plan: {applied.get('plan_id')}")
                print(f"  New Version ID:         {applied.get('version_id')}")
                print(f"  Registered Dataset ID:  {applied.get('registered_dataset_id')}")
                print(f"  Output Path:            {applied.get('path')}")
                print(f"  Content Hash:           {applied.get('content_hash') or 'N/A'}")
                p_steps = applied.get("pipeline_steps", [])
                print(f"  Pipeline Steps:         {len(p_steps)} recorded for fold execution")
                print(f"  Original Path Intact:   {applied.get('original_path_intact')} (immutable)")
            return 0
        except PreparationValidationError as exc:
            print(f"Validation error applying plan: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Error applying preparation plan: {exc}", file=sys.stderr)
            return 1

    elif prep_action == "create":
        dataset_id = getattr(args, "dataset", None) or getattr(args, "dataset_id", None)
        if not dataset_id:
            print("Error: Dataset ID is required.", file=sys.stderr)
            return 1

        steps_raw = getattr(args, "steps", None)
        if not steps_raw:
            print("Error: --steps is required (JSON string or path to JSON file).", file=sys.stderr)
            return 1

        try:
            raw_stripped = steps_raw.strip()
            if raw_stripped.startswith(("[", "{")):
                steps_data = json.loads(raw_stripped)
            else:
                is_file = False
                try:
                    steps_file_path = Path(steps_raw)
                    if steps_file_path.is_file():
                        is_file = True
                        steps_data = json.loads(steps_file_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
                if not is_file:
                    steps_data = json.loads(steps_raw)

            if isinstance(steps_data, dict):
                steps_data = [steps_data]
            elif not isinstance(steps_data, list):
                print("Error: steps must be a JSON array of step objects.", file=sys.stderr)
                return 1

            name = getattr(args, "name", None)
            notes = getattr(args, "notes", None)

            created_id: str = cmd.dispatch(
                CreatePreparationPlanCommand(
                    dataset_id=dataset_id,
                    steps=tuple(steps_data),
                    name=name,
                    notes=notes,
                )
            )
            if getattr(args, "json", False):
                print(json.dumps({"plan_id": created_id, "status": "draft"}))
            else:
                print(f"Created Preparation Plan: {created_id}")
            return 0
        except Exception as exc:
            print(f"Error creating preparation plan: {exc}", file=sys.stderr)
            return 1

    elif prep_action in ("plans", "list-plans"):
        dataset_id = getattr(args, "dataset", None) or getattr(args, "dataset_id", None)
        try:
            plans: list[dict[str, Any]] = qry.dispatch(ListPreparationPlansQuery(dataset_id=dataset_id))
            if getattr(args, "json", False):
                print(json.dumps(plans, indent=2, default=str))
            else:
                if not plans:
                    print("No preparation plans found.")
                    return 0
                print(f"Preparation Plans ({len(plans)}):")
                print("-" * 75)
                print(f"{'Plan ID':<20} {'Dataset':<16} {'Status':<12} {'Steps':<6} {'Name'}")
                print("-" * 75)
                for p in plans:
                    steps_cnt = len(p.get("steps", []))
                    name = p.get("name") or "-"
                    print(f"{p.get('id', ''):<20} {p.get('dataset_id', ''):<16} {str(p.get('status', '')).upper():<12} {steps_cnt:<6} {name}")
            return 0
        except Exception as exc:
            print(f"Error listing preparation plans: {exc}", file=sys.stderr)
            return 1

    elif prep_action == "show-plan":
        plan_id = getattr(args, "plan_id", None)
        if not plan_id:
            print("Error: Plan ID is required.", file=sys.stderr)
            return 1
        try:
            plan_data: dict[str, Any] | None = qry.dispatch(GetPreparationPlanQuery(plan_id=plan_id))
            if not plan_data:
                print(f"Preparation plan not found: {plan_id}", file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                print(json.dumps(plan_data, indent=2, default=str))
            else:
                print(f"Preparation Plan: {plan_data.get('id')}")
                print(f"  Name:       {plan_data.get('name') or '-'}")
                print(f"  Dataset ID: {plan_data.get('dataset_id')}")
                print(f"  Status:     {str(plan_data.get('status', '')).upper()}")
                steps = plan_data.get("steps", [])
                print(f"  Steps ({len(steps)}):")
                for idx, s in enumerate(steps, 1):
                    scope = s.get("scope", "dataset")
                    op = s.get("operation", "")
                    cols = ", ".join(s.get("columns", [])) or "all"
                    req = " (requires approval)" if s.get("requires_approval") else ""
                    print(f"    {idx}. [{scope}] {op} on ({cols}){req}")
            return 0
        except Exception as exc:
            print(f"Error showing preparation plan: {exc}", file=sys.stderr)
            return 1

    elif prep_action in ("versions", "list-versions"):
        parent_id = (
            getattr(args, "parent_dataset_id", None)
            or getattr(args, "parent", None)
            or getattr(args, "dataset", None)
            or getattr(args, "dataset_id", None)
        )
        if not parent_id:
            print("Error: Parent dataset ID is required. Use `automl data prepare versions <parent_dataset_id>`.", file=sys.stderr)
            return 1
        try:
            versions: list[dict[str, Any]] = qry.dispatch(ListDatasetVersionsQuery(parent_dataset_id=parent_id))
            if getattr(args, "json", False):
                print(json.dumps(versions, indent=2, default=str))
            else:
                if not versions:
                    print(f"No prepared versions found for dataset {parent_id}.")
                    return 0
                print(f"Dataset Versions for {parent_id} ({len(versions)}):")
                print("-" * 80)
                print(f"{'Version ID':<20} {'Registered ID':<20} {'Plan ID':<20} {'Hash':<12}")
                print("-" * 80)
                for v in versions:
                    chash = str(v.get("content_hash") or "N/A")[:10]
                    print(f"{v.get('id', ''):<20} {v.get('registered_dataset_id', ''):<20} {v.get('plan_id', ''):<20} {chash:<12}")
            return 0
        except Exception as exc:
            print(f"Error listing dataset versions: {exc}", file=sys.stderr)
            return 1

    else:
        print("Error: Unknown data preparation action. Use `automl data --help`.", file=sys.stderr)
        return 1
