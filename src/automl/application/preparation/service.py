"""PreparationService: analyze → suggest → preview → validate → apply."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from automl.application.preparation.commands import (
    AnalyzeDataQualityCommand,
    ApplyPreparationPlanCommand,
    CreatePreparationPlanCommand,
    PreviewPreparationPlanCommand,
    SuggestPreparationPlanCommand,
    ValidatePreparationPlanCommand,
)
from automl.application.preparation.queries import (
    GetDatasetVersionQuery,
    GetPreparationPlanQuery,
    ListDatasetVersionsQuery,
    ListPreparationPlansQuery,
)
from automl.domain.datasets.content_hash import try_compute_file_content_hash
from automl.domain.preparation.models import (
    APPROVAL_DEFAULT_OPERATIONS,
    DatasetVersion,
    PlanStatus,
    PreparationOperation,
    PreparationPlan,
    PreparationScope,
    PreparationStep,
    QualityIssue,
    QualityReport,
)
from automl.domain.preparation.validation import (
    PreparationValidationError,
    validate_plan,
)
from automl.infrastructure.preparation.json_store import JsonPreparationStore

if TYPE_CHECKING:
    from automl.application.services.workspace import AutoMLWorkspace

logger = logging.getLogger(__name__)


class PreparationService:
    """
    Application orchestrator for Data Preparation.
    Preview never mutates stored datasets; apply creates a new version.
    """

    def __init__(self, workspace: AutoMLWorkspace, store: JsonPreparationStore | None = None) -> None:
        self.workspace = workspace
        self.store = store or JsonPreparationStore(workspace.root_dir)

    # --- commands ---

    def analyze(self, cmd: AnalyzeDataQualityCommand) -> dict[str, Any]:
        ds = self.workspace.get_dataset(cmd.dataset_id)
        df = pd.read_csv(ds.path) if str(ds.path).endswith(".csv") else pd.read_parquet(ds.path)
        issues: list[QualityIssue] = []
        n_rows, n_cols = len(df), int(df.shape[1])

        null_frac = df.isna().mean()
        for col, frac in null_frac.items():
            if float(frac) > 0:
                issues.append(
                    QualityIssue(
                        code="nulls",
                        severity="warning" if float(frac) < 0.5 else "error",
                        message=f"Column '{col}' has {float(frac):.1%} missing values",
                        columns=(str(col),),
                        evidence={"null_fraction": float(frac)},
                    )
                )

        for col in df.columns:
            nunique = int(df[col].nunique(dropna=True))
            if n_rows > 1 and nunique <= 1:
                issues.append(
                    QualityIssue(
                        code="constant_column",
                        severity="warning",
                        message=f"Column '{col}' is constant",
                        columns=(str(col),),
                        evidence={"nunique": nunique},
                    )
                )

        dup_count = int(df.duplicated().sum())
        if dup_count:
            issues.append(
                QualityIssue(
                    code="duplicate_rows",
                    severity="info",
                    message=f"{dup_count} duplicate row candidate(s); dedupe policy required",
                    evidence={"duplicate_rows": dup_count},
                )
            )

        report = QualityReport(
            dataset_id=cmd.dataset_id,
            issues=issues,
            row_count=n_rows,
            column_count=n_cols,
        )
        return report.to_dict()

    def suggest(self, cmd: SuggestPreparationPlanCommand) -> dict[str, Any]:
        report = self.analyze(AnalyzeDataQualityCommand(dataset_id=cmd.dataset_id))
        steps: list[PreparationStep] = []

        for issue in report["issues"]:
            code = issue["code"]
            cols = tuple(issue.get("columns") or ())
            if code == "nulls" and cmd.include_pipeline_marks and cols:
                steps.append(
                    PreparationStep(
                        operation=PreparationOperation.MARK_IMPUTE,
                        scope=PreparationScope.TRAINING_PIPELINE,
                        columns=cols,
                        rationale=issue["message"],
                        requires_approval=False,
                    )
                )
            elif code == "constant_column" and cols:
                steps.append(
                    PreparationStep(
                        operation=PreparationOperation.DROP_COLUMNS,
                        scope=PreparationScope.DATASET,
                        columns=cols,
                        rationale=issue["message"],
                        requires_approval=True,
                    )
                )
            elif code == "duplicate_rows":
                steps.append(
                    PreparationStep(
                        operation=PreparationOperation.DEDUPE,
                        scope=PreparationScope.DATASET,
                        params={"policy": "flag_only"},
                        rationale=issue["message"],
                        requires_approval=False,
                    )
                )

        if not steps:
            # Minimal no-op-safe suggestion: normalize whitespace on object cols later via cast skip
            steps.append(
                PreparationStep(
                    operation=PreparationOperation.NORMALIZE_WHITESPACE,
                    scope=PreparationScope.DATASET,
                    columns=(),
                    rationale="No severe issues; optional whitespace normalize on all string columns",
                    requires_approval=False,
                )
            )

        plan = PreparationPlan.create(
            dataset_id=cmd.dataset_id,
            steps=steps,
            name=cmd.name or f"suggested_{cmd.dataset_id[:8]}",
        )
        plan.status = PlanStatus.SUGGESTED
        plan.lineage["source"] = "heuristic_suggest"
        plan.lineage["issue_codes"] = [i["code"] for i in report["issues"]]
        self.store.save_plan(plan)
        return plan.to_dict()

    def create_plan(self, cmd: CreatePreparationPlanCommand) -> str:
        steps = [PreparationStep.from_dict(dict(s)) for s in cmd.steps]
        plan = PreparationPlan.create(
            dataset_id=cmd.dataset_id,
            steps=steps,
            name=cmd.name or "",
            notes=cmd.notes,
        )
        validate_plan(plan)
        self.store.save_plan(plan)
        return plan.id

    def preview(self, cmd: PreviewPreparationPlanCommand) -> dict[str, Any]:
        plan = self._require_plan(cmd.plan_id)
        warnings = validate_plan(plan)
        ds = self.workspace.get_dataset(plan.dataset_id)
        df = self._load_frame(ds.path)
        before = {"rows": len(df), "columns": int(df.shape[1]), "column_names": list(df.columns)}

        working = df.copy()
        pipeline_steps: list[dict[str, Any]] = []
        dataset_steps_applied = 0
        for step in plan.steps:
            if step.scope == PreparationScope.TRAINING_PIPELINE:
                pipeline_steps.append(step.to_dict())
                continue
            working = self._apply_dataset_step(working, step)
            dataset_steps_applied += 1

        after = {
            "rows": len(working),
            "columns": int(working.shape[1]),
            "column_names": list(working.columns),
        }
        plan.status = PlanStatus.PREVIEWED
        plan.touch()
        self.store.save_plan(plan)

        touched_cols = sorted({str(c) for step in plan.steps for c in step.columns if c})
        rows_touched = abs(after["rows"] - before["rows"])

        return {
            "plan_id": plan.id,
            "dataset_id": plan.dataset_id,
            "before": before,
            "after": after,
            "before_row_count": before["rows"],
            "before_col_count": before["columns"],
            "after_row_count": after["rows"],
            "after_col_count": after["columns"],
            "rows_delta": after["rows"] - before["rows"],
            "columns_delta": after["columns"] - before["columns"],
            "touched_columns": touched_cols,
            "rows_touched": rows_touched,
            "dataset_steps_applied_in_preview": dataset_steps_applied,
            "pipeline_steps_deferred": pipeline_steps,
            "warnings": warnings,
            "mutated_stored_dataset": False,
        }

    def validate(self, cmd: ValidatePreparationPlanCommand) -> dict[str, Any]:
        plan = self._require_plan(cmd.plan_id)
        warnings = validate_plan(plan)
        plan.status = PlanStatus.VALIDATED
        plan.touch()
        self.store.save_plan(plan)
        return {"plan_id": plan.id, "ok": True, "warnings": warnings, "status": plan.status.value}

    def apply(self, cmd: ApplyPreparationPlanCommand) -> dict[str, Any]:
        plan = self._require_plan(cmd.plan_id)
        warnings = validate_plan(plan)

        needs_approval = any(
            s.requires_approval or s.operation in APPROVAL_DEFAULT_OPERATIONS
            for s in plan.steps
            if s.scope == PreparationScope.DATASET
        )
        if needs_approval and not cmd.approved:
            raise PreparationValidationError(
                "Plan includes steps that require approval; pass approved=True after review "
                "(Propose ≠ Accept)."
            )

        ds = self.workspace.get_dataset(plan.dataset_id)
        original_path = str(Path(ds.path).resolve())
        df = self._load_frame(ds.path)
        working = df.copy()
        pipeline_steps: list[dict[str, Any]] = []
        digests: list[str] = []

        for step in plan.steps:
            digests.append(self._step_digest(step))
            if step.scope == PreparationScope.TRAINING_PIPELINE:
                pipeline_steps.append(step.to_dict())
                continue
            working = self._apply_dataset_step(working, step)

        out_dir = Path(self.workspace.root_dir) / "prepared"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{plan.id}.csv"
        working.to_csv(out_path, index=False)
        content_hash = try_compute_file_content_hash(out_path)

        registered = self.workspace.register_dataset(
            name=f"{getattr(ds, 'name', plan.dataset_id)}_prepared",
            path=str(out_path),
            target=getattr(ds, "target_column", None) or getattr(ds, "target", None),
            task_type=getattr(ds, "task_type", None),
        )

        version = DatasetVersion.create(
            parent_dataset_id=plan.dataset_id,
            plan_id=plan.id,
            path=str(out_path.resolve()),
            content_hash=content_hash,
            step_digests=digests,
            registered_dataset_id=registered.id,
            lineage={
                "parent_path": original_path,
                "pipeline_steps": pipeline_steps,
                "notes": cmd.notes,
            },
        )
        self.store.save_version(version)

        plan.status = PlanStatus.APPLIED
        plan.lineage["applied_version_id"] = version.id
        plan.lineage["registered_dataset_id"] = registered.id
        plan.lineage["pipeline_steps"] = pipeline_steps
        plan.touch()
        self.store.save_plan(plan)

        # Integrity: original file still present
        if not Path(original_path).exists():
            raise RuntimeError("Original dataset path missing after apply — immutability violated")

        return {
            "plan_id": plan.id,
            "version_id": version.id,
            "registered_dataset_id": registered.id,
            "path": version.path,
            "content_hash": content_hash,
            "pipeline_steps": pipeline_steps,
            "warnings": warnings,
            "original_path_intact": True,
        }

    # --- queries ---

    def get_plan(self, q: GetPreparationPlanQuery) -> dict[str, Any] | None:
        plan = self.store.get_plan(q.plan_id)
        return plan.to_dict() if plan else None

    def list_plans(self, q: ListPreparationPlansQuery) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self.store.list_plans(q.dataset_id)]

    def list_versions(self, q: ListDatasetVersionsQuery) -> list[dict[str, Any]]:
        return [v.to_dict() for v in self.store.list_versions(q.parent_dataset_id)]

    def get_version(self, q: GetDatasetVersionQuery) -> dict[str, Any] | None:
        ver = self.store.get_version(q.version_id)
        return ver.to_dict() if ver else None

    # --- helpers ---

    def _require_plan(self, plan_id: str) -> PreparationPlan:
        plan = self.store.get_plan(plan_id)
        if plan is None:
            raise KeyError(f"PreparationPlan not found: {plan_id}")
        return plan

    def _load_frame(self, path: str | Path) -> pd.DataFrame:
        p = Path(path)
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p)

    def _step_digest(self, step: PreparationStep) -> str:
        raw = json_dumps_stable(step.to_dict())
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _apply_dataset_step(self, df: pd.DataFrame, step: PreparationStep) -> pd.DataFrame:
        op = step.operation
        if op == PreparationOperation.DROP_COLUMNS:
            cols = [c for c in step.columns if c in df.columns]
            return df.drop(columns=cols)
        if op == PreparationOperation.KEEP_COLUMNS:
            cols = [c for c in step.columns if c in df.columns]
            # Always keep intersection; empty would wipe frame — validate prevents empty
            return df.loc[:, cols]
        if op == PreparationOperation.NORMALIZE_WHITESPACE:
            out = df.copy()
            targets = list(step.columns) if step.columns else [
                c for c in out.columns if out[c].dtype == object
            ]
            for col in targets:
                if col in out.columns:
                    out[col] = out[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            return out
        if op == PreparationOperation.REPLACE_VALUES:
            out = df.copy()
            mapping = dict((step.params or {}).get("mapping") or {})
            for col in step.columns or out.columns:
                if col in out.columns:
                    out[col] = out[col].replace(mapping)
            return out
        if op == PreparationOperation.CAST_TYPE:
            out = df.copy()
            dtype = (step.params or {}).get("dtype")
            if not dtype:
                raise PreparationValidationError("cast_type requires params.dtype")
            for col in step.columns:
                if col in out.columns:
                    out[col] = out[col].astype(dtype)
            return out
        if op == PreparationOperation.PARSE_DATES:
            out = df.copy()
            for col in step.columns:
                if col in out.columns:
                    out[col] = pd.to_datetime(out[col], errors="coerce")
            return out
        if op == PreparationOperation.DEDUPE:
            policy = (step.params or {}).get("policy")
            if policy == "flag_only":
                return df  # preview/apply no row drop
            if policy == "keep_first":
                return df.drop_duplicates(keep="first")
            if policy == "drop":
                return df.drop_duplicates(keep=False)
            raise PreparationValidationError(f"Unknown dedupe policy: {policy}")
        raise PreparationValidationError(
            f"Operation {op.value} is not executable at dataset scope in apply/preview"
        )


def json_dumps_stable(obj: Any) -> str:
    import json

    return json.dumps(obj, sort_keys=True, default=str)
