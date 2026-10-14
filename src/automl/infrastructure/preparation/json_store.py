"""Filesystem JSON persistence for preparation plans and dataset versions."""
from __future__ import annotations

import json
from pathlib import Path

from automl.domain.preparation.models import DatasetVersion, PreparationPlan


class JsonPreparationStore:
    """Stores plans/versions under ``<root>/preparation/``."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root = Path(root_dir) / "preparation"
        self.plans_dir = self.root / "plans"
        self.versions_dir = self.root / "versions"
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def save_plan(self, plan: PreparationPlan) -> None:
        path = self.plans_dir / f"{plan.id}.json"
        path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    def get_plan(self, plan_id: str) -> PreparationPlan | None:
        path = self.plans_dir / f"{plan_id}.json"
        if not path.is_file():
            return None
        return PreparationPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_plans(self, dataset_id: str | None = None) -> list[PreparationPlan]:
        plans: list[PreparationPlan] = []
        for path in sorted(self.plans_dir.glob("pplan_*.json")):
            plan = PreparationPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if dataset_id is None or plan.dataset_id == dataset_id:
                plans.append(plan)
        return plans

    def save_version(self, version: DatasetVersion) -> None:
        path = self.versions_dir / f"{version.id}.json"
        path.write_text(json.dumps(version.to_dict(), indent=2), encoding="utf-8")

    def get_version(self, version_id: str) -> DatasetVersion | None:
        path = self.versions_dir / f"{version_id}.json"
        if not path.is_file():
            return None
        return DatasetVersion.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_versions(self, parent_dataset_id: str) -> list[DatasetVersion]:
        out: list[DatasetVersion] = []
        for path in sorted(self.versions_dir.glob("dsver_*.json")):
            ver = DatasetVersion.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if ver.parent_dataset_id == parent_dataset_id:
                out.append(ver)
        return out

    def find_version_for_dataset(self, dataset_id: str) -> DatasetVersion | None:
        """Find the DatasetVersion that produced registered_dataset_id or has id == dataset_id."""
        for path in sorted(self.versions_dir.glob("dsver_*.json")):
            ver = DatasetVersion.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if ver.registered_dataset_id == dataset_id or ver.id == dataset_id:
                return ver
        return None

    def find_version_by_path(self, file_path: str | Path) -> DatasetVersion | None:
        """Find the DatasetVersion whose output path matches the given file path."""
        target_path = str(Path(file_path).resolve())
        for path in sorted(self.versions_dir.glob("dsver_*.json")):
            ver = DatasetVersion.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if str(Path(ver.path).resolve()) == target_path:
                return ver
        return None

