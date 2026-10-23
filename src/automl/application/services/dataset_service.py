"""Application service for dataset registration, profiling, task planning, and framing questionnaires."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from automl.domain.datasets.profile import Dataset, DatasetProfile
from automl.domain.features.feature import Feature
from automl.domain.features.registry import FeatureRegistry
from automl.domain.tasks.problem_definition import ProblemDefinition
from automl.domain.tasks.task_type import TaskType
from automl.engine.planning.task_planner import plan_from_dataframe
from automl.engine.profiling.dataset_profiler import (
    infer_task_type,
    load_dataframe,
    profile_dataset,
)

if TYPE_CHECKING:
    from automl.application.services.workspace import AutoMLWorkspace


class DatasetService:
    """Application service for dataset persistence, feature registry hydration, and problem framing."""

    def __init__(self, workspace: AutoMLWorkspace) -> None:
        self.workspace = workspace

    def _hydrate_feature_registry(self, dataset_id: str) -> None:
        ws = self.workspace
        features = list(ws.repository.list_features(dataset_id))
        profile = ws.repository.get_dataset_profile(dataset_id)
        if profile is None:
            try:
                dataset = self._get_dataset(dataset_id)
                if dataset and Path(dataset.path).exists():
                    profile = profile_dataset(dataset)
                    ws.repository.save_dataset_profile(profile)
            except Exception:
                pass

        registry = FeatureRegistry(features)
        known_names = {f.name for f in registry.list()}

        if profile:
            id_names = set(profile.identifier_column_names)
            for f in registry.list():
                if f.semantic_type == "unknown" and f.name in id_names:
                    f.semantic_type = "identifier"
                    ws.repository.save_feature(f)

            target = profile.target_column
            for column in profile.columns:
                if target is not None and column.name == target:
                    continue
                if column.name not in known_names:
                    feature = Feature(
                        id=f"feat_{column.name}",
                        dataset_id=dataset_id,
                        name=column.name,
                        physical_dtype=column.dtype,
                        semantic_type="identifier" if column.is_identifier else "unknown",
                    )
                    registry.register(feature)
                    ws.repository.save_feature(feature)
                    known_names.add(column.name)

        ws._feature_registries[dataset_id] = registry

    def _get_dataset(self, dataset_id: str) -> Dataset:
        ws = self.workspace
        dataset = ws._datasets.get(dataset_id) or ws.repository.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(f"Dataset not found: {dataset_id}")
        ws._datasets[dataset_id] = dataset
        return dataset

    def get_dataset(self, dataset_id: str) -> Dataset:
        return self._get_dataset(dataset_id)

    def list_datasets(self) -> list[Dataset]:
        ws = self.workspace
        return (
            ws.repository.list_datasets(workspace_id=ws.id)
            if hasattr(ws.repository, "list_datasets")
            else list(ws._datasets.values())
        )

    def get_dataset_profile(self, dataset_id: str) -> DatasetProfile | None:
        return self.workspace.repository.get_dataset_profile(dataset_id)

    def save_dataset_profile(self, profile: DatasetProfile) -> None:
        self.workspace.repository.save_dataset_profile(profile)

    def register_dataset(
        self,
        name: str,
        path: str | Path | Any,
        target: str | None = None,
        task_type: str | None = None,
    ) -> Dataset:
        ws = self.workspace
        dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
        if isinstance(path, pd.DataFrame):
            df = path.copy()
            ds_dir = ws.root_dir / "datasets"
            ds_dir.mkdir(parents=True, exist_ok=True)
            try:
                save_path = ds_dir / f"{dataset_id}.parquet"
                df.to_parquet(save_path)
            except Exception:
                save_path = ds_dir / f"{dataset_id}.csv"
                df.to_csv(save_path, index=False)
            resolved = str(save_path.resolve())
        else:
            resolved = str(Path(path).resolve())
            df = load_dataframe(resolved)
        if task_type:
            resolved_task = TaskType.parse(task_type)
        elif target is not None and target in df.columns:
            resolved_task = TaskType.parse(infer_task_type(df[target]))
        else:
            resolved_task = None

        dataset = Dataset(
            id=dataset_id,
            workspace_id=ws.id,
            name=name,
            path=resolved,
            target_column=target,
            task_type=resolved_task.value if resolved_task else None,
        )
        profile = profile_dataset(dataset)
        ws._datasets[dataset_id] = dataset
        ws.repository.save_dataset(dataset)
        ws.repository.save_dataset_profile(profile)

        problem = plan_from_dataframe(dataset, df, task_type=resolved_task)
        ws.repository.save_problem_definition(problem)

        registry = FeatureRegistry()
        for column in profile.columns:
            if target is not None and column.name == target:
                continue
            feature = Feature(
                id=f"feat_{column.name}",
                dataset_id=dataset_id,
                name=column.name,
                physical_dtype=column.dtype,
                semantic_type="identifier" if column.is_identifier else "unknown",
            )
            registry.register(feature)
            ws.repository.save_feature(feature)
        ws._feature_registries[dataset_id] = registry
        ws._emit("DatasetRegistered", {"dataset_id": dataset_id, "name": name})
        return dataset

    def get_task_plan(self, dataset_id: str) -> ProblemDefinition:
        ws = self.workspace
        problem = ws.repository.get_problem_definition(dataset_id)
        if problem is not None:
            return problem
        dataset = self._get_dataset(dataset_id)
        df = load_dataframe(dataset.path)
        problem = plan_from_dataframe(dataset, df)
        ws.repository.save_problem_definition(problem)
        return problem

    def get_dataset_questionnaire(
        self,
        dataset_id: str,
        auto_generate: bool = True,
        llm_provider: Any = None,
    ) -> Any | None:
        """Retrieve dataset framing questionnaire, optionally generating it if not found."""
        q = self.workspace.repository.get_dataset_questionnaire(dataset_id)
        if q is not None:
            return q
        if not auto_generate:
            return None
        return self.generate_dataset_questionnaire(dataset_id, llm_provider=llm_provider)

    def generate_dataset_questionnaire(
        self,
        dataset_id: str,
        llm_provider: Any = None,
    ) -> Any:
        """Synthesize and persist a problem context questionnaire for the dataset."""
        ws = self.workspace
        dataset = self._get_dataset(dataset_id)
        profile = ws.repository.get_dataset_profile(dataset_id)
        profile_summary = profile.to_dict() if profile else None
        columns = [c.name for c in profile.columns] if profile else []
        target_col = getattr(dataset, "target_column", getattr(dataset, "target", ""))

        from automl.engine.planning.questionnaire_advisor import QuestionnaireAdvisor

        advisor = QuestionnaireAdvisor(llm_provider=llm_provider)
        q = advisor.infer(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            column_names=columns,
            target_column=target_col,
            profile_summary=profile_summary,
        )
        ws.repository.save_dataset_questionnaire(q)
        return q

    def save_dataset_questionnaire(self, questionnaire: Any) -> None:
        """Explicitly persist an edited or reviewed questionnaire."""
        self.workspace.repository.save_dataset_questionnaire(questionnaire)

    def get_meta_knowledge(
        self,
        dataset_id: str,
        run_id: str | None = None,
    ) -> Any:
        from automl.engine.meta_learning.extractor import extract_fingerprint
        from automl.engine.meta_learning.knowledge_base import MetaKnowledgeBase

        ws = self.workspace
        dataset = self._get_dataset(dataset_id)
        profile = ws.repository.get_dataset_profile(dataset_id)
        if profile is None:
            profile = profile_dataset(dataset)
            ws.repository.save_dataset_profile(profile)

        df = None
        try:
            df = load_dataframe(dataset.path)
        except Exception:
            pass

        fingerprint = extract_fingerprint(dataset, profile, df=df)

        trials = []
        try:
            runs = ws.repository.list_runs()
            for r in runs:
                if r.dataset_id == dataset_id:
                    results = ws.repository.get_leaderboard(r.id)
                    trials.extend(results)
        except Exception:
            pass

        kb = MetaKnowledgeBase()
        return kb.synthesize(fingerprint, workspace_trials=trials if trials else None)

    def list_task_types(self) -> list[dict[str, Any]]:
        from automl.domain.tasks.task_type import TASK_CATALOG, TaskType

        rows = []
        for task in TaskType:
            catalog = TASK_CATALOG[task]
            rows.append(
                {
                    "task_type": task.value,
                    "label": catalog["label"],
                    "description": catalog["description"],
                    "default_metric": catalog["default_metric"],
                    "metrics": catalog["metrics"],
                    "models": catalog["model_ids"],
                }
            )
        return rows
