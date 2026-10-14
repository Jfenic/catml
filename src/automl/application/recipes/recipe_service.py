from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from automl.application.recipes.commands import (
    ArchiveExperimentRecipeCommand,
    CreateExperimentRecipeFromEvidenceCommand,
    PromoteExperimentRecipeCommand,
    SelectExperimentRecipeCommand,
    SubmitExperimentFromRecipeCommand,
)
from automl.application.recipes.queries import (
    ExperimentRecipeDTO,
    GetExperimentRecipeQuery,
    ListExperimentRecipesQuery,
    ListVerifiedImprovementsQuery,
    VerifiedImprovementDTO,
)
from automl.domain.datasets.content_hash import try_compute_file_content_hash
from automl.domain.experiments.validation import require_known_validation_strategy
from automl.domain.recipes.ports import ExperimentRecipeRepositoryPort
from automl.domain.recipes.recipe import ExperimentRecipe, RecipeStatus

if TYPE_CHECKING:
    from automl.application.services.workspace import AutoMLWorkspace
    from automl.domain.analysis.ports import StudyRepositoryPort

logger = logging.getLogger(__name__)


class RecipeService:
    """
    Application service orchestrating reproducible experiment recipes,
    connecting verified evidence from CATML Explore with AutoML training and promotion.
    """

    def __init__(
        self,
        recipe_repository: ExperimentRecipeRepositoryPort,
        study_repository: StudyRepositoryPort,
        workspace: AutoMLWorkspace,
    ) -> None:
        self.recipe_repo = recipe_repository
        self.study_repo = study_repository
        self.workspace = workspace

    def create_recipe_from_evidence(
        self,
        cmd: CreateExperimentRecipeFromEvidenceCommand,
    ) -> str:
        """
        Creates a draft reproducible ExperimentRecipe from verified hypothesis evidence.
        Does not train models automatically (strict Propose != Accept).
        """
        evidence = self.study_repo.get_evidence_link(cmd.evidence_link_id)
        if not evidence:
            raise ValueError(f"Evidence link not found: {cmd.evidence_link_id}")

        hypothesis = self.study_repo.get_hypothesis(evidence.hypothesis_id)
        if not hypothesis:
            raise ValueError(f"Hypothesis not found for evidence: {evidence.hypothesis_id}")

        study = self.study_repo.get_study(hypothesis.study_id)
        if not study:
            raise ValueError(f"Study not found for hypothesis: {hypothesis.study_id}")

        dataset_id = study.data_source.dataset_id
        dataset = self.workspace.get_dataset(dataset_id)
        dataset_content_hash = None
        if dataset is not None:
            dataset_content_hash = getattr(dataset, "content_hash", None) or try_compute_file_content_hash(
                getattr(dataset, "path", None)
            )

        feature_names: list[str] = []
        derived_features: list[dict[str, Any]] = []
        column_transforms: list[dict[str, Any]] = []
        drop_features: list[str] = []
        model_ids: list[str] = list(cmd.model_ids or [])
        validation_strategy: str = require_known_validation_strategy(
            cmd.validation_strategy or "kfold",
            default="kfold",
        )
        metric: str = cmd.metric or evidence.metric
        cv_folds: int = cmd.cv_folds or 5
        random_seed: int = cmd.random_seed or 42

        # 1. Inspect candidate experiment if available
        candidate_exp = None
        if evidence.experiment_id:
            candidate_exp = self.workspace.get_experiment(evidence.experiment_id)

        if candidate_exp is not None:
            feature_names = list(candidate_exp.feature_names)
            derived_features = [dict(df) for df in candidate_exp.derived_features]
            column_transforms = [dict(ct) for ct in candidate_exp.column_transforms]
            if not model_ids:
                model_ids = list(candidate_exp.model_ids)
            if not cmd.metric:
                metric = candidate_exp.metric
            if not cmd.validation_strategy:
                validation_strategy = require_known_validation_strategy(
                    candidate_exp.validation_strategy,
                    default="kfold",
                )

            # Infer dropped features by checking baseline experiment if recorded
            baseline_exp_id = (
                hypothesis.experiment_delta.get("baseline_experiment_id")
                if isinstance(hypothesis.experiment_delta, dict)
                else None
            )
            if baseline_exp_id:
                baseline_exp = self.workspace.get_experiment(baseline_exp_id)
                if baseline_exp:
                    drop_features = [f for f in baseline_exp.feature_names if f not in feature_names]
        else:
            # Fallback to translate via HypothesisExperimentTranslator
            from automl.engine.analysis.hypothesis_translator import (
                HypothesisExperimentTranslator,
            )

            base_reg = self.workspace.get_feature_registry(dataset_id)
            base_features = [f.name for f in base_reg.list()]
            target_col = study.target_column
            if target_col and target_col in base_features:
                base_features.remove(target_col)
            task_type = getattr(dataset, "task_type", None) if dataset else None
            default_models = list(model_ids) if model_ids else None
            spec = HypothesisExperimentTranslator.translate(
                hypothesis,
                base_features,
                task_type=task_type,
                default_models=default_models,
            )
            feature_names = list(spec.feature_names)
            derived_features = [dict(df) for df in spec.derived_features]
            column_transforms = [dict(ct) for ct in spec.column_transforms]
            drop_features = list(spec.drop_features)
            if not model_ids:
                model_ids = list(spec.model_ids)

        if not column_transforms and dataset_id:
            try:
                from automl.domain.preparation.pipeline_bridge import (
                    pipeline_steps_to_column_transforms,
                )
                from automl.infrastructure.preparation.json_store import JsonPreparationStore

                prep_store = JsonPreparationStore(self.workspace.root_dir)
                version = prep_store.find_version_for_dataset(dataset_id)
                if not version:
                    ds = self.workspace.get_dataset(dataset_id)
                    if ds and hasattr(ds, "path"):
                        version = prep_store.find_version_by_path(str(ds.path))
                if version and version.lineage and version.lineage.get("pipeline_steps"):
                    column_transforms = pipeline_steps_to_column_transforms(
                        version.lineage["pipeline_steps"]
                    )
            except Exception:
                pass

        if not drop_features and isinstance(hypothesis.experiment_delta, dict):
            drop_features = list(hypothesis.experiment_delta.get("drop_features", []))

        recipe_name = cmd.recipe_name or f"Recipe: {hypothesis.proposed_action} ({hypothesis.id})"
        lineage = {
            "evidence_link_id": evidence.id,
            "hypothesis_id": hypothesis.id,
            "study_id": study.id,
            "finding_id": hypothesis.finding_id,
            "baseline_score": evidence.baseline_score,
            "candidate_score": evidence.candidate_score,
            "delta_primary_metric": evidence.candidate_score - evidence.baseline_score,
            "accepted": evidence.accepted,
            "notes": cmd.notes or evidence.notes,
            **dict(cmd.extra_lineage),
        }

        recipe = ExperimentRecipe.create(
            name=recipe_name,
            dataset_id=dataset_id,
            dataset_content_hash=dataset_content_hash,
            feature_names=feature_names,
            derived_features=derived_features,
            column_transforms=column_transforms,
            drop_features=drop_features,
            model_ids=model_ids,
            validation_strategy=validation_strategy,
            cv_folds=cv_folds,
            metric=metric,
            random_seed=random_seed,
            evidence_link_id=evidence.id,
            hypothesis_id=hypothesis.id,
            study_id=study.id,
            lineage=lineage,
            status=RecipeStatus.DRAFT,
        )

        self.recipe_repo.save(recipe)
        return recipe.id

    def submit_experiment_from_recipe(
        self,
        cmd: SubmitExperimentFromRecipeCommand,
    ) -> list[str]:
        """
        Submits and runs one or more experiments reproducing the recipe configuration.
        """
        recipe = self.recipe_repo.get(cmd.recipe_id)
        if not recipe:
            raise ValueError(f"Experiment recipe not found: {cmd.recipe_id}")

        if recipe.status == RecipeStatus.ARCHIVED:
            raise ValueError(f"Cannot submit archived recipe: {recipe.id}")

        # Resolve Run
        if cmd.run_id:
            run = self.workspace.get_run(cmd.run_id)
            if not run:
                raise ValueError(f"Specified run not found: {cmd.run_id}")
        else:
            runs = self.workspace.list_runs(recipe.dataset_id)
            active_runs = [
                r for r in runs
                if getattr(r, "is_active", False)
                or getattr(r.status, "value", str(r.status)) in ("CREATED", "RUNNING", "EXPERIMENTING")
            ]
            if active_runs:
                run = active_runs[0]
            elif runs:
                run = runs[0]
            else:
                dataset = self.workspace.get_dataset(recipe.dataset_id)
                if not dataset:
                    raise ValueError(f"Dataset not found for recipe: {recipe.dataset_id}")
                run = self.workspace.create_run(dataset)

        # Restore declared recipe execution contracts onto the run before creating experiments.
        val_strategy = require_known_validation_strategy(
            recipe.validation_strategy,
            default="kfold",
        )
        run.config.metric = recipe.metric
        run.config.cv_folds = int(recipe.cv_folds)
        run.config.random_seed = int(recipe.random_seed)
        run.config.validation_strategy = val_strategy
        self.workspace.save_run(run)

        # Optional integrity check: warn (do not hard-fail) when dataset bytes drifted.
        dataset = self.workspace.get_dataset(recipe.dataset_id)
        if recipe.dataset_content_hash and dataset is not None:
            current_hash = try_compute_file_content_hash(getattr(dataset, "path", None))
            if current_hash and current_hash != recipe.dataset_content_hash:
                logger.warning(
                    "Recipe %s dataset content_hash mismatch (recipe=%s current=%s)",
                    recipe.id,
                    recipe.dataset_content_hash[:12],
                    current_hash[:12],
                )
                recipe.lineage.setdefault("integrity_warnings", []).append({
                    "type": "dataset_content_hash_mismatch",
                    "expected": recipe.dataset_content_hash,
                    "actual": current_hash,
                })

        models_to_run = list(cmd.model_ids or recipe.model_ids or ["random_forest"])
        new_experiment_ids: list[str] = []

        for model_id in models_to_run:
            exp_name = f"{recipe.name} [{model_id}]"
            experiment = self.workspace.create_experiment(
                run=run,
                name=exp_name,
                model_ids=[model_id],
                feature_names=recipe.feature_names,
                validation_strategy=val_strategy,
                hypothesis=f"Recipe {recipe.id}: {recipe.name}",
                column_transforms=recipe.column_transforms,
                derived_features=recipe.derived_features,
                recipe_id=recipe.id,
            )
            self.workspace.run_experiment(run, experiment)
            recipe.submit(experiment.id)
            new_experiment_ids.append(experiment.id)

        if cmd.notes:
            recipe.lineage.setdefault("history", []).append({
                "action": "submitted",
                "run_id": run.id,
                "experiment_ids": new_experiment_ids,
                "notes": cmd.notes,
            })

        self.recipe_repo.save(recipe)
        return new_experiment_ids

    def select_recipe(self, cmd: SelectExperimentRecipeCommand) -> str:
        """Explicitly selects recipe for execution."""
        recipe = self.recipe_repo.get(cmd.recipe_id)
        if not recipe:
            raise ValueError(f"Experiment recipe not found: {cmd.recipe_id}")
        recipe.select(notes=cmd.notes)
        self.recipe_repo.save(recipe)
        return recipe.id

    def promote_recipe(self, cmd: PromoteExperimentRecipeCommand) -> str:
        """Promotes recipe after empirical validation."""
        recipe = self.recipe_repo.get(cmd.recipe_id)
        if not recipe:
            raise ValueError(f"Experiment recipe not found: {cmd.recipe_id}")
        recipe.promote(reason=cmd.reason, notes=cmd.notes)
        self.recipe_repo.save(recipe)
        return recipe.id

    def archive_recipe(self, cmd: ArchiveExperimentRecipeCommand) -> str:
        """Archives recipe."""
        recipe = self.recipe_repo.get(cmd.recipe_id)
        if not recipe:
            raise ValueError(f"Experiment recipe not found: {cmd.recipe_id}")
        recipe.archive()
        self.recipe_repo.save(recipe)
        return recipe.id

    def get_recipe(self, q: GetExperimentRecipeQuery) -> ExperimentRecipeDTO | None:
        """Retrieves a recipe as DTO."""
        recipe = self.recipe_repo.get(q.recipe_id)
        if not recipe:
            return None
        return ExperimentRecipeDTO.from_domain(recipe)

    def list_recipes(self, q: ListExperimentRecipesQuery) -> list[ExperimentRecipeDTO]:
        """Lists recipes optionally filtered by dataset ID and status."""
        recipes = self.recipe_repo.list(dataset_id=q.dataset_id, status=q.status)
        return [ExperimentRecipeDTO.from_domain(r) for r in recipes]

    def list_verified_improvements(
        self,
        q: ListVerifiedImprovementsQuery,
    ) -> list[VerifiedImprovementDTO]:
        """Lists verified improvements across studies, filtering by evidence links."""
        if q.study_id:
            hypotheses = self.study_repo.list_hypotheses(q.study_id)
            hyp_map = {h.id: h for h in hypotheses}
            links = []
            for h in hypotheses:
                links.extend(self.study_repo.list_evidence_links(hypothesis_id=h.id))
        else:
            links = self.study_repo.list_evidence_links()
            hyp_map = {}
            for l in links:
                if l.hypothesis_id not in hyp_map:
                    h = self.study_repo.get_hypothesis(l.hypothesis_id)
                    if h:
                        hyp_map[l.hypothesis_id] = h

        results: list[VerifiedImprovementDTO] = []
        for l in links:
            if q.accepted_only and not l.accepted:
                continue
            h = hyp_map.get(l.hypothesis_id)
            if not h:
                continue
            study = self.study_repo.get_study(h.study_id)
            if not study:
                continue
            if q.dataset_id and study.data_source.dataset_id != q.dataset_id:
                continue

            delta = float(l.candidate_score - l.baseline_score)
            base_exp_id = (
                h.experiment_delta.get("baseline_experiment_id")
                if isinstance(h.experiment_delta, dict)
                else None
            )

            results.append(
                VerifiedImprovementDTO(
                    evidence_link_id=l.id,
                    study_id=h.study_id,
                    hypothesis_id=h.id,
                    dataset_id=study.data_source.dataset_id,
                    suggested_action=h.proposed_action,
                    hypothesis_description=h.description,
                    accepted=l.accepted,
                    delta_primary_metric=delta,
                    baseline_score=float(l.baseline_score),
                    candidate_score=float(l.candidate_score),
                    metric_name=l.metric,
                    candidate_experiment_id=l.experiment_id,
                    baseline_experiment_id=base_exp_id,
                    created_at=l.created_at,
                )
            )

        return results
