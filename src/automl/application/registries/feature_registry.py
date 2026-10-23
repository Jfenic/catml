from __future__ import annotations

from typing import TYPE_CHECKING

from automl.application.commands.workspace_commands import (
    CreateFeatureSetCommand,
    ExcludeFeatureCommand,
    GenerateTemporalFeaturesCommand,
    PlanAblationExperimentsCommand,
    PrioritizeFeatureCommand,
    PromoteCandidateFeatureSetCommand,
    SelectFeaturesCommand,
)
from automl.application.queries.workspace_queries import (
    DetectTemporalStructureQuery,
    GetFeatureEvidenceQuery,
    GetFeatureRankingQuery,
    GetMetaKnowledgeQuery,
    ListCandidateFeatureSetsQuery,
    ListFeatureSetsQuery,
)

if TYPE_CHECKING:
    from automl.application.bus.command_bus import CommandBus
    from automl.application.bus.query_bus import QueryBus
    from automl.application.services.workspace import AutoMLWorkspace


def register_feature_handlers(
    workspace: AutoMLWorkspace,
    command_bus: CommandBus,
    query_bus: QueryBus,
) -> None:
    """Registers feature engineering, selection, ablation, and metadata handlers."""
    # Commands
    command_bus.register(
        ExcludeFeatureCommand,
        lambda cmd: workspace.exclude_feature(cmd.dataset_id, cmd.feature_name, run_id=cmd.run_id),
    )
    command_bus.register(
        PrioritizeFeatureCommand,
        lambda cmd: workspace.prioritize_feature(
            cmd.dataset_id,
            cmd.feature_name,
            score=cmd.score,
            run_id=cmd.run_id,
        ),
    )
    command_bus.register(
        CreateFeatureSetCommand,
        lambda cmd: workspace.create_feature_set(
            cmd.dataset_id,
            cmd.name,
            cmd.feature_names,
            lineage=cmd.lineage,
        ),
    )
    command_bus.register(
        SelectFeaturesCommand,
        lambda cmd: workspace.select_features(cmd.run_id, strategy=cmd.strategy),
    )
    command_bus.register(
        PlanAblationExperimentsCommand,
        lambda cmd: workspace.plan_ablation_experiments(
            cmd.run_id,
            base_feature_names=cmd.base_feature_names,
            model_ids=cmd.model_ids,
            max_features=cmd.max_features,
            auto_enqueue=cmd.auto_enqueue,
        ),
    )
    command_bus.register(
        PromoteCandidateFeatureSetCommand,
        lambda cmd: workspace.promote_candidate_feature_set(
            cmd.run_id,
            cmd.candidate_id,
            new_name=cmd.new_name,
            allow_unverified_promotion=cmd.allow_unverified_promotion,
        ),
    )
    command_bus.register(
        GenerateTemporalFeaturesCommand,
        lambda cmd: workspace.generate_temporal_features(
            run_id=cmd.run_id,
            dataset_id=cmd.dataset_id,
            max_lags=cmd.max_lags,
            include_lags=cmd.include_lags,
            include_deltas=cmd.include_deltas,
            include_cyclical=cmd.include_cyclical,
        ),
    )

    # Queries
    query_bus.register(
        DetectTemporalStructureQuery,
        lambda q: workspace.detect_temporal_structure(q.dataset_id),
    )
    query_bus.register(
        GetMetaKnowledgeQuery,
        lambda q: workspace.get_meta_knowledge(q.dataset_id, run_id=q.run_id),
    )
    query_bus.register(
        ListFeatureSetsQuery,
        lambda q: workspace.list_feature_sets(q.dataset_id),
    )
    query_bus.register(
        GetFeatureEvidenceQuery,
        lambda q: workspace.get_feature_evidence(q.run_id, q.feature_id),
    )
    query_bus.register(
        ListCandidateFeatureSetsQuery,
        lambda q: workspace.list_candidate_feature_sets(q.run_id),
    )
    query_bus.register(
        GetFeatureRankingQuery,
        lambda q: workspace.get_feature_ranking(q.run_id, method=q.method),
    )
