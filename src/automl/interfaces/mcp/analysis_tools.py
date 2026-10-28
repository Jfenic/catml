"""Analysis and Explore tools for CATML Model Context Protocol (MCP) server."""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

try:
    import mcp.types as types
except ImportError:
    types = Any  # type: ignore

from automl.application.agents.contracts import ToolCallContext, ToolInvocation
from automl.application.analysis.commands import (
    CreateHypothesisCommand,
    CreateStudyCommand,
    RunAnalysisCommand,
    VerifyHypothesisCommand,
)
from automl.application.analysis.queries import (
    GetAnalysisRunQuery,
    GetEvidenceLinkQuery,
    GetHypothesisQuery,
    GetStudyQuery,
    ListEvidenceLinksQuery,
    ListFindingsQuery,
    ListHypothesesQuery,
    ListStudiesQuery,
    ListVisualizationsQuery,
)
from automl.application.bus.command_bus import CommandBus
from automl.application.bus.query_bus import QueryBus
from automl.domain.agents.entities import AgentPermission, ToolErrorCode

logger = logging.getLogger("catml.mcp.analysis")


def _compact_finding(f: dict[str, Any]) -> dict[str, Any]:
    """Prunes bulky data payloads (matrices, raw arrays) from findings metrics for LLM token efficiency."""
    res = dict(f)
    if "metrics" in res and isinstance(res["metrics"], dict):
        cleaned_metrics = {}
        for k, v in res["metrics"].items():
            if isinstance(v, list) and len(v) > 10:
                cleaned_metrics[k] = f"[truncated array: {len(v)} elements]"
            elif isinstance(v, dict) and len(v) > 10:
                cleaned_metrics[k] = f"[truncated dict/matrix: {len(v)} keys]"
            else:
                cleaned_metrics[k] = v
        res["metrics"] = cleaned_metrics
    return res


def register_analysis_mcp_tools(
    server: Any,
    query_bus: QueryBus,
    command_bus: CommandBus,
    executor: Any = None,
) -> None:
    """Register CATML Explore (Fases E1 a E4) tools and resources on an MCPServer instance."""
    if not hasattr(types, "CallToolResult"):
        return

    def _execute_or_dispatch(
        tool_name: str,
        arguments: dict[str, Any],
        scope_id: str,
        idempotency_key: str | None,
        approval_id: str | None,
        fallback_fn: Any,
    ) -> types.CallToolResult:
        active_executor = executor or getattr(server, "executor", None)
        if active_executor is not None:
            inv = ToolInvocation(
                tool_name=tool_name,
                arguments=arguments,
                context=ToolCallContext(
                    actor="mcp_agent",
                    workspace_path=".",
                    run_id=scope_id,
                    correlation_id=f"mcp-{uuid.uuid4().hex[:8]}",
                    permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
                    approval_id=approval_id,
                ),
                idempotency_key=idempotency_key,
                approval_id=approval_id,
            )
            res = active_executor.execute(inv)
            if res.success:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(res.data, default=str, indent=2))],
                    is_error=False,
                )
            if res.error and res.error.code == ToolErrorCode.APPROVAL_REQUIRED:
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "status": "PENDING_APPROVAL",
                                    "approval_id": res.approval_id,
                                    "message": f"Action '{tool_name}' requires human approval. Run: automl agent approve {res.approval_id}",
                                    "details": res.error.details,
                                },
                                indent=2,
                            ),
                        )
                    ],
                    is_error=False,
                )
            err_msg = res.error.message if res.error else "Unknown error"
            err_code = res.error.code.value if res.error else ToolErrorCode.INTERNAL_ERROR.value
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{err_code}] {err_msg}")],
                is_error=True,
            )
        return fallback_fn()

    @server.tool(
        name="analysis_create_study",
        description="Create an exploratory study specification for a dataset (unsupervised or with an optional target column).",
    )
    def analysis_create_study(
        dataset_id: str,
        name: str = "",
        target_column: str | None = None,
        idempotency_key: str | None = None,
        approval_id: str | None = None,
    ) -> types.CallToolResult:
        def _fallback():
            try:
                cmd = CreateStudyCommand(
                    dataset_id=dataset_id,
                    name=name,
                    target_column=target_column if target_column else None,
                )
                study_res = command_bus.dispatch(cmd)
                study_id = study_res if isinstance(study_res, str) else getattr(study_res, "id", str(study_res))
                study = query_bus.dispatch(GetStudyQuery(study_id=study_id))
                data = study if isinstance(study, dict) else (study.to_dict() if hasattr(study, "to_dict") else dict(study))
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(data, default=str, indent=2))],
                    is_error=False,
                )
            except Exception as e:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                    is_error=True,
                )

        args: dict[str, Any] = {"dataset_id": dataset_id, "name": name}
        if target_column is not None:
            args["target_column"] = target_column

        return _execute_or_dispatch(
            "analysis_create_study",
            args,
            scope_id=dataset_id,
            idempotency_key=idempotency_key,
            approval_id=approval_id,
            fallback_fn=_fallback,
        )

    @server.tool(
        name="analysis_run_study",
        description="Execute deterministic statistical and exploratory analysis on an existing study.",
    )
    def analysis_run_study(
        study_id: str,
        idempotency_key: str | None = None,
        approval_id: str | None = None,
    ) -> types.CallToolResult:
        def _fallback():
            try:
                cmd = RunAnalysisCommand(study_id=study_id)
                run_res = command_bus.dispatch(cmd)
                run_id = run_res if isinstance(run_res, str) else getattr(run_res, "id", str(run_res))
                run_data = query_bus.dispatch(GetAnalysisRunQuery(run_id=run_id))
                findings = query_bus.dispatch(ListFindingsQuery(study_id=study_id, run_id=run_id))
                payload = {
                    "study_id": study_id,
                    "run_id": run_id,
                    "run": run_data,
                    "findings_count": len(findings) if findings else 0,
                }
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(payload, default=str, indent=2))],
                    is_error=False,
                )
            except KeyError as ke:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] {str(ke)}")],
                    is_error=True,
                )
            except Exception as e:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                    is_error=True,
                )

        return _execute_or_dispatch(
            "analysis_run_study",
            {"study_id": study_id},
            scope_id=study_id,
            idempotency_key=idempotency_key,
            approval_id=approval_id,
            fallback_fn=_fallback,
        )

    @server.tool(
        name="analysis_get_study",
        description="Retrieve details and configuration of an exploratory study by its ID.",
    )
    def analysis_get_study(study_id: str) -> types.CallToolResult:
        try:
            study = query_bus.dispatch(GetStudyQuery(study_id=study_id))
            if not study:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] Study '{study_id}' not found.")],
                    is_error=True,
                )
            data = study.to_dict() if hasattr(study, "to_dict") else dict(study)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(data, default=str, indent=2))],
                is_error=False,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    @server.tool(
        name="analysis_list_studies",
        description="List all exploratory studies in the workspace, optionally filtered by dataset ID.",
    )
    def analysis_list_studies(dataset_id: str | None = None) -> types.CallToolResult:
        try:
            studies = query_bus.dispatch(ListStudiesQuery(dataset_id=dataset_id))
            data = [s.to_dict() if hasattr(s, "to_dict") else dict(s) for s in (studies or [])]
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(data, default=str, indent=2))],
                is_error=False,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    @server.tool(
        name="analysis_get_findings",
        description="Retrieve deterministic statistical findings (correlations, outliers, distributions, tests) for a study with token efficiency, filtering, and pagination.",
    )
    def analysis_get_findings(
        study_id: str,
        finding_type: str | None = None,
        category: str | None = None,
        min_significance: float | None = None,
        severity: str | None = None,
        limit: int = 20,
        offset: int = 0,
        compact: bool = True,
        envelope: bool = False,
    ) -> types.CallToolResult:
        try:
            study = query_bus.dispatch(GetStudyQuery(study_id=study_id))
            if not study:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] Study '{study_id}' not found.")],
                    is_error=True,
                )

            # Query all matching findings to support accurate pagination counting
            all_findings = query_bus.dispatch(
                ListFindingsQuery(
                    study_id=study_id,
                    finding_type=finding_type,
                    category=category,
                    min_significance=min_significance,
                    severity=severity,
                )
            )
            raw_list = [f.to_dict() if hasattr(f, "to_dict") else dict(f) for f in (all_findings or [])]
            total_count = len(raw_list)

            # Slicing
            start = max(0, offset)
            end = start + limit if limit > 0 else total_count
            sliced = raw_list[start:end]

            if compact:
                sliced = [_compact_finding(f) for f in sliced]

            if envelope:
                payload = {
                    "study_id": study_id,
                    "total": total_count,
                    "limit": limit,
                    "offset": start,
                    "has_more": end < total_count,
                    "findings": sliced,
                }
                result_text = json.dumps(payload, default=str, indent=2)
            else:
                result_text = json.dumps(sliced, default=str, indent=2)

            return types.CallToolResult(
                content=[types.TextContent(type="text", text=result_text)],
                is_error=False,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    @server.tool(
        name="analysis_get_visualizations",
        description="Retrieve declarative visualization specifications (e.g. correlation heatmaps, distributions, scatter plots) for a study.",
    )
    def analysis_get_visualizations(
        study_id: str,
        visualization_type: str | None = None,
        chart_id: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> types.CallToolResult:
        try:
            study = query_bus.dispatch(GetStudyQuery(study_id=study_id))
            if not study:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] Study '{study_id}' not found.")],
                    is_error=True,
                )

            specs = query_bus.dispatch(
                ListVisualizationsQuery(
                    study_id=study_id,
                    visualization_type=visualization_type,
                    chart_id=chart_id,
                    limit=limit,
                    offset=offset,
                )
            )
            data = [s.to_dict() if hasattr(s, "to_dict") else dict(s) for s in (specs or [])]
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(data, default=str, indent=2))],
                is_error=False,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    @server.tool(
        name="analysis_propose_experiment",
        description="Propose an actionable machine learning experiment or feature engineering hypothesis derived from a statistical finding (Principle: Propose ≠ Accept).",
    )
    def analysis_propose_experiment(
        study_id: str,
        finding_id: str,
        hypothesis_description: str,
        proposed_action: str = "feature_engineering",
        transformation_spec: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        approval_id: str | None = None,
    ) -> types.CallToolResult:
        def _fallback():
            try:
                study = query_bus.dispatch(GetStudyQuery(study_id=study_id))
                if not study:
                    return types.CallToolResult(
                        content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] Study '{study_id}' not found.")],
                        is_error=True,
                    )

                findings = query_bus.dispatch(ListFindingsQuery(study_id=study_id))
                f_list = [f.to_dict() if hasattr(f, "to_dict") else dict(f) for f in (findings or [])]
                target_finding = next((f for f in f_list if f.get("id") == finding_id), None)
                if not target_finding:
                    return types.CallToolResult(
                        content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] Finding '{finding_id}' not found in study '{study_id}'.")],
                        is_error=True,
                    )

                cmd = CreateHypothesisCommand(
                    study_id=study_id,
                    finding_id=finding_id,
                    description=hypothesis_description,
                    proposed_action=proposed_action,
                    experiment_delta=transformation_spec or {},
                )
                hyp_id = command_bus.dispatch(cmd)
                payload = {
                    "hypothesis_id": hyp_id,
                    "study_id": study_id,
                    "finding_id": finding_id,
                    "description": hypothesis_description,
                    "proposed_action": proposed_action,
                    "status": "proposed",
                    "experiment_delta": transformation_spec or {},
                    "note": "Hypothesis recorded under 'Propose ≠ Accept' principle. Requires empirical validation trial before promotion.",
                }
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(payload, default=str, indent=2))],
                    is_error=False,
                )
            except Exception as e:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                    is_error=True,
                )

        return _execute_or_dispatch(
            "analysis_propose_experiment",
            {
                "study_id": study_id,
                "finding_id": finding_id,
                "hypothesis_description": hypothesis_description,
                "proposed_action": proposed_action,
                "transformation_spec": transformation_spec or {},
            },
            scope_id=study_id,
            idempotency_key=idempotency_key,
            approval_id=approval_id,
            fallback_fn=_fallback,
        )

    # -----------------------------------------------------------------------
    # Resources
    # -----------------------------------------------------------------------
    @server.resource("catml://studies/{study_id}")
    def study_detail_resource(study_id: str) -> str:
        """Resource exposing study metadata as JSON."""
        try:
            study = query_bus.dispatch(GetStudyQuery(study_id=study_id))
            if not study:
                return json.dumps({"error": f"Study '{study_id}' not found", "code": ToolErrorCode.NOT_FOUND.value})
            data = study.to_dict() if hasattr(study, "to_dict") else dict(study)
            return json.dumps(data, default=str, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "code": ToolErrorCode.INTERNAL_ERROR.value})

    @server.resource("catml://studies/{study_id}/findings")
    def study_findings_resource(study_id: str) -> str:
        """Resource exposing study statistical findings as JSON."""
        try:
            findings = query_bus.dispatch(ListFindingsQuery(study_id=study_id))
            data = [f.to_dict() if hasattr(f, "to_dict") else dict(f) for f in (findings or [])]
            return json.dumps(data, default=str, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "code": ToolErrorCode.INTERNAL_ERROR.value})

    @server.resource("catml://studies/{study_id}/summary")
    def study_summary_resource(study_id: str) -> str:
        """Resource exposing high-density, token-efficient executive statistical summary as JSON."""
        try:
            study = query_bus.dispatch(GetStudyQuery(study_id=study_id))
            if not study:
                return json.dumps({"error": f"Study '{study_id}' not found", "code": ToolErrorCode.NOT_FOUND.value})
            s_data = study.to_dict() if hasattr(study, "to_dict") else dict(study)
            findings = query_bus.dispatch(ListFindingsQuery(study_id=study_id))
            f_list = [f.to_dict() if hasattr(f, "to_dict") else dict(f) for f in (findings or [])]
            viz = query_bus.dispatch(ListVisualizationsQuery(study_id=study_id))
            hyps = query_bus.dispatch(ListHypothesesQuery(study_id=study_id))

            findings_by_type: dict[str, int] = {}
            sig_count = 0
            key_signals: list[dict[str, Any]] = []

            for f in f_list:
                ft = str(f.get("finding_type", "other")).upper()
                findings_by_type[ft] = findings_by_type.get(ft, 0) + 1
                p = f.get("p_value")
                p_fdr = (f.get("metrics") or {}).get("p_value_fdr")
                is_sig = (p is not None and p <= 0.05) or (p_fdr is not None and p_fdr <= 0.05)
                if is_sig:
                    sig_count += 1
                    if len(key_signals) < 10:
                        key_signals.append({
                            "id": f.get("id"),
                            "type": ft,
                            "column": f.get("column_name"),
                            "secondary_column": f.get("secondary_column"),
                            "p_value": p,
                            "p_value_fdr": p_fdr,
                            "effect_size": f.get("effect_size"),
                            "summary": f.get("summary"),
                        })

            summary = {
                "study_id": study_id,
                "name": s_data.get("name"),
                "dataset_id": (s_data.get("data_source") or {}).get("dataset_id") or s_data.get("dataset_id"),
                "target_column": s_data.get("target_column"),
                "status": s_data.get("status"),
                "total_findings": len(f_list),
                "significant_findings": sig_count,
                "findings_by_type": findings_by_type,
                "key_signals": key_signals,
                "visualizations_count": len(viz or []),
                "hypotheses_count": len(hyps or []),
            }
            return json.dumps(summary, default=str, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "code": ToolErrorCode.INTERNAL_ERROR.value})

    @server.tool(
        name="analysis_verify_hypothesis",
        description="Empirically verify a proposed hypothesis against AutoML baseline on identical validation split (Principle: Propose ≠ Accept). Promotes to 'accepted' only if metric delta > min_improvement.",
    )
    def analysis_verify_hypothesis(
        hypothesis_id: str,
        run_id: str | None = None,
        min_improvement: float = 0.0,
        idempotency_key: str | None = None,
        approval_id: str | None = None,
    ) -> types.CallToolResult:
        def _fallback():
            try:
                cmd = VerifyHypothesisCommand(
                    hypothesis_id=hypothesis_id,
                    run_id=run_id,
                    min_improvement=min_improvement,
                )
                link_data = command_bus.dispatch(cmd)
                payload = link_data if isinstance(link_data, dict) else (link_data.to_dict() if hasattr(link_data, "to_dict") else dict(link_data))
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(payload, default=str, indent=2))],
                    is_error=False,
                )
            except KeyError as ke:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] {str(ke)}")],
                    is_error=True,
                )
            except Exception as e:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                    is_error=True,
                )

        args: dict[str, Any] = {
            "hypothesis_id": hypothesis_id,
            "min_improvement": min_improvement,
        }
        if run_id is not None:
            args["run_id"] = run_id

        return _execute_or_dispatch(
            "analysis_verify_hypothesis",
            args,
            scope_id=run_id or hypothesis_id,
            idempotency_key=idempotency_key,
            approval_id=approval_id,
            fallback_fn=_fallback,
        )

    @server.resource("catml://studies/{study_id}/visualizations")
    def study_visualizations_resource(study_id: str) -> str:
        """Resource exposing declarative visualization specs list as JSON."""
        try:
            viz = query_bus.dispatch(ListVisualizationsQuery(study_id=study_id))
            data = [v.to_dict() if hasattr(v, "to_dict") else dict(v) for v in (viz or [])]
            return json.dumps(data, default=str, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "code": ToolErrorCode.INTERNAL_ERROR.value})

    @server.resource("catml://studies/{study_id}/evidence")
    def study_evidence_resource(study_id: str) -> str:
        """Resource exposing empirical evidence verification links for a study as JSON."""
        try:
            links = query_bus.dispatch(ListEvidenceLinksQuery(study_id=study_id))
            data = [l.to_dict() if hasattr(l, "to_dict") else dict(l) for l in (links or [])]
            return json.dumps(data, default=str, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "code": ToolErrorCode.INTERNAL_ERROR.value})
