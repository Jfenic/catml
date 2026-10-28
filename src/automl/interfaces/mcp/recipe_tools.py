"""Experiment Recipe tools for CATML Model Context Protocol (MCP) server."""
from __future__ import annotations

from dataclasses import asdict
import json
import logging
import uuid
from typing import Any

try:
    import mcp.types as types
except ImportError:
    types = Any  # type: ignore

from automl.application.agents.contracts import ToolCallContext, ToolInvocation
from automl.application.bus.command_bus import CommandBus
from automl.application.bus.query_bus import QueryBus
from automl.application.recipes.commands import (
    ArchiveExperimentRecipeCommand,
    CreateExperimentRecipeFromEvidenceCommand,
    PromoteExperimentRecipeCommand,
    SelectExperimentRecipeCommand,
    SubmitExperimentFromRecipeCommand,
)
from automl.application.recipes.queries import (
    GetExperimentRecipeQuery,
    ListExperimentRecipesQuery,
    ListVerifiedImprovementsQuery,
)
from automl.domain.agents.entities import AgentPermission, ToolErrorCode

logger = logging.getLogger("catml.mcp.recipes")


def register_recipe_mcp_tools(
    server: Any,
    query_bus: QueryBus,
    command_bus: CommandBus,
    executor: Any = None,
) -> None:
    """Register Experiment Recipe tools on an MCPServer instance."""
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
        name="recipe_list_verified_improvements",
        description="List empirical improvements verified from statistical hypotheses (Propose != Accept evidence).",
    )
    def recipe_list_verified_improvements(
        study_id: str | None = None,
        dataset_id: str | None = None,
        accepted_only: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> types.CallToolResult:
        try:
            query = ListVerifiedImprovementsQuery(
                study_id=study_id,
                dataset_id=dataset_id,
                accepted_only=accepted_only,
            )
            improvements = query_bus.dispatch(query)
            paged = improvements[offset : offset + limit] if limit > 0 else improvements
            data = [asdict(i) for i in paged]
            payload = {
                "total": len(improvements),
                "limit": limit,
                "offset": offset,
                "items": data,
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

    @server.tool(
        name="recipe_create_from_evidence",
        description="Create a draft reproducible ExperimentRecipe from verified hypothesis evidence (does not run training).",
    )
    def recipe_create_from_evidence(
        evidence_link_id: str,
        recipe_name: str | None = None,
        notes: str | None = None,
        model_ids: list[str] | None = None,
        idempotency_key: str | None = None,
        approval_id: str | None = None,
    ) -> types.CallToolResult:
        def _fallback():
            try:
                cmd = CreateExperimentRecipeFromEvidenceCommand(
                    evidence_link_id=evidence_link_id,
                    recipe_name=recipe_name,
                    notes=notes,
                    model_ids=model_ids,
                )
                recipe_id = command_bus.dispatch(cmd)
                recipe = query_bus.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
                data = asdict(recipe) if recipe else {"recipe_id": recipe_id}
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(data, default=str, indent=2))],
                    is_error=False,
                )
            except ValueError as ve:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] {str(ve)}")],
                    is_error=True,
                )
            except Exception as e:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                    is_error=True,
                )

        args: dict[str, Any] = {"evidence_link_id": evidence_link_id}
        if recipe_name is not None:
            args["name"] = recipe_name
        if notes is not None:
            args["description"] = notes
        if model_ids is not None:
            args["model_ids"] = model_ids

        return _execute_or_dispatch(
            "recipe_create_from_evidence",
            args,
            scope_id=evidence_link_id,
            idempotency_key=idempotency_key,
            approval_id=approval_id,
            fallback_fn=_fallback,
        )

    @server.tool(
        name="recipe_get",
        description="Retrieve full configuration and status of an ExperimentRecipe by ID.",
    )
    def recipe_get(recipe_id: str) -> types.CallToolResult:
        try:
            recipe = query_bus.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
            if not recipe:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] Recipe not found: {recipe_id}")],
                    is_error=True,
                )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(asdict(recipe), default=str, indent=2))],
                is_error=False,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    @server.tool(
        name="recipe_list",
        description="List registered experiment recipes, optionally filtered by dataset ID and status.",
    )
    def recipe_list(
        dataset_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> types.CallToolResult:
        try:
            recipes = query_bus.dispatch(
                ListExperimentRecipesQuery(
                    dataset_id=dataset_id,
                    status=status,
                )
            )
            paged = recipes[offset : offset + limit] if limit > 0 else recipes
            data = [asdict(r) for r in paged]
            payload = {
                "total": len(recipes),
                "limit": limit,
                "offset": offset,
                "items": data,
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

    @server.tool(
        name="recipe_select",
        description="Explicitly select a recipe for trial/evaluation.",
    )
    def recipe_select(
        recipe_id: str,
        notes: str | None = None,
        idempotency_key: str | None = None,
        approval_id: str | None = None,
    ) -> types.CallToolResult:
        def _fallback():
            try:
                cmd = SelectExperimentRecipeCommand(recipe_id=recipe_id, notes=notes)
                command_bus.dispatch(cmd)
                recipe = query_bus.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
                data = asdict(recipe) if recipe else {"status": "selected", "recipe_id": recipe_id}
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(data, default=str, indent=2))],
                    is_error=False,
                )
            except ValueError as ve:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] {str(ve)}")],
                    is_error=True,
                )
            except Exception as e:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                    is_error=True,
                )

        args: dict[str, Any] = {"recipe_id": recipe_id}
        if notes is not None:
            args["notes"] = notes
        return _execute_or_dispatch(
            "recipe_select",
            args,
            scope_id=recipe_id,
            idempotency_key=idempotency_key,
            approval_id=approval_id,
            fallback_fn=_fallback,
        )

    @server.tool(
        name="recipe_submit",
        description="Submit and execute training experiments reproducing an experiment recipe configuration.",
    )
    def recipe_submit(
        recipe_id: str,
        run_id: str | None = None,
        model_ids: list[str] | None = None,
        notes: str | None = None,
        idempotency_key: str | None = None,
        approval_id: str | None = None,
    ) -> types.CallToolResult:
        def _fallback():
            try:
                cmd = SubmitExperimentFromRecipeCommand(
                    recipe_id=recipe_id,
                    run_id=run_id,
                    model_ids=model_ids,
                    notes=notes,
                )
                exp_ids = command_bus.dispatch(cmd)
                recipe = query_bus.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
                payload = {
                    "status": "submitted",
                    "recipe_id": recipe_id,
                    "executed_experiments": exp_ids,
                    "recipe": asdict(recipe) if recipe else None,
                }
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(payload, default=str, indent=2))],
                    is_error=False,
                )
            except ValueError as ve:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] {str(ve)}")],
                    is_error=True,
                )
            except Exception as e:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                    is_error=True,
                )

        args: dict[str, Any] = {"recipe_id": recipe_id}
        if run_id is not None:
            args["run_id"] = run_id
        if model_ids is not None:
            args["model_ids"] = model_ids
        if notes is not None:
            args["notes"] = notes

        return _execute_or_dispatch(
            "recipe_submit",
            args,
            scope_id=recipe_id,
            idempotency_key=idempotency_key,
            approval_id=approval_id,
            fallback_fn=_fallback,
        )

    @server.tool(
        name="recipe_promote",
        description="Promote an experiment recipe as a reusable production candidate with audit reasoning.",
    )
    def recipe_promote(
        recipe_id: str,
        reason: str | None = None,
        notes: str | None = None,
        idempotency_key: str | None = None,
        approval_id: str | None = None,
    ) -> types.CallToolResult:
        def _fallback():
            try:
                cmd = PromoteExperimentRecipeCommand(
                    recipe_id=recipe_id,
                    reason=reason,
                    notes=notes,
                )
                command_bus.dispatch(cmd)
                recipe = query_bus.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
                data = asdict(recipe) if recipe else {"status": "promoted", "recipe_id": recipe_id}
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(data, default=str, indent=2))],
                    is_error=False,
                )
            except ValueError as ve:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] {str(ve)}")],
                    is_error=True,
                )
            except Exception as e:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                    is_error=True,
                )

        args: dict[str, Any] = {"recipe_id": recipe_id}
        if reason is not None:
            args["reason"] = reason
        if notes is not None:
            args["notes"] = notes

        return _execute_or_dispatch(
            "recipe_promote",
            args,
            scope_id=recipe_id,
            idempotency_key=idempotency_key,
            approval_id=approval_id,
            fallback_fn=_fallback,
        )

    @server.tool(
        name="recipe_archive",
        description="Archive an experiment recipe.",
    )
    def recipe_archive(
        recipe_id: str,
        idempotency_key: str | None = None,
        approval_id: str | None = None,
    ) -> types.CallToolResult:
        def _fallback():
            try:
                cmd = ArchiveExperimentRecipeCommand(recipe_id=recipe_id)
                command_bus.dispatch(cmd)
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps({"status": "archived", "recipe_id": recipe_id}))],
                    is_error=False,
                )
            except ValueError as ve:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] {str(ve)}")],
                    is_error=True,
                )
            except Exception as e:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                    is_error=True,
                )

        return _execute_or_dispatch(
            "recipe_archive",
            {"recipe_id": recipe_id},
            scope_id=recipe_id,
            idempotency_key=idempotency_key,
            approval_id=approval_id,
            fallback_fn=_fallback,
        )
