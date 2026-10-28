"""Tool executor dispatching invocations through policy checks, scope isolation, and QueryBus."""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from automl.application.agents.contracts import (
    ApprovalRequest,
    OperationRecord,
    ToolCallContext,
    ToolDefinition,
    ToolError,
    ToolInvocation,
    ToolResult,
)
from automl.application.agents.policy import (
    PolicyEvaluator,
    compute_arguments_hash,
)
from automl.application.agents.ports import AgentLedgerPort
from automl.application.agents.registry import ToolRegistry
from automl.application.agents.schemas import (
    TOOL_SCHEMAS,
    validate_arguments,
)
from automl.application.bus.command_bus import CommandBus
from automl.application.bus.query_bus import QueryBus
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    OptimizeExperimentCommand,
    PrioritizeFeatureCommand,
    RunExperimentCommand,
)
from automl.application.queries.workspace_queries import (
    GetDatasetProfileQuery,
    GetFeatureEvidenceQuery,
    GetFeatureRankingQuery,
    GetLeaderboardQuery,
    ListExperimentsQuery,
    ListModelsQuery,
    ListPluginsQuery,
)
from automl.domain.agents.entities import (
    AgentBudget,
    AgentPermission,
    ApprovalStatus,
    OperationStatus,
    PolicyDecisionType,
    ToolEffect,
    ToolErrorCode,
)

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes agent tools with schema validation, policy authorization, scope checking, and error mapping."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy_evaluator: PolicyEvaluator | None = None,
        ledger: AgentLedgerPort | None = None,
        default_budget: AgentBudget | None = None,
        approval_ttl_seconds: float = 86400.0,
    ):
        self.registry = registry
        self.policy_evaluator = policy_evaluator or PolicyEvaluator()
        self.ledger = ledger
        self.default_budget = default_budget or AgentBudget()
        self.approval_ttl_seconds = float(approval_ttl_seconds)

    def execute(
        self,
        invocation: ToolInvocation,
        budget: AgentBudget | None = None,
    ) -> ToolResult:
        """Execute a tool invocation adhering to CQRS, authorization, and idempotency rules."""
        start_time = time.perf_counter()
        req_id = invocation.idempotency_key or str(uuid.uuid4())
        active_budget = budget or self.default_budget

        # 1. Lookup tool definition
        tool_def = self.registry.get_definition(invocation.tool_name)
        if not tool_def:
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                error=ToolError(
                    code=ToolErrorCode.NOT_FOUND,
                    message=f"Tool '{invocation.tool_name}' is not registered in the catalog",
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )

        handler = self.registry.get_handler(invocation.tool_name)
        if not handler:
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                error=ToolError(
                    code=ToolErrorCode.INTERNAL_ERROR,
                    message=f"No execution handler registered for tool '{invocation.tool_name}'",
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )

        # 2. Validate input schema
        if tool_def.input_schema:
            validation_errors = validate_arguments(tool_def.input_schema, invocation.arguments)
            if validation_errors:
                return ToolResult(
                    request_id=req_id,
                    tool_name=invocation.tool_name,
                    success=False,
                    error=ToolError(
                        code=ToolErrorCode.INVALID_ARGUMENT,
                        message=f"Invalid arguments for tool '{invocation.tool_name}': {'; '.join(validation_errors)}",
                        details={"validation_errors": validation_errors},
                        correlation_id=invocation.context.correlation_id,
                    ),
                    execution_time_seconds=time.perf_counter() - start_time,
                )

        # 3. Scope Isolation (Prevent issue #14: run cross-contamination)
        target_run_id = invocation.arguments.get("run_id")
        if target_run_id and invocation.context.run_id:
            if target_run_id != invocation.context.run_id:
                return ToolResult(
                    request_id=req_id,
                    tool_name=invocation.tool_name,
                    success=False,
                    error=ToolError(
                        code=ToolErrorCode.SCOPE_VIOLATION,
                        message=f"Scope violation: cannot access run_id '{target_run_id}' from context authorized for '{invocation.context.run_id}'",
                        correlation_id=invocation.context.correlation_id,
                    ),
                    execution_time_seconds=time.perf_counter() - start_time,
                )

        # 3b. Check deadline before approval and policy evaluation
        if invocation.context.deadline is not None and time.time() > invocation.context.deadline:
            op_id = None
            if self.ledger and invocation.idempotency_key:
                op_id = f"op-{uuid.uuid4().hex[:12]}"
                try:
                    self.ledger.record_operation(
                        OperationRecord(
                            operation_id=op_id,
                            run_id=invocation.context.run_id,
                            actor=invocation.context.actor,
                            idempotency_key=invocation.idempotency_key,
                            action=invocation.tool_name,
                            arguments_hash=compute_arguments_hash(
                                action=invocation.tool_name,
                                actor=invocation.context.actor,
                                run_id=invocation.context.run_id,
                                arguments=invocation.arguments,
                                policy_version=getattr(getattr(self.policy_evaluator, "config", None), "policy_version", "1.0.0"),
                                max_cost=tool_def.cost_estimate,
                            ),
                            arguments=invocation.arguments,
                            status=OperationStatus.TIMED_OUT,
                            error_code=ToolErrorCode.DEADLINE_EXCEEDED.value,
                            error_message="Operation deadline expired before execution",
                        )
                    )
                except Exception as exc:
                    logger.warning("Failed to record timed out operation: %s", exc)
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                operation_id=op_id,
                error=ToolError(
                    code=ToolErrorCode.DEADLINE_EXCEEDED,
                    message="Operation deadline exceeded",
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )

        # 4. Check for pre-existing approval
        approval_id = (
            invocation.approval_id
            or getattr(invocation.context, "approval_id", None)
            or invocation.arguments.get("approval_id")
        )
        pre_approved = False
        if approval_id:
            if not self.ledger:
                return ToolResult(
                    request_id=req_id,
                    tool_name=invocation.tool_name,
                    success=False,
                    approval_id=approval_id,
                    error=ToolError(
                        code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
                        message="Ledger is required to verify approval requests",
                        correlation_id=invocation.context.correlation_id,
                    ),
                    execution_time_seconds=time.perf_counter() - start_time,
                )
            approval_record = self.ledger.get_approval(approval_id)
            if not approval_record:
                return ToolResult(
                    request_id=req_id,
                    tool_name=invocation.tool_name,
                    success=False,
                    approval_id=approval_id,
                    error=ToolError(
                        code=ToolErrorCode.NOT_FOUND,
                        message=f"Approval request '{approval_id}' not found",
                        correlation_id=invocation.context.correlation_id,
                    ),
                    execution_time_seconds=time.perf_counter() - start_time,
                )

            # Check expiration
            if approval_record.expires_at:
                try:
                    exp_dt = datetime.fromisoformat(approval_record.expires_at.replace("Z", "+00:00"))
                    if datetime.now(timezone.utc) > exp_dt:
                        self.ledger.update_approval_status(approval_id, ApprovalStatus.EXPIRED)
                        return ToolResult(
                            request_id=req_id,
                            tool_name=invocation.tool_name,
                            success=False,
                            approval_id=approval_id,
                            error=ToolError(
                                code=ToolErrorCode.PERMISSION_DENIED,
                                message=f"Approval request '{approval_id}' has expired",
                                correlation_id=invocation.context.correlation_id,
                            ),
                            execution_time_seconds=time.perf_counter() - start_time,
                        )
                except Exception:
                    pass

            if approval_record.status == ApprovalStatus.PENDING:
                return ToolResult(
                    request_id=req_id,
                    tool_name=invocation.tool_name,
                    success=False,
                    approval_id=approval_id,
                    error=ToolError(
                        code=ToolErrorCode.APPROVAL_REQUIRED,
                        message=f"Approval request '{approval_id}' is still pending reviewer resolution",
                        details={"approval_id": approval_id, "status": "pending"},
                        correlation_id=invocation.context.correlation_id,
                    ),
                    execution_time_seconds=time.perf_counter() - start_time,
                )
            elif approval_record.status == ApprovalStatus.REJECTED:
                return ToolResult(
                    request_id=req_id,
                    tool_name=invocation.tool_name,
                    success=False,
                    approval_id=approval_id,
                    error=ToolError(
                        code=ToolErrorCode.PERMISSION_DENIED,
                        message=f"Approval request '{approval_id}' was rejected by reviewer '{approval_record.reviewer}'",
                        details={"approval_id": approval_id, "status": "rejected"},
                        correlation_id=invocation.context.correlation_id,
                    ),
                    execution_time_seconds=time.perf_counter() - start_time,
                )
            elif approval_record.status == ApprovalStatus.REVOKED:
                return ToolResult(
                    request_id=req_id,
                    tool_name=invocation.tool_name,
                    success=False,
                    approval_id=approval_id,
                    error=ToolError(
                        code=ToolErrorCode.PERMISSION_DENIED,
                        message=f"Approval request '{approval_id}' was revoked",
                        details={"approval_id": approval_id, "status": "revoked"},
                        correlation_id=invocation.context.correlation_id,
                    ),
                    execution_time_seconds=time.perf_counter() - start_time,
                )
            elif approval_record.status == ApprovalStatus.EXPIRED:
                return ToolResult(
                    request_id=req_id,
                    tool_name=invocation.tool_name,
                    success=False,
                    approval_id=approval_id,
                    error=ToolError(
                        code=ToolErrorCode.PERMISSION_DENIED,
                        message=f"Approval request '{approval_id}' has expired",
                        details={"approval_id": approval_id, "status": "expired"},
                        correlation_id=invocation.context.correlation_id,
                    ),
                    execution_time_seconds=time.perf_counter() - start_time,
                )
            elif approval_record.status == ApprovalStatus.APPROVED:
                # Expiration check: verify approval has not elapsed TTL
                if approval_record.expires_at:
                    try:
                        exp_dt = datetime.fromisoformat(approval_record.expires_at.replace("Z", "+00:00"))
                        if datetime.now(timezone.utc) > exp_dt:
                            if self.ledger:
                                self.ledger.update_approval_status(
                                    approval_id=approval_id,
                                    status=ApprovalStatus.EXPIRED,
                                    reviewer=approval_record.reviewer,
                                    reason="Approval request has expired",
                                )
                            return ToolResult(
                                request_id=req_id,
                                tool_name=invocation.tool_name,
                                success=False,
                                approval_id=approval_id,
                                error=ToolError(
                                    code=ToolErrorCode.PERMISSION_DENIED,
                                    message=f"Approval request '{approval_id}' has expired",
                                    details={"approval_id": approval_id, "status": "expired"},
                                    correlation_id=invocation.context.correlation_id,
                                ),
                                execution_time_seconds=time.perf_counter() - start_time,
                            )
                    except Exception:
                        pass

                # Anti-tampering check: verify that arguments match approved payload
                expected_hash = compute_arguments_hash(
                    action=invocation.tool_name,
                    actor=approval_record.actor,
                    run_id=approval_record.run_id,
                    arguments=invocation.arguments,
                    policy_version=approval_record.policy_version,
                    max_cost=approval_record.max_cost,
                )
                if expected_hash != approval_record.arguments_hash:
                    return ToolResult(
                        request_id=req_id,
                        tool_name=invocation.tool_name,
                        success=False,
                        approval_id=approval_id,
                        error=ToolError(
                            code=ToolErrorCode.PERMISSION_DENIED,
                            message="Approval validation failed: invocation arguments do not match approved request",
                            details={"approval_id": approval_id},
                            correlation_id=invocation.context.correlation_id,
                        ),
                        execution_time_seconds=time.perf_counter() - start_time,
                    )
                pre_approved = True

        # Resolve cost estimate (dynamic for optimize_experiment based on n_trials)
        cost_estimate = dict(tool_def.cost_estimate) if tool_def.cost_estimate else {}
        if invocation.tool_name == "optimize_experiment" and "n_trials" in invocation.arguments:
            cost_estimate["trials"] = int(invocation.arguments["n_trials"])

        # Compute effective budget considering active reservations
        if self.ledger and invocation.context.run_id:
            active_reserved = self.ledger.get_active_reserved_budget(invocation.context.run_id)
            effective_budget = AgentBudget(
                max_experiments=active_budget.max_experiments,
                max_trials=active_budget.max_trials,
                max_folds=active_budget.max_folds,
                max_duration_seconds=active_budget.max_duration_seconds,
                max_llm_calls=active_budget.max_llm_calls,
                max_tokens=active_budget.max_tokens,
                consumed_experiments=active_budget.consumed_experiments + active_reserved.get("experiments", 0),
                consumed_trials=active_budget.consumed_trials + active_reserved.get("trials", 0),
                consumed_folds=active_budget.consumed_folds + active_reserved.get("folds", 0),
                consumed_duration_seconds=active_budget.consumed_duration_seconds + active_reserved.get("duration_seconds", 0.0),
                consumed_llm_calls=active_budget.consumed_llm_calls + active_reserved.get("llm_calls", 0),
                consumed_tokens=active_budget.consumed_tokens + active_reserved.get("tokens", 0),
            )
        else:
            effective_budget = active_budget

        # 5. Policy evaluation
        decision = self.policy_evaluator.evaluate(
            tool_def,
            invocation.context,
            effective_budget,
            invocation.arguments,
        )

        if decision.decision == PolicyDecisionType.DENY:
            err_code = ToolErrorCode.PERMISSION_DENIED
            if "Budget exceeded" in decision.reason:
                err_code = ToolErrorCode.BUDGET_EXCEEDED
            elif "Scope violation" in decision.reason:
                err_code = ToolErrorCode.SCOPE_VIOLATION

            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                error=ToolError(
                    code=err_code,
                    message=decision.reason,
                    details=decision.to_dict(),
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )

        if decision.decision == PolicyDecisionType.REQUIRE_APPROVAL and not pre_approved:
            arg_hash = compute_arguments_hash(
                action=invocation.tool_name,
                actor=invocation.context.actor,
                run_id=invocation.context.run_id,
                arguments=invocation.arguments,
                policy_version=decision.policy_version,
                max_cost=cost_estimate,
            )
            appr_id = f"appr-{arg_hash[:16]}"
            if self.ledger:
                now_utc = datetime.now(timezone.utc)
                expires_at_iso = (now_utc + timedelta(seconds=self.approval_ttl_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
                approval_req = ApprovalRequest(
                    approval_id=appr_id,
                    action=invocation.tool_name,
                    actor=invocation.context.actor,
                    run_id=invocation.context.run_id,
                    arguments_hash=arg_hash,
                    arguments=invocation.arguments,
                    policy_version=decision.policy_version,
                    max_cost=cost_estimate,
                    expires_at=expires_at_iso,
                    status=ApprovalStatus.PENDING,
                )
                self.ledger.save_approval(approval_req)

            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                approval_id=appr_id,
                error=ToolError(
                    code=ToolErrorCode.APPROVAL_REQUIRED,
                    message=decision.reason,
                    details={**decision.to_dict(), "approval_id": appr_id},
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )

        # 6. Idempotency & Deduplication
        idempotency_key = invocation.idempotency_key
        is_mutating = tool_def.effect in (ToolEffect.PROPOSE, ToolEffect.MUTATE)
        if is_mutating and not idempotency_key:
            idempotency_key = f"auto-{req_id}"

        operation_id: str | None = None
        current_arg_hash = compute_arguments_hash(
            action=invocation.tool_name,
            actor=invocation.context.actor,
            run_id=invocation.context.run_id,
            arguments=invocation.arguments,
            policy_version=decision.policy_version,
            max_cost=cost_estimate,
        )

        if self.ledger and idempotency_key:
            existing_op = self.ledger.get_operation_by_idempotency_key(
                run_id=invocation.context.run_id,
                action=invocation.tool_name,
                idempotency_key=idempotency_key,
            )
            if existing_op is not None:
                # Check for argument mismatch -> CONFLICT
                if existing_op.arguments_hash != current_arg_hash:
                    return ToolResult(
                        request_id=req_id,
                        tool_name=invocation.tool_name,
                        success=False,
                        operation_id=existing_op.operation_id,
                        error=ToolError(
                            code=ToolErrorCode.CONFLICT,
                            message=(
                                f"Idempotency key '{idempotency_key}' was already used with "
                                f"different arguments for action '{invocation.tool_name}'"
                            ),
                            correlation_id=invocation.context.correlation_id,
                        ),
                        execution_time_seconds=time.perf_counter() - start_time,
                    )

                # Matching arguments: deduplication ("Duplicados no repiten efectos")
                if existing_op.status in (OperationStatus.CANCEL_REQUESTED, OperationStatus.CANCELLED):
                    if existing_op.status == OperationStatus.CANCEL_REQUESTED and self.ledger:
                        self.ledger.update_operation_status(
                            operation_id=existing_op.operation_id,
                            status=OperationStatus.CANCELLED,
                            error_code=ToolErrorCode.CONFLICT.value,
                            error_message="Operation cancelled",
                        )
                    return ToolResult(
                        request_id=req_id,
                        tool_name=invocation.tool_name,
                        success=False,
                        operation_id=existing_op.operation_id,
                        error=ToolError(
                            code=ToolErrorCode.CONFLICT,
                            message="Operation cancelled",
                            correlation_id=invocation.context.correlation_id,
                        ),
                        execution_time_seconds=time.perf_counter() - start_time,
                    )
                elif existing_op.status == OperationStatus.TIMED_OUT:
                    return ToolResult(
                        request_id=req_id,
                        tool_name=invocation.tool_name,
                        success=False,
                        operation_id=existing_op.operation_id,
                        error=ToolError(
                            code=ToolErrorCode.DEADLINE_EXCEEDED,
                            message=existing_op.error_message or "Operation timed out",
                            correlation_id=invocation.context.correlation_id,
                        ),
                        execution_time_seconds=time.perf_counter() - start_time,
                    )
                elif existing_op.status == OperationStatus.RECOVERY_REQUIRED:
                    return ToolResult(
                        request_id=req_id,
                        tool_name=invocation.tool_name,
                        success=False,
                        operation_id=existing_op.operation_id,
                        error=ToolError(
                            code=ToolErrorCode.CONFLICT,
                            message="Operation requires recovery before it can be re-executed",
                            correlation_id=invocation.context.correlation_id,
                        ),
                        execution_time_seconds=time.perf_counter() - start_time,
                    )
                elif existing_op.status == OperationStatus.SUCCEEDED:
                    cached_data = None
                    if existing_op.result_ref:
                        try:
                            cached_data = json.loads(existing_op.result_ref)
                        except Exception:
                            cached_data = {"result_ref": existing_op.result_ref}
                    return ToolResult(
                        request_id=req_id,
                        tool_name=invocation.tool_name,
                        success=True,
                        data=cached_data,
                        operation_id=existing_op.operation_id,
                        execution_time_seconds=time.perf_counter() - start_time,
                    )
                elif existing_op.status in (OperationStatus.RUNNING, OperationStatus.PENDING):
                    return ToolResult(
                        request_id=req_id,
                        tool_name=invocation.tool_name,
                        success=False,
                        operation_id=existing_op.operation_id,
                        error=ToolError(
                            code=ToolErrorCode.CONFLICT,
                            message=f"Operation '{existing_op.operation_id}' is already in progress with status '{existing_op.status.value}'",
                            correlation_id=invocation.context.correlation_id,
                        ),
                        execution_time_seconds=time.perf_counter() - start_time,
                    )
                elif existing_op.status == OperationStatus.FAILED:
                    err_code = ToolErrorCode.INTERNAL_ERROR
                    if existing_op.error_code:
                        try:
                            err_code = ToolErrorCode(existing_op.error_code)
                        except Exception:
                            pass
                    return ToolResult(
                        request_id=req_id,
                        tool_name=invocation.tool_name,
                        success=False,
                        operation_id=existing_op.operation_id,
                        error=ToolError(
                            code=err_code,
                            message=existing_op.error_message or "Operation previously failed",
                            correlation_id=invocation.context.correlation_id,
                        ),
                        execution_time_seconds=time.perf_counter() - start_time,
                    )

        # 7. Durable intention before mutating ("Intención duradera antes de mutar")
        if self.ledger and is_mutating:
            operation_id = f"op-{uuid.uuid4().hex[:12]}"
            op_record = OperationRecord(
                operation_id=operation_id,
                run_id=invocation.context.run_id,
                actor=invocation.context.actor,
                idempotency_key=idempotency_key or operation_id,
                action=invocation.tool_name,
                arguments_hash=current_arg_hash,
                arguments=invocation.arguments,
                status=OperationStatus.RUNNING,
                reserved_budget=cost_estimate,
                consumed_budget={},
            )
            try:
                self.ledger.record_operation(op_record)
            except Exception as exc:
                return ToolResult(
                    request_id=req_id,
                    tool_name=invocation.tool_name,
                    success=False,
                    error=ToolError(
                        code=ToolErrorCode.INTERNAL_ERROR,
                        message=f"Failed to record operation intention in ledger: {exc!s}",
                        correlation_id=invocation.context.correlation_id,
                    ),
                    execution_time_seconds=time.perf_counter() - start_time,
                )

        # 8. Check deadline before handler execution ("Timeout no se presenta como cancelación")
        if invocation.context.deadline is not None and time.time() > invocation.context.deadline:
            if self.ledger and operation_id:
                self.ledger.update_operation_status(
                    operation_id=operation_id,
                    status=OperationStatus.TIMED_OUT,
                    error_code=ToolErrorCode.DEADLINE_EXCEEDED.value,
                    error_message="Operation exceeded invocation context deadline before execution",
                )
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                operation_id=operation_id,
                error=ToolError(
                    code=ToolErrorCode.DEADLINE_EXCEEDED,
                    message="Operation deadline exceeded",
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )

        # Check cooperative cancellation before dispatching handler
        if self.ledger and operation_id:
            is_cancelled = False
            if hasattr(self.ledger, "is_cancellation_requested") and self.ledger.is_cancellation_requested(operation_id):
                is_cancelled = True
            else:
                current_op = self.ledger.get_operation(operation_id)
                if current_op and current_op.status in (OperationStatus.CANCEL_REQUESTED, OperationStatus.CANCELLED):
                    is_cancelled = True

            if is_cancelled:
                self.ledger.update_operation_status(
                    operation_id=operation_id,
                    status=OperationStatus.CANCELLED,
                    error_code=ToolErrorCode.CONFLICT.value,
                    error_message="Operation was cancelled prior to execution",
                )
                return ToolResult(
                    request_id=req_id,
                    tool_name=invocation.tool_name,
                    success=False,
                    operation_id=operation_id,
                    error=ToolError(
                        code=ToolErrorCode.CONFLICT,
                        message="Operation cancelled",
                        correlation_id=invocation.context.correlation_id,
                    ),
                    execution_time_seconds=time.perf_counter() - start_time,
                )

        # 9. Execute Handler
        try:
            raw_data = handler(invocation.arguments, invocation.context)
            if cost_estimate:
                active_budget.consume(cost_estimate)

            if self.ledger and operation_id:
                if hasattr(self.ledger, "is_cancellation_requested") and self.ledger.is_cancellation_requested(operation_id):
                    self.ledger.update_operation_status(
                        operation_id=operation_id,
                        status=OperationStatus.CANCELLED,
                        error_code=ToolErrorCode.CANCELLED.value,
                        error_message="Operation was cancelled during execution",
                    )
                    return ToolResult(
                        request_id=req_id,
                        tool_name=invocation.tool_name,
                        success=False,
                        operation_id=operation_id,
                        error=ToolError(
                            code=ToolErrorCode.CANCELLED,
                            message="Operation was cancelled during execution",
                            correlation_id=invocation.context.correlation_id,
                        ),
                        execution_time_seconds=time.perf_counter() - start_time,
                    )

                result_ref_str = (
                    json.dumps(raw_data)
                    if isinstance(raw_data, (dict, list))
                    else str(raw_data)
                )
                self.ledger.update_operation_status(
                    operation_id=operation_id,
                    status=OperationStatus.SUCCEEDED,
                    consumed=cost_estimate,
                    result_ref=result_ref_str,
                )

            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=True,
                data=raw_data,
                operation_id=operation_id,
                execution_time_seconds=time.perf_counter() - start_time,
            )
        except TimeoutError as exc:
            err_msg = str(exc) or "Operation timed out"
            if self.ledger and operation_id:
                self.ledger.update_operation_status(
                    operation_id=operation_id,
                    status=OperationStatus.TIMED_OUT,
                    error_code=ToolErrorCode.DEADLINE_EXCEEDED.value,
                    error_message=err_msg,
                )
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                operation_id=operation_id,
                error=ToolError(
                    code=ToolErrorCode.DEADLINE_EXCEEDED,
                    message=err_msg,
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )
        except (InterruptedError, KeyboardInterrupt):
            err_msg = f"Operation '{invocation.tool_name}' was interrupted or cancelled"
            if self.ledger and operation_id:
                self.ledger.update_operation_status(
                    operation_id=operation_id,
                    status=OperationStatus.CANCELLED,
                    error_code=ToolErrorCode.CANCELLED.value,
                    error_message=err_msg,
                )
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                operation_id=operation_id,
                error=ToolError(
                    code=ToolErrorCode.CANCELLED,
                    message=err_msg,
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )
        except KeyError as exc:
            err_msg = str(exc).strip("'")
            if self.ledger and operation_id:
                self.ledger.update_operation_status(
                    operation_id=operation_id,
                    status=OperationStatus.FAILED,
                    error_code=ToolErrorCode.NOT_FOUND.value,
                    error_message=err_msg,
                )
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                operation_id=operation_id,
                error=ToolError(
                    code=ToolErrorCode.NOT_FOUND,
                    message=err_msg,
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )
        except ValueError as exc:
            err_msg = str(exc)
            if self.ledger and operation_id:
                self.ledger.update_operation_status(
                    operation_id=operation_id,
                    status=OperationStatus.FAILED,
                    error_code=ToolErrorCode.INVALID_ARGUMENT.value,
                    error_message=err_msg,
                )
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                operation_id=operation_id,
                error=ToolError(
                    code=ToolErrorCode.INVALID_ARGUMENT,
                    message=err_msg,
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )
        except PermissionError as exc:
            err_msg = str(exc)
            if self.ledger and operation_id:
                self.ledger.update_operation_status(
                    operation_id=operation_id,
                    status=OperationStatus.FAILED,
                    error_code=ToolErrorCode.PERMISSION_DENIED.value,
                    error_message=err_msg,
                )
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                operation_id=operation_id,
                error=ToolError(
                    code=ToolErrorCode.PERMISSION_DENIED,
                    message=err_msg,
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )
        except Exception as exc:
            if self.ledger and operation_id and self.ledger.is_cancellation_requested(operation_id):
                err_msg = f"Operation '{invocation.tool_name}' was cancelled"
                self.ledger.update_operation_status(
                    operation_id=operation_id,
                    status=OperationStatus.CANCELLED,
                    error_code=ToolErrorCode.CANCELLED.value,
                    error_message=err_msg,
                )
                return ToolResult(
                    request_id=req_id,
                    tool_name=invocation.tool_name,
                    success=False,
                    operation_id=operation_id,
                    error=ToolError(
                        code=ToolErrorCode.CANCELLED,
                        message=err_msg,
                        correlation_id=invocation.context.correlation_id,
                    ),
                    execution_time_seconds=time.perf_counter() - start_time,
                )

            err_msg = f"Internal error executing '{invocation.tool_name}': {exc!s}"
            if self.ledger and operation_id:
                self.ledger.update_operation_status(
                    operation_id=operation_id,
                    status=OperationStatus.FAILED,
                    error_code=ToolErrorCode.INTERNAL_ERROR.value,
                    error_message=err_msg,
                )
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                operation_id=operation_id,
                error=ToolError(
                    code=ToolErrorCode.INTERNAL_ERROR,
                    message=err_msg,
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )


def create_read_only_tool_registry(query_bus: QueryBus) -> ToolRegistry:
    """Build and populate ToolRegistry with all canonical V0.9 read-only query tools."""
    registry = ToolRegistry()

    # 1. get_dataset_profile
    def _handle_get_dataset_profile(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        profile = query_bus.dispatch(GetDatasetProfileQuery(dataset_id=args["dataset_id"]))
        if profile is None:
            raise KeyError(f"Dataset profile not found: {args['dataset_id']}")
        return {
            "dataset_id": profile.dataset_id,
            "n_rows": getattr(profile, "row_count", 0),
            "n_columns": getattr(profile, "column_count", 0),
            "target_column": getattr(profile, "target_column", ""),
            "task_type": getattr(profile, "task_type", ""),
            "columns": [c.name for c in getattr(profile, "columns", [])],
        }

    registry.register(
        ToolDefinition(
            name="get_dataset_profile",
            version="1.0.0",
            description="Retrieve metadata, dimensions, and column profile of a dataset (no raw data leakage)",
            effect=ToolEffect.READ,
            permission_required=AgentPermission.READ_ONLY,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["get_dataset_profile"]["input"],
            output_schema=TOOL_SCHEMAS["get_dataset_profile"]["output"],
        ),
        _handle_get_dataset_profile,
    )

    # 2. list_models
    def _handle_list_models(args: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        task_type = args.get("task_type")
        models = query_bus.dispatch(ListModelsQuery(run_id=ctx.run_id, task_type=task_type))
        return {"models": list(models) if models else []}

    registry.register(
        ToolDefinition(
            name="list_models",
            version="1.0.0",
            description="List compatible ML models available for the active task type and run",
            effect=ToolEffect.READ,
            permission_required=AgentPermission.READ_ONLY,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["list_models"]["input"],
            output_schema=TOOL_SCHEMAS["list_models"]["output"],
        ),
        _handle_list_models,
    )

    # 3. list_plugins
    def _handle_list_plugins(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        plugin_type = args.get("plugin_type")
        plugins = query_bus.dispatch(ListPluginsQuery(plugin_type=plugin_type))
        normalized = []
        for p in plugins or []:
            ptype = p.plugin_type.value if hasattr(p.plugin_type, "value") else str(p.plugin_type)
            normalized.append(
                {
                    "plugin_id": getattr(p, "plugin_id", ""),
                    "name": getattr(p, "name", ""),
                    "version": getattr(p, "version", "1.0.0"),
                    "plugin_type": ptype,
                }
            )
        return {"plugins": normalized}

    registry.register(
        ToolDefinition(
            name="list_plugins",
            version="1.0.0",
            description="List installed AutoML plugins (models, metrics, preprocessors, modalities)",
            effect=ToolEffect.READ,
            permission_required=AgentPermission.READ_ONLY,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["list_plugins"]["input"],
            output_schema=TOOL_SCHEMAS["list_plugins"]["output"],
        ),
        _handle_list_plugins,
    )

    # 4. list_experiments
    def _handle_list_experiments(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        run_id = args["run_id"]
        experiments = query_bus.dispatch(ListExperimentsQuery(run_id=run_id))
        normalized = []
        for exp in experiments or []:
            status_val = exp.status.value if hasattr(exp.status, "value") else str(exp.status)
            normalized.append(
                {
                    "experiment_id": exp.id,
                    "run_id": exp.run_id,
                    "name": exp.name,
                    "model_ids": getattr(exp, "model_ids", []),
                    "metric": getattr(exp, "metric", ""),
                    "status": status_val,
                }
            )
        return {"experiments": normalized}

    registry.register(
        ToolDefinition(
            name="list_experiments",
            version="1.0.0",
            description="List all experiments configured or executed within a specific run",
            effect=ToolEffect.READ,
            permission_required=AgentPermission.READ_ONLY,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["list_experiments"]["input"],
            output_schema=TOOL_SCHEMAS["list_experiments"]["output"],
        ),
        _handle_list_experiments,
    )

    # 5. get_leaderboard
    def _handle_get_leaderboard(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        run_id = args["run_id"]
        top_k = args.get("top_k")
        results = query_bus.dispatch(GetLeaderboardQuery(run_id=run_id)) or []
        if top_k:
            results = results[:top_k]
        normalized = []
        for r in results:
            normalized.append(
                {
                    "trial_id": getattr(r, "trial_id", ""),
                    "experiment_id": getattr(r, "experiment_id", ""),
                    "model_id": getattr(r, "model_id", ""),
                    "metric": getattr(r, "primary_metric", ""),
                    "score": getattr(r, "primary_score", 0.0),
                    "training_time_seconds": getattr(r, "training_time_seconds", 0.0),
                }
            )
        return {"leaderboard": normalized}

    registry.register(
        ToolDefinition(
            name="get_leaderboard",
            version="1.0.0",
            description="Get ranked leaderboard of evaluated trials and metric scores for a run",
            effect=ToolEffect.READ,
            permission_required=AgentPermission.READ_ONLY,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["get_leaderboard"]["input"],
            output_schema=TOOL_SCHEMAS["get_leaderboard"]["output"],
        ),
        _handle_get_leaderboard,
    )

    # 6. get_feature_evidence
    def _handle_get_feature_evidence(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        run_id = args["run_id"]
        feature_id = args["feature_id"]
        evidence = query_bus.dispatch(GetFeatureEvidenceQuery(run_id=run_id, feature_id=feature_id))
        if evidence is None:
            raise KeyError(f"Feature evidence not found for '{feature_id}' in run '{run_id}'")
        if hasattr(evidence, "to_dict"):
            evidence_data = evidence.to_dict()
        else:
            evidence_data = {"feature_id": feature_id}
        return {"evidence": evidence_data}

    registry.register(
        ToolDefinition(
            name="get_feature_evidence",
            version="1.0.0",
            description="Get statistical and mathematical evidence for a specific feature in a run",
            effect=ToolEffect.READ,
            permission_required=AgentPermission.READ_ONLY,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["get_feature_evidence"]["input"],
            output_schema=TOOL_SCHEMAS["get_feature_evidence"]["output"],
        ),
        _handle_get_feature_evidence,
    )

    # 7. get_feature_ranking
    def _handle_get_feature_ranking(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        run_id = args["run_id"]
        ranks = query_bus.dispatch(GetFeatureRankingQuery(run_id=run_id)) or []
        normalized = []
        for r in ranks:
            if hasattr(r, "to_dict"):
                normalized.append(r.to_dict())
            else:
                normalized.append(
                    {
                        "feature_name": getattr(r, "feature_name", ""),
                        "score": getattr(r, "score", 0.0),
                        "rank": getattr(r, "rank", 0),
                        "method": getattr(r, "method", "ensemble"),
                    }
                )
        return {"ranks": normalized}

    registry.register(
        ToolDefinition(
            name="get_feature_ranking",
            version="1.0.0",
            description="Get ranked list of features ordered by importance scores across methods",
            effect=ToolEffect.READ,
            permission_required=AgentPermission.READ_ONLY,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["get_feature_ranking"]["input"],
            output_schema=TOOL_SCHEMAS["get_feature_ranking"]["output"],
        ),
        _handle_get_feature_ranking,
    )

    return registry


def create_full_tool_registry(
    query_bus: QueryBus,
    command_bus: CommandBus,
    workspace: Any = None,
    run_dataset_resolver: Callable[[str], str] | None = None,
    ledger: AgentLedgerPort | None = None,
) -> ToolRegistry:
    """Build ToolRegistry with safe query tools, authorized mutating tools, and operations tools."""
    registry = create_read_only_tool_registry(query_bus)

    # 8. create_experiment
    def _handle_create_experiment(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        run_id = args["run_id"]
        model_name = args["model_name"]
        feature_names = args.get("feature_names")

        cmd = CreateExperimentCommand(
            run_id=run_id,
            name=f"exp_{model_name}_{uuid.uuid4().hex[:6]}",
            model_ids=[model_name],
            feature_names=feature_names,
            hypothesis=f"Agent candidate experiment with model {model_name}",
        )
        res = command_bus.dispatch(cmd)
        exp_id = getattr(res, "id", None)
        if not exp_id:
            if isinstance(res, dict):
                exp_id = res.get("experiment_id") or res.get("id")
            elif isinstance(res, str):
                exp_id = res
            else:
                exp_id = str(res)
        return {"experiment_id": str(exp_id)}

    registry.register(
        ToolDefinition(
            name="create_experiment",
            version="1.0.0",
            description="Propose and create an experiment candidate with specified model and features",
            effect=ToolEffect.MUTATE,
            permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
            cost_estimate={"experiments": 1},
            input_schema=TOOL_SCHEMAS["create_experiment"]["input"],
            output_schema=TOOL_SCHEMAS["create_experiment"]["output"],
        ),
        _handle_create_experiment,
    )

    # 9. prioritize_feature
    def _handle_prioritize_feature(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        run_id = args["run_id"]
        feature_name = args["feature_name"]
        priority = args["priority"]
        score_map = {"high": 2.0, "medium": 1.0, "low": 0.5}
        score = score_map.get(priority, 1.0)

        dataset_id = ""
        if run_dataset_resolver:
            dataset_id = run_dataset_resolver(run_id)
        elif workspace and hasattr(workspace, "_get_run"):
            run = workspace._get_run(run_id)
            dataset_id = getattr(run, "dataset_id", run_id)
        else:
            dataset_id = args.get("dataset_id") or run_id

        cmd = PrioritizeFeatureCommand(
            dataset_id=dataset_id,
            feature_name=feature_name,
            score=score,
            run_id=run_id,
        )
        command_bus.dispatch(cmd)
        return {"feature_name": feature_name, "priority": priority}

    registry.register(
        ToolDefinition(
            name="prioritize_feature",
            version="1.0.0",
            description="Prioritize a specific feature to adjust its weight in candidate selection",
            effect=ToolEffect.MUTATE,
            permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["prioritize_feature"]["input"],
            output_schema=TOOL_SCHEMAS["prioritize_feature"]["output"],
        ),
        _handle_prioritize_feature,
    )

    # 10. run_experiment
    def _handle_run_experiment(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        run_id = args["run_id"]
        experiment_id = args["experiment_id"]

        cmd = RunExperimentCommand(
            run_id=run_id,
            experiment_id=experiment_id,
        )
        trials = command_bus.dispatch(cmd)
        if not trials:
            raise ValueError(f"Experiment '{experiment_id}' did not produce any trial results")

        best_trial = trials[-1] if isinstance(trials, list) else trials
        trial_id = getattr(best_trial, "trial_id", getattr(best_trial, "id", str(best_trial)))
        score = getattr(best_trial, "primary_score", getattr(best_trial, "score", 0.0))

        return {
            "trial_id": str(trial_id),
            "metric_value": float(score),
        }

    registry.register(
        ToolDefinition(
            name="run_experiment",
            version="1.0.0",
            description="Execute training and validation for a configured experiment candidate",
            effect=ToolEffect.MUTATE,
            permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
            cost_estimate={"trials": 1, "fits": 1},
            input_schema=TOOL_SCHEMAS["run_experiment"]["input"],
            output_schema=TOOL_SCHEMAS["run_experiment"]["output"],
        ),
        _handle_run_experiment,
    )

    if ledger:
        # 11. get_operation_status
        def _handle_get_operation_status(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
            op = ledger.get_operation(args["operation_id"])
            if not op:
                raise KeyError(f"Operation not found: {args['operation_id']}")
            return op.to_dict()

        registry.register(
            ToolDefinition(
                name="get_operation_status",
                version="1.0.0",
                description="Inspect execution status, budget consumption, and result or error of a recorded agent operation",
                effect=ToolEffect.READ,
                permission_required=AgentPermission.READ_ONLY,
                cost_estimate={},
                input_schema=TOOL_SCHEMAS["get_operation_status"]["input"],
                output_schema=TOOL_SCHEMAS["get_operation_status"]["output"],
            ),
            _handle_get_operation_status,
        )

        # 12. list_operations
        def _handle_list_operations(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
            run_id = args["run_id"]
            status_arg = args.get("status")
            status_filter = OperationStatus(status_arg.lower()) if status_arg else None
            ops = ledger.list_operations(run_id=run_id, status=status_filter)
            return {"operations": [o.to_dict() for o in ops]}

        registry.register(
            ToolDefinition(
                name="list_operations",
                version="1.0.0",
                description="List recorded agent operations for a run, optionally filtered by status",
                effect=ToolEffect.READ,
                permission_required=AgentPermission.READ_ONLY,
                cost_estimate={},
                input_schema=TOOL_SCHEMAS["list_operations"]["input"],
                output_schema=TOOL_SCHEMAS["list_operations"]["output"],
            ),
            _handle_list_operations,
        )

        # 13. cancel_operation
        def _handle_cancel_operation(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
            operation_id = args["operation_id"]
            reason = args.get("reason")
            force = args.get("force", False)
            op = ledger.get_operation(operation_id)
            if not op:
                raise KeyError(f"Operation not found: {operation_id}")
            if op.status in (
                OperationStatus.SUCCEEDED,
                OperationStatus.FAILED,
                OperationStatus.CANCELLED,
                OperationStatus.TIMED_OUT,
            ):
                raise ValueError(
                    f"Cannot cancel operation '{operation_id}' with terminal status '{op.status.value}'"
                )
            target_status = OperationStatus.CANCELLED if force else OperationStatus.CANCEL_REQUESTED
            error_msg = f"Cancellation requested: {reason}" if reason else "Cancellation requested"
            resolved = ledger.update_operation_status(operation_id, target_status, error_message=error_msg)

            # Real process & run cancellation when workspace is available
            if workspace and op.run_id:
                try:
                    workspace.cancel_run(op.run_id)
                except Exception:
                    pass
                if hasattr(workspace, "jobs") and workspace.jobs:
                    try:
                        from automl.application.commands.job_commands import ControlJobCommand
                        active_jobs = []
                        if hasattr(workspace.jobs, "list"):
                            active_jobs = workspace.jobs.list(op.run_id)
                        for j in active_jobs:
                            jid = j.get("id") if isinstance(j, dict) else getattr(j, "id", None)
                            if jid:
                                workspace.jobs.control(ControlJobCommand(job_id=jid, action="cancel"))
                    except Exception:
                        pass

            return {"operation_id": resolved.operation_id, "status": resolved.status.value}

        registry.register(
            ToolDefinition(
                name="cancel_operation",
                version="1.0.0",
                description="Request cooperative cancellation or force cancellation of an active agent operation",
                effect=ToolEffect.MUTATE,
                permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
                cost_estimate={},
                input_schema=TOOL_SCHEMAS["cancel_operation"]["input"],
                output_schema=TOOL_SCHEMAS["cancel_operation"]["output"],
            ),
            _handle_cancel_operation,
        )
    # 11. optimize_experiment
    def _handle_optimize_experiment(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        run_id = args["run_id"]
        experiment_id = args["experiment_id"]
        n_trials = args.get("n_trials", 10)

        cmd = OptimizeExperimentCommand(
            run_id=run_id,
            experiment_id=experiment_id,
            n_trials=n_trials,
        )
        res = command_bus.dispatch(cmd)
        best_trial_id = ""
        best_score = 0.0
        if isinstance(res, dict):
            best_trial_id = res.get("best_trial_id") or ""
            if not best_trial_id and res.get("trials"):
                best_trial_id = res["trials"][0].get("trial_id", "")
            best_score = float(res.get("best_score", 0.0))

        return {
            "best_trial_id": str(best_trial_id),
            "best_score": float(best_score),
        }

    registry.register(
        ToolDefinition(
            name="optimize_experiment",
            version="1.0.0",
            description="Run hyperparameter optimization trials for a candidate experiment",
            effect=ToolEffect.MUTATE,
            permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
            cost_estimate={"trials": 10},
            input_schema=TOOL_SCHEMAS["optimize_experiment"]["input"],
            output_schema=TOOL_SCHEMAS["optimize_experiment"]["output"],
        ),
        _handle_optimize_experiment,
    )

    # 15. analysis_create_study
    def _handle_analysis_create_study(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        from automl.application.analysis.commands import CreateStudyCommand
        from automl.application.analysis.queries import GetStudyQuery

        cmd = CreateStudyCommand(
            dataset_id=args["dataset_id"],
            name=args.get("name", ""),
            target_column=args.get("target_column"),
        )
        study_res = command_bus.dispatch(cmd)
        study_id = study_res if isinstance(study_res, str) else getattr(study_res, "id", str(study_res))
        study = query_bus.dispatch(GetStudyQuery(study_id=study_id))
        return study if isinstance(study, dict) else (study.to_dict() if hasattr(study, "to_dict") else dict(study))

    registry.register(
        ToolDefinition(
            name="analysis_create_study",
            version="1.0.0",
            description="Create an exploratory study specification for a dataset",
            effect=ToolEffect.PROPOSE,
            permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["analysis_create_study"]["input"],
            output_schema=TOOL_SCHEMAS["analysis_create_study"]["output"],
        ),
        _handle_analysis_create_study,
    )

    # 16. analysis_run_study
    def _handle_analysis_run_study(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        from automl.application.analysis.commands import RunAnalysisCommand
        from automl.application.analysis.queries import GetAnalysisRunQuery, ListFindingsQuery

        study_id = args["study_id"]
        cmd = RunAnalysisCommand(study_id=study_id)
        run_res = command_bus.dispatch(cmd)
        run_id = run_res if isinstance(run_res, str) else getattr(run_res, "id", str(run_res))
        run_data = query_bus.dispatch(GetAnalysisRunQuery(run_id=run_id))
        findings = query_bus.dispatch(ListFindingsQuery(study_id=study_id, run_id=run_id))
        return {
            "study_id": study_id,
            "run_id": run_id,
            "run": run_data,
            "findings_count": len(findings) if findings else 0,
        }

    registry.register(
        ToolDefinition(
            name="analysis_run_study",
            version="1.0.0",
            description="Execute deterministic statistical and exploratory analysis on an existing study",
            effect=ToolEffect.PROPOSE,
            permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
            cost_estimate={"duration_seconds": 15.0},
            input_schema=TOOL_SCHEMAS["analysis_run_study"]["input"],
            output_schema=TOOL_SCHEMAS["analysis_run_study"]["output"],
        ),
        _handle_analysis_run_study,
    )

    # 17. analysis_propose_experiment
    def _handle_analysis_propose_experiment(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        from automl.application.analysis.commands import CreateHypothesisCommand
        from automl.application.analysis.queries import ListFindingsQuery

        study_id = args["study_id"]
        finding_id = args.get("finding_id", "")
        desc = args.get("hypothesis_description") or args.get("description") or args.get("hypothesis", "")
        action = args.get("proposed_action", "feature_engineering")
        delta = args.get("transformation_spec") or args.get("experiment_delta") or {}

        if finding_id:
            findings = query_bus.dispatch(ListFindingsQuery(study_id=study_id))
            target_finding = next((f for f in (findings or []) if getattr(f, "id", None) == finding_id or (isinstance(f, dict) and f.get("id") == finding_id)), None)
            if not target_finding:
                raise KeyError(f"Finding '{finding_id}' not found in study '{study_id}'.")

        cmd = CreateHypothesisCommand(
            study_id=study_id,
            finding_id=finding_id,
            description=desc,
            proposed_action=action,
            experiment_delta=delta,
        )
        hyp_id = command_bus.dispatch(cmd)
        return {
            "hypothesis_id": str(hyp_id),
            "study_id": study_id,
            "finding_id": finding_id,
            "description": desc,
            "proposed_action": action,
            "status": "proposed",
            "experiment_delta": delta,
            "note": "Hypothesis recorded under 'Propose ≠ Accept' principle. Requires empirical validation trial before promotion.",
        }

    registry.register(
        ToolDefinition(
            name="analysis_propose_experiment",
            version="1.0.0",
            description="Propose an actionable machine learning experiment hypothesis derived from a statistical finding",
            effect=ToolEffect.PROPOSE,
            permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["analysis_propose_experiment"]["input"],
            output_schema=TOOL_SCHEMAS["analysis_propose_experiment"]["output"],
        ),
        _handle_analysis_propose_experiment,
    )

    # 18. analysis_verify_hypothesis
    def _handle_analysis_verify_hypothesis(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        from automl.application.analysis.commands import VerifyHypothesisCommand

        cmd = VerifyHypothesisCommand(
            hypothesis_id=args["hypothesis_id"],
            run_id=args.get("run_id"),
            min_improvement=float(args.get("min_improvement", 0.0)),
        )
        link_data = command_bus.dispatch(cmd)
        return link_data if isinstance(link_data, dict) else (link_data.to_dict() if hasattr(link_data, "to_dict") else dict(link_data))

    registry.register(
        ToolDefinition(
            name="analysis_verify_hypothesis",
            version="1.0.0",
            description="Empirically verify a proposed hypothesis against AutoML baseline on identical validation split",
            effect=ToolEffect.PROPOSE,
            permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
            cost_estimate={"trials": 1, "duration_seconds": 30.0},
            input_schema=TOOL_SCHEMAS["analysis_verify_hypothesis"]["input"],
            output_schema=TOOL_SCHEMAS["analysis_verify_hypothesis"]["output"],
        ),
        _handle_analysis_verify_hypothesis,
    )

    # 19. recipe_create_from_evidence
    def _handle_recipe_create_from_evidence(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        from dataclasses import asdict, is_dataclass
        from automl.application.recipes.commands import CreateExperimentRecipeFromEvidenceCommand
        from automl.application.recipes.queries import GetExperimentRecipeQuery

        cmd = CreateExperimentRecipeFromEvidenceCommand(
            evidence_link_id=args["evidence_link_id"],
            recipe_name=args.get("name"),
            notes=args.get("description"),
            model_ids=args.get("model_ids"),
        )
        recipe_id = command_bus.dispatch(cmd)
        recipe = query_bus.dispatch(GetExperimentRecipeQuery(recipe_id=str(recipe_id)))
        if recipe is None:
            return {"recipe_id": str(recipe_id), "status": "draft"}
        if isinstance(recipe, dict):
            return recipe
        if hasattr(recipe, "to_dict"):
            return recipe.to_dict()
        if is_dataclass(recipe):
            return asdict(recipe)
        return dict(recipe)

    registry.register(
        ToolDefinition(
            name="recipe_create_from_evidence",
            version="1.0.0",
            description="Create a draft reproducible ExperimentRecipe from verified hypothesis evidence",
            effect=ToolEffect.PROPOSE,
            permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["recipe_create_from_evidence"]["input"],
            output_schema=TOOL_SCHEMAS["recipe_create_from_evidence"]["output"],
        ),
        _handle_recipe_create_from_evidence,
    )

    # 20. recipe_select
    def _handle_recipe_select(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        from automl.application.recipes.commands import SelectExperimentRecipeCommand

        recipe_id = args["recipe_id"]
        cmd = SelectExperimentRecipeCommand(recipe_id=recipe_id, notes=args.get("notes"))
        command_bus.dispatch(cmd)
        return {"recipe_id": recipe_id, "status": "selected"}

    registry.register(
        ToolDefinition(
            name="recipe_select",
            version="1.0.0",
            description="Explicitly select a recipe for trial/evaluation",
            effect=ToolEffect.PROPOSE,
            permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["recipe_select"]["input"],
            output_schema=TOOL_SCHEMAS["recipe_select"]["output"],
        ),
        _handle_recipe_select,
    )

    # 21. recipe_submit
    def _handle_recipe_submit(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        from dataclasses import asdict, is_dataclass
        from automl.application.recipes.commands import SubmitExperimentFromRecipeCommand
        from automl.application.recipes.queries import GetExperimentRecipeQuery

        recipe_id = args["recipe_id"]
        cmd = SubmitExperimentFromRecipeCommand(
            recipe_id=recipe_id,
            run_id=args.get("run_id"),
            model_ids=args.get("model_ids"),
            notes=args.get("notes"),
        )
        exp_ids = command_bus.dispatch(cmd)
        recipe = query_bus.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
        rec_data = None
        if recipe is not None:
            if is_dataclass(recipe):
                rec_data = asdict(recipe)
            elif hasattr(recipe, "to_dict"):
                rec_data = recipe.to_dict()
            elif isinstance(recipe, dict):
                rec_data = recipe
        executed = [str(e) for e in (exp_ids if isinstance(exp_ids, list) else [exp_ids])]
        return {
            "status": "submitted",
            "recipe_id": recipe_id,
            "executed_experiments": executed,
            "experiment_ids": executed,
            "recipe": rec_data,
        }

    registry.register(
        ToolDefinition(
            name="recipe_submit",
            version="1.0.0",
            description="Submit and execute training experiments reproducing an experiment recipe configuration",
            effect=ToolEffect.MUTATE,
            permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
            cost_estimate={"experiments": 1, "trials": 2, "duration_seconds": 60.0},
            input_schema=TOOL_SCHEMAS["recipe_submit"]["input"],
            output_schema=TOOL_SCHEMAS["recipe_submit"]["output"],
        ),
        _handle_recipe_submit,
    )

    # 22. recipe_promote
    def _handle_recipe_promote(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        from dataclasses import asdict, is_dataclass
        from automl.application.recipes.commands import PromoteExperimentRecipeCommand
        from automl.application.recipes.queries import GetExperimentRecipeQuery

        recipe_id = args["recipe_id"]
        cmd = PromoteExperimentRecipeCommand(
            recipe_id=recipe_id,
            reason=args.get("reason"),
            notes=args.get("notes"),
        )
        command_bus.dispatch(cmd)
        recipe = query_bus.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
        if recipe is not None:
            if is_dataclass(recipe):
                return asdict(recipe)
            elif hasattr(recipe, "to_dict"):
                return recipe.to_dict()
            elif isinstance(recipe, dict):
                return recipe
        return {
            "recipe_id": recipe_id,
            "status": "promoted",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }

    registry.register(
        ToolDefinition(
            name="recipe_promote",
            version="1.0.0",
            description="Promote an experiment recipe as a reusable production candidate with audit reasoning",
            effect=ToolEffect.MUTATE,
            permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["recipe_promote"]["input"],
            output_schema=TOOL_SCHEMAS["recipe_promote"]["output"],
        ),
        _handle_recipe_promote,
    )

    # 23. recipe_archive
    def _handle_recipe_archive(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        from automl.application.recipes.commands import ArchiveExperimentRecipeCommand

        recipe_id = args["recipe_id"]
        cmd = ArchiveExperimentRecipeCommand(recipe_id=recipe_id)
        command_bus.dispatch(cmd)
        return {"recipe_id": recipe_id, "status": "archived"}

    registry.register(
        ToolDefinition(
            name="recipe_archive",
            version="1.0.0",
            description="Archive an experiment recipe",
            effect=ToolEffect.MUTATE,
            permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["recipe_archive"]["input"],
            output_schema=TOOL_SCHEMAS["recipe_archive"]["output"],
        ),
        _handle_recipe_archive,
    )

    return registry
