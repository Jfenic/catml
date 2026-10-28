"""Unit and integration tests for P0 Agent Governance Audit Fixes.

Validates:
1. AgentBudget fold accounting: max_folds, consumed_folds, remaining_folds(), can_afford().
2. Approval TTL expiration: dynamic TTL calculation, ledger update to EXPIRED, PERMISSION_DENIED.
3. Unified MCP governance: recipe_submit & analysis_verify_hypothesis require approval under ToolExecutor.
4. Real run cancellation in cancel_operation calling workspace.cancel_run and JobExecutor.
5. MCP optimize_experiment exposed and executed via ToolExecutor.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from automl.application.agents.contracts import (
    ApprovalRequest,
    ToolCallContext,
    ToolInvocation,
)
from automl.application.agents.executor import ToolExecutor, create_full_tool_registry
from automl.application.agents.policy import AgentPolicyConfig, PolicyEvaluator
from automl.application.bootstrap import build_application
from automl.domain.agents.entities import (
    AgentBudget,
    AgentPermission,
    ApprovalStatus,
    OperationStatus,
    ToolErrorCode,
)
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger
from automl.interfaces.mcp.server import HAS_MCP, create_mcp_server


# =========================================================================
# 1. Fold Budget Accounting
# =========================================================================

def test_agent_budget_fold_limit_enforcement():
    """Verify max_folds restricts executions when fold limit is exceeded."""
    budget = AgentBudget(
        max_experiments=10,
        max_trials=50,
        max_duration_seconds=3600.0,
        max_folds=5,
    )
    assert budget.consumed_folds == 0
    assert budget.remaining_folds() == 5

    # Afford within limit
    can, err = budget.can_afford({"folds": 3})
    assert can is True
    assert err is None

    # Consume 3 folds
    budget.consume({"folds": 3})
    assert budget.consumed_folds == 3
    assert budget.remaining_folds() == 2

    # Cannot afford 3 more folds (3 + 3 = 6 > 5)
    can, err = budget.can_afford({"folds": 3})
    assert can is False
    assert err is not None
    assert "fold" in err.lower()

    # Can afford 2 folds
    can, err = budget.can_afford({"folds": 2})
    assert can is True

    # Budget dictionary includes folds
    b_dict = budget.to_dict()
    assert b_dict["max_folds"] == 5
    assert b_dict["consumed_folds"] == 3
    assert budget.remaining_folds() == 2


def test_sqlite_agent_ledger_budget_includes_folds(tmp_path: Path):
    """Verify SqliteAgentLedger active budget calculation handles folds."""
    db_path = tmp_path / "ledger_folds.db"
    ledger = SqliteAgentLedger(db_path)
    totals = ledger.get_active_reserved_budget(run_id="run_test")
    assert "folds" in totals
    assert totals["folds"] == 0


# =========================================================================
# 2. Approval Dynamic TTL & Expiration
# =========================================================================

def test_approval_ttl_generation_and_expiration_rejection(tmp_path: Path):
    """Verify approvals have dynamic TTL and are rejected with PERMISSION_DENIED once expired."""
    db_path = tmp_path / "ledger_ttl.db"
    ledger = SqliteAgentLedger(db_path)
    policy_eval = PolicyEvaluator(
        AgentPolicyConfig(
            require_approval_for_mutations=True,
            default_mode=AgentPermission.EXECUTE_WITHIN_BUDGET,
        )
    )
    ws, cb, qb = build_application(root_dir=str(tmp_path / "ws_ttl"))
    registry = create_full_tool_registry(query_bus=qb, command_bus=cb)

    # Create executor with 2-second TTL
    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=policy_eval,
        ledger=ledger,
        approval_ttl_seconds=2.0,
    )

    ctx = ToolCallContext(
        actor="test_agent",
        workspace_path=str(tmp_path),
        run_id="run_ttl",
        correlation_id="corr_ttl",
        permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )
    inv = ToolInvocation(
        tool_name="create_experiment",
        arguments={"run_id": "run_ttl", "model_name": "rf"},
        context=ctx,
    )

    # 1. Propose mutation -> generates approval request with dynamic TTL
    res = executor.execute(inv)
    assert res.success is False
    assert res.approval_id is not None
    appr_id = res.approval_id

    req = ledger.get_approval(appr_id)
    assert req is not None
    assert req.expires_at is not None
    assert not req.expires_at.startswith("2099")  # Verified: Not hardcoded 2099!

    # 2. Approve via reviewer
    ledger.update_approval_status(appr_id, ApprovalStatus.APPROVED, reviewer="lead_ds")

    # 3. Simulate passage of time past TTL
    past_time = (datetime.now(timezone.utc) - timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with ledger._connect() as conn:
        conn.execute("UPDATE agent_approvals SET expires_at = ? WHERE approval_id = ?", (past_time, appr_id))

    # 4. Attempt to execute using expired approval -> fails closed
    inv_with_appr = ToolInvocation(
        tool_name="create_experiment",
        arguments={"run_id": "run_ttl", "model_name": "rf"},
        context=ctx,
        approval_id=appr_id,
    )
    res_expired = executor.execute(inv_with_appr)
    assert res_expired.success is False
    assert res_expired.error is not None
    assert res_expired.error.code == ToolErrorCode.PERMISSION_DENIED
    assert "expired" in res_expired.error.message.lower()

    # Ledger record was updated to EXPIRED
    updated_req = ledger.get_approval(appr_id)
    assert updated_req.status == ApprovalStatus.EXPIRED


# =========================================================================
# 3. MCP Unified Governance on Analysis & Recipe Mutations
# =========================================================================

@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_governance_unification_recipe_submit(tmp_path: Path):
    """Verify recipe_submit passes through ToolExecutor approval governance."""
    ws, cb, qb = build_application(root_dir=str(tmp_path / "ws"))
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb, require_approval_for_mutations=True)

    # Calling recipe_submit without approval returns PENDING_APPROVAL
    res = asyncio.run(server.call_tool("recipe_submit", {"recipe_id": "rcp_dummy"}))
    assert res.is_error is False
    payload = json.loads(res.content[0].text)
    assert payload["status"] == "PENDING_APPROVAL"
    assert "approval_id" in payload
    assert payload["approval_id"].startswith("appr-")


@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_governance_unification_analysis_verify(tmp_path: Path):
    """Verify analysis_verify_hypothesis passes through ToolExecutor governance and ledger."""
    ws, cb, qb = build_application(root_dir=str(tmp_path / "ws"))
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb)

    # Calling analysis_verify_hypothesis routes through ToolExecutor
    res = asyncio.run(
        server.call_tool(
            "analysis_verify_hypothesis",
            {"hypothesis_id": "hyp_nonexistent", "idempotency_key": "idemp_verify_1"},
        )
    )
    # Target hypothesis does not exist, so it fails cleanly with ToolError (NOT_FOUND)
    assert res.is_error is True
    assert "NOT_FOUND" in res.content[0].text or "not found" in res.content[0].text.lower()

    # And ledger recorded the operation attempt
    op = server.ledger.get_operation_by_idempotency_key(
        run_id="hyp_nonexistent",
        action="analysis_verify_hypothesis",
        idempotency_key="idemp_verify_1",
    )
    assert op is not None
    assert op.action == "analysis_verify_hypothesis"


# =========================================================================
# 4. MCP optimize_experiment Exposure
# =========================================================================

@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_optimize_experiment_exposed_and_governed(tmp_path: Path):
    """Verify optimize_experiment is exposed on MCP server and routes through ToolExecutor."""
    ws, cb, qb = build_application(root_dir=str(tmp_path / "ws"))
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb, require_approval_for_mutations=True)

    tools = asyncio.run(server.list_tools())
    tool_names = {t.name for t in tools}
    assert "optimize_experiment" in tool_names

    # Calling without approval returns PENDING_APPROVAL
    res = asyncio.run(
        server.call_tool(
            "optimize_experiment",
            {"run_id": "run_opt", "experiment_id": "exp_opt", "n_trials": 5},
        )
    )
    assert res.is_error is False
    payload = json.loads(res.content[0].text)
    assert payload["status"] == "PENDING_APPROVAL"
    assert "approval_id" in payload


# =========================================================================
# 5. Real Run & Process Cancellation in cancel_operation
# =========================================================================

def test_cancel_operation_triggers_real_cancellation(tmp_path: Path):
    """Verify cancel_operation triggers workspace.cancel_run and cancels active jobs."""
    ws, cb, qb = build_application(root_dir=str(tmp_path / "ws"))
    ws.cancel_run = MagicMock()
    ws.jobs = MagicMock()
    active_job = MagicMock()
    active_job.id = "job_123"
    active_job.status = "running"
    active_job.metadata = {"run_id": "run_target"}
    ws.jobs.list.return_value = [active_job]

    ledger = SqliteAgentLedger(tmp_path / "ledger_cancel.db")
    policy_eval = PolicyEvaluator(AgentPolicyConfig(require_approval_for_mutations=False))
    registry = create_full_tool_registry(
        query_bus=qb,
        command_bus=cb,
        workspace=ws,
        ledger=ledger,
    )
    executor = ToolExecutor(registry=registry, policy_evaluator=policy_eval, ledger=ledger)

    # Record a running operation
    from automl.application.agents.contracts import OperationRecord
    op_rec = OperationRecord(
        operation_id="op_cancel_1",
        run_id="run_target",
        actor="test_agent",
        idempotency_key="key_cancel_1",
        action="run_experiment",
        arguments_hash="hash_123",
        arguments={"run_id": "run_target"},
        status=OperationStatus.RUNNING,
    )
    ledger.record_operation(op_rec)

    # Cancel the operation
    ctx = ToolCallContext(
        actor="test_agent",
        workspace_path=str(tmp_path),
        run_id="run_target",
        correlation_id="corr_cancel",
        permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )
    inv = ToolInvocation(
        tool_name="cancel_operation",
        arguments={"operation_id": "op_cancel_1", "reason": "User abort", "force": True},
        context=ctx,
    )
    res = executor.execute(inv)
    assert res.success is True
    assert res.data["status"] == "cancelled"

    # Verify workspace.cancel_run was called
    ws.cancel_run.assert_called_once_with("run_target")

    # Verify job cancellation command was dispatched to JobExecutor
    ws.jobs.control.assert_called_once()
    called_cmd = ws.jobs.control.call_args[0][0]
    assert called_cmd.job_id == "job_123"
    assert called_cmd.action == "cancel"
