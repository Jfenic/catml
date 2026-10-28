"""Unit and integration tests for Package A2 (mutating tools, policy approval, ledger idempotency)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import uuid

import pytest

from automl.domain.agents.entities import (
    AgentBudget,
    AgentPermission,
    ApprovalStatus,
    OperationStatus,
    ToolEffect,
    ToolErrorCode,
)
from automl.application.agents.contracts import (
    ApprovalRequest,
    OperationRecord,
    ToolCallContext,
    ToolDefinition,
    ToolInvocation,
)
from automl.application.agents.policy import (
    AgentPolicyConfig,
    PolicyEvaluator,
    compute_arguments_hash,
)
from automl.application.agents.executor import (
    ToolExecutor,
    create_full_tool_registry,
    create_read_only_tool_registry,
)
from automl.application.agents.registry import ToolRegistry
from automl.application.bus.command_bus import CommandBus
from automl.application.bus.query_bus import QueryBus
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    PrioritizeFeatureCommand,
    RunExperimentCommand,
)
from automl.domain.experiments.trial import TrialResult
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger


@pytest.fixture
def temp_ledger(tmp_path: Path) -> SqliteAgentLedger:
    db_file = tmp_path / "test_agent_ledger.db"
    return SqliteAgentLedger(db_file)


@pytest.fixture
def mock_buses():
    q_bus = QueryBus()
    c_bus = CommandBus()
    return q_bus, c_bus


def test_create_full_tool_registry_catalog(mock_buses):
    """Verify that create_full_tool_registry registers both queries and mutating tools."""
    q_bus, c_bus = mock_buses
    registry = create_full_tool_registry(q_bus, c_bus)

    tools = registry.list_tools()
    assert len(tools) >= 11

    tool_names = {t.name for t in tools}
    assert "get_dataset_profile" in tool_names
    assert "list_models" in tool_names
    assert "create_experiment" in tool_names
    assert "prioritize_feature" in tool_names
    assert "run_experiment" in tool_names
    assert "optimize_experiment" in tool_names

    create_exp_def = registry.get_definition("create_experiment")
    assert create_exp_def is not None
    assert create_exp_def.effect == ToolEffect.MUTATE
    assert create_exp_def.permission_required == AgentPermission.EXECUTE_WITHIN_BUDGET
    assert create_exp_def.cost_estimate == {"experiments": 1}

    opt_exp_def = registry.get_definition("optimize_experiment")
    assert opt_exp_def is not None
    assert opt_exp_def.effect == ToolEffect.MUTATE
    assert opt_exp_def.permission_required == AgentPermission.EXECUTE_WITHIN_BUDGET


def test_authorized_mutating_execution_and_ledger_audit(mock_buses, temp_ledger):
    """Verify executing mutating tools dispatches commands and records in SQLite ledger."""
    q_bus, c_bus = mock_buses

    # Register mock command handlers
    dispatched_commands = []

    def _handle_create_exp(cmd: CreateExperimentCommand):
        dispatched_commands.append(cmd)
        return {"id": "exp_abc123"}

    c_bus.register(CreateExperimentCommand, _handle_create_exp)

    registry = create_full_tool_registry(q_bus, c_bus)
    policy_config = AgentPolicyConfig(
        require_approval_for_mutations=False,
        default_mode=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )
    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(policy_config),
        ledger=temp_ledger,
    )

    ctx = ToolCallContext(
        actor="agent_planner",
        workspace_path="/tmp/test",
        run_id="run_test_1",
        correlation_id="corr-1",
        permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )

    invocation = ToolInvocation(
        tool_name="create_experiment",
        arguments={
            "run_id": "run_test_1",
            "model_name": "ridge",
            "feature_names": ["feat_1", "feat_2"],
        },
        context=ctx,
        idempotency_key="idem-key-1",
    )

    result = executor.execute(invocation)
    assert result.success is True
    assert result.data == {"experiment_id": "exp_abc123"}
    assert result.operation_id is not None
    assert len(dispatched_commands) == 1

    # Verify ledger record
    op_record = temp_ledger.get_operation(result.operation_id)
    assert op_record is not None
    assert op_record.action == "create_experiment"
    assert op_record.status == OperationStatus.SUCCEEDED
    assert op_record.idempotency_key == "idem-key-1"
    assert op_record.run_id == "run_test_1"


def test_idempotency_deduplication_exact_match(mock_buses, temp_ledger):
    """Verify duplicate invocation with same idempotency key does NOT re-execute handler."""
    q_bus, c_bus = mock_buses
    call_count = 0

    def _handle_create_exp(cmd: CreateExperimentCommand):
        nonlocal call_count
        call_count += 1
        return {"id": "exp_unique_1"}

    c_bus.register(CreateExperimentCommand, _handle_create_exp)

    registry = create_full_tool_registry(q_bus, c_bus)
    policy_config = AgentPolicyConfig(require_approval_for_mutations=False)
    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(policy_config),
        ledger=temp_ledger,
    )

    ctx = ToolCallContext(
        actor="agent_planner",
        workspace_path="/tmp",
        run_id="run_1",
        correlation_id="c-1",
        permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )

    inv = ToolInvocation(
        tool_name="create_experiment",
        arguments={"run_id": "run_1", "model_name": "ridge"},
        context=ctx,
        idempotency_key="key-exact-match",
    )

    # First call: executes handler
    res1 = executor.execute(inv)
    assert res1.success is True
    assert res1.data == {"experiment_id": "exp_unique_1"}
    assert call_count == 1

    # Second call with identical key and arguments: cached, handler NOT called!
    res2 = executor.execute(inv)
    assert res2.success is True
    assert res2.data == {"experiment_id": "exp_unique_1"}
    assert res2.operation_id == res1.operation_id
    assert call_count == 1  # Deduplicated! "Duplicados no repiten efectos"


def test_idempotency_conflict_on_different_payload(mock_buses, temp_ledger):
    """Verify reusing an idempotency key with different arguments returns CONFLICT."""
    q_bus, c_bus = mock_buses
    c_bus.register(CreateExperimentCommand, lambda cmd: {"id": "exp_1"})

    registry = create_full_tool_registry(q_bus, c_bus)
    policy_config = AgentPolicyConfig(require_approval_for_mutations=False)
    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(policy_config),
        ledger=temp_ledger,
    )

    ctx = ToolCallContext(
        actor="agent_planner",
        workspace_path="/tmp",
        run_id="run_1",
        correlation_id="c-1",
        permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )

    inv1 = ToolInvocation(
        tool_name="create_experiment",
        arguments={"run_id": "run_1", "model_name": "ridge"},
        context=ctx,
        idempotency_key="shared-key",
    )
    res1 = executor.execute(inv1)
    assert res1.success is True

    # Same key, different arguments (model_name="random_forest")
    inv2 = ToolInvocation(
        tool_name="create_experiment",
        arguments={"run_id": "run_1", "model_name": "random_forest"},
        context=ctx,
        idempotency_key="shared-key",
    )
    res2 = executor.execute(inv2)
    assert res2.success is False
    assert res2.error is not None
    assert res2.error.code == ToolErrorCode.CONFLICT
    assert "already used with different arguments" in res2.error.message


def test_approval_lifecycle_propose_only_to_execution(mock_buses, temp_ledger):
    """Verify human approval requirement, pending state, and execution once approved."""
    q_bus, c_bus = mock_buses
    dispatched = []
    c_bus.register(CreateExperimentCommand, lambda cmd: dispatched.append(cmd) or {"id": "exp_approved_1"})

    registry = create_full_tool_registry(q_bus, c_bus)
    # Default policy requires approval for propose_only or mutating operations
    policy_config = AgentPolicyConfig(
        require_approval_for_mutations=True,
        default_mode=AgentPermission.PROPOSE_ONLY,
    )
    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(policy_config),
        ledger=temp_ledger,
    )

    ctx = ToolCallContext(
        actor="junior_agent",
        workspace_path="/tmp",
        run_id="run_appr_1",
        correlation_id="corr-appr",
        permission=AgentPermission.PROPOSE_ONLY,
    )

    inv = ToolInvocation(
        tool_name="create_experiment",
        arguments={"run_id": "run_appr_1", "model_name": "ridge"},
        context=ctx,
    )

    # 1. First execution: requires approval
    res1 = executor.execute(inv)
    assert res1.success is False
    assert res1.error is not None
    assert res1.error.code == ToolErrorCode.APPROVAL_REQUIRED
    assert res1.approval_id is not None
    assert len(dispatched) == 0  # Command NOT dispatched yet

    approval_id = res1.approval_id

    # Verify pending approval saved in ledger
    appr_record = temp_ledger.get_approval(approval_id)
    assert appr_record is not None
    assert appr_record.status == ApprovalStatus.PENDING

    # 2. Invocations while still pending return APPROVAL_REQUIRED
    inv_with_appr = ToolInvocation(
        tool_name="create_experiment",
        arguments={"run_id": "run_appr_1", "model_name": "ridge"},
        context=ctx,
        approval_id=approval_id,
    )
    res_pending = executor.execute(inv_with_appr)
    assert res_pending.success is False
    assert res_pending.error.code == ToolErrorCode.APPROVAL_REQUIRED
    assert "still pending reviewer resolution" in res_pending.error.message

    # 3. Reviewer approves request
    temp_ledger.update_approval_status(
        approval_id=approval_id,
        status=ApprovalStatus.APPROVED,
        reviewer="lead_engineer",
    )

    # 4. Invocations with approved approval_id now succeed!
    res_approved = executor.execute(inv_with_appr)
    assert res_approved.success is True
    assert res_approved.data == {"experiment_id": "exp_approved_1"}
    assert len(dispatched) == 1


def test_approval_anti_tampering_check(mock_buses, temp_ledger):
    """Verify modifying arguments after approval causes PERMISSION_DENIED."""
    q_bus, c_bus = mock_buses
    c_bus.register(CreateExperimentCommand, lambda cmd: {"id": "exp_tamper"})

    registry = create_full_tool_registry(q_bus, c_bus)
    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(AgentPolicyConfig(require_approval_for_mutations=True)),
        ledger=temp_ledger,
    )

    ctx = ToolCallContext(
        actor="agent_1",
        workspace_path="/tmp",
        run_id="run_1",
        correlation_id="c-1",
        permission=AgentPermission.PROPOSE_ONLY,
    )

    # Request approval for ridge
    res1 = executor.execute(
        ToolInvocation(
            tool_name="create_experiment",
            arguments={"run_id": "run_1", "model_name": "ridge"},
            context=ctx,
        )
    )
    approval_id = res1.approval_id

    # Reviewer approves ridge
    temp_ledger.update_approval_status(approval_id, ApprovalStatus.APPROVED, reviewer="human")

    # Caller tries to submit with random_forest instead
    tampered_inv = ToolInvocation(
        tool_name="create_experiment",
        arguments={"run_id": "run_1", "model_name": "random_forest"},
        context=ctx,
        approval_id=approval_id,
    )
    res_tampered = executor.execute(tampered_inv)
    assert res_tampered.success is False
    assert res_tampered.error.code == ToolErrorCode.PERMISSION_DENIED
    assert "arguments do not match approved request" in res_tampered.error.message


def test_approval_rejection_and_revocation(mock_buses, temp_ledger):
    """Verify rejected or revoked approvals reject execution."""
    q_bus, c_bus = mock_buses
    registry = create_full_tool_registry(q_bus, c_bus)
    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(AgentPolicyConfig(require_approval_for_mutations=True)),
        ledger=temp_ledger,
    )

    ctx = ToolCallContext(
        actor="a",
        workspace_path="/tmp",
        run_id="r",
        correlation_id="c",
        permission=AgentPermission.PROPOSE_ONLY,
    )
    res = executor.execute(
        ToolInvocation(
            tool_name="create_experiment",
            arguments={"run_id": "r", "model_name": "ridge"},
            context=ctx,
        )
    )
    appr_id = res.approval_id
    assert appr_id is not None

    # Reject
    temp_ledger.update_approval_status(appr_id, ApprovalStatus.REJECTED, reviewer="admin")
    res_rej = executor.execute(
        ToolInvocation(
            tool_name="create_experiment",
            arguments={"run_id": "r", "model_name": "ridge"},
            context=ctx,
            approval_id=appr_id,
        )
    )
    assert res_rej.success is False
    assert res_rej.error.code == ToolErrorCode.PERMISSION_DENIED
    assert "rejected by reviewer" in res_rej.error.message

    # Revoke
    temp_ledger.update_approval_status(appr_id, ApprovalStatus.REVOKED)
    res_rev = executor.execute(
        ToolInvocation(
            tool_name="create_experiment",
            arguments={"run_id": "r", "model_name": "ridge"},
            context=ctx,
            approval_id=appr_id,
        )
    )
    assert res_rev.success is False
    assert res_rev.error.code == ToolErrorCode.PERMISSION_DENIED
    assert "revoked" in res_rev.error.message


def test_model_denial_policy_enforcement(mock_buses):
    """Verify blacklisted models in policy are rejected before execution."""
    q_bus, c_bus = mock_buses
    registry = create_full_tool_registry(q_bus, c_bus)

    config = AgentPolicyConfig(
        denied_models=["svc", "kmeans"],
        require_approval_for_mutations=False,
    )
    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(config),
    )

    ctx = ToolCallContext(
        actor="planner",
        workspace_path="/tmp",
        run_id="run_1",
        correlation_id="c-1",
        permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )

    # Denied model
    res_denied = executor.execute(
        ToolInvocation(
            tool_name="create_experiment",
            arguments={"run_id": "run_1", "model_name": "svc"},
            context=ctx,
        )
    )
    assert res_denied.success is False
    assert res_denied.error.code == ToolErrorCode.PERMISSION_DENIED
    assert "prohibited by policy" in res_denied.error.message

    # Whitelist enforcement
    config_white = AgentPolicyConfig(
        allowed_models=["ridge", "logistic_regression"],
        require_approval_for_mutations=False,
    )
    executor_white = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(config_white),
    )
    res_not_white = executor_white.execute(
        ToolInvocation(
            tool_name="create_experiment",
            arguments={"run_id": "run_1", "model_name": "random_forest"},
            context=ctx,
        )
    )
    assert res_not_white.success is False
    assert res_not_white.error.code == ToolErrorCode.PERMISSION_DENIED
    assert "not in allowed models whitelist" in res_not_white.error.message


def test_budget_exhaustion_stops_execution(mock_buses):
    """Verify exhausting the budget stops mutating tools."""
    q_bus, c_bus = mock_buses
    registry = create_full_tool_registry(q_bus, c_bus)

    budget = AgentBudget(max_experiments=1, max_trials=1)
    budget.consume({"experiments": 1})  # 1 of 1 consumed!

    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(AgentPolicyConfig(require_approval_for_mutations=False)),
        default_budget=budget,
    )

    ctx = ToolCallContext(
        actor="planner",
        workspace_path="/tmp",
        run_id="run_1",
        correlation_id="c-1",
        permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )

    res = executor.execute(
        ToolInvocation(
            tool_name="create_experiment",
            arguments={"run_id": "run_1", "model_name": "ridge"},
            context=ctx,
        )
    )
    assert res.success is False
    assert res.error.code == ToolErrorCode.BUDGET_EXCEEDED
    assert "Budget exceeded" in res.error.message


def test_prioritize_feature_and_run_experiment_handlers(mock_buses, temp_ledger):
    """Verify prioritize_feature and run_experiment mutating tools."""
    q_bus, c_bus = mock_buses

    c_bus.register(
        PrioritizeFeatureCommand,
        lambda cmd: None,
    )
    c_bus.register(
        RunExperimentCommand,
        lambda cmd: [
            TrialResult(
                trial_id="trial_001",
                experiment_id=cmd.experiment_id,
                model_id="ridge",
                primary_metric="roc_auc",
                primary_score=0.912,
            )
        ],
    )

    registry = create_full_tool_registry(
        q_bus,
        c_bus,
        run_dataset_resolver=lambda run_id: "ds_test_123",
    )
    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(AgentPolicyConfig(require_approval_for_mutations=False)),
        ledger=temp_ledger,
    )

    ctx = ToolCallContext(
        actor="agent",
        workspace_path="/tmp",
        run_id="run_1",
        correlation_id="c-1",
        permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )

    # 1. prioritize_feature
    res_pf = executor.execute(
        ToolInvocation(
            tool_name="prioritize_feature",
            arguments={"run_id": "run_1", "feature_name": "col_a", "priority": "high"},
            context=ctx,
        )
    )
    assert res_pf.success is True
    assert res_pf.data == {"feature_name": "col_a", "priority": "high"}

    # 2. run_experiment
    res_re = executor.execute(
        ToolInvocation(
            tool_name="run_experiment",
            arguments={"run_id": "run_1", "experiment_id": "exp_01"},
            context=ctx,
        )
    )
    assert res_re.success is True
    assert res_re.data == {"trial_id": "trial_001", "metric_value": 0.912}


def test_crash_simulation_and_ledger_failure_recovery(mock_buses, temp_ledger):
    """Verify durable intention prevents execution if ledger fails, and execution errors update ledger."""
    q_bus, c_bus = mock_buses

    # Handler that fails
    def _faulty_handler(cmd):
        raise ValueError("Simulated training crash in trainer")

    c_bus.register(RunExperimentCommand, _faulty_handler)

    registry = create_full_tool_registry(q_bus, c_bus)
    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(AgentPolicyConfig(require_approval_for_mutations=False)),
        ledger=temp_ledger,
    )

    ctx = ToolCallContext(
        actor="agent",
        workspace_path="/tmp",
        run_id="run_fail",
        correlation_id="c-fail",
        permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )

    inv = ToolInvocation(
        tool_name="run_experiment",
        arguments={"run_id": "run_fail", "experiment_id": "exp_fail"},
        context=ctx,
        idempotency_key="idem-fail-1",
    )

    res = executor.execute(inv)
    assert res.success is False
    assert res.error.code == ToolErrorCode.INVALID_ARGUMENT
    assert "Simulated training crash" in res.error.message

    # Verify operation was recorded as FAILED in ledger
    op = temp_ledger.get_operation_by_idempotency_key("run_fail", "run_experiment", "idem-fail-1")
    assert op is not None
    assert op.status == OperationStatus.FAILED
    assert op.error_code == ToolErrorCode.INVALID_ARGUMENT.value
    assert "Simulated training crash" in op.error_message


def test_approval_expired_rejection(mock_buses, temp_ledger):
    """Verify expired approvals are marked EXPIRED and rejected with PERMISSION_DENIED."""
    q_bus, c_bus = mock_buses
    registry = create_full_tool_registry(q_bus, c_bus)
    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(AgentPolicyConfig(require_approval_for_mutations=True)),
        ledger=temp_ledger,
    )

    ctx = ToolCallContext(actor="a", workspace_path="/tmp", run_id="r", correlation_id="c")
    expired_req = ApprovalRequest(
        approval_id="appr-expired-123",
        action="create_experiment",
        actor="a",
        run_id="r",
        arguments_hash="dummy_hash",
        arguments={"run_id": "r", "model_name": "ridge"},
        policy_version="1.0.0",
        max_cost={},
        expires_at="2020-01-01T00:00:00Z",
        status=ApprovalStatus.APPROVED,
    )
    temp_ledger.save_approval(expired_req)

    res = executor.execute(
        ToolInvocation(
            tool_name="create_experiment",
            arguments={"run_id": "r", "model_name": "ridge"},
            context=ctx,
            approval_id="appr-expired-123",
        )
    )
    assert res.success is False
    assert res.error.code == ToolErrorCode.PERMISSION_DENIED
    assert "has expired" in res.error.message

    updated = temp_ledger.get_approval("appr-expired-123")
    assert updated.status == ApprovalStatus.EXPIRED


def test_approval_not_found_or_missing_ledger(mock_buses, temp_ledger):
    """Verify errors when approval is not found or ledger is missing."""
    q_bus, c_bus = mock_buses
    registry = create_full_tool_registry(q_bus, c_bus)

    # 1. No ledger
    exec_no_ledger = ToolExecutor(registry=registry)
    ctx = ToolCallContext(actor="a", workspace_path="/tmp", run_id="r", correlation_id="c")
    res_no_ledg = exec_no_ledger.execute(
        ToolInvocation(
            tool_name="create_experiment",
            arguments={"run_id": "r", "model_name": "ridge"},
            context=ctx,
            approval_id="nonexistent-appr",
        )
    )
    assert res_no_ledg.success is False
    assert res_no_ledg.error.code == ToolErrorCode.DEPENDENCY_UNAVAILABLE

    # 2. Ledger present, approval not found
    exec_with_ledg = ToolExecutor(registry=registry, ledger=temp_ledger)
    res_not_found = exec_with_ledg.execute(
        ToolInvocation(
            tool_name="create_experiment",
            arguments={"run_id": "r", "model_name": "ridge"},
            context=ctx,
            approval_id="nonexistent-appr",
        )
    )
    assert res_not_found.success is False
    assert res_not_found.error.code == ToolErrorCode.NOT_FOUND


def test_idempotency_running_operation_conflict(mock_buses, temp_ledger):
    """Verify operation with status RUNNING returns CONFLICT on duplicate invocation."""
    q_bus, c_bus = mock_buses
    registry = create_full_tool_registry(q_bus, c_bus)
    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(AgentPolicyConfig(require_approval_for_mutations=False)),
        ledger=temp_ledger,
    )

    ctx = ToolCallContext(
        actor="a",
        workspace_path="/tmp",
        run_id="r",
        correlation_id="c",
        permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )
    arg_hash = compute_arguments_hash(
        action="create_experiment",
        actor="a",
        run_id="r",
        arguments={"run_id": "r", "model_name": "ridge"},
        policy_version="1.0.0",
        max_cost={"experiments": 1},
    )
    op = OperationRecord(
        operation_id="op-running-001",
        run_id="r",
        actor="a",
        idempotency_key="key-running",
        action="create_experiment",
        arguments_hash=arg_hash,
        arguments={"run_id": "r", "model_name": "ridge"},
        status=OperationStatus.RUNNING,
    )
    temp_ledger.record_operation(op)

    res = executor.execute(
        ToolInvocation(
            tool_name="create_experiment",
            arguments={"run_id": "r", "model_name": "ridge"},
            context=ctx,
            idempotency_key="key-running",
        )
    )
    assert res.success is False
    assert res.error.code == ToolErrorCode.CONFLICT
    assert "already in progress" in res.error.message


def test_prioritize_feature_with_workspace_resolution(mock_buses):
    """Verify prioritize_feature resolves dataset_id via workspace._get_run."""
    q_bus, c_bus = mock_buses
    dispatched_cmd = []
    c_bus.register(PrioritizeFeatureCommand, lambda cmd: dispatched_cmd.append(cmd))

    class MockRun:
        dataset_id = "ds_from_workspace_run"

    class MockWorkspace:
        def _get_run(self, run_id: str):
            return MockRun()

    registry = create_full_tool_registry(q_bus, c_bus, workspace=MockWorkspace())
    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(AgentPolicyConfig(require_approval_for_mutations=False)),
    )

    ctx = ToolCallContext(
        actor="a",
        workspace_path="/tmp",
        run_id="r_ws",
        correlation_id="c",
        permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )
    res = executor.execute(
        ToolInvocation(
            tool_name="prioritize_feature",
            arguments={"run_id": "r_ws", "feature_name": "col_z", "priority": "medium"},
            context=ctx,
        )
    )
    assert res.success is True
    assert len(dispatched_cmd) == 1
    assert dispatched_cmd[0].dataset_id == "ds_from_workspace_run"
    assert dispatched_cmd[0].score == 1.0


def test_run_experiment_empty_trials_error(mock_buses):
    """Verify run_experiment raises ValueError if no trials are returned."""
    q_bus, c_bus = mock_buses
    c_bus.register(RunExperimentCommand, lambda cmd: [])

    registry = create_full_tool_registry(q_bus, c_bus)
    executor = ToolExecutor(
        registry=registry,
        policy_evaluator=PolicyEvaluator(AgentPolicyConfig(require_approval_for_mutations=False)),
    )

    ctx = ToolCallContext(
        actor="a",
        workspace_path="/tmp",
        run_id="r",
        correlation_id="c",
        permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )
    res = executor.execute(
        ToolInvocation(
            tool_name="run_experiment",
            arguments={"run_id": "r", "experiment_id": "exp_empty"},
            context=ctx,
        )
    )
    assert res.success is False
    assert res.error.code == ToolErrorCode.INVALID_ARGUMENT
    assert "did not produce any trial results" in res.error.message

