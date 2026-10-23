"""CLI subcommands for managing autonomous agent sessions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from automl.domain.agents.entities import ApprovalStatus
from automl.application.agents.contracts import AgentSessionState, SessionStepResult
from automl.application.agents.executor import ToolExecutor, create_full_tool_registry
from automl.application.agents.orchestrator.session_manager import AgentSessionManager
from automl.application.agents.orchestrator.state_machine import SessionStatus
from automl.application.agents.specialists.context_builder import ContextBuilder
from automl.application.agents.specialists.critic import Critic
from automl.application.agents.specialists.planner import Planner
from automl.application.bootstrap import build_application
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger

try:
    from automl.application.agents.orchestrator.graph import (
        LangGraphAgentOrchestrator,
        is_langgraph_available,
    )
    from automl.infrastructure.database.sqlite_checkpoint_saver import SqliteCheckpointSaver
except ImportError:  # pragma: no cover
    LangGraphAgentOrchestrator = None  # type: ignore[assignment, misc]
    SqliteCheckpointSaver = None  # type: ignore[assignment, misc]

    def is_langgraph_available() -> bool:  # type: ignore[misc]
        return False


def _resolve_workspace_and_ledger(
    workspace_path: str | Path | None = None,
) -> tuple[Any, SqliteAgentLedger, AgentSessionManager]:
    """Resolve workspace, SQLite ledger, and initialized AgentSessionManager."""
    if workspace_path:
        w_path = Path(workspace_path)
        if w_path.is_file() or str(w_path).endswith(".db"):
            workspace_dir = w_path.parent
            ledger_file = w_path
        else:
            workspace_dir = w_path
            ledger_file = w_path / "agent_ledger.db"
    else:
        workspace_dir = Path.cwd() / ".automl" / "default"
        ledger_file = workspace_dir / "agent_ledger.db"

    ws, cmd, qry = build_application(root_dir=str(workspace_dir))
    ledger = SqliteAgentLedger(ledger_file)

    tool_registry = create_full_tool_registry(
        query_bus=qry, command_bus=cmd, workspace=ws, ledger=ledger
    )
    executor = ToolExecutor(registry=tool_registry, ledger=ledger)

    manager = AgentSessionManager(
        workspace=ws,
        session_store=ledger,
        approval_store=ledger,
        context_builder=ContextBuilder(),
        planner=Planner(),
        critic=Critic(),
        executor=executor,
    )
    return ws, ledger, manager


def session_start_cli(args: argparse.Namespace) -> int:
    """Start a new autonomous agent optimization session."""
    try:
        _, _, manager = _resolve_workspace_and_ledger(getattr(args, "workspace", None))
        run_id = args.run_id
        goal = getattr(args, "goal", None) or f"Autonomous optimization for run {run_id}"
        max_iterations = getattr(args, "max_iterations", 10)
        target_score = getattr(args, "target_score", None)
        patience = getattr(args, "patience", 3)

        session = manager.create_session(
            run_id=run_id,
            goal=goal,
            max_iterations=max_iterations,
            target_score=target_score,
            patience=patience,
        )

        if getattr(args, "json", False):
            sys.stdout.write(json.dumps(session.to_dict(), indent=2) + "\n")
        else:
            sys.stdout.write("\n=== Agent Session Created ===\n")
            sys.stdout.write(f"Session ID:      {session.session_id}\n")
            sys.stdout.write(f"Run ID:          {session.run_id}\n")
            sys.stdout.write(f"Goal:            {session.goal}\n")
            sys.stdout.write(f"Status:          {session.status.upper()}\n")
            sys.stdout.write(f"Max Iterations:  {session.max_iterations}\n")
            if target_score is not None:
                sys.stdout.write(f"Target Score:    {target_score}\n")
            sys.stdout.write(f"Checkpoint ID:   {session.checkpoint_id}\n\n")
            sys.stdout.write("To advance one step:   automl agent session step " + session.session_id + "\n")
            sys.stdout.write("To run autonomously:   automl agent session resume " + session.session_id + "\n")

        return 0
    except Exception as exc:
        sys.stderr.write(f"Error starting agent session: {exc}\n")
        return 1


def session_step_cli(args: argparse.Namespace) -> int:
    """Execute a single deterministic step in the session."""
    try:
        _, _, manager = _resolve_workspace_and_ledger(getattr(args, "workspace", None))
        session_id = args.session_id
        result = manager.step(session_id)

        if getattr(args, "json", False):
            sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
        else:
            sys.stdout.write(f"\n=== Session Step {result.step_number} [{result.state.upper()}] ===\n")
            if result.proposal:
                sys.stdout.write(f"Specialist:   {result.proposal.specialist_name}\n")
                sys.stdout.write(f"Hypothesis:   {result.proposal.hypothesis}\n")
                sys.stdout.write(f"Action:       {result.proposal.action_type}\n")
            if result.feedback:
                improv_label = "YES" if result.feedback.is_improvement else "NO"
                sys.stdout.write(f"Improvement:  {improv_label}\n")
                if result.feedback.current_score is not None:
                    sys.stdout.write(f"Score:        {result.feedback.current_score:.4f}\n")
                sys.stdout.write(f"Critic Rec:   {result.feedback.recommendation}\n")
                if result.feedback.variance_observation:
                    sys.stdout.write(f"Variance:     {result.feedback.variance_observation}\n")
            if result.stop_reason:
                sys.stdout.write(f"Stop Reason:  {result.stop_reason}\n")
            sys.stdout.write(f"Checkpoint:   {result.checkpoint_id}\n\n")

        return 0
    except Exception as exc:
        sys.stderr.write(f"Error executing session step: {exc}\n")
        return 1


def session_resume_cli(args: argparse.Namespace) -> int:
    """Resume session and execute autonomous loop until completion or intervention."""
    try:
        ws, ledger, manager = _resolve_workspace_and_ledger(getattr(args, "workspace", None))
        session_id = args.session_id
        max_steps = getattr(args, "max_steps", None)
        engine = getattr(args, "engine", "deterministic")

        if engine == "langgraph":
            if not is_langgraph_available() or LangGraphAgentOrchestrator is None or SqliteCheckpointSaver is None:
                sys.stderr.write("Error: LangGraph not available. Install via `pip install 'catml[agents]'`.\n")
                return 1

            checkpointer = SqliteCheckpointSaver(ledger.db_path)
            orchestrator = LangGraphAgentOrchestrator(
                workspace=ws,
                session_store=ledger,
                checkpointer=checkpointer.saver,
                approval_store=ledger,
                operation_store=ledger,
                context_builder=manager.context_builder,
                planner=manager.planner,
                critic=manager.critic,
                executor=manager.executor,
            )

            session = manager.get_session(session_id)
            if session and session.status == SessionStatus.WAITING_APPROVAL.value and session.pending_approval_id:
                appr = ledger.get_approval(session.pending_approval_id)
                if appr and appr.status == ApprovalStatus.APPROVED:
                    res_state = orchestrator.resume(session_id, human_decision=True)
                elif appr and appr.status == ApprovalStatus.REJECTED:
                    res_state = orchestrator.resume(session_id, human_decision=False)
                else:
                    sys.stdout.write(f"\nSession {session_id} is waiting for human approval (`automl agent approvals list`).\n")
                    return 0
            else:
                res_state = orchestrator.run(session_id)

            if getattr(args, "json", False):
                sys.stdout.write(json.dumps(dict(res_state), indent=2, default=str) + "\n")
            else:
                sys.stdout.write(f"\n=== LangGraph Autonomous Execution: {session_id} ===\n")
                sys.stdout.write(f"Status:       {res_state.get('status', '').upper()}\n")
                sys.stdout.write(f"Iterations:   {res_state.get('iteration_count', 0)} / {res_state.get('max_iterations', 10)}\n")
                if res_state.get("stop_reason"):
                    sys.stdout.write(f"Stop Reason:  {res_state.get('stop_reason')}\n")
                if res_state.get("checkpoint_id"):
                    sys.stdout.write(f"Checkpoint:   {res_state.get('checkpoint_id')}\n")
            return 0

        results = manager.resume(session_id, max_steps=max_steps)

        if getattr(args, "json", False):
            payload = [r.to_dict() for r in results]
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        else:
            sys.stdout.write(f"\n=== Resumed Session {session_id} ({len(results)} steps executed) ===\n")
            for r in results:
                action_str = r.action_taken or "none"
                improv_str = "improved" if r.feedback and r.feedback.is_improvement else "no-improv"
                score_str = f"score={r.feedback.current_score:.4f}" if r.feedback and r.feedback.current_score is not None else ""
                sys.stdout.write(f"  Step {r.step_number:2d}: [{r.state:16s}] action={action_str:20s} {improv_str:10s} {score_str}\n")

            last_result = results[-1] if results else None
            if last_result and last_result.stop_reason:
                sys.stdout.write(f"\nSession terminated: {last_result.stop_reason}\n")
            elif last_result and last_result.state == "waiting_approval":
                sys.stdout.write("\nSession paused: waiting for human approval (`automl agent approvals list`).\n")

        return 0
    except Exception as exc:
        sys.stderr.write(f"Error resuming agent session: {exc}\n")
        return 1


def session_status_cli(args: argparse.Namespace) -> int:
    """Inspect status, state, and checkpoint of an agent session."""
    try:
        _, ledger, manager = _resolve_workspace_and_ledger(getattr(args, "workspace", None))
        session_id = args.session_id
        session = manager.get_session(session_id)
        if session is None:
            sys.stderr.write(f"Error: Session '{session_id}' not found.\n")
            return 1

        engine = getattr(args, "engine", "deterministic")
        graph_data: dict[str, Any] | None = None
        if engine == "langgraph" and is_langgraph_available() and SqliteCheckpointSaver is not None:
            try:
                cp = SqliteCheckpointSaver(ledger.db_path)
                graph_data = cp.get_latest_state(session_id)
            except Exception:
                pass

        if getattr(args, "json", False):
            payload = session.to_dict()
            if graph_data:
                payload["langgraph_state"] = graph_data
            sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
        else:
            sys.stdout.write("\n=== Agent Session Details ===\n")
            sys.stdout.write(f"Session ID:      {session.session_id}\n")
            sys.stdout.write(f"Run ID:          {session.run_id}\n")
            sys.stdout.write(f"Status:          {session.status.upper()}\n")
            sys.stdout.write(f"Goal:            {session.goal}\n")
            sys.stdout.write(f"Progress:        {session.iteration_count} / {session.max_iterations} iterations\n")
            if session.stop_reason:
                sys.stdout.write(f"Stop Reason:     {session.stop_reason}\n")
            if session.pending_approval_id:
                sys.stdout.write(f"Pending Approval:{session.pending_approval_id}\n")
            sys.stdout.write(f"Checkpoint ID:   {session.checkpoint_id}\n")
            sys.stdout.write(f"Updated At:      {session.updated_at}\n")
            if graph_data:
                sys.stdout.write(f"LangGraph State: Synchronized ({len(graph_data)} channels persisted)\n\n")
            else:
                sys.stdout.write("\n")

        return 0
    except Exception as exc:
        sys.stderr.write(f"Error inspecting session status: {exc}\n")
        return 1


def session_stop_cli(args: argparse.Namespace) -> int:
    """Stop an active agent session."""
    try:
        _, _, manager = _resolve_workspace_and_ledger(getattr(args, "workspace", None))
        session_id = args.session_id
        reason = getattr(args, "reason", None) or "manually_stopped"
        session = manager.stop_session(session_id, reason=reason)

        if getattr(args, "json", False):
            sys.stdout.write(json.dumps(session.to_dict(), indent=2) + "\n")
        else:
            sys.stdout.write(f"\nSession {session.session_id} stopped ({session.stop_reason}).\n")

        return 0
    except Exception as exc:
        sys.stderr.write(f"Error stopping agent session: {exc}\n")
        return 1


def register_agent_session_subparser(agent_parser: argparse.ArgumentParser) -> None:
    """Register 'session' subcommand under 'automl agent'."""
    # Find existing subparsers for agent_parser if present
    sub = getattr(agent_parser, "_subparsers", None)
    if sub is None:
        return

    # Look for action action in subparsers
    session_parser = None
    for action in sub._actions:
        if isinstance(action, argparse._SubParsersAction):
            session_parser = action.add_parser("session", help="Manage autonomous agent sessions and cycles")
            break

    if session_parser is None:
        return

    session_sub = session_parser.add_subparsers(dest="session_subcommand", required=True)

    # 1. start
    start_p = session_sub.add_parser("start", help="Start a new autonomous agent session")
    start_p.add_argument("--run-id", required=True, help="Target run ID to optimize")
    start_p.add_argument("--goal", help="Optional scientific goal for this session")
    start_p.add_argument("--max-iterations", type=int, default=10, help="Maximum iteration steps (default: 10)")
    start_p.add_argument("--target-score", type=float, help="Target metric score threshold for early success stopping")
    start_p.add_argument("--patience", type=int, default=3, help="Stagnation patience before stopping (default: 3)")
    start_p.add_argument("--workspace", help="Path to workspace directory or ledger db file")
    start_p.add_argument("--json", action="store_true", help="Output in JSON format")
    start_p.set_defaults(func=session_start_cli)

    # 2. step
    step_p = session_sub.add_parser("step", help="Execute one single deterministic step in the session")
    step_p.add_argument("session_id", help="Session ID to step")
    step_p.add_argument("--workspace", help="Path to workspace directory or ledger db file")
    step_p.add_argument("--json", action="store_true", help="Output in JSON format")
    step_p.set_defaults(func=session_step_cli)

    # 3. resume
    resume_p = session_sub.add_parser("resume", help="Resume session and execute autonomous loop")
    resume_p.add_argument("session_id", help="Session ID to resume")
    resume_p.add_argument("--max-steps", type=int, help="Optional maximum steps to run in this invocation")
    resume_p.add_argument(
        "--engine",
        choices=["deterministic", "langgraph"],
        default="deterministic",
        help="Execution engine: deterministic state machine or LangGraph StateGraph (default: deterministic)",
    )
    resume_p.add_argument("--workspace", help="Path to workspace directory or ledger db file")
    resume_p.add_argument("--json", action="store_true", help="Output in JSON format")
    resume_p.set_defaults(func=session_resume_cli)

    # 4. status
    status_p = session_sub.add_parser("status", help="Inspect status and checkpoint of an agent session")
    status_p.add_argument("session_id", help="Session ID to inspect")
    status_p.add_argument(
        "--engine",
        choices=["deterministic", "langgraph"],
        default="deterministic",
        help="Inspect deterministic ledger state or LangGraph checkpointer state (default: deterministic)",
    )
    status_p.add_argument("--workspace", help="Path to workspace directory or ledger db file")
    status_p.add_argument("--json", action="store_true", help="Output in JSON format")
    status_p.set_defaults(func=session_status_cli)

    # 5. stop
    stop_p = session_sub.add_parser("stop", help="Force stop an active agent session")
    stop_p.add_argument("session_id", help="Session ID to stop")
    stop_p.add_argument("--reason", default="manually_stopped", help="Reason for stopping")
    stop_p.add_argument("--workspace", help="Path to workspace directory or ledger db file")
    stop_p.add_argument("--json", action="store_true", help="Output in JSON format")
    stop_p.set_defaults(func=session_stop_cli)
