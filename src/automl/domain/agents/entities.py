"""Pure domain entities, value objects, and enums for the agentic subsystem."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ToolEffect(str, Enum):
    """Effect classification for agent tools."""
    READ = "read"
    PROPOSE = "propose"
    MUTATE = "mutate"


class AgentPermission(str, Enum):
    """Authorization permission levels for agent operations."""
    READ_ONLY = "read_only"
    PROPOSE_ONLY = "propose_only"
    EXECUTE_WITHIN_BUDGET = "execute_within_budget"
    ADMIN = "admin"


class PolicyDecisionType(str, Enum):
    """Decision outcomes produced by policy evaluation."""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class ApprovalStatus(str, Enum):
    """Lifecycle status for human-in-the-loop or policy approval requests."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REVOKED = "revoked"


class OperationStatus(str, Enum):
    """Execution status for recorded agent operations."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    RECOVERY_REQUIRED = "recovery_required"


class ToolErrorCode(str, Enum):
    """Canonical error codes for tool and agent failures."""
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    NOT_FOUND = "NOT_FOUND"
    SCOPE_VIOLATION = "SCOPE_VIOLATION"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    CONFLICT = "CONFLICT"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    CANCELLED = "CANCELLED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass
class RunLease:
    """Exclusive writer lease for a run to prevent concurrent state corruption."""
    run_id: str
    owner_id: str
    acquired_at: str
    expires_at: str
    heartbeat_at: str

    def is_expired(self, now_iso: str | None = None) -> bool:
        from datetime import datetime, timezone
        now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00")) if now_iso else datetime.now(timezone.utc)
        exp_dt = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        return now_dt > exp_dt

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunLease:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})



@dataclass
class AgentBudget:
    """Finite operational budget enforced across agent executions."""
    max_experiments: int = 5
    max_trials: int = 20
    max_folds: int = 5
    max_duration_seconds: float = 600.0
    max_llm_calls: int = 50
    max_tokens: int = 100_000
    consumed_experiments: int = 0
    consumed_trials: int = 0
    consumed_folds: int = 0
    consumed_duration_seconds: float = 0.0
    consumed_llm_calls: int = 0
    consumed_tokens: int = 0

    def remaining_experiments(self) -> int:
        return max(0, self.max_experiments - self.consumed_experiments)

    def remaining_trials(self) -> int:
        return max(0, self.max_trials - self.consumed_trials)

    def remaining_folds(self) -> int:
        return max(0, self.max_folds - self.consumed_folds)

    def remaining_duration_seconds(self) -> float:
        return max(0.0, self.max_duration_seconds - self.consumed_duration_seconds)

    def remaining_llm_calls(self) -> int:
        return max(0, self.max_llm_calls - self.consumed_llm_calls)

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.consumed_tokens)

    def can_afford(self, cost: dict[str, Any]) -> tuple[bool, str | None]:
        """Check if an estimated cost can be afforded within remaining budget."""
        req_exp = cost.get("experiments", 0)
        req_trials = cost.get("trials", 0)
        req_folds = cost.get("folds", 0)
        req_duration = cost.get("duration_seconds", 0.0)
        req_calls = cost.get("llm_calls", 0)
        req_tokens = cost.get("tokens", 0)

        if self.consumed_experiments + req_exp > self.max_experiments:
            return False, f"Experiment budget exceeded: requires {req_exp}, remaining {self.remaining_experiments()}"
        if self.consumed_trials + req_trials > self.max_trials:
            return False, f"Trial budget exceeded: requires {req_trials}, remaining {self.remaining_trials()}"
        if self.consumed_folds + req_folds > self.max_folds:
            return False, f"Fold budget exceeded: requires {req_folds}, remaining {self.remaining_folds()}"
        if self.consumed_duration_seconds + req_duration > self.max_duration_seconds:
            return False, f"Duration budget exceeded: requires {req_duration}s, remaining {self.remaining_duration_seconds()}s"
        if self.consumed_llm_calls + req_calls > self.max_llm_calls:
            return False, f"LLM calls budget exceeded: requires {req_calls}, remaining {self.remaining_llm_calls()}"
        if self.consumed_tokens + req_tokens > self.max_tokens:
            return False, f"Tokens budget exceeded: requires {req_tokens}, remaining {self.remaining_tokens()}"

        return True, None

    def consume(self, cost: dict[str, Any]) -> None:
        """Record consumed resources."""
        self.consumed_experiments += cost.get("experiments", 0)
        self.consumed_trials += cost.get("trials", 0)
        self.consumed_folds += cost.get("folds", 0)
        self.consumed_duration_seconds += float(cost.get("duration_seconds", 0.0))
        self.consumed_llm_calls += cost.get("llm_calls", 0)
        self.consumed_tokens += cost.get("tokens", 0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentBudget:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Hypothesis:
    """Scientific experiment hypothesis proposed by the agent (Propose != Accept)."""
    hypothesis_id: str
    run_id: str
    reasoning: str
    candidate_config: dict[str, Any]
    target_metric: str
    metric_direction: str = "maximize"
    baseline_metric: float | None = None
    verification_criteria: dict[str, Any] = field(default_factory=dict)
    status: str = "proposed"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hypothesis:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
