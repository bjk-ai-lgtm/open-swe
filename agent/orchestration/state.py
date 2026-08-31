"""Deterministic execution state machine for orchestrated tasks."""

from dataclasses import dataclass, replace
from enum import StrEnum

from agent.routing import ExecutionPlan, SpecialistRole


class TaskStatus(StrEnum):
    """Lifecycle states for one orchestrated task."""

    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    RETRY_REQUIRED = "retry-required"
    ESCALATION_REQUIRED = "escalation-required"
    SUCCEEDED = "succeeded"
    QUARANTINED = "quarantined"


TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.QUARANTINED,
    }
)


@dataclass(frozen=True)
class ExecutionState:
    """Runtime-owned state for one task execution."""

    plan: ExecutionPlan
    status: TaskStatus = TaskStatus.PENDING
    attempt: int = 0
    validation_failures: int = 0
    escalated: bool = False
    last_failure: str | None = None

    @property
    def terminal(self) -> bool:
        """Return whether this task can no longer transition normally."""
        return self.status in TERMINAL_STATUSES

    @property
    def current_role(self) -> SpecialistRole:
        """Return the role expected to act in the current phase."""
        if self.status is TaskStatus.VALIDATING and self.plan.validation.validator is not None:
            return self.plan.validation.validator

        return self.plan.primary_role


def create_execution_state(plan: ExecutionPlan) -> ExecutionState:
    """Create initial state for an execution plan."""
    return ExecutionState(plan=plan)


def begin_attempt(state: ExecutionState) -> ExecutionState:
    """Begin an initial or retry implementation attempt."""
    if state.status not in {
        TaskStatus.PENDING,
        TaskStatus.RETRY_REQUIRED,
    }:
        raise ValueError(f"Cannot begin normal attempt from status {state.status}")

    return replace(
        state,
        status=TaskStatus.RUNNING,
        attempt=state.attempt + 1,
    )


def begin_escalated_attempt(state: ExecutionState) -> ExecutionState:
    """Begin the stronger-model attempt after normal retries are exhausted."""
    if state.status is not TaskStatus.ESCALATION_REQUIRED:
        raise ValueError(f"Cannot begin escalated attempt from status {state.status}")

    return replace(
        state,
        status=TaskStatus.RUNNING,
        attempt=state.attempt + 1,
        escalated=True,
    )


def mark_execution_complete(state: ExecutionState) -> ExecutionState:
    """Mark implementation work complete and enter validation when required."""
    if state.status is not TaskStatus.RUNNING:
        raise ValueError(f"Cannot complete execution from status {state.status}")

    if state.plan.validation.required:
        return replace(
            state,
            status=TaskStatus.VALIDATING,
        )

    return replace(
        state,
        status=TaskStatus.SUCCEEDED,
    )


def record_validation_result(
    state: ExecutionState,
    *,
    passed: bool,
    failure_reason: str | None = None,
) -> ExecutionState:
    """Apply a deterministic QA result to the task state."""
    if state.status is not TaskStatus.VALIDATING:
        raise ValueError(f"Cannot record validation from status {state.status}")

    if passed:
        return replace(
            state,
            status=TaskStatus.SUCCEEDED,
            last_failure=None,
        )

    failures = state.validation_failures + 1
    reason = failure_reason or "Validation failed"

    if state.escalated:
        return replace(
            state,
            status=TaskStatus.QUARANTINED,
            validation_failures=failures,
            last_failure=reason,
        )

    if failures <= state.plan.validation.max_retries:
        return replace(
            state,
            status=TaskStatus.RETRY_REQUIRED,
            validation_failures=failures,
            last_failure=reason,
        )

    return replace(
        state,
        status=TaskStatus.ESCALATION_REQUIRED,
        validation_failures=failures,
        last_failure=reason,
    )


def quarantine_task(
    state: ExecutionState,
    *,
    reason: str,
) -> ExecutionState:
    """Quarantine one failed task without affecting other tasks."""
    if state.terminal:
        raise ValueError(f"Cannot quarantine terminal task with status {state.status}")

    return replace(
        state,
        status=TaskStatus.QUARANTINED,
        last_failure=reason,
    )
