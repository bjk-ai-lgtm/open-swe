"""Runtime-owned coordinator for one orchestrated software task."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from agent.routing import (
    SpecialistRole,
    build_execution_plan,
    model_for_role,
)
from agent.validation import (
    CommandRunner,
    ValidationCheck,
    ValidationReport,
    apply_validation_report,
    run_validation,
    sandbox_runner_for_thread,
)

from .state import (
    ExecutionState,
    TaskStatus,
    begin_attempt,
    begin_escalated_attempt,
    create_execution_state,
    mark_execution_complete,
    quarantine_task,
    record_execution_failure,
)


@dataclass(frozen=True)
class SpecialistExecutionResult:
    """Result returned by a specialist execution adapter."""

    success: bool
    summary: str = ""
    failure_reason: str | None = None


class SpecialistExecutor(Protocol):
    """Adapter responsible for invoking the selected specialist."""

    async def execute(
        self,
        *,
        thread_id: str,
        work_dir: str,
        task: str,
        role: SpecialistRole,
        model_id: str,
        attempt: int,
        escalation_level: int,
        previous_failure: str | None,
    ) -> SpecialistExecutionResult:
        """Execute one specialist attempt."""
        ...


@dataclass(frozen=True)
class AttemptRecord:
    """Observable evidence for one orchestrated attempt."""

    attempt: int
    role: SpecialistRole
    model_id: str
    escalation_level: int
    execution: SpecialistExecutionResult
    validation: ValidationReport | None


@dataclass(frozen=True)
class CoordinatorResult:
    """Final task state plus its attempt history."""

    state: ExecutionState
    attempts: tuple[AttemptRecord, ...]


RunnerFactory = Callable[[str], Awaitable[CommandRunner]]


async def _default_runner_factory(thread_id: str) -> CommandRunner:
    return await sandbox_runner_for_thread(thread_id)


async def run_orchestrated_task(
    *,
    thread_id: str,
    task: str,
    work_dir: str,
    executor: SpecialistExecutor,
    checks: Sequence[ValidationCheck],
    runner_factory: RunnerFactory = _default_runner_factory,
) -> CoordinatorResult:
    """Run one task through routing, execution, validation, and escalation."""
    if not thread_id.strip():
        raise ValueError("Thread ID cannot be empty")

    if not task.strip():
        raise ValueError("Task cannot be empty")

    if not work_dir.strip():
        raise ValueError("Work directory cannot be empty")

    plan = build_execution_plan(task)

    if plan.validation.required and not checks:
        raise ValueError("At least one validation check is required for this task")

    state = create_execution_state(plan)
    history: list[AttemptRecord] = []

    while not state.terminal:
        if state.status in {
            TaskStatus.PENDING,
            TaskStatus.RETRY_REQUIRED,
        }:
            state = begin_attempt(state)
        elif state.status is TaskStatus.ESCALATION_REQUIRED:
            state = begin_escalated_attempt(state)
        else:
            raise RuntimeError(f"Coordinator cannot begin attempt from status {state.status}")

        model_id = model_for_role(
            state.plan.primary_role,
            escalation_level=state.escalation_level,
        )

        try:
            execution = await executor.execute(
                thread_id=thread_id,
                work_dir=work_dir,
                task=task,
                role=state.plan.primary_role,
                model_id=model_id,
                attempt=state.attempt,
                escalation_level=state.escalation_level,
                previous_failure=state.last_failure,
            )
        except Exception as exc:
            execution = SpecialistExecutionResult(
                success=False,
                failure_reason=(f"Specialist executor raised {type(exc).__name__}: {exc}"),
            )

        if not execution.success:
            state = record_execution_failure(
                state,
                failure_reason=execution.failure_reason,
            )

            history.append(
                AttemptRecord(
                    attempt=state.attempt,
                    role=state.plan.primary_role,
                    model_id=model_id,
                    escalation_level=state.escalation_level,
                    execution=execution,
                    validation=None,
                )
            )
            continue

        state = mark_execution_complete(state)
        validation_report: ValidationReport | None = None

        if state.status is TaskStatus.VALIDATING:
            try:
                runner = await runner_factory(thread_id)
                validation_report = await run_validation(
                    runner,
                    checks,
                    work_dir=work_dir,
                )
                state = apply_validation_report(
                    state,
                    validation_report,
                )
            except Exception as exc:
                state = quarantine_task(
                    state,
                    reason=(f"Validation infrastructure failure: {type(exc).__name__}: {exc}"),
                )

        history.append(
            AttemptRecord(
                attempt=state.attempt,
                role=state.plan.primary_role,
                model_id=model_id,
                escalation_level=state.escalation_level,
                execution=execution,
                validation=validation_report,
            )
        )

    return CoordinatorResult(
        state=state,
        attempts=tuple(history),
    )
