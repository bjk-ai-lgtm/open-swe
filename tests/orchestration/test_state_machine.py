import pytest

from agent.orchestration import (
    TaskStatus,
    begin_attempt,
    begin_escalated_attempt,
    create_execution_state,
    mark_execution_complete,
    record_validation_result,
)
from agent.routing import SpecialistRole, build_execution_plan


def backend_state():
    plan = build_execution_plan("Implement a REST API endpoint backed by the database.")
    return create_execution_state(plan)


def test_backend_happy_path_requires_qa_before_success() -> None:
    state = backend_state()

    state = begin_attempt(state)

    assert state.status is TaskStatus.RUNNING
    assert state.current_role is SpecialistRole.BACKEND
    assert state.attempt == 1

    state = mark_execution_complete(state)

    assert state.status is TaskStatus.VALIDATING
    assert state.current_role is SpecialistRole.QA

    state = record_validation_result(state, passed=True)

    assert state.status is TaskStatus.SUCCEEDED
    assert state.terminal is True


def test_backend_validation_failure_requests_retry() -> None:
    state = begin_attempt(backend_state())
    state = mark_execution_complete(state)

    state = record_validation_result(
        state,
        passed=False,
        failure_reason="API contract test failed",
    )

    assert state.status is TaskStatus.RETRY_REQUIRED
    assert state.validation_failures == 1
    assert state.last_failure == "API contract test failed"


def test_backend_exhausts_retries_then_requires_escalation() -> None:
    state = backend_state()

    for _ in range(3):
        state = begin_attempt(state)
        state = mark_execution_complete(state)
        state = record_validation_result(
            state,
            passed=False,
            failure_reason="Tests still failing",
        )

    assert state.status is TaskStatus.ESCALATION_REQUIRED
    assert state.validation_failures == 3
    assert state.attempt == 3


def test_failed_first_escalation_requests_next_escalation() -> None:
    state = backend_state()

    for _ in range(3):
        state = begin_attempt(state)
        state = mark_execution_complete(state)
        state = record_validation_result(
            state,
            passed=False,
            failure_reason="Tests still failing",
        )

    state = begin_escalated_attempt(state)

    assert state.escalated is True
    assert state.escalation_level == 1
    assert state.status is TaskStatus.RUNNING

    state = mark_execution_complete(state)

    state = record_validation_result(
        state,
        passed=False,
        failure_reason="Sol could not fix the task",
    )

    assert state.status is TaskStatus.ESCALATION_REQUIRED
    assert state.terminal is False
    assert state.escalation_level == 1


def test_failed_final_escalation_is_quarantined() -> None:
    state = backend_state()

    for _ in range(3):
        state = begin_attempt(state)
        state = mark_execution_complete(state)
        state = record_validation_result(
            state,
            passed=False,
            failure_reason="Tests still failing",
        )

    state = begin_escalated_attempt(state)
    state = mark_execution_complete(state)
    state = record_validation_result(
        state,
        passed=False,
        failure_reason="Sol could not fix the task",
    )

    assert state.status is TaskStatus.ESCALATION_REQUIRED

    state = begin_escalated_attempt(state)

    assert state.escalation_level == 2
    assert state.status is TaskStatus.RUNNING

    state = mark_execution_complete(state)

    state = record_validation_result(
        state,
        passed=False,
        failure_reason="Opus could not fix the task",
    )

    assert state.status is TaskStatus.QUARANTINED
    assert state.terminal is True
    assert state.last_failure == "Opus could not fix the task"


def test_research_task_succeeds_without_qa_gate() -> None:
    plan = build_execution_plan("Research the official documentation and compare versions.")
    state = create_execution_state(plan)

    state = begin_attempt(state)
    state = mark_execution_complete(state)

    assert state.status is TaskStatus.SUCCEEDED
    assert state.validation_failures == 0


def test_terminal_task_cannot_be_started_again() -> None:
    plan = build_execution_plan("Research the official documentation and compare versions.")
    state = create_execution_state(plan)
    state = begin_attempt(state)
    state = mark_execution_complete(state)

    with pytest.raises(ValueError):
        begin_attempt(state)
