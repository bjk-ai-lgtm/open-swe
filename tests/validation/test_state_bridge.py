from agent.orchestration import (
    TaskStatus,
    begin_attempt,
    create_execution_state,
    mark_execution_complete,
)
from agent.routing import build_execution_plan
from agent.validation import (
    CheckResult,
    CommandResult,
    ValidationCheck,
    ValidationReport,
    apply_validation_report,
)


def validating_backend_state():
    plan = build_execution_plan("Implement a REST API endpoint backed by the database.")

    state = create_execution_state(plan)
    state = begin_attempt(state)

    return mark_execution_complete(state)


def test_passing_report_completes_task() -> None:
    state = validating_backend_state()

    check = ValidationCheck(
        name="tests",
        command=("pytest", "-q"),
    )

    report = ValidationReport(
        results=(
            CheckResult(
                check=check,
                command_result=CommandResult(exit_code=0),
            ),
        )
    )

    state = apply_validation_report(state, report)

    assert state.status is TaskStatus.SUCCEEDED
    assert state.terminal is True


def test_failing_report_requests_retry() -> None:
    state = validating_backend_state()

    check = ValidationCheck(
        name="tests",
        command=("pytest", "-q"),
    )

    report = ValidationReport(
        results=(
            CheckResult(
                check=check,
                command_result=CommandResult(
                    exit_code=1,
                    stderr="test failure",
                ),
            ),
        )
    )

    state = apply_validation_report(state, report)

    assert state.status is TaskStatus.RETRY_REQUIRED
    assert state.validation_failures == 1
    assert "exit code 1" in state.last_failure
