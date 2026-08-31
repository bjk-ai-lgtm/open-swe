"""Bridge deterministic validation into orchestration state."""

from agent.orchestration import ExecutionState, record_validation_result

from .types import ValidationReport


def apply_validation_report(
    state: ExecutionState,
    report: ValidationReport,
) -> ExecutionState:
    """Apply deterministic validation evidence to task state."""
    return record_validation_result(
        state,
        passed=report.passed,
        failure_reason=(None if report.passed else report.failure_summary()),
    )
