"""Deterministic validation executor."""

from collections.abc import Sequence

from .runner import CommandRunner
from .types import CheckResult, ValidationCheck, ValidationReport


async def run_validation(
    runner: CommandRunner,
    checks: Sequence[ValidationCheck],
    *,
    work_dir: str,
) -> ValidationReport:
    """Execute validation checks and return deterministic evidence."""
    results: list[CheckResult] = []

    for check in checks:
        command_result = await runner.run(
            check,
            work_dir=work_dir,
        )

        results.append(
            CheckResult(
                check=check,
                command_result=command_result,
            )
        )

    return ValidationReport(results=tuple(results))
