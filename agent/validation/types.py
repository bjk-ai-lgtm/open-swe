"""Types used by deterministic validation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationCheck:
    """One deterministic command that must be evaluated."""

    name: str
    command: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class CommandResult:
    """Result returned by an isolated command runner."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


@dataclass(frozen=True)
class CheckResult:
    """Result of one validation check."""

    check: ValidationCheck
    command_result: CommandResult

    @property
    def passed(self) -> bool:
        return self.command_result.exit_code == 0 and not self.command_result.timed_out


@dataclass(frozen=True)
class ValidationReport:
    """Aggregated deterministic validation result."""

    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results if result.check.required)

    @property
    def failed_results(self) -> tuple[CheckResult, ...]:
        return tuple(
            result for result in self.results if result.check.required and not result.passed
        )

    def failure_summary(self) -> str:
        if self.passed:
            return ""

        parts: list[str] = []

        for result in self.failed_results:
            command = " ".join(result.check.command)

            if result.command_result.timed_out:
                reason = "timed out"
            else:
                reason = f"exit code {result.command_result.exit_code}"

            parts.append(f"{result.check.name}: {command} ({reason})")

        return "; ".join(parts)
