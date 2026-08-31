from agent.validation import (
    CommandResult,
    ValidationCheck,
    run_validation,
)


class FakeRunner:
    def __init__(
        self,
        results: dict[str, CommandResult],
    ) -> None:
        self.results = results
        self.calls: list[tuple[str, str]] = []

    async def run(
        self,
        check: ValidationCheck,
        *,
        work_dir: str,
    ) -> CommandResult:
        self.calls.append((check.name, work_dir))
        return self.results[check.name]


async def test_all_required_checks_pass() -> None:
    checks = (
        ValidationCheck(
            name="tests",
            command=("pytest", "-q"),
        ),
        ValidationCheck(
            name="ruff",
            command=("ruff", "check", "."),
        ),
    )

    runner = FakeRunner(
        {
            "tests": CommandResult(exit_code=0),
            "ruff": CommandResult(exit_code=0),
        }
    )

    report = await run_validation(
        runner,
        checks,
        work_dir="/workspace/project",
    )

    assert report.passed is True
    assert report.failed_results == ()
    assert runner.calls == [
        ("tests", "/workspace/project"),
        ("ruff", "/workspace/project"),
    ]


async def test_required_failure_fails_report() -> None:
    checks = (
        ValidationCheck(
            name="tests",
            command=("pytest", "-q"),
        ),
        ValidationCheck(
            name="ruff",
            command=("ruff", "check", "."),
        ),
    )

    runner = FakeRunner(
        {
            "tests": CommandResult(
                exit_code=1,
                stderr="1 test failed",
            ),
            "ruff": CommandResult(exit_code=0),
        }
    )

    report = await run_validation(
        runner,
        checks,
        work_dir="/workspace/project",
    )

    assert report.passed is False
    assert len(report.failed_results) == 1
    assert "tests" in report.failure_summary()
    assert "exit code 1" in report.failure_summary()


async def test_optional_failure_does_not_fail_report() -> None:
    checks = (
        ValidationCheck(
            name="tests",
            command=("pytest", "-q"),
        ),
        ValidationCheck(
            name="advisory",
            command=("example", "check"),
            required=False,
        ),
    )

    runner = FakeRunner(
        {
            "tests": CommandResult(exit_code=0),
            "advisory": CommandResult(exit_code=2),
        }
    )

    report = await run_validation(
        runner,
        checks,
        work_dir="/workspace/project",
    )

    assert report.passed is True


async def test_timeout_fails_required_check() -> None:
    checks = (
        ValidationCheck(
            name="tests",
            command=("pytest", "-q"),
        ),
    )

    runner = FakeRunner(
        {
            "tests": CommandResult(
                exit_code=124,
                timed_out=True,
            ),
        }
    )

    report = await run_validation(
        runner,
        checks,
        work_dir="/workspace/project",
    )

    assert report.passed is False
    assert "timed out" in report.failure_summary()
