from agent.orchestration import TaskStatus
from agent.orchestration.coordinator import (
    SpecialistExecutionResult,
    run_orchestrated_task,
)
from agent.validation import (
    CommandResult,
    ValidationCheck,
)

CHECKS = (
    ValidationCheck(
        name="tests",
        command=("pytest", "-q"),
    ),
)


class SuccessfulExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return SpecialistExecutionResult(
            success=True,
            summary="implementation complete",
        )


class FailOnceExecutor(SuccessfulExecutor):
    async def execute(self, **kwargs):
        self.calls.append(kwargs)

        if len(self.calls) == 1:
            raise RuntimeError("temporary specialist failure")

        return SpecialistExecutionResult(
            success=True,
            summary="recovered",
        )


class SequencedRunner:
    def __init__(self, exit_codes: list[int]) -> None:
        self.exit_codes = list(exit_codes)

    async def run(self, check, *, work_dir):
        if not self.exit_codes:
            raise AssertionError("No fake validation result remaining")

        exit_code = self.exit_codes.pop(0)

        return CommandResult(
            exit_code=exit_code,
            stdout="pass" if exit_code == 0 else "fail",
        )


def runner_factory(runner):
    async def factory(thread_id):
        return runner

    return factory


async def test_happy_path_completes_backend_task() -> None:
    executor = SuccessfulExecutor()
    runner = SequencedRunner([0])

    result = await run_orchestrated_task(
        thread_id="thread-1",
        task="Implement a REST API endpoint backed by the database.",
        work_dir="/workspace/project",
        executor=executor,
        checks=CHECKS,
        runner_factory=runner_factory(runner),
    )

    assert result.state.status is TaskStatus.SUCCEEDED
    assert len(result.attempts) == 1
    assert result.attempts[0].model_id == "openai:gpt-5.6-terra"


async def test_validation_failures_drive_full_escalation_chain() -> None:
    executor = SuccessfulExecutor()
    runner = SequencedRunner([1, 1, 1, 1, 1])

    result = await run_orchestrated_task(
        thread_id="thread-2",
        task="Implement a REST API endpoint backed by the database.",
        work_dir="/workspace/project",
        executor=executor,
        checks=CHECKS,
        runner_factory=runner_factory(runner),
    )

    assert result.state.status is TaskStatus.QUARANTINED

    assert [attempt.model_id for attempt in result.attempts] == [
        "openai:gpt-5.6-terra",
        "openai:gpt-5.6-terra",
        "openai:gpt-5.6-terra",
        "openai:gpt-5.6-sol",
        "anthropic:claude-opus-5",
    ]


async def test_executor_exception_is_contained_and_retried() -> None:
    executor = FailOnceExecutor()
    runner = SequencedRunner([0])

    result = await run_orchestrated_task(
        thread_id="thread-3",
        task="Implement a REST API endpoint backed by the database.",
        work_dir="/workspace/project",
        executor=executor,
        checks=CHECKS,
        runner_factory=runner_factory(runner),
    )

    assert result.state.status is TaskStatus.SUCCEEDED
    assert result.state.execution_failures == 1
    assert len(result.attempts) == 2

    assert [attempt.model_id for attempt in result.attempts] == [
        "openai:gpt-5.6-terra",
        "openai:gpt-5.6-terra",
    ]

    assert result.attempts[0].execution.success is False
    assert "RuntimeError" in (result.attempts[0].execution.failure_reason or "")
