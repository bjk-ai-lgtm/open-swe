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


async def test_required_validation_rejects_empty_checks() -> None:
    executor = SuccessfulExecutor()

    try:
        await run_orchestrated_task(
            thread_id="thread-empty-validation",
            task=("Implement a REST API endpoint backed by the database."),
            work_dir="/workspace/project",
            executor=executor,
            checks=(),
        )
    except ValueError as exc:
        assert str(exc) == ("At least one validation check is required for this task")
    else:
        raise AssertionError("Expected required validation to reject empty checks")

    assert executor.calls == []

async def test_successful_task_is_published_only_after_validation() -> None:
    events = []

    class Executor:
        async def execute(self, **kwargs):
            del kwargs
            events.append("execute")
            return SpecialistExecutionResult(
                success=True,
                summary="done",
            )

    class RecordingRunner:
        async def run(self, check, *, work_dir):
            del check, work_dir
            events.append("validate")
            return CommandResult(
                exit_code=0,
                stdout="ok",
            )

    async def publisher(**kwargs):
        del kwargs
        events.append("publish")

    result = await run_orchestrated_task(
        thread_id="thread-publish",
        task="Implement a REST API endpoint backed by the database.",
        work_dir="/workspace/project",
        executor=Executor(),
        checks=CHECKS,
        runner_factory=runner_factory(RecordingRunner()),
        publisher=publisher,
    )

    assert result.state.status is TaskStatus.SUCCEEDED
    assert events == [
        "execute",
        "validate",
        "publish",
    ]


async def test_failed_task_is_never_published() -> None:
    published = []

    class Executor:
        async def execute(self, **kwargs):
            del kwargs
            return SpecialistExecutionResult(
                success=False,
                failure_reason="implementation failed",
            )

    async def publisher(**kwargs):
        published.append(kwargs)

    result = await run_orchestrated_task(
        thread_id="thread-no-publish",
        task="Implement a REST API endpoint backed by the database.",
        work_dir="/workspace/project",
        executor=Executor(),
        checks=CHECKS,
        runner_factory=runner_factory(SequencedRunner([])),
        publisher=publisher,
    )

    assert result.state.status is TaskStatus.QUARANTINED
    assert published == []

async def test_publication_failure_is_quarantined() -> None:
    events = []

    class Executor:
        async def execute(self, **kwargs):
            del kwargs
            events.append("execute")
            return SpecialistExecutionResult(
                success=True,
                summary="done",
            )

    class RecordingRunner:
        async def run(self, check, *, work_dir):
            del check, work_dir
            events.append("validate")
            return CommandResult(
                exit_code=0,
                stdout="ok",
            )

    async def publisher(**kwargs):
        del kwargs
        events.append("publish")
        raise RuntimeError("remote unavailable")

    result = await run_orchestrated_task(
        thread_id="thread-publish-fail",
        task="Implement a REST API endpoint backed by the database.",
        work_dir="/workspace/project",
        executor=Executor(),
        checks=CHECKS,
        runner_factory=runner_factory(RecordingRunner()),
        publisher=publisher,
    )

    assert result.state.status is TaskStatus.QUARANTINED
    assert result.state.last_failure == (
        "Publication failed: RuntimeError: remote unavailable"
    )
    assert events == [
        "execute",
        "validate",
        "publish",
    ]
