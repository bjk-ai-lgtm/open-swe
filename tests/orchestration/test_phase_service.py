from typing import cast

from agent.orchestration.coordinator import (
    SpecialistExecutionResult,
)
from agent.orchestration.server_bridge import (
    PreparedRun,
    RoutedOrchestrationService,
)
from agent.orchestration.specialist_executor import (
    OpenSWESpecialistExecutor,
)
from agent.routing import SpecialistRole
from agent.validation import (
    CommandResult,
    ValidationCheck,
)
from agent.validation.runner import (
    CommandRunner,
)

CHECK = ValidationCheck(
    name="unit",
    command=("pytest", "-q"),
)


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "thread_id": thread_id,
                "work_dir": work_dir,
                "task": task,
                "role": role,
                "model_id": model_id,
                "attempt": attempt,
                "escalation_level":
                    escalation_level,
                "previous_failure":
                    previous_failure,
            }
        )

        return SpecialistExecutionResult(
            success=True,
            summary="done",
        )


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[
            tuple[ValidationCheck, str]
        ] = []

    async def run(
        self,
        check: ValidationCheck,
        *,
        work_dir: str,
    ) -> CommandResult:
        self.calls.append(
            (
                check,
                work_dir,
            )
        )

        return CommandResult(
            exit_code=0,
        )


async def test_phase_service_exposes_runtime_boundaries() -> None:
    fake_executor = FakeExecutor()
    runner = FakeRunner()

    guard_calls = 0
    preparation_calls: list[str] = []
    publication_calls: list[
        dict[str, str]
    ] = []

    def guard() -> None:
        nonlocal guard_calls
        guard_calls += 1

    async def preparer(
        work_dir: str,
    ) -> PreparedRun:
        preparation_calls.append(
            work_dir
        )

        return PreparedRun(
            work_dir=(
                "/workspace/open-swe"
            ),
            checks=(CHECK,),
        )

    async def runner_factory(
        thread_id: str,
    ) -> CommandRunner:
        assert thread_id == "thread-1"
        return runner

    async def publisher(
        *,
        thread_id: str,
        task: str,
        work_dir: str,
    ) -> None:
        publication_calls.append(
            {
                "thread_id": thread_id,
                "task": task,
                "work_dir": work_dir,
            }
        )

    service = RoutedOrchestrationService(
        thread_id="thread-1",
        executor=cast(
            OpenSWESpecialistExecutor,
            fake_executor,
        ),
        checks=(),
        runner_factory=runner_factory,
        run_preparer=preparer,
        publisher=publisher,
        execution_guard=guard,
    )

    prepared = await service.prepare_run(
        "/workspace"
    )

    assert guard_calls == 1
    assert preparation_calls == [
        "/workspace"
    ]

    assert prepared == PreparedRun(
        work_dir="/workspace/open-swe",
        checks=(CHECK,),
    )

    execution = await service.execute_attempt(
        task="Implement API",
        work_dir=prepared.work_dir,
        role=SpecialistRole.BACKEND,
        model_id="openai:gpt-5.6-terra",
        attempt=1,
        escalation_level=0,
        previous_failure=None,
    )

    assert execution.success is True
    assert execution.summary == "done"

    assert fake_executor.calls == [
        {
            "thread_id": "thread-1",
            "work_dir":
                "/workspace/open-swe",
            "task": "Implement API",
            "role":
                SpecialistRole.BACKEND,
            "model_id":
                "openai:gpt-5.6-terra",
            "attempt": 1,
            "escalation_level": 0,
            "previous_failure": None,
        }
    ]

    report = await service.validate_attempt(
        work_dir=prepared.work_dir,
        checks=prepared.checks,
    )

    assert report.passed is True

    assert runner.calls == [
        (
            CHECK,
            "/workspace/open-swe",
        )
    ]

    await service.publish_task(
        task="Implement API",
        work_dir=prepared.work_dir,
    )

    assert publication_calls == [
        {
            "thread_id": "thread-1",
            "task": "Implement API",
            "work_dir":
                "/workspace/open-swe",
        }
    ]


async def test_phase_service_publish_without_publisher_is_noop() -> None:
    service = RoutedOrchestrationService(
        thread_id="thread-1",
        executor=cast(
            OpenSWESpecialistExecutor,
            FakeExecutor(),
        ),
        checks=(),
        execution_guard=lambda: None,
    )

    await service.publish_task(
        task="Research task",
        work_dir="/workspace",
    )
