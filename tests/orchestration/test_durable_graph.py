from collections import deque
from collections.abc import Sequence

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from agent.orchestration.coordinator import (
    SpecialistExecutionResult,
)
from agent.orchestration.durable_graph import (
    build_durable_orchestrator_graph,
)
from agent.orchestration.server_bridge import (
    PreparedRun,
)
from agent.routing import SpecialistRole
from agent.validation import (
    CheckResult,
    CommandResult,
    ValidationCheck,
    ValidationReport,
)

CHECK = ValidationCheck(
    name="pytest",
    command=("pytest", "-q"),
)


def report(
    *,
    exit_code: int,
) -> ValidationReport:
    return ValidationReport(
        results=(
            CheckResult(
                check=CHECK,
                command_result=CommandResult(
                    exit_code=exit_code,
                ),
            ),
        )
    )


class FakePhaseService:
    def __init__(
        self,
        *,
        executions: Sequence[
            SpecialistExecutionResult
        ],
        reports: Sequence[
            ValidationReport
        ] = (),
    ) -> None:
        self.executions = deque(
            executions
        )

        self.reports = deque(
            reports
        )

        self.events: list[str] = []

        self.execute_calls: list[
            dict[str, object]
        ] = []

        self.validation_calls = 0
        self.publication_calls = 0

    async def prepare_run(
        self,
        work_dir: str,
    ) -> PreparedRun:
        self.events.append(
            "prepare"
        )

        assert work_dir == "/workspace"

        return PreparedRun(
            work_dir=(
                "/workspace/open-swe"
            ),
            checks=(CHECK,),
        )

    async def execute_attempt(
        self,
        *,
        task: str,
        work_dir: str,
        role: SpecialistRole,
        model_id: str,
        attempt: int,
        escalation_level: int,
        previous_failure: str | None,
    ) -> SpecialistExecutionResult:
        self.events.append(
            "execute"
        )

        self.execute_calls.append(
            {
                "task": task,
                "work_dir": work_dir,
                "role": role,
                "model_id": model_id,
                "attempt": attempt,
                "escalation_level":
                    escalation_level,
                "previous_failure":
                    previous_failure,
            }
        )

        return self.executions.popleft()

    async def validate_attempt(
        self,
        *,
        work_dir: str,
        checks: Sequence[
            ValidationCheck
        ],
    ) -> ValidationReport:
        self.events.append(
            "validate"
        )

        self.validation_calls += 1

        assert work_dir == (
            "/workspace/open-swe"
        )

        assert tuple(checks) == (
            CHECK,
        )

        return self.reports.popleft()

    async def publish_task(
        self,
        *,
        task: str,
        work_dir: str,
    ) -> None:
        self.events.append(
            "publish"
        )

        self.publication_calls += 1

        assert task
        assert work_dir == (
            "/workspace/open-swe"
        )


async def test_durable_graph_exposes_phase_nodes() -> None:
    service = FakePhaseService(
        executions=(
            SpecialistExecutionResult(
                success=True,
            ),
        ),
    )

    graph = build_durable_orchestrator_graph(
        service=service,
        work_dir="/workspace",
    )

    nodes = set(
        graph.get_graph().nodes
    )

    assert {
        "prepare",
        "begin_attempt",
        "execute",
        "validate",
        "publish",
        "finalize",
    }.issubset(nodes)


async def test_durable_graph_runs_validated_task() -> None:
    service = FakePhaseService(
        executions=(
            SpecialistExecutionResult(
                success=True,
                summary="done",
            ),
        ),
        reports=(
            report(
                exit_code=0
            ),
        ),
    )

    graph = build_durable_orchestrator_graph(
        service=service,
        work_dir="/workspace",
    )

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Implement a REST "
                        "API backed by "
                        "the database."
                    )
                )
            ]
        }
    )

    assert service.events == [
        "prepare",
        "execute",
        "validate",
        "publish",
    ]

    assert result[
        "orchestration_status"
    ] == "succeeded"

    assert result[
        "orchestration_attempts"
    ] == 1

    assert result[
        "orchestration_phase"
    ] == "finished"

    assert result[
        "durable_execution"
    ]["status"] == "succeeded"

    message = result[
        "messages"
    ][-1]

    assert isinstance(
        message,
        AIMessage,
    )

    assert message.content == "done"


async def test_durable_graph_retries_execution_failure() -> None:
    service = FakePhaseService(
        executions=(
            SpecialistExecutionResult(
                success=False,
                failure_reason="boom",
            ),
            SpecialistExecutionResult(
                success=True,
                summary="recovered",
            ),
        ),
        reports=(
            report(
                exit_code=0
            ),
        ),
    )

    graph = build_durable_orchestrator_graph(
        service=service,
        work_dir="/workspace",
    )

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Implement an API "
                        "endpoint."
                    )
                )
            ]
        }
    )

    assert result[
        "orchestration_status"
    ] == "succeeded"

    assert result[
        "orchestration_attempts"
    ] == 2

    assert len(
        service.execute_calls
    ) == 2

    assert service.execute_calls[
        0
    ]["attempt"] == 1

    assert service.execute_calls[
        1
    ]["attempt"] == 2

    assert service.execute_calls[
        1
    ]["previous_failure"] == (
        "boom"
    )

    assert service.validation_calls == 1
    assert service.publication_calls == 1


async def test_durable_graph_retries_after_validation_failure() -> None:
    service = FakePhaseService(
        executions=(
            SpecialistExecutionResult(
                success=True,
                summary="first",
            ),
            SpecialistExecutionResult(
                success=True,
                summary="second",
            ),
        ),
        reports=(
            report(
                exit_code=1
            ),
            report(
                exit_code=0
            ),
        ),
    )

    graph = build_durable_orchestrator_graph(
        service=service,
        work_dir="/workspace",
    )

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Implement an API "
                        "and database "
                        "endpoint."
                    )
                )
            ]
        }
    )

    assert result[
        "orchestration_status"
    ] == "succeeded"

    assert result[
        "orchestration_attempts"
    ] == 2

    assert service.validation_calls == 2
    assert service.publication_calls == 1

    assert service.events == [
        "prepare",
        "execute",
        "validate",
        "execute",
        "validate",
        "publish",
    ]


async def test_durable_graph_skips_validation_when_not_required() -> None:
    service = FakePhaseService(
        executions=(
            SpecialistExecutionResult(
                success=True,
                summary="research done",
            ),
        ),
    )

    graph = build_durable_orchestrator_graph(
        service=service,
        work_dir="/workspace",
    )

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Research dependency "
                        "compatibility."
                    )
                )
            ]
        }
    )

    assert result[
        "orchestration_status"
    ] == "succeeded"

    assert service.validation_calls == 0
    assert service.publication_calls == 1

    assert service.events == [
        "prepare",
        "execute",
        "publish",
    ]


async def test_durable_graph_rejects_missing_human_task() -> None:
    service = FakePhaseService(
        executions=(),
    )

    graph = build_durable_orchestrator_graph(
        service=service,
        work_dir="/workspace",
    )

    result = await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="No task"
                )
            ]
        }
    )

    assert service.events == []

    assert result[
        "orchestration_status"
    ] == "invalid-input"

    assert result[
        "orchestration_attempts"
    ] == 0

    assert result[
        "orchestration_phase"
    ] == "finished"
