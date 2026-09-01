from langchain_core.messages import AIMessage, HumanMessage

from agent.orchestration.coordinator import (
    AttemptRecord,
    CoordinatorResult,
    SpecialistExecutionResult,
)
from agent.orchestration.graph import build_orchestrator_graph
from agent.orchestration.state import ExecutionState, TaskStatus
from agent.routing import SpecialistRole, build_execution_plan


def make_result(
    *,
    status: TaskStatus,
    summary: str = "",
    attempt: int = 1,
    escalation_level: int = 0,
    last_failure: str | None = None,
) -> CoordinatorResult:
    plan = build_execution_plan("Implement a REST API backed by the database.")

    state = ExecutionState(
        plan=plan,
        status=status,
        attempt=attempt,
        escalation_level=escalation_level,
        last_failure=last_failure,
    )

    record = AttemptRecord(
        attempt=attempt,
        role=SpecialistRole.BACKEND,
        model_id="openai:gpt-5.6-terra",
        escalation_level=escalation_level,
        execution=SpecialistExecutionResult(
            success=status is TaskStatus.SUCCEEDED,
            summary=summary,
            failure_reason=last_failure,
        ),
        validation=None,
    )

    return CoordinatorResult(
        state=state,
        attempts=(record,),
    )


class FakeService:
    def __init__(self, result: CoordinatorResult) -> None:
        self.result = result
        self.calls = []

    async def run(self, *, task, work_dir):
        self.calls.append(
            {
                "task": task,
                "work_dir": work_dir,
            }
        )

        return self.result


async def test_graph_runs_successful_task() -> None:
    service = FakeService(
        make_result(
            status=TaskStatus.SUCCEEDED,
            summary="Implementation complete.",
        )
    )

    graph = build_orchestrator_graph(
        service=service,
        work_dir="/workspace/project",
    )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=("Implement a REST API backed by the database."))]}
    )

    assert service.calls == [
        {
            "task": ("Implement a REST API backed by the database."),
            "work_dir": "/workspace/project",
        }
    ]

    assert result["orchestration_status"] == "succeeded"
    assert result["orchestration_attempts"] == 1
    assert result["messages"][-1].content == ("Implementation complete.")


async def test_graph_exposes_quarantine_state() -> None:
    service = FakeService(
        make_result(
            status=TaskStatus.QUARANTINED,
            attempt=5,
            escalation_level=2,
            last_failure="tests failed",
        )
    )

    graph = build_orchestrator_graph(
        service=service,
        work_dir="/workspace/project",
    )

    result = await graph.ainvoke({"messages": [HumanMessage(content="Implement an API.")]})

    assert result["orchestration_status"] == "quarantined"
    assert result["orchestration_attempts"] == 5
    assert result["orchestration_escalation_level"] == 2
    assert result["orchestration_last_failure"] == ("tests failed")

    assert "tests failed" in result["messages"][-1].content


async def test_graph_uses_latest_human_task() -> None:
    service = FakeService(
        make_result(
            status=TaskStatus.SUCCEEDED,
            summary="done",
        )
    )

    graph = build_orchestrator_graph(
        service=service,
        work_dir="/workspace/project",
    )

    await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="Old task"),
                AIMessage(content="Old response"),
                HumanMessage(content="Latest task"),
            ]
        }
    )

    assert service.calls == [
        {
            "task": "Latest task",
            "work_dir": "/workspace/project",
        }
    ]


async def test_graph_rejects_missing_human_task() -> None:
    service = FakeService(
        make_result(
            status=TaskStatus.SUCCEEDED,
        )
    )

    graph = build_orchestrator_graph(
        service=service,
        work_dir="/workspace/project",
    )

    result = await graph.ainvoke(
        {
            "messages": [
                AIMessage(content="No task here"),
            ]
        }
    )

    assert service.calls == []
    assert result["orchestration_status"] == "invalid-input"
    assert result["orchestration_attempts"] == 0
