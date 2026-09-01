import pytest
from langchain_core.messages import HumanMessage

from agent.orchestration.bootstrap import (
    OrchestratorRuntimeContext,
)
from agent.orchestration.coordinator import (
    AttemptRecord,
    CoordinatorResult,
    SpecialistExecutionResult,
)
from agent.orchestration.factory import get_orchestrator
from agent.orchestration.state import ExecutionState, TaskStatus
from agent.routing import SpecialistRole, build_execution_plan


def successful_result() -> CoordinatorResult:
    plan = build_execution_plan("Implement a REST API backed by the database.")

    state = ExecutionState(
        plan=plan,
        status=TaskStatus.SUCCEEDED,
        attempt=1,
    )

    attempt = AttemptRecord(
        attempt=1,
        role=SpecialistRole.BACKEND,
        model_id="openai:gpt-5.6-terra",
        escalation_level=0,
        execution=SpecialistExecutionResult(
            success=True,
            summary="factory task complete",
        ),
        validation=None,
    )

    return CoordinatorResult(
        state=state,
        attempts=(attempt,),
    )


class FakeService:
    def __init__(self):
        self.calls = []

    async def run(self, *, task, work_dir):
        self.calls.append(
            {
                "task": task,
                "work_dir": work_dir,
            }
        )
        return successful_result()


async def test_factory_builds_execution_graph_from_runtime_context():
    backend = object()

    context = OrchestratorRuntimeContext(
        thread_id="thread-factory",
        sandbox_backend=backend,
        work_dir="/workspace/project",
    )

    service = FakeService()
    captured = {}

    async def bootstrap(config):
        captured["bootstrap_config"] = config
        return context

    def service_factory(received_context, **kwargs):
        captured["context"] = received_context
        captured["kwargs"] = kwargs
        return service

    config = {
        "configurable": {
            "thread_id": "thread-factory",
            "orchestrator_validation_profile": ("open-swe-python"),
        }
    }

    graph = await get_orchestrator(
        config,
        bootstrap=bootstrap,
        service_factory=service_factory,
    )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=("Implement a REST API backed by the database."))]}
    )

    assert captured["context"] is context
    checks = captured["kwargs"]["checks"]
    assert checks

    assert service.calls == [
        {
            "task": ("Implement a REST API backed by the database."),
            "work_dir": "/workspace/project",
        }
    ]

    assert result["orchestration_status"] == "succeeded"
    assert result["messages"][-1].content == ("factory task complete")


async def test_factory_schema_load_does_not_build_runtime_service():
    calls = []

    async def bootstrap(config):
        return None

    def service_factory(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("service factory must not run for schema loads")

    graph = await get_orchestrator(
        {
            "configurable": {},
        },
        bootstrap=bootstrap,
        service_factory=service_factory,
    )

    assert graph is not None
    assert calls == []


async def test_factory_defaults_to_no_validation_profile():
    context = OrchestratorRuntimeContext(
        thread_id="thread-none",
        sandbox_backend=object(),
        work_dir="/workspace/project",
    )

    captured = {}

    async def bootstrap(config):
        return context

    def service_factory(received_context, **kwargs):
        captured["checks"] = kwargs["checks"]
        return FakeService()

    await get_orchestrator(
        {
            "configurable": {
                "thread_id": "thread-none",
            }
        },
        bootstrap=bootstrap,
        service_factory=service_factory,
    )

    assert captured["checks"] == ()


async def test_factory_rejects_unknown_validation_profile():
    async def bootstrap(config):
        raise AssertionError("bootstrap must not run for invalid profile")

    with pytest.raises(
        ValueError,
        match="Unsupported validation profile",
    ):
        await get_orchestrator(
            {
                "configurable": {
                    "thread_id": "thread-invalid",
                    "orchestrator_validation_profile": ("javascript-magic"),
                }
            },
            bootstrap=bootstrap,
        )


async def test_factory_dry_run_does_not_build_service():
    service_calls = []

    async def bootstrap(config):
        raise AssertionError("dry-run must not bootstrap a sandbox")

    def service_factory(*args, **kwargs):
        service_calls.append((args, kwargs))
        raise AssertionError("dry-run must not build execution service")

    graph = await get_orchestrator(
        {
            "configurable": {
                "thread_id": "thread-dry-run",
                "orchestrator_dry_run": True,
            }
        },
        bootstrap=bootstrap,
        service_factory=service_factory,
    )

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content=("Implement a REST API endpoint backed by the database."))
            ]
        }
    )

    assert service_calls == []
    assert result["orchestration_status"] == ("planned")


async def test_factory_passes_runtime_dependencies_to_service():
    from agent.orchestration.runtime_dependencies import (
        SpecialistRuntimeDependencies,
    )

    context = OrchestratorRuntimeContext(
        thread_id="thread-dependencies",
        sandbox_backend=object(),
        work_dir="/workspace/project",
    )

    captured = {}

    async def bootstrap(config):
        return context

    def dependency_factory(config, received_context):
        assert received_context is context

        return SpecialistRuntimeDependencies(
            tools=("tool-a", "tool-b"),
            skills=("skill-a",),
            middleware=("middleware-a",),
            use_gateway=True,
            model_effort="high",
        )

    def service_factory(received_context, **kwargs):
        captured.update(kwargs)
        return FakeService()

    await get_orchestrator(
        {
            "configurable": {
                "thread_id": "thread-dependencies",
            }
        },
        bootstrap=bootstrap,
        dependency_factory=dependency_factory,
        service_factory=service_factory,
    )

    assert captured["tools"] == (
        "tool-a",
        "tool-b",
    )
    assert captured["skills"] == ["skill-a"]
    assert captured["middleware"] == ("middleware-a",)
    assert captured["use_gateway"] is True
    assert captured["model_effort"] == "high"
