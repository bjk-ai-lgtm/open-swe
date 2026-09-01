import pytest
from langchain_core.messages import AIMessage

from agent.orchestration.bootstrap import (
    OrchestratorRuntimeContext,
)
from agent.orchestration.execution_safety import (
    UnsafeExecutionEnvironmentError,
)
from agent.orchestration.runtime_service import (
    build_runtime_orchestration_service,
)
from agent.validation import (
    CommandResult,
    ValidationCheck,
)


class FakeAgent:
    async def ainvoke(self, state, config):
        return {
            "messages": [
                AIMessage(content="done"),
            ]
        }


class FakeRunner:
    async def run(self, check, *, work_dir):
        return CommandResult(
            exit_code=0,
            stdout="passed",
        )


async def test_runtime_service_uses_bootstrapped_context() -> None:
    backend = object()
    captured_models = []

    context = OrchestratorRuntimeContext(
        thread_id="thread-runtime",
        sandbox_backend=backend,
        work_dir="/workspace/project",
    )

    def model_factory(model_id):
        captured_models.append(model_id)
        return object()

    def agent_factory(**kwargs):
        assert kwargs["backend"] is backend
        return FakeAgent()

    async def runner_factory(thread_id):
        assert thread_id == "thread-runtime"
        return FakeRunner()

    service = build_runtime_orchestration_service(
        context,
        tools=[],
        model_factory=model_factory,
        agent_factory=agent_factory,
        runner_factory=runner_factory,
        execution_guard=lambda: None,
        checks=(
            ValidationCheck(
                name="tests",
                command=("pytest", "-q"),
            ),
        ),
    )

    assert service.thread_id == "thread-runtime"

    result = await service.run(
        task=("Implement a REST API backed by the database."),
        work_dir=context.work_dir,
    )

    assert result.state.status.value == "succeeded"

    assert captured_models == [
        "openai:gpt-5.6-terra",
    ]


def test_runtime_service_preserves_exact_backend() -> None:
    backend = object()

    context = OrchestratorRuntimeContext(
        thread_id="thread-backend",
        sandbox_backend=backend,
        work_dir="/workspace/project",
    )

    service = build_runtime_orchestration_service(
        context,
        tools=[],
        checks=(),
        model_factory=lambda model_id: object(),
    )

    assert service.executor._backend is backend


async def test_runtime_service_blocks_local_before_model(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SANDBOX_TYPE", "local")

    backend = object()
    captured_models = []

    context = OrchestratorRuntimeContext(
        thread_id="thread-local-blocked",
        sandbox_backend=backend,
        work_dir="/workspace/project",
    )

    def model_factory(model_id):
        captured_models.append(model_id)
        return object()

    service = build_runtime_orchestration_service(
        context,
        tools=[],
        checks=(),
        model_factory=model_factory,
    )

    with pytest.raises(
        UnsafeExecutionEnvironmentError,
        match="SANDBOX_TYPE=local",
    ):
        await service.run(
            task="Implement a REST API backed by the database.",
            work_dir=context.work_dir,
        )

    assert captured_models == []
