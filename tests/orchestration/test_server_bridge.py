from langchain_core.messages import AIMessage

from agent.orchestration.server_bridge import (
    build_server_orchestration_service,
)
from agent.validation import (
    CommandResult,
    ValidationCheck,
)


class FakeAgent:
    def __init__(self, captured):
        self.captured = captured

    async def ainvoke(self, state, config):
        self.captured["invoke_state"] = state
        self.captured["invoke_config"] = config

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


async def test_server_bridge_runs_complete_pipeline() -> None:
    captured = {}
    backend = object()

    def model_factory(model_id):
        captured["model_id"] = model_id
        return object()

    def agent_factory(**kwargs):
        captured["agent_kwargs"] = kwargs
        return FakeAgent(captured)

    async def runner_factory(thread_id):
        captured["runner_thread_id"] = thread_id
        return FakeRunner()

    service = build_server_orchestration_service(
        thread_id="thread-server",
        backend=backend,
        tools=[],
        skills=["/skills/project/"],
        middleware=[],
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

    result = await service.run(
        task="Implement a REST API backed by the database.",
        work_dir="/workspace/project",
    )

    assert result.state.status.value == "succeeded"
    assert result.state.attempt == 1

    assert captured["model_id"] == "openai:gpt-5.6-terra"
    assert captured["runner_thread_id"] == "thread-server"

    assert captured["agent_kwargs"]["backend"] is backend
    assert captured["agent_kwargs"]["skills"] == [
        "/skills/project/",
    ]

    config = captured["invoke_config"]

    assert config["configurable"]["thread_id"] == ("thread-server")


async def test_server_bridge_preserves_retry_model_routing() -> None:
    model_ids = []

    class Agent:
        async def ainvoke(self, state, config):
            return {"messages": []}

    def model_factory(model_id):
        model_ids.append(model_id)
        return object()

    def agent_factory(**kwargs):
        return Agent()

    class Runner:
        def __init__(self):
            self.calls = 0

        async def run(self, check, *, work_dir):
            self.calls += 1

            return CommandResult(
                exit_code=0 if self.calls == 4 else 1,
            )

    runner = Runner()

    async def runner_factory(thread_id):
        return runner

    service = build_server_orchestration_service(
        thread_id="thread-retry",
        backend=object(),
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

    result = await service.run(
        task="Implement a REST API backed by the database.",
        work_dir="/workspace/project",
    )

    assert result.state.status.value == "succeeded"

    assert model_ids == [
        "openai:gpt-5.6-terra",
        "openai:gpt-5.6-terra",
        "openai:gpt-5.6-terra",
        "openai:gpt-5.6-sol",
    ]


async def test_server_bridge_runs_execution_guard_before_coordinator() -> None:
    calls = []

    def execution_guard():
        calls.append("guard")
        raise RuntimeError("blocked")

    def model_factory(model_id):
        calls.append(("model", model_id))
        return object()

    service = build_server_orchestration_service(
        thread_id="thread-guard",
        backend=object(),
        tools=[],
        model_factory=model_factory,
        execution_guard=execution_guard,
        checks=(),
    )

    try:
        await service.run(
            task="Implement a REST API backed by the database.",
            work_dir="/workspace/project",
        )
    except RuntimeError as exc:
        assert str(exc) == "blocked"
    else:
        raise AssertionError("execution guard must block the run")

    assert calls == ["guard"]
