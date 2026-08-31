from langchain_core.messages import AIMessage

from agent.orchestration.specialist_executor import OpenSWESpecialistExecutor
from agent.routing import SpecialistRole


class FakeAgent:
    def __init__(self) -> None:
        self.calls = []

    async def ainvoke(self, state, config):
        self.calls.append((state, config))

        return {
            "messages": [
                AIMessage(content="Implementation complete"),
            ]
        }


async def test_executor_uses_coordinator_selected_model() -> None:
    backend = object()
    fake_agent = FakeAgent()

    model_calls = []
    agent_calls = []

    fake_model = object()

    def model_factory(model_id):
        model_calls.append(model_id)
        return fake_model

    def agent_factory(**kwargs):
        agent_calls.append(kwargs)
        return fake_agent

    executor = OpenSWESpecialistExecutor(
        backend=backend,
        tools=[],
        model_factory=model_factory,
        agent_factory=agent_factory,
    )

    result = await executor.execute(
        thread_id="thread-1",
        work_dir="/workspace/project",
        task="Implement a REST API endpoint backed by the database.",
        role=SpecialistRole.BACKEND,
        model_id="openai:gpt-5.6-sol",
        attempt=4,
        escalation_level=1,
        previous_failure="pytest failed",
    )

    assert result.success is True
    assert result.summary == "Implementation complete"

    assert model_calls == [
        "openai:gpt-5.6-sol",
    ]

    assert len(agent_calls) == 1
    assert agent_calls[0]["model"] is fake_model
    assert agent_calls[0]["backend"] is backend

    assert "Backend Engineering Specialist" in (agent_calls[0]["system_prompt"])

    state, config = fake_agent.calls[0]

    prompt = state["messages"][0].content

    assert "Attempt: 4" in prompt
    assert "Capability escalation level: 1" in prompt
    assert "pytest failed" in prompt

    assert config["metadata"]["open_swe_specialist_role"] == ("backend-engineer")

    assert config["metadata"]["open_swe_model_id"] == ("openai:gpt-5.6-sol")


async def test_executor_supports_general_fallback_role() -> None:
    fake_agent = FakeAgent()

    def model_factory(model_id):
        return object()

    def agent_factory(**kwargs):
        assert kwargs["system_prompt"]
        return fake_agent

    executor = OpenSWESpecialistExecutor(
        backend=object(),
        tools=[],
        model_factory=model_factory,
        agent_factory=agent_factory,
    )

    result = await executor.execute(
        thread_id="thread-general",
        work_dir="/workspace/project",
        task="Help improve this project.",
        role=SpecialistRole.GENERAL,
        model_id="openai:gpt-5.6-sol",
        attempt=1,
        escalation_level=0,
        previous_failure=None,
    )

    assert result.success is True


async def test_executor_blocks_specialist_subdelegation() -> None:
    captured = {}

    def model_factory(model_id):
        return object()

    class FakeAgent:
        async def ainvoke(self, state, config):
            return {"messages": []}

    def agent_factory(**kwargs):
        captured.update(kwargs)
        return FakeAgent()

    executor = OpenSWESpecialistExecutor(
        backend=object(),
        tools=[],
        model_factory=model_factory,
        agent_factory=agent_factory,
    )

    await executor.execute(
        thread_id="thread-guard",
        work_dir="/workspace/project",
        task="Implement an API.",
        role=SpecialistRole.BACKEND,
        model_id="openai:gpt-5.6-terra",
        attempt=1,
        escalation_level=0,
        previous_failure=None,
    )

    middleware_names = {type(middleware).__name__ for middleware in captured["middleware"]}

    assert "SpecialistNoDelegationMiddleware" in middleware_names
