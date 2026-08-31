from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel

from agent.specialist_agents import (
    backend_subagent,
    qa_subagent,
    research_subagent,
)


def make_model() -> BaseChatModel:
    return MagicMock(spec=BaseChatModel)


def test_specialist_agents_have_unique_names() -> None:
    model = make_model()

    agents = [
        backend_subagent(model, tools=[]),
        research_subagent(model, tools=[]),
        qa_subagent(model, tools=[]),
    ]

    names = [agent["name"] for agent in agents]

    assert names == [
        "backend-engineer",
        "research-specialist",
        "qa-engineer",
    ]

    assert len(names) == len(set(names))


def test_backend_agent_preserves_dependencies() -> None:
    model = make_model()
    tool = MagicMock()

    agent = backend_subagent(
        model,
        tools=[tool],
        skills=["/skills/test"],
    )

    assert agent["model"] is model
    assert agent["tools"] == [tool]
    assert agent["skills"] == ["/skills/test"]


def test_specialist_prompts_are_role_specific() -> None:
    model = make_model()

    backend = backend_subagent(model, tools=[])
    research = research_subagent(model, tools=[])
    qa = qa_subagent(model, tools=[])

    assert "Backend Engineering Specialist" in backend["system_prompt"]
    assert "Technical Research Specialist" in research["system_prompt"]
    assert "Quality Assurance Specialist" in qa["system_prompt"]


def test_research_agent_requires_evidence() -> None:
    model = make_model()
    agent = research_subagent(model, tools=[])

    assert "official sources" in agent["system_prompt"]
    assert "Never fabricate" in agent["system_prompt"]


def test_qa_agent_requires_verification() -> None:
    model = make_model()
    agent = qa_subagent(model, tools=[])

    assert "unproven until verified" in agent["system_prompt"]
    assert "Return PASS only" in agent["system_prompt"]
