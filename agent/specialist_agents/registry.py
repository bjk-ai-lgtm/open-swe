"""Registry for the custom Open SWE specialist agents."""

from collections.abc import Sequence
from typing import Any

from deepagents.middleware.subagents import SubAgent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel

from .backend import backend_subagent
from .qa import qa_subagent
from .research import research_subagent


def build_v01_specialists(
    model: BaseChatModel,
    *,
    tools: Sequence[Any],
    skills: list[str] | None = None,
    middleware: Sequence[AgentMiddleware[Any, Any, Any]] | None = None,
) -> list[SubAgent]:
    """Build the specialist set enabled in Open SWE Custom v0.1."""
    return [
        backend_subagent(
            model,
            tools=tools,
            skills=skills,
            middleware=middleware,
        ),
        research_subagent(
            model,
            tools=tools,
            skills=skills,
            middleware=middleware,
        ),
        qa_subagent(
            model,
            tools=tools,
            skills=skills,
            middleware=middleware,
        ),
    ]
