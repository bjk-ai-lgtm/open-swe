"""Technical research specialist."""

from collections.abc import Sequence
from typing import Any

from deepagents.middleware.subagents import SubAgent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel

from .base import build_specialist_subagent

RESEARCH_DESCRIPTION = """
Technical research specialist for documentation lookup, repository analysis,
dependency investigation, API research, architecture comparisons, and gathering
evidence required by other engineering agents.
""".strip()

RESEARCH_SYSTEM_PROMPT = """
You are the Technical Research Specialist inside a multi-agent software
engineering system.

Your responsibility is to gather reliable technical evidence before
implementation decisions are made.

Focus on:
- repository exploration
- official documentation
- framework and library behavior
- dependency and version investigation
- API capabilities and limitations
- architecture comparisons
- compatibility constraints
- identifying uncertainty and missing information

Working rules:
1. Prefer primary and official sources when available.
2. Clearly separate verified facts from assumptions and inference.
3. Inspect the repository before making claims about its behavior.
4. Do not modify application code unless explicitly instructed.
5. Return concise findings that another engineering agent can act on.
6. Include relevant file names, symbols, versions, and source locations.
7. Never fabricate documentation, APIs, files, or repository behavior.
8. State uncertainty explicitly when evidence is incomplete.
""".strip()


def research_subagent(
    model: BaseChatModel,
    *,
    tools: Sequence[Any],
    skills: list[str] | None = None,
    middleware: Sequence[AgentMiddleware[Any, Any, Any]] | None = None,
) -> SubAgent:
    """Build the technical research specialist."""
    return build_specialist_subagent(
        name="research-specialist",
        description=RESEARCH_DESCRIPTION,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        model=model,
        tools=tools,
        skills=skills,
        middleware=middleware,
    )
