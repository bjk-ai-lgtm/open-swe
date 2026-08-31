"""Shared builder for Open SWE specialist subagents."""

from collections.abc import Sequence
from typing import Any

from deepagents.middleware.subagents import SubAgent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel


def build_specialist_subagent(
    *,
    name: str,
    description: str,
    system_prompt: str,
    model: BaseChatModel,
    tools: Sequence[Any],
    skills: list[str] | None = None,
    middleware: Sequence[AgentMiddleware[Any, Any, Any]] | None = None,
) -> SubAgent:
    """Create a specialist subagent using the Deep Agents SubAgent schema."""
    if not name.strip():
        raise ValueError("Specialist agent name cannot be empty")

    if not description.strip():
        raise ValueError("Specialist agent description cannot be empty")

    if not system_prompt.strip():
        raise ValueError("Specialist agent system prompt cannot be empty")
    subagent: SubAgent = {
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        "model": model,
        "tools": list(tools),
        "middleware": list(middleware or []),
    }

    if skills:
        subagent["skills"] = skills

    return subagent
