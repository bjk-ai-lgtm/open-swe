"""Backend engineering specialist."""

from collections.abc import Sequence
from typing import Any

from deepagents.middleware.subagents import SubAgent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel

from .base import build_specialist_subagent

BACKEND_DESCRIPTION = """
Backend engineering specialist for server-side implementation, APIs,
databases, business logic, data models, integrations, debugging, migrations,
performance, and backend-focused tests.
""".strip()

BACKEND_SYSTEM_PROMPT = """
You are the Backend Engineering Specialist inside a multi-agent software
engineering system.

Your responsibility is backend implementation and diagnosis.

Focus on:
- server-side application code
- API design and implementation
- database access and data models
- business logic
- integrations
- migrations
- backend performance
- backend unit and integration tests
- debugging server-side failures

Working rules:
1. Inspect the repository before modifying code.
2. Preserve existing architecture and conventions unless the task requires change.
3. Prefer the smallest correct implementation.
4. Do not invent APIs, schemas, files, or dependencies without checking the repository.
5. Run relevant tests and static checks after implementation when possible.
6. Report exactly what changed and any remaining uncertainty.
7. Do not claim success when verification failed.
8. Escalate unclear requirements instead of silently guessing.
""".strip()


def backend_subagent(
    model: BaseChatModel,
    *,
    tools: Sequence[Any],
    skills: list[str] | None = None,
    middleware: Sequence[AgentMiddleware[Any, Any, Any]] | None = None,
) -> SubAgent:
    """Build the backend engineering specialist."""
    return build_specialist_subagent(
        name="backend-engineer",
        description=BACKEND_DESCRIPTION,
        system_prompt=BACKEND_SYSTEM_PROMPT,
        model=model,
        tools=tools,
        skills=skills,
        middleware=middleware,
    )
