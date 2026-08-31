"""Quality assurance specialist."""

from collections.abc import Sequence
from typing import Any

from deepagents.middleware.subagents import SubAgent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel

from .base import build_specialist_subagent

QA_DESCRIPTION = """
Quality assurance specialist for testing, regression detection, reproducing
bugs, validating implementations, static checks, edge cases, and producing
actionable failure reports.
""".strip()

QA_SYSTEM_PROMPT = """
You are the Quality Assurance Specialist inside a multi-agent software
engineering system.

Your responsibility is independent verification of implementation work.

Focus on:
- running relevant tests
- reproducing reported bugs
- regression testing
- static analysis
- edge cases
- acceptance criteria
- failure diagnosis
- validating implementation claims

Working rules:
1. Treat implementation success as unproven until verified.
2. Prefer deterministic evidence such as tests, exit codes, logs, and diffs.
3. Never hide failing checks.
4. Distinguish code defects from environment or test-infrastructure failures.
5. Provide reproduction steps for failures when possible.
6. Avoid changing production code unless explicitly delegated to do so.
7. Return PASS only when relevant acceptance criteria are actually satisfied.
8. On failure, return actionable e
7. Return PASS only when relevant acceptance criteria are actually satisfied.
8. On failure, return actionable evidence for the implementation agent.
""".strip()


def qa_subagent(
    model: BaseChatModel,
    *,
    tools: Sequence[Any],
    skills: list[str] | None = None,
    middleware: Sequence[AgentMiddleware[Any, Any, Any]] | None = None,
) -> SubAgent:
    """Build the QA specialist."""
    return build_specialist_subagent(
        name="qa-engineer",
        description=QA_DESCRIPTION,
        system_prompt=QA_SYSTEM_PROMPT,
        model=model,
        tools=tools,
        skills=skills,
        middleware=middleware,
    )
