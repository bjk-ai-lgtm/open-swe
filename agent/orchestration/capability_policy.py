"""Fail-closed capability policy for routed specialists."""

from collections.abc import Sequence
from typing import Any

from agent.routing import SpecialistRole

ROLE_TOOL_ALLOWLIST: dict[
    SpecialistRole,
    frozenset[str],
] = {
    SpecialistRole.BACKEND: frozenset(
        {
            "http_request",
            "fetch_url",
        }
    ),
    SpecialistRole.RESEARCH: frozenset(
        {
            "http_request",
            "fetch_url",
            "web_search",
        }
    ),
    SpecialistRole.QA: frozenset(
        {
            "http_request",
            "fetch_url",
        }
    ),
    SpecialistRole.GENERAL: frozenset(
        {
            "http_request",
            "fetch_url",
            "web_search",
        }
    ),
}


def registered_tool_name(tool: Any) -> str | None:
    """Return the registered name of a runtime tool."""
    name = getattr(tool, "name", None) or getattr(tool, "__name__", None)

    if not isinstance(name, str) or not name:
        return None

    return name


def filter_tools_for_role(
    role: SpecialistRole,
    tools: Sequence[Any],
) -> tuple[Any, ...]:
    """Return only capabilities explicitly allowed for a role."""
    allowed = ROLE_TOOL_ALLOWLIST[role]

    return tuple(tool for tool in tools if registered_tool_name(tool) in allowed)
