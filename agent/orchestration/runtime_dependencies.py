"""Runtime dependencies exposed to routed specialist agents."""

from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langgraph.graph.state import RunnableConfig

from agent.tools import fetch_url, http_request, web_search

from .bootstrap import OrchestratorRuntimeContext


@dataclass(frozen=True)
class SpecialistRuntimeDependencies:
    """Open SWE capabilities available to one routed specialist."""

    tools: tuple[Any, ...] = ()
    skills: tuple[str, ...] = ()
    middleware: tuple[
        AgentMiddleware[Any, Any, Any],
        ...,
    ] = ()
    use_gateway: bool | None = None
    model_effort: str | None = None


def build_specialist_runtime_dependencies(
    config: RunnableConfig,
    context: OrchestratorRuntimeContext,
) -> SpecialistRuntimeDependencies:
    """Build the minimal safe Open SWE specialist environment."""
    del config, context

    return SpecialistRuntimeDependencies(
        tools=(
            http_request,
            fetch_url,
            web_search,
        ),
    )
