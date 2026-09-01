"""Standalone deterministic orchestration graph entrypoint."""

from agent.orchestration.factory import get_orchestrator
from agent.utils.tracing import traced_graph_factory

ORCHESTRATOR_TRACING_PROJECT = "open-swe-orchestrator"

traced_orchestrator = traced_graph_factory(
    get_orchestrator,
    ORCHESTRATOR_TRACING_PROJECT,
)

__all__ = [
    "get_orchestrator",
    "traced_orchestrator",
]
