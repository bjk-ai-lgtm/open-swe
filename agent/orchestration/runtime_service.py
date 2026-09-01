"""Bind bootstrapped Open SWE resources to the orchestration service."""

from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from agent.validation import (
    OPEN_SWE_PYTHON_CHECKS,
    ValidationCheck,
)

from .bootstrap import OrchestratorRuntimeContext
from .server_bridge import (
    ModelFactory,
    RoutedOrchestrationService,
    RunnerFactory,
    build_server_orchestration_service,
)
from .specialist_executor import AgentFactory


def build_runtime_orchestration_service(
    context: OrchestratorRuntimeContext,
    *,
    tools: Sequence[Any] = (),
    skills: list[str] | None = None,
    middleware: Sequence[AgentMiddleware[Any, Any, Any]] | None = None,
    use_gateway: bool | None = None,
    checks: Sequence[ValidationCheck] = OPEN_SWE_PYTHON_CHECKS,
    model_factory: ModelFactory | None = None,
    agent_factory: AgentFactory | None = None,
    runner_factory: RunnerFactory | None = None,
) -> RoutedOrchestrationService:
    """Build an orchestration service bound to one Open SWE thread."""
    if not context.thread_id.strip():
        raise ValueError("Runtime context thread ID cannot be empty")

    if not context.work_dir.strip():
        raise ValueError("Runtime context work directory cannot be empty")

    return build_server_orchestration_service(
        thread_id=context.thread_id,
        backend=context.sandbox_backend,
        tools=tools,
        skills=skills,
        middleware=middleware,
        use_gateway=use_gateway,
        checks=checks,
        model_factory=model_factory,
        agent_factory=agent_factory,
        runner_factory=runner_factory,
    )
