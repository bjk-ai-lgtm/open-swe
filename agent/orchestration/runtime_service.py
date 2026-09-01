"""Bind bootstrapped Open SWE resources to the orchestration service."""

from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from agent.runtime import DEFAULT_LLM_MAX_TOKENS
from agent.validation import ValidationCheck

from .bootstrap import OrchestratorRuntimeContext
from .model_factory import build_orchestration_model_factory
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
    model_effort: str | None = None,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    checks: Sequence[ValidationCheck],
    model_factory: ModelFactory | None = None,
    agent_factory: AgentFactory | None = None,
    runner_factory: RunnerFactory | None = None,
) -> RoutedOrchestrationService:
    """Build an orchestration service bound to one Open SWE thread."""
    if not context.thread_id.strip():
        raise ValueError("Runtime context thread ID cannot be empty")

    if not context.work_dir.strip():
        raise ValueError("Runtime context work directory cannot be empty")

    if model_factory is None:
        model_factory = build_orchestration_model_factory(
            use_gateway=use_gateway,
            model_effort=model_effort,
            max_tokens=max_tokens,
        )

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
