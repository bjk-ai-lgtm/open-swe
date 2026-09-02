"""Bridge Open SWE server dependencies into the custom orchestrator."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel

from agent.validation import (
    OPEN_SWE_PYTHON_CHECKS,
    CommandRunner,
    ValidationCheck,
)

from .coordinator import (
    CoordinatorResult,
    TaskPublisher,
    run_orchestrated_task,
)
from .execution_safety import assert_isolated_execution_environment
from .specialist_executor import (
    AgentFactory,
    OpenSWESpecialistExecutor,
)

ModelFactory = Callable[[str], BaseChatModel]
RunnerFactory = Callable[[str], Awaitable[CommandRunner]]
ExecutionGuard = Callable[[], None]


@dataclass(frozen=True)
class PreparedRun:
    """Runtime-owned inputs prepared immediately before coordination."""

    work_dir: str
    checks: tuple[ValidationCheck, ...]


RunPreparer = Callable[[str], Awaitable[PreparedRun]]


@dataclass
class RoutedOrchestrationService:
    """Thread-bound runtime service for deterministic orchestration."""

    thread_id: str
    executor: OpenSWESpecialistExecutor
    checks: tuple[ValidationCheck, ...]
    runner_factory: RunnerFactory | None = None
    run_preparer: RunPreparer | None = None
    publisher: TaskPublisher | None = None
    execution_guard: ExecutionGuard = assert_isolated_execution_environment

    async def run(
        self,
        *,
        task: str,
        work_dir: str,
    ) -> CoordinatorResult:
        """Run one task through the custom orchestration pipeline."""
        self.execution_guard()

        effective_work_dir = work_dir
        effective_checks = self.checks

        if self.run_preparer is not None:
            prepared = await self.run_preparer(work_dir)

            if not prepared.work_dir.strip():
                raise ValueError(
                    "Prepared runtime work directory cannot be empty"
                )

            effective_work_dir = prepared.work_dir
            effective_checks = prepared.checks

        kwargs: dict[str, Any] = {
            "thread_id": self.thread_id,
            "task": task,
            "work_dir": effective_work_dir,
            "executor": self.executor,
            "checks": effective_checks,
        }

        if self.runner_factory is not None:
            kwargs["runner_factory"] = self.runner_factory

        if self.publisher is not None:
            kwargs["publisher"] = self.publisher

        return await run_orchestrated_task(**kwargs)


def build_server_orchestration_service(
    *,
    thread_id: str,
    backend: BackendProtocol,
    tools: Sequence[Any],
    skills: list[str] | None = None,
    middleware: Sequence[AgentMiddleware[Any, Any, Any]] | None = None,
    use_gateway: bool | None = None,
    checks: Sequence[ValidationCheck] = OPEN_SWE_PYTHON_CHECKS,
    model_factory: ModelFactory | None = None,
    agent_factory: AgentFactory | None = None,
    runner_factory: RunnerFactory | None = None,
    run_preparer: RunPreparer | None = None,
    publisher: TaskPublisher | None = None,
    execution_guard: ExecutionGuard = assert_isolated_execution_environment,
) -> RoutedOrchestrationService:
    """Build a coordinator service from the live Open SWE server context."""
    if not thread_id.strip():
        raise ValueError("Thread ID cannot be empty")

    executor_kwargs: dict[str, Any] = {
        "backend": backend,
        "tools": tools,
        "skills": skills,
        "middleware": middleware,
        "use_gateway": use_gateway,
        "model_factory": model_factory,
    }

    if agent_factory is not None:
        executor_kwargs["agent_factory"] = agent_factory

    executor = OpenSWESpecialistExecutor(**executor_kwargs)

    return RoutedOrchestrationService(
        thread_id=thread_id,
        executor=executor,
        checks=tuple(checks),
        runner_factory=runner_factory,
        run_preparer=run_preparer,
        publisher=publisher,
        execution_guard=execution_guard,
    )
