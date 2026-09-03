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
    ValidationReport,
    run_validation,
    sandbox_runner_for_thread,
)

from .coordinator import (
    CoordinatorResult,
    SpecialistExecutionResult,
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

    async def prepare_run(
        self,
        work_dir: str,
    ) -> PreparedRun:
        """Prepare deterministic runtime inputs for one task."""
        self.execution_guard()

        prepared = PreparedRun(
            work_dir=work_dir,
            checks=self.checks,
        )

        if self.run_preparer is not None:
            prepared = await self.run_preparer(
                work_dir
            )

        if not prepared.work_dir.strip():
            raise ValueError(
                "Prepared runtime work directory cannot be empty"
            )

        return prepared

    async def execute_attempt(
        self,
        *,
        task: str,
        work_dir: str,
        role,
        model_id: str,
        attempt: int,
        escalation_level: int,
        previous_failure: str | None,
    ) -> SpecialistExecutionResult:
        """Execute exactly one specialist attempt."""
        return await self.executor.execute(
            thread_id=self.thread_id,
            work_dir=work_dir,
            task=task,
            role=role,
            model_id=model_id,
            attempt=attempt,
            escalation_level=escalation_level,
            previous_failure=previous_failure,
        )

    async def validate_attempt(
        self,
        *,
        work_dir: str,
        checks: Sequence[ValidationCheck],
    ) -> ValidationReport:
        """Execute deterministic validation for one phase."""
        if self.runner_factory is None:
            runner = await sandbox_runner_for_thread(
                self.thread_id
            )
        else:
            runner = await self.runner_factory(
                self.thread_id
            )

        return await run_validation(
            runner,
            checks,
            work_dir=work_dir,
        )

    async def publish_task(
        self,
        *,
        task: str,
        work_dir: str,
    ) -> None:
        """Publish one validated task when a publisher is configured."""
        if self.publisher is None:
            return

        await self.publisher(
            thread_id=self.thread_id,
            task=task,
            work_dir=work_dir,
        )

    async def run(
        self,
        *,
        task: str,
        work_dir: str,
    ) -> CoordinatorResult:
        """Run one task through the custom orchestration pipeline."""
        prepared = await self.prepare_run(
            work_dir
        )

        kwargs: dict[str, Any] = {
            "thread_id": self.thread_id,
            "task": task,
            "work_dir": prepared.work_dir,
            "executor": self.executor,
            "checks": prepared.checks,
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
