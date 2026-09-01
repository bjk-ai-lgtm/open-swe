"""LangGraph factory for the standalone deterministic orchestrator."""

from collections.abc import Awaitable, Callable

from langgraph.graph.state import RunnableConfig

from .bootstrap import (
    OrchestratorRuntimeContext,
    bootstrap_orchestrator_runtime,
)
from .graph import build_orchestrator_graph
from .runtime_service import (
    build_runtime_orchestration_service,
)
from .server_bridge import RoutedOrchestrationService
from .validation_profiles import (
    ValidationProfile,
    validation_checks_for_profile,
)

BootstrapFactory = Callable[
    [RunnableConfig],
    Awaitable[OrchestratorRuntimeContext | None],
]

ServiceFactory = Callable[
    ...,
    RoutedOrchestrationService,
]

_VALIDATION_PROFILE_KEY = "orchestrator_validation_profile"

_DRY_RUN_KEY = "orchestrator_dry_run"

_SCHEMA_WORK_DIR = "/schema-only"


class _NonExecutingService:
    """Service placeholder for non-executing graph modes."""

    async def run(
        self,
        *,
        task: str,
        work_dir: str,
    ):
        raise RuntimeError("Non-executing orchestrator graph cannot execute tasks")


def _validation_profile_from_config(
    config: RunnableConfig,
) -> ValidationProfile:
    configurable = config.get("configurable") or {}

    raw_profile = configurable.get(
        _VALIDATION_PROFILE_KEY,
        ValidationProfile.NONE.value,
    )

    if not isinstance(raw_profile, str):
        raise ValueError("Orchestrator validation profile must be a string")

    try:
        return ValidationProfile(raw_profile)
    except ValueError as exc:
        raise ValueError(f"Unsupported validation profile: {raw_profile}") from exc


def _dry_run_from_config(
    config: RunnableConfig,
) -> bool:
    configurable = config.get("configurable") or {}

    raw_value = configurable.get(
        _DRY_RUN_KEY,
        False,
    )

    if not isinstance(raw_value, bool):
        raise ValueError("orchestrator_dry_run must be a boolean")

    return raw_value


async def get_orchestrator(
    config: RunnableConfig,
    *,
    bootstrap: BootstrapFactory = (bootstrap_orchestrator_runtime),
    service_factory: ServiceFactory = (build_runtime_orchestration_service),
):
    """Build the standalone orchestration graph."""
    profile = _validation_profile_from_config(config)

    dry_run = _dry_run_from_config(config)

    checks = validation_checks_for_profile(profile)

    if dry_run:
        return build_orchestrator_graph(
            service=_NonExecutingService(),
            work_dir="/dry-run",
            dry_run=True,
        )

    context = await bootstrap(config)

    if context is None:
        return build_orchestrator_graph(
            service=_NonExecutingService(),
            work_dir=_SCHEMA_WORK_DIR,
            dry_run=False,
        )

    service = service_factory(
        context,
        tools=(),
        checks=checks,
    )

    return build_orchestrator_graph(
        service=service,
        work_dir=context.work_dir,
    )
