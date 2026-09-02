"""LangGraph factory for the standalone deterministic orchestrator."""

from collections.abc import Awaitable, Callable

from langgraph.graph.state import RunnableConfig

from agent.validation import ValidationCheck

from .bootstrap import (
    OrchestratorRuntimeContext,
    bootstrap_orchestrator_runtime,
)
from .graph import build_orchestrator_graph
from .run_preparation import prepare_runtime_run
from .runtime_dependencies import (
    SpecialistRuntimeDependencies,
    build_specialist_runtime_dependencies,
)
from .runtime_service import (
    build_runtime_orchestration_service,
)
from .server_bridge import RoutedOrchestrationService
from .validation_profiles import ValidationProfile
from .validation_resolution import resolve_validation_checks

BootstrapFactory = Callable[
    [RunnableConfig],
    Awaitable[OrchestratorRuntimeContext | None],
]

ServiceFactory = Callable[
    ...,
    RoutedOrchestrationService,
]

DependencyFactory = Callable[
    [RunnableConfig, OrchestratorRuntimeContext],
    SpecialistRuntimeDependencies,
]

ValidationResolver = Callable[..., Awaitable[tuple[ValidationCheck, ...]]]

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
    dependency_factory: DependencyFactory = (build_specialist_runtime_dependencies),
    validation_resolver: ValidationResolver = resolve_validation_checks,
):
    """Build the standalone orchestration graph."""
    profile = _validation_profile_from_config(config)

    dry_run = _dry_run_from_config(config)

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

    async def run_preparer(work_dir: str):
        return await prepare_runtime_run(
            config,
            context,
            profile,
            requested_work_dir=work_dir,
            validation_resolver=validation_resolver,
        )

    dependencies = dependency_factory(
        config,
        context,
    )

    service = service_factory(
        context,
        tools=dependencies.tools,
        skills=(list(dependencies.skills) if dependencies.skills else None),
        middleware=(dependencies.middleware if dependencies.middleware else None),
        use_gateway=dependencies.use_gateway,
        model_effort=dependencies.model_effort,
        checks=(),
        run_preparer=run_preparer,
    )

    return build_orchestrator_graph(
        service=service,
        work_dir=context.work_dir,
    )
