"""Resolve validation checks from explicit or repository-aware profiles."""

from collections.abc import Awaitable, Callable

from agent.validation import ValidationCheck

from .bootstrap import OrchestratorRuntimeContext
from .repository_inspection import read_repository_metadata
from .repository_validation import validation_checks_for_repository
from .validation_profiles import ValidationProfile, validation_checks_for_profile

RepositoryInspector = Callable[..., Awaitable[dict[str, str]]]


async def resolve_validation_checks(
    profile: ValidationProfile,
    context: OrchestratorRuntimeContext,
    *,
    repository_inspector: RepositoryInspector = read_repository_metadata,
) -> tuple[ValidationCheck, ...]:
    """Resolve checks without letting the LLM choose validation commands."""
    if profile is ValidationProfile.AUTO:
        files = await repository_inspector(
            context.sandbox_backend,
            work_dir=context.work_dir,
        )
        return validation_checks_for_repository(files)

    return validation_checks_for_profile(profile)
