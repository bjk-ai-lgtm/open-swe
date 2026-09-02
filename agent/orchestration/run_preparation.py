"""Prepare repository-bound runtime inputs immediately before execution."""

from collections.abc import Awaitable, Callable
from dataclasses import replace

from langgraph.graph.state import RunnableConfig

from agent.validation import ValidationCheck

from .bootstrap import OrchestratorRuntimeContext
from .publication_target import resolve_publication_target
from .repository_workspace import (
    RepositoryWorkspace,
    prepare_repository_workspace,
)
from .server_bridge import PreparedRun
from .validation_profiles import ValidationProfile
from .validation_resolution import resolve_validation_checks

WorkspacePreparer = Callable[..., Awaitable[RepositoryWorkspace]]
ValidationResolver = Callable[
    ...,
    Awaitable[tuple[ValidationCheck, ...]],
]


async def prepare_runtime_run(
    config: RunnableConfig,
    context: OrchestratorRuntimeContext,
    profile: ValidationProfile,
    *,
    requested_work_dir: str,
    workspace_preparer: WorkspacePreparer = (
        prepare_repository_workspace
    ),
    validation_resolver: ValidationResolver = (
        resolve_validation_checks
    ),
) -> PreparedRun:
    """Prepare one executable task on a deterministic repository checkout."""
    target = resolve_publication_target(
        config,
        work_dir=requested_work_dir,
    )

    workspace = await workspace_preparer(
        context.sandbox_backend,
        target,
        thread_id=context.thread_id,
    )

    repo_context = replace(
        context,
        work_dir=workspace.target.repo_dir,
    )

    checks = await validation_resolver(
        profile,
        repo_context,
    )

    return PreparedRun(
        work_dir=repo_context.work_dir,
        checks=checks,
    )
