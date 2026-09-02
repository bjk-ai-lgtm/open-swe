import pytest

from agent.orchestration.bootstrap import (
    OrchestratorRuntimeContext,
)
from agent.orchestration.publication_target import (
    PublicationTargetError,
)
from agent.orchestration.repository_workspace import (
    RepositoryWorkspace,
)
from agent.orchestration.run_preparation import (
    prepare_runtime_run,
)
from agent.orchestration.validation_profiles import (
    ValidationProfile,
)
from agent.validation import ValidationCheck


@pytest.mark.asyncio
async def test_run_preparation_resolves_repo_root_before_validation() -> None:
    backend = object()

    context = OrchestratorRuntimeContext(
        thread_id="thread-runtime-prep",
        sandbox_backend=backend,
        work_dir="/workspace",
    )

    captured = {}

    async def workspace_preparer(
        received_backend,
        target,
        *,
        thread_id,
    ):
        captured["backend"] = received_backend
        captured["target"] = target
        captured["thread_id"] = thread_id

        return RepositoryWorkspace(
            target=target,
            head_branch="open-swe/thread-runtime-prep",
            base_branch="custom/v0.1",
        )

    async def validation_resolver(
        profile,
        received_context,
    ):
        captured["profile"] = profile
        captured["validation_context"] = received_context

        return (
            ValidationCheck(
                name="repo-tests",
                command=("pytest", "-q"),
            ),
        )

    prepared = await prepare_runtime_run(
        {
            "configurable": {
                "repo": {
                    "owner": "bjk-ai-lgtm",
                    "name": "open-swe",
                },
                "base_branch": "custom/v0.1",
            }
        },
        context,
        ValidationProfile.AUTO,
        requested_work_dir="/workspace",
        workspace_preparer=workspace_preparer,
        validation_resolver=validation_resolver,
    )

    assert captured["backend"] is backend
    assert captured["thread_id"] == "thread-runtime-prep"
    assert captured["target"].repo_dir == (
        "/workspace/open-swe"
    )

    assert captured["profile"] is ValidationProfile.AUTO
    assert captured["validation_context"].work_dir == (
        "/workspace/open-swe"
    )

    assert prepared.work_dir == "/workspace/open-swe"
    assert len(prepared.checks) == 1
    assert prepared.checks[0].name == "repo-tests"

    assert context.work_dir == "/workspace"


@pytest.mark.asyncio
async def test_run_preparation_fails_before_workspace_without_repo_metadata() -> None:
    context = OrchestratorRuntimeContext(
        thread_id="thread-no-repo",
        sandbox_backend=object(),
        work_dir="/workspace",
    )

    calls = []

    async def workspace_preparer(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(
            "workspace preparation must not run"
        )

    with pytest.raises(
        PublicationTargetError,
        match="does not identify a target repository",
    ):
        await prepare_runtime_run(
            {
                "configurable": {},
            },
            context,
            ValidationProfile.AUTO,
            requested_work_dir="/workspace",
            workspace_preparer=workspace_preparer,
        )

    assert calls == []
