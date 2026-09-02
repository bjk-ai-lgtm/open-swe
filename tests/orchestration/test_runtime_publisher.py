from collections import deque

import pytest

from agent.orchestration.bootstrap import (
    OrchestratorRuntimeContext,
)
from agent.orchestration.repository_workspace import (
    RepositoryWorkspace,
)
from agent.orchestration.runtime_publisher import (
    OpenSWERuntimePublisher,
    RuntimePublicationError,
)
from agent.orchestration.sandbox_command import (
    SandboxCommandResult,
)


class SequenceExecutor:
    def __init__(self, steps):
        self.steps = deque(steps)
        self.commands = []

    async def __call__(
        self,
        backend,
        command,
        *,
        timeout,
        sandbox_type,
    ):

        del backend, timeout, sandbox_type

        self.commands.append(command)

        if not self.steps:
            raise AssertionError(
                f"Unexpected command: {command}"
            )

        expected, exit_code, output = self.steps.popleft()

        assert expected in command

        return SandboxCommandResult(
            exit_code=exit_code,
            output=output,
        )


def make_publisher(
    steps,
    *,
    pr_result=None,
):
    backend = object()

    context = OrchestratorRuntimeContext(
        thread_id="thread-1",
        sandbox_backend=backend,
        work_dir="/workspace",
    )

    config = {
        "configurable": {
            "thread_id": "thread-1",
            "repo": {
                "owner": "bjk-ai-lgtm",
                "name": "open-swe",
            },
            "base_branch": "custom/v0.1",
        }
    }

    executor = SequenceExecutor(steps)
    opened = []

    async def inspector(
        received_backend,
        target,
        *,
        sandbox_type,
    ):
        assert received_backend is backend
        assert sandbox_type is None

        return RepositoryWorkspace(
            target=target,
            head_branch="open-swe/thread-1",
            base_branch="custom/v0.1",
        )

    async def pr_opener(**kwargs):
        opened.append(kwargs)

        if pr_result is not None:
            return pr_result

        return {
            "success": True,
            "created": True,
            "url": "https://example.test/pr/1",
            "number": 1,
        }

    publisher = OpenSWERuntimePublisher(
        config=config,
        context=context,
        workspace_inspector=inspector,
        command_executor=executor,
        pr_opener=pr_opener,
    )

    return publisher, executor, opened


async def test_publisher_commits_pushes_and_opens_pr() -> None:
    publisher, executor, opened = make_publisher(
        [
            ("git status --porcelain", 0, " M app.py\n"),
            ("git add -A", 0, ""),
            ("git diff --cached --quiet", 1, ""),
            ("git commit -m", 0, ""),
            ("git status --porcelain", 0, ""),
            ("git merge-base --is-ancestor", 0, ""),
            ("git rev-list --count", 0, "1\n"),
            (
                "git diff --name-only",
                0,
                "",
            ),
            ("git rev-parse HEAD", 0, "abc123\n"),
            ("git ls-remote --heads", 0, ""),
            ("git push origin", 0, ""),
            (
                "git ls-remote --heads",
                0,
                "abc123\trefs/heads/open-swe/thread-1\n",
            ),
        ]
    )

    await publisher(
        thread_id="thread-1",
        task="Implement the API",
        work_dir="/workspace/open-swe",
    )

    assert not executor.steps
    assert len(opened) == 1
    assert opened[0]["head"] == "open-swe/thread-1"
    assert opened[0]["base"] == "custom/v0.1"
    assert opened[0]["draft"] is True


async def test_publisher_noops_when_branch_has_no_commits() -> None:
    publisher, executor, opened = make_publisher(
        [
            ("git status --porcelain", 0, ""),
            ("git merge-base --is-ancestor", 0, ""),
            ("git rev-list --count", 0, "0\n"),
        ]
    )

    await publisher(
        thread_id="thread-1",
        task="Research the implementation",
        work_dir="/workspace/open-swe",
    )

    assert not executor.steps
    assert opened == []
    assert not any(
        "git push" in command
        for command in executor.commands
    )


async def test_publisher_reconciles_ambiguous_push_failure() -> None:
    publisher, executor, opened = make_publisher(
        [
            ("git status --porcelain", 0, ""),
            ("git merge-base --is-ancestor", 0, ""),
            ("git rev-list --count", 0, "1\n"),
            ("git diff --name-only", 0, ""),
            ("git rev-parse HEAD", 0, "abc123\n"),
            ("git ls-remote --heads", 0, ""),
            (
                "git push origin",
                1,
                "session cleanup failed",
            ),
            (
                "git ls-remote --heads",
                0,
                "abc123\trefs/heads/open-swe/thread-1\n",
            ),
        ]
    )

    await publisher(
        thread_id="thread-1",
        task="Implement the API",
        work_dir="/workspace/open-swe",
    )

    assert not executor.steps
    assert len(opened) == 1


async def test_publisher_rejects_non_fast_forward_remote() -> None:
    publisher, executor, opened = make_publisher(
        [
            ("git status --porcelain", 0, ""),
            ("git merge-base --is-ancestor", 0, ""),
            ("git rev-list --count", 0, "1\n"),
            ("git diff --name-only", 0, ""),
            ("git rev-parse HEAD", 0, "abc123\n"),
            (
                "git ls-remote --heads",
                0,
                "def456\trefs/heads/open-swe/thread-1\n",
            ),
            ("git fetch origin", 0, ""),
            ("git merge-base --is-ancestor", 1, ""),
        ]
    )

    with pytest.raises(
        RuntimePublicationError,
        match="not an ancestor",
    ):
        await publisher(
            thread_id="thread-1",
            task="Implement the API",
            work_dir="/workspace/open-swe",
        )

    assert not executor.steps
    assert opened == []
    assert not any(
        "git push" in command
        for command in executor.commands
    )


async def test_publisher_blocks_workflow_changes() -> None:
    publisher, executor, opened = make_publisher(
        [
            ("git status --porcelain", 0, ""),
            ("git merge-base --is-ancestor", 0, ""),
            ("git rev-list --count", 0, "1\n"),
            (
                "git diff --name-only",
                0,
                ".github/workflows/ci.yml\n",
            ),
        ]
    )

    with pytest.raises(
        RuntimePublicationError,
        match="workflow changes",
    ):
        await publisher(
            thread_id="thread-1",
            task="Change CI",
            work_dir="/workspace/open-swe",
        )

    assert not executor.steps
    assert opened == []


async def test_publisher_surfaces_native_pr_failure() -> None:
    publisher, executor, opened = make_publisher(
        [
            ("git status --porcelain", 0, ""),
            ("git merge-base --is-ancestor", 0, ""),
            ("git rev-list --count", 0, "1\n"),
            ("git diff --name-only", 0, ""),
            ("git rev-parse HEAD", 0, "abc123\n"),
            (
                "git ls-remote --heads",
                0,
                "abc123\trefs/heads/open-swe/thread-1\n",
            ),
        ],
        pr_result={
            "success": False,
            "error": "GitHub unavailable",
        },
    )

    with pytest.raises(
        RuntimePublicationError,
        match="GitHub unavailable",
    ):
        await publisher(
            thread_id="thread-1",
            task="Implement the API",
            work_dir="/workspace/open-swe",
        )

    assert not executor.steps
    assert len(opened) == 1


async def test_publisher_rejects_wrong_prepared_work_dir() -> None:
    publisher, executor, opened = make_publisher([])

    with pytest.raises(
        RuntimePublicationError,
        match="does not match",
    ):
        await publisher(
            thread_id="thread-1",
            task="Implement the API",
            work_dir="/workspace/other",
        )

    assert executor.commands == []
    assert opened == []
