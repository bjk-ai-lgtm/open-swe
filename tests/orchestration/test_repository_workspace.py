import pytest

from agent.orchestration.publication_target import PublicationTarget
from agent.orchestration.repository_workspace import (
    RepositoryWorkspaceError,
    inspect_repository_workspace,
)


class Result:
    def __init__(self, exit_code=0, output=""):
        self.exit_code = exit_code
        self.output = output


class Backend:
    def __init__(self, results):
        self.results = list(results)
        self.commands = []

    async def aexecute(self, command, timeout=None):
        self.commands.append((command, timeout))

        if not self.results:
            raise AssertionError("No fake command result remaining")

        return self.results.pop(0)


def target(
    *,
    head="feature/example",
    base="custom/v0.1",
):
    return PublicationTarget(
        owner="bjk-ai-lgtm",
        repo="open-swe",
        repo_dir="/workspace/open-swe",
        configured_head=head,
        configured_base=base,
    )


@pytest.mark.asyncio
async def test_valid_existing_checkout_passes_preflight() -> None:
    backend = Backend(
        [
            Result(),
            Result(output="https://github.com/bjk-ai-lgtm/open-swe.git\n"),
            Result(output="feature/example\n"),
            Result(output="custom/v0.1\n"),
            Result(output="abc123\n"),
        ]
    )

    workspace = await inspect_repository_workspace(
        backend,
        target(),
    )

    assert workspace.head_branch == "feature/example"
    assert workspace.base_branch == "custom/v0.1"


@pytest.mark.asyncio
async def test_base_branch_can_come_from_origin_head() -> None:
    backend = Backend(
        [
            Result(),
            Result(output="git@github.com:bjk-ai-lgtm/open-swe.git\n"),
            Result(output="feature/example\n"),
            Result(output="origin/custom/v0.1\n"),
            Result(output="custom/v0.1\n"),
            Result(output="abc123\n"),
        ]
    )

    workspace = await inspect_repository_workspace(
        backend,
        target(base=None),
    )

    assert workspace.base_branch == "custom/v0.1"


@pytest.mark.asyncio
async def test_wrong_origin_is_rejected() -> None:
    backend = Backend(
        [
            Result(),
            Result(output="https://github.com/other/project.git\n"),
        ]
    )

    with pytest.raises(
        RepositoryWorkspaceError,
        match="does not match configured target",
    ):
        await inspect_repository_workspace(
            backend,
            target(),
        )


@pytest.mark.asyncio
async def test_detached_head_is_rejected() -> None:
    backend = Backend(
        [
            Result(),
            Result(output="https://github.com/bjk-ai-lgtm/open-swe.git\n"),
            Result(output="HEAD\n"),
        ]
    )

    with pytest.raises(
        RepositoryWorkspaceError,
        match="detached HEAD",
    ):
        await inspect_repository_workspace(
            backend,
            target(head=None),
        )


@pytest.mark.asyncio
async def test_missing_checkout_fails_closed() -> None:
    backend = Backend(
        [
            Result(exit_code=1),
        ]
    )

    with pytest.raises(
        RepositoryWorkspaceError,
        match="Repository command failed",
    ):
        await inspect_repository_workspace(
            backend,
            target(),
        )


class ScriptedBackend:
    def __init__(self, steps):
        self.steps = list(steps)
        self.commands = []

    async def aexecute(self, command, timeout=None):
        self.commands.append((command, timeout))

        if not self.steps:
            raise AssertionError(
                f"Unexpected command: {command}"
            )

        expected, result = self.steps.pop(0)

        assert expected in command

        return result


@pytest.mark.asyncio
async def test_fresh_clone_creates_thread_stable_branch() -> None:
    from agent.orchestration.repository_workspace import (
        prepare_repository_workspace,
    )

    thread_id = "11111111-2222-3333-4444-555555555555"
    desired = f"open-swe/{thread_id}"

    backend = ScriptedBackend(
        [
            ("git check-ref-format --branch", Result()),
            ("test -d", Result(exit_code=1)),
            ("test -e", Result(exit_code=1)),
            ("git clone --origin origin", Result()),
            (
                "git remote get-url origin",
                Result(
                    output=(
                        "https://github.com/"
                        "bjk-ai-lgtm/open-swe.git\n"
                    )
                ),
            ),
            ("git fetch origin --prune", Result()),
            (
                "symbolic-ref --quiet --short",
                Result(output="origin/custom/v0.1\n"),
            ),
            ("git check-ref-format --branch", Result()),
            (
                "git rev-parse --verify",
                Result(output="base-sha\n"),
            ),
            (
                "git status --porcelain",
                Result(output=""),
            ),
            (
                "git rev-parse --abbrev-ref HEAD",
                Result(output="custom/v0.1\n"),
            ),
            (
                "refs/heads/open-swe/",
                Result(exit_code=1),
            ),
            (
                "refs/remotes/origin/open-swe/",
                Result(exit_code=1),
            ),
            (
                "git checkout -b",
                Result(),
            ),
            ("test -d", Result()),
            (
                "git remote get-url origin",
                Result(
                    output=(
                        "https://github.com/"
                        "bjk-ai-lgtm/open-swe.git\n"
                    )
                ),
            ),
            (
                "git rev-parse --abbrev-ref HEAD",
                Result(output=f"{desired}\n"),
            ),
            (
                "symbolic-ref --quiet --short",
                Result(output="origin/custom/v0.1\n"),
            ),
            ("git check-ref-format --branch", Result()),
            (
                "git rev-parse --verify",
                Result(output="base-sha\n"),
            ),
        ]
    )

    workspace = await prepare_repository_workspace(
        backend,
        target(
            head=None,
            base=None,
        ),
        thread_id=thread_id,
        sandbox_type="langsmith",
    )

    assert workspace.head_branch == desired
    assert workspace.base_branch == "custom/v0.1"
    assert backend.steps == []

    first_command = backend.commands[0][0]
    assert first_command.startswith(
        "git check-ref-format --branch "
    )
    assert "cd /workspace/open-swe" not in first_command


@pytest.mark.asyncio
async def test_dirty_wrong_branch_fails_without_checkout() -> None:
    from agent.orchestration.repository_workspace import (
        prepare_repository_workspace,
    )

    backend = ScriptedBackend(
        [
            ("git check-ref-format --branch", Result()),
            ("test -d", Result()),
            (
                "git remote get-url origin",
                Result(
                    output=(
                        "https://github.com/"
                        "bjk-ai-lgtm/open-swe.git\n"
                    )
                ),
            ),
            ("git fetch origin --prune", Result()),
            ("git check-ref-format --branch", Result()),
            (
                "git rev-parse --verify",
                Result(output="base-sha\n"),
            ),
            (
                "git status --porcelain",
                Result(output=" M file.py\n"),
            ),
            (
                "git rev-parse --abbrev-ref HEAD",
                Result(output="different-branch\n"),
            ),
        ]
    )

    with pytest.raises(
        RepositoryWorkspaceError,
        match="Cannot switch branches",
    ):
        await prepare_repository_workspace(
            backend,
            target(),
            thread_id=(
                "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            ),
            sandbox_type="langsmith",
        )

    assert backend.steps == []
