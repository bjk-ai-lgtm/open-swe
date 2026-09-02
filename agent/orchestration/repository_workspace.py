"""Prepare and validate a repository checkout for runtime-owned publication."""

import shlex
from dataclasses import dataclass

from deepagents.backends.protocol import SandboxBackendProtocol

from .publication_target import PublicationTarget
from .sandbox_command import execute_control_plane_command


class RepositoryWorkspaceError(RuntimeError):
    """Raised when a repository workspace cannot be prepared safely."""


@dataclass(frozen=True)
class RepositoryWorkspace:
    """Validated Git checkout used by deterministic publication."""

    target: PublicationTarget
    head_branch: str
    base_branch: str


def _github_repo_from_remote(remote_url: str) -> tuple[str, str]:
    value = remote_url.strip()

    prefixes = (
        "https://github.com/",
        "ssh://git@github.com/",
        "git@github.com:",
    )

    path = None

    for prefix in prefixes:
        if value.startswith(prefix):
            path = value[len(prefix):]
            break

    if path is None:
        raise RepositoryWorkspaceError(
            "Origin remote is not a supported GitHub repository URL"
        )

    if path.endswith(".git"):
        path = path[:-4]

    parts = path.split("/")

    if len(parts) != 2 or not all(parts):
        raise RepositoryWorkspaceError(
            "Origin remote does not identify exactly one GitHub repository"
        )

    return parts[0], parts[1]


async def _execute(
    backend: SandboxBackendProtocol,
    command: str,
    *,
    timeout: int = 30,
    sandbox_type: str | None = None,
):
    return await execute_control_plane_command(
        backend,
        command,
        timeout=timeout,
        sandbox_type=sandbox_type,
    )


async def _run(
    backend: SandboxBackendProtocol,
    command: str,
    *,
    timeout: int = 30,
    sandbox_type: str | None = None,
) -> str:
    result = await _execute(
        backend,
        command,
        timeout=timeout,
        sandbox_type=sandbox_type,
    )

    if result.exit_code != 0:
        raise RepositoryWorkspaceError(
            f"Repository command failed with exit code "
            f"{result.exit_code}: {command}"
        )

    return result.output.strip()


async def _succeeds(
    backend: SandboxBackendProtocol,
    command: str,
    *,
    timeout: int = 30,
    sandbox_type: str | None = None,
) -> bool:
    result = await _execute(
        backend,
        command,
        timeout=timeout,
        sandbox_type=sandbox_type,
    )

    return result.exit_code == 0


async def _assert_origin_matches(
    backend: SandboxBackendProtocol,
    target: PublicationTarget,
    *,
    sandbox_type: str | None = None,
) -> None:
    repo_dir = shlex.quote(target.repo_dir)

    remote_url = await _run(
        backend,
        f"cd {repo_dir} && git remote get-url origin",
        sandbox_type=sandbox_type,
    )

    remote_owner, remote_repo = _github_repo_from_remote(
        remote_url
    )

    if (
        remote_owner.lower() != target.owner.lower()
        or remote_repo.lower() != target.repo.lower()
    ):
        raise RepositoryWorkspaceError(
            "Origin remote does not match configured target repository"
        )


async def _validate_branch(
    backend: SandboxBackendProtocol,
    branch: str,
    *,
    sandbox_type: str | None = None,
) -> None:
    await _run(
        backend,
        f"git check-ref-format --branch {shlex.quote(branch)}",
        sandbox_type=sandbox_type,
    )


async def _resolve_base_branch(
    backend: SandboxBackendProtocol,
    target: PublicationTarget,
    *,
    sandbox_type: str | None = None,
) -> str:
    repo_dir = shlex.quote(target.repo_dir)

    if target.configured_base is not None:
        base = target.configured_base
    else:
        remote_head = await _run(
            backend,
            (
                f"cd {repo_dir} && "
                "git symbolic-ref --quiet --short "
                "refs/remotes/origin/HEAD"
            ),
            sandbox_type=sandbox_type,
        )

        prefix = "origin/"

        if not remote_head.startswith(prefix):
            raise RepositoryWorkspaceError(
                "origin/HEAD did not resolve to an origin branch"
            )

        base = remote_head[len(prefix):]

    if not base:
        raise RepositoryWorkspaceError(
            "Base branch could not be resolved"
        )

    await _validate_branch(
        backend,
        base,
        sandbox_type=sandbox_type,
    )

    await _run(
        backend,
        (
            f"cd {repo_dir} && "
            "git rev-parse --verify "
            f"{shlex.quote(f'refs/remotes/origin/{base}^{{commit}}')}"
        ),
        sandbox_type=sandbox_type,
    )

    return base


async def _sync_current_branch(
    backend: SandboxBackendProtocol,
    *,
    repo_dir: str,
    branch: str,
    dirty: bool,
    sandbox_type: str | None = None,
) -> None:
    rendered_repo = shlex.quote(repo_dir)
    remote_ref = f"refs/remotes/origin/{branch}"
    rendered_remote = shlex.quote(remote_ref)

    remote_exists = await _succeeds(
        backend,
        (
            f"cd {rendered_repo} && "
            f"git show-ref --verify --quiet {rendered_remote}"
        ),
        sandbox_type=sandbox_type,
    )

    if not remote_exists:
        return

    local_sha = await _run(
        backend,
        f"cd {rendered_repo} && git rev-parse HEAD",
        sandbox_type=sandbox_type,
    )

    remote_sha = await _run(
        backend,
        (
            f"cd {rendered_repo} && "
            f"git rev-parse {rendered_remote}"
        ),
        sandbox_type=sandbox_type,
    )

    if local_sha == remote_sha:
        return

    if dirty:
        raise RepositoryWorkspaceError(
            "Dirty checkout cannot be synchronized safely"
        )

    local_is_behind = await _succeeds(
        backend,
        (
            f"cd {rendered_repo} && "
            f"git merge-base --is-ancestor HEAD {rendered_remote}"
        ),
        sandbox_type=sandbox_type,
    )

    if local_is_behind:
        await _run(
            backend,
            (
                f"cd {rendered_repo} && "
                f"git merge --ff-only {rendered_remote}"
            ),
            sandbox_type=sandbox_type,
        )
        return

    remote_is_behind = await _succeeds(
        backend,
        (
            f"cd {rendered_repo} && "
            f"git merge-base --is-ancestor {rendered_remote} HEAD"
        ),
        sandbox_type=sandbox_type,
    )

    if remote_is_behind:
        return

    raise RepositoryWorkspaceError(
        "Local and remote thread branches have diverged"
    )


async def inspect_repository_workspace(
    backend: SandboxBackendProtocol,
    target: PublicationTarget,
    *,
    sandbox_type: str | None = None,
) -> RepositoryWorkspace:
    """Validate an existing checkout without mutating repository state."""
    repo_dir = shlex.quote(target.repo_dir)

    await _run(
        backend,
        f"test -d {repo_dir}/.git",
        sandbox_type=sandbox_type,
    )

    await _assert_origin_matches(
        backend,
        target,
        sandbox_type=sandbox_type,
    )

    head = await _run(
        backend,
        f"cd {repo_dir} && git rev-parse --abbrev-ref HEAD",
        sandbox_type=sandbox_type,
    )

    if not head or head == "HEAD":
        raise RepositoryWorkspaceError(
            "Repository checkout is in detached HEAD state"
        )

    if (
        target.configured_head is not None
        and target.configured_head != head
    ):
        raise RepositoryWorkspaceError(
            "Configured head branch does not match checked-out branch"
        )

    base = await _resolve_base_branch(
        backend,
        target,
        sandbox_type=sandbox_type,
    )

    if head == base:
        raise RepositoryWorkspaceError(
            "Publication head branch cannot equal base branch"
        )

    return RepositoryWorkspace(
        target=target,
        head_branch=head,
        base_branch=base,
    )


async def prepare_repository_workspace(
    backend: SandboxBackendProtocol,
    target: PublicationTarget,
    *,
    thread_id: str,
    sandbox_type: str | None = None,
) -> RepositoryWorkspace:
    """Clone/synchronize the repo and prepare one thread-stable branch."""
    thread_id = thread_id.strip()

    if not thread_id:
        raise ValueError(
            "Repository workspace thread ID cannot be empty"
        )

    desired_head = (
        target.configured_head
        or f"open-swe/{thread_id}"
    )

    await _validate_branch(
        backend,
        desired_head,
        sandbox_type=sandbox_type,
    )

    rendered_repo = shlex.quote(target.repo_dir)

    repo_exists = await _succeeds(
        backend,
        f"test -d {rendered_repo}/.git",
        sandbox_type=sandbox_type,
    )

    if not repo_exists:
        path_exists = await _succeeds(
            backend,
            f"test -e {rendered_repo}",
            sandbox_type=sandbox_type,
        )

        if path_exists:
            raise RepositoryWorkspaceError(
                "Repository target path exists but is not a Git checkout"
            )

        clone_url = (
            f"https://github.com/"
            f"{target.owner}/{target.repo}.git"
        )

        await _run(
            backend,
            (
                "git clone --origin origin "
                f"{shlex.quote(clone_url)} "
                f"{rendered_repo}"
            ),
            timeout=240,
            sandbox_type=sandbox_type,
        )

    await _assert_origin_matches(
        backend,
        target,
        sandbox_type=sandbox_type,
    )

    await _run(
        backend,
        f"cd {rendered_repo} && git fetch origin --prune",
        timeout=240,
        sandbox_type=sandbox_type,
    )

    base = await _resolve_base_branch(
        backend,
        target,
        sandbox_type=sandbox_type,
    )

    dirty_output = await _run(
        backend,
        (
            f"cd {rendered_repo} && "
            "git status --porcelain --untracked-files=normal"
        ),
        sandbox_type=sandbox_type,
    )

    dirty = bool(dirty_output)

    current_head = await _run(
        backend,
        (
            f"cd {rendered_repo} && "
            "git rev-parse --abbrev-ref HEAD"
        ),
        sandbox_type=sandbox_type,
    )

    if current_head == desired_head:
        await _sync_current_branch(
            backend,
            repo_dir=target.repo_dir,
            branch=desired_head,
            dirty=dirty,
            sandbox_type=sandbox_type,
        )
    else:
        if dirty:
            raise RepositoryWorkspaceError(
                "Cannot switch branches while preserving dirty checkout"
            )

        local_ref = f"refs/heads/{desired_head}"
        remote_ref = f"refs/remotes/origin/{desired_head}"

        local_exists = await _succeeds(
            backend,
            (
                f"cd {rendered_repo} && "
                "git show-ref --verify --quiet "
                f"{shlex.quote(local_ref)}"
            ),
            sandbox_type=sandbox_type,
        )

        remote_exists = await _succeeds(
            backend,
            (
                f"cd {rendered_repo} && "
                "git show-ref --verify --quiet "
                f"{shlex.quote(remote_ref)}"
            ),
            sandbox_type=sandbox_type,
        )

        if local_exists:
            await _run(
                backend,
                (
                    f"cd {rendered_repo} && "
                    f"git checkout {shlex.quote(desired_head)}"
                ),
                sandbox_type=sandbox_type,
            )

            await _sync_current_branch(
                backend,
                repo_dir=target.repo_dir,
                branch=desired_head,
                dirty=False,
                sandbox_type=sandbox_type,
            )

        elif remote_exists:
            await _run(
                backend,
                (
                    f"cd {rendered_repo} && "
                    "git checkout -b "
                    f"{shlex.quote(desired_head)} "
                    "--track "
                    f"{shlex.quote(f'origin/{desired_head}')}"
                ),
                sandbox_type=sandbox_type,
            )

        else:
            await _run(
                backend,
                (
                    f"cd {rendered_repo} && "
                    "git checkout -b "
                    f"{shlex.quote(desired_head)} "
                    f"{shlex.quote(f'origin/{base}')}"
                ),
                sandbox_type=sandbox_type,
            )

    workspace = await inspect_repository_workspace(
        backend,
        target,
        sandbox_type=sandbox_type,
    )

    if workspace.head_branch != desired_head:
        raise RepositoryWorkspaceError(
            "Prepared repository branch does not match thread branch"
        )

    return workspace
