"""Runtime-owned Git publication after deterministic validation."""

import posixpath
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langgraph.graph.state import RunnableConfig

from agent.tools.open_pull_request import open_pull_request

from .bootstrap import OrchestratorRuntimeContext
from .publication_target import resolve_publication_target
from .repository_workspace import (
    RepositoryWorkspace,
    inspect_repository_workspace,
)
from .sandbox_command import (
    SandboxCommandResult,
    execute_control_plane_command,
)

CommandExecutor = Callable[..., Awaitable[SandboxCommandResult]]
WorkspaceInspector = Callable[..., Awaitable[RepositoryWorkspace]]
PullRequestOpener = Callable[..., Awaitable[dict[str, Any]]]


class RuntimePublicationError(RuntimeError):
    """Raised when validated changes cannot be published safely."""


def _task_title(task: str) -> str:
    for line in task.splitlines():
        normalized = " ".join(line.split())
        if normalized:
            return normalized[:120]

    return "Open SWE automated change"


def _commit_subject(task: str) -> str:
    title = _task_title(task)
    prefix = "open-swe: "
    return f"{prefix}{title}"[:72]


def _pull_request_body(task: str) -> str:
    return (
        "Automated changes produced by the Open SWE orchestrator.\n\n"
        "## Task\n\n"
        f"{task.strip()}"
    )


@dataclass
class OpenSWERuntimePublisher:
    """Commit, push, reconcile, and open a PR for one validated task."""

    config: RunnableConfig
    context: OrchestratorRuntimeContext
    workspace_inspector: WorkspaceInspector = (
        inspect_repository_workspace
    )
    command_executor: CommandExecutor = (
        execute_control_plane_command
    )
    pr_opener: PullRequestOpener = open_pull_request
    sandbox_type: str | None = None

    async def _execute(
        self,
        command: str,
        *,
        timeout: int = 60,
    ) -> SandboxCommandResult:
        return await self.command_executor(
            self.context.sandbox_backend,
            command,
            timeout=timeout,
            sandbox_type=self.sandbox_type,
        )

    async def _run(
        self,
        command: str,
        *,
        timeout: int = 60,
    ) -> str:
        result = await self._execute(
            command,
            timeout=timeout,
        )

        if result.exit_code != 0:
            detail = result.output.strip()
            suffix = f": {detail}" if detail else ""

            raise RuntimePublicationError(
                "Publication command failed with exit code "
                f"{result.exit_code}: {command}{suffix}"
            )

        return result.output.strip()

    async def _remote_sha(
        self,
        *,
        repo_dir: str,
        branch: str,
    ) -> str | None:
        rendered_repo = shlex.quote(repo_dir)
        remote_ref = f"refs/heads/{branch}"

        output = await self._run(
            (
                f"cd {rendered_repo} && "
                "git ls-remote --heads origin "
                f"{shlex.quote(remote_ref)}"
            ),
            timeout=120,
        )

        for line in output.splitlines():
            parts = line.split()

            if (
                len(parts) >= 2
                and parts[1] == remote_ref
            ):
                return parts[0]

        return None

    async def _commit_pending_changes(
        self,
        *,
        repo_dir: str,
        task: str,
    ) -> None:
        rendered_repo = shlex.quote(repo_dir)

        dirty = await self._run(

                f"cd {rendered_repo} && "
                "git status --porcelain "
                "--untracked-files=normal"

        )

        if not dirty:
            return

        await self._run(
            f"cd {rendered_repo} && git add -A"
        )

        staged = await self._execute(

                f"cd {rendered_repo} && "
                "git diff --cached --quiet"

        )

        if staged.exit_code == 1:
            await self._run(
                (
                    f"cd {rendered_repo} && "
                    "git commit -m "
                    f"{shlex.quote(_commit_subject(task))}"
                ),
                timeout=120,
            )
        elif staged.exit_code != 0:
            raise RuntimePublicationError(
                "Unable to determine whether staged "
                "publication changes exist"
            )

        remaining = await self._run(

                f"cd {rendered_repo} && "
                "git status --porcelain "
                "--untracked-files=normal"

        )

        if remaining:
            raise RuntimePublicationError(
                "Repository remained dirty after runtime commit"
            )

    async def _push_reconciled(
        self,
        *,
        repo_dir: str,
        branch: str,
        local_sha: str,
    ) -> None:
        rendered_repo = shlex.quote(repo_dir)
        head_ref = f"refs/heads/{branch}"
        tracking_ref = f"refs/remotes/origin/{branch}"

        remote_sha = await self._remote_sha(
            repo_dir=repo_dir,
            branch=branch,
        )

        if remote_sha == local_sha:
            return

        if remote_sha is not None:
            await self._run(
                (
                    f"cd {rendered_repo} && "
                    "git fetch origin "
                    f"{shlex.quote(head_ref)}:"
                    f"{shlex.quote(tracking_ref)}"
                ),
                timeout=240,
            )

            ancestor = await self._execute(

                    f"cd {rendered_repo} && "
                    "git merge-base --is-ancestor "
                    f"{shlex.quote(tracking_ref)} HEAD"

            )

            if ancestor.exit_code != 0:
                raise RuntimePublicationError(
                    "Remote publication branch is not an "
                    "ancestor of the validated local branch"
                )

        push_result = await self._execute(
            (
                f"cd {rendered_repo} && "
                "git push origin "
                f"HEAD:{shlex.quote(head_ref)}"
            ),
            timeout=240,
        )

        reconciled_sha = await self._remote_sha(
            repo_dir=repo_dir,
            branch=branch,
        )

        if reconciled_sha == local_sha:
            return

        detail = push_result.output.strip()
        suffix = f": {detail}" if detail else ""

        raise RuntimePublicationError(
            "Git push did not reconcile to the validated "
            f"local HEAD{suffix}"
        )

    async def __call__(
        self,
        *,
        thread_id: str,
        task: str,
        work_dir: str,
    ) -> None:
        if thread_id != self.context.thread_id:
            raise RuntimePublicationError(
                "Publication thread does not match runtime context"
            )

        if not task.strip():
            raise ValueError("Publication task cannot be empty")

        target = resolve_publication_target(
            self.config,
            work_dir=self.context.work_dir,
        )

        if posixpath.normpath(work_dir) != target.repo_dir:
            raise RuntimePublicationError(
                "Publication work directory does not match "
                "the prepared repository target"
            )

        workspace = await self.workspace_inspector(
            self.context.sandbox_backend,
            target,
            sandbox_type=self.sandbox_type,
        )

        expected_head = (
            target.configured_head
            or f"open-swe/{thread_id}"
        )

        if workspace.head_branch != expected_head:
            raise RuntimePublicationError(
                "Publication branch does not match "
                "the thread-stable branch"
            )

        rendered_repo = shlex.quote(target.repo_dir)
        base_ref = (
            f"refs/remotes/origin/{workspace.base_branch}"
        )

        await self._commit_pending_changes(
            repo_dir=target.repo_dir,
            task=task,
        )

        base_ancestor = await self._execute(

                f"cd {rendered_repo} && "
                "git merge-base --is-ancestor "
                f"{shlex.quote(base_ref)} HEAD"

        )

        if base_ancestor.exit_code != 0:
            raise RuntimePublicationError(
                "Validated publication branch no longer "
                "descends from the configured base branch"
            )

        commit_count_raw = await self._run(

                f"cd {rendered_repo} && "
                "git rev-list --count "
                f"{shlex.quote(f'{base_ref}..HEAD')}"

        )

        try:
            commit_count = int(commit_count_raw)
        except ValueError as exc:
            raise RuntimePublicationError(
                "Git returned an invalid publication commit count"
            ) from exc

        # Research/no-code tasks are valid successful no-op publications.
        if commit_count == 0:
            return

        workflow_changes = await self._run(

                f"cd {rendered_repo} && "
                "git diff --name-only "
                f"{shlex.quote(f'{base_ref}...HEAD')} "
                "-- .github/workflows"

        )

        if workflow_changes:
            raise RuntimePublicationError(
                "Publishing GitHub Actions workflow changes "
                "requires explicit approval"
            )

        local_sha = await self._run(
            f"cd {rendered_repo} && git rev-parse HEAD"
        )

        await self._push_reconciled(
            repo_dir=target.repo_dir,
            branch=workspace.head_branch,
            local_sha=local_sha,
        )

        result = await self.pr_opener(
            owner=target.owner,
            repo=target.repo,
            head=workspace.head_branch,
            base=workspace.base_branch,
            title=_task_title(task),
            body=_pull_request_body(task),
            draft=True,
        )

        if (
            not isinstance(result, dict)
            or result.get("success") is not True
        ):
            reason = (
                result.get("error")
                if isinstance(result, dict)
                else None
            )

            raise RuntimePublicationError(
                "Native Open SWE pull request creation failed"
                + (f": {reason}" if reason else "")
            )


def build_runtime_publisher(
    config: RunnableConfig,
    context: OrchestratorRuntimeContext,
) -> OpenSWERuntimePublisher:
    """Build the publication adapter for one runtime thread."""
    return OpenSWERuntimePublisher(
        config=config,
        context=context,
    )
