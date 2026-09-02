"""Resolve deterministic repository targets for runtime-owned publication."""

import posixpath
import re
from dataclasses import dataclass
from typing import Any

from langgraph.graph.state import RunnableConfig

_GITHUB_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


class PublicationTargetError(RuntimeError):
    """Raised when a publication target cannot be resolved safely."""


@dataclass(frozen=True)
class PublicationTarget:
    """Repository identity and optional configured branch hints."""

    owner: str
    repo: str
    repo_dir: str
    configured_head: str | None
    configured_base: str | None


def _non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    value = value.strip()
    return value or None


def _github_segment(value: Any, *, field: str) -> str:
    segment = _non_empty_string(value)

    if segment is None or not _GITHUB_SEGMENT.fullmatch(segment):
        raise PublicationTargetError(
            f"Invalid or missing GitHub repository {field}"
        )

    if segment in {".", ".."}:
        raise PublicationTargetError(
            f"Invalid GitHub repository {field}"
        )

    return segment


def resolve_publication_target(
    config: RunnableConfig,
    *,
    work_dir: str,
) -> PublicationTarget:
    """Resolve publication metadata without guessing missing branch information."""
    if not isinstance(work_dir, str) or not work_dir.strip():
        raise ValueError("Publication work directory cannot be empty")

    configurable = config.get("configurable")

    if not isinstance(configurable, dict):
        raise PublicationTargetError(
            "Runnable config does not contain publication metadata"
        )

    repo_config = configurable.get("repo")

    if not isinstance(repo_config, dict):
        raise PublicationTargetError(
            "Runnable config does not identify a target repository"
        )

    owner = _github_segment(
        repo_config.get("owner"),
        field="owner",
    )
    repo = _github_segment(
        repo_config.get("name"),
        field="name",
    )

    return PublicationTarget(
        owner=owner,
        repo=repo,
        repo_dir=posixpath.join(
            posixpath.normpath(work_dir),
            repo,
        ),
        configured_head=_non_empty_string(
            configurable.get("branch_name")
        ),
        configured_base=_non_empty_string(
            configurable.get("base_branch")
        ),
    )
