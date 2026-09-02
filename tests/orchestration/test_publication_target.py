import pytest

from agent.orchestration.publication_target import (
    PublicationTargetError,
    resolve_publication_target,
)


def test_resolves_github_issue_style_repository_context() -> None:
    target = resolve_publication_target(
        {
            "configurable": {
                "source": "github",
                "repo": {
                    "owner": "langchain-ai",
                    "name": "open-swe",
                },
            }
        },
        work_dir="/workspace",
    )

    assert target.owner == "langchain-ai"
    assert target.repo == "open-swe"
    assert target.repo_dir == "/workspace/open-swe"
    assert target.configured_head is None
    assert target.configured_base is None


def test_preserves_explicit_branch_metadata() -> None:
    target = resolve_publication_target(
        {
            "configurable": {
                "repo": {
                    "owner": "bjk-ai-lgtm",
                    "name": "open-swe",
                },
                "branch_name": "feature/example",
                "base_branch": "custom/v0.1",
            }
        },
        work_dir="/home/daytona",
    )

    assert target.configured_head == "feature/example"
    assert target.configured_base == "custom/v0.1"


def test_missing_repository_metadata_fails_closed() -> None:
    with pytest.raises(
        PublicationTargetError,
        match="does not identify a target repository",
    ):
        resolve_publication_target(
            {"configurable": {}},
            work_dir="/workspace",
        )


def test_unsafe_repository_name_is_rejected() -> None:
    with pytest.raises(
        PublicationTargetError,
        match="repository name",
    ):
        resolve_publication_target(
            {
                "configurable": {
                    "repo": {
                        "owner": "example",
                        "name": "../escape",
                    },
                }
            },
            work_dir="/workspace",
        )


def test_empty_work_dir_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="work directory cannot be empty",
    ):
        resolve_publication_target(
            {
                "configurable": {
                    "repo": {
                        "owner": "example",
                        "name": "project",
                    },
                }
            },
            work_dir="",
        )
