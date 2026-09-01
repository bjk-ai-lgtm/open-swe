from dataclasses import dataclass

import pytest

from agent.orchestration.repository_inspection import (
    RepositoryInspectionError,
    read_repository_metadata,
)


@dataclass
class FakeResponse:
    exit_code: int
    output: str


class FakeBackend:
    def __init__(
        self,
        files=None,
        *,
        exit_code=0,
        malformed=False,
    ):
        self.files = files or {}
        self.exit_code = exit_code
        self.malformed = malformed
        self.calls = []

    async def aexecute(self, command, timeout):
        self.calls.append((command, timeout))

        if self.exit_code:
            return FakeResponse(
                exit_code=self.exit_code,
                output="inspection failed",
            )

        if self.malformed:
            return FakeResponse(
                exit_code=0,
                output="unexpected-output",
            )

        for filename, content in self.files.items():
            if filename in command:
                if filename in {
                    "pyproject.toml",
                    "package.json",
                    "go.mod",
                }:
                    return FakeResponse(
                        exit_code=0,
                        output=f"1\n{content}",
                    )

                return FakeResponse(
                    exit_code=0,
                    output="1\n",
                )

        return FakeResponse(
            exit_code=0,
            output="0\n",
        )


async def test_reads_known_repository_metadata():
    backend = FakeBackend(
        {
            "pyproject.toml": "[tool.pytest.ini_options]\n",
            "uv.lock": "",
        }
    )

    files = await read_repository_metadata(
        backend,
        work_dir="/workspace/project",
    )

    assert files == {
        "pyproject.toml": "[tool.pytest.ini_options]\n",
        "uv.lock": "",
    }


async def test_quotes_repository_work_directory():
    backend = FakeBackend()

    await read_repository_metadata(
        backend,
        work_dir="/workspace/project with spaces",
    )

    assert backend.calls
    assert "cd '/workspace/project with spaces'" in backend.calls[0][0]


async def test_rejects_empty_work_directory():
    backend = FakeBackend()

    with pytest.raises(
        ValueError,
        match="work directory cannot be empty",
    ):
        await read_repository_metadata(
            backend,
            work_dir=" ",
        )

    assert backend.calls == []


async def test_command_failure_is_not_treated_as_missing_metadata():
    backend = FakeBackend(exit_code=23)

    with pytest.raises(
        RepositoryInspectionError,
        match="exit code 23",
    ):
        await read_repository_metadata(
            backend,
            work_dir="/workspace/project",
        )


async def test_malformed_inspection_output_fails_closed():
    backend = FakeBackend(malformed=True)

    with pytest.raises(
        RepositoryInspectionError,
        match="Malformed repository inspection output",
    ):
        await read_repository_metadata(
            backend,
            work_dir="/workspace/project",
        )
