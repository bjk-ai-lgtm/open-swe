from deepagents.backends.protocol import ExecuteResponse

from agent.validation import (
    CommandResult,
    SandboxCommandRunner,
    ValidationCheck,
)


class FakeSandboxBackend:
    def __init__(self, response: ExecuteResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, int | None]] = []

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        self.calls.append((command, timeout))
        return self.response


async def test_runs_command_through_sandbox_backend() -> None:
    backend = FakeSandboxBackend(
        ExecuteResponse(
            output="34 passed\n",
            exit_code=0,
            truncated=False,
        )
    )

    runner = SandboxCommandRunner(
        backend,
        timeout_seconds=180,
    )

    result = await runner.run(
        ValidationCheck(
            name="tests",
            command=("uv", "run", "pytest", "-q"),
        ),
        work_dir="/workspace/project",
    )

    assert result == CommandResult(
        exit_code=0,
        stdout="34 passed\n",
        stderr="",
        timed_out=False,
    )

    assert backend.calls == [
        (
            "cd /workspace/project && uv run pytest -q",
            180,
        )
    ]


async def test_quotes_command_arguments_and_work_directory() -> None:
    backend = FakeSandboxBackend(
        ExecuteResponse(
            output="ok",
            exit_code=0,
            truncated=False,
        )
    )

    runner = SandboxCommandRunner(backend)

    await runner.run(
        ValidationCheck(
            name="example",
            command=(
                "python",
                "some file.py",
                "value; echo unsafe",
            ),
        ),
        work_dir="/workspace/project with spaces",
    )

    command, _ = backend.calls[0]

    assert command == (
        "cd '/workspace/project with spaces' && python 'some file.py' 'value; echo unsafe'"
    )


async def test_maps_backend_timeout_to_command_result() -> None:
    backend = FakeSandboxBackend(
        ExecuteResponse(
            output="Error: Command timed out after 300 seconds.",
            exit_code=124,
            truncated=False,
        )
    )

    runner = SandboxCommandRunner(backend)

    result = await runner.run(
        ValidationCheck(
            name="tests",
            command=("pytest", "-q"),
        ),
        work_dir="/workspace/project",
    )

    assert result.exit_code == 124
    assert result.timed_out is True


async def test_non_timeout_exit_124_is_not_automatically_timeout() -> None:
    backend = FakeSandboxBackend(
        ExecuteResponse(
            output="Application returned status 124",
            exit_code=124,
            truncated=False,
        )
    )

    runner = SandboxCommandRunner(backend)

    result = await runner.run(
        ValidationCheck(
            name="custom",
            command=("example",),
        ),
        work_dir="/workspace/project",
    )

    assert result.exit_code == 124
    assert result.timed_out is False
