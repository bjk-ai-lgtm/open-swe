from dataclasses import dataclass
from typing import Any

import pytest

from agent.orchestration.git_transport import (
    configure_git_transport,
)
from agent.utils.sandbox_state import SandboxBackendProxy


@dataclass
class _ExecResult:
    exit_code: int = 0
    result: str = ""


@dataclass
class _SessionResponse:
    cmd_id: str = "git-command-1"
    exit_code: int | None = None
    output: str | None = None


@dataclass
class _Command:
    exit_code: int | None = 0


@dataclass
class _Logs:
    output: str = ""
    stdout: str = ""
    stderr: str = ""


class _Process:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        stdout: str = "",
        stderr: str = "",
    ):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

        self.created_sessions: list[str] = []
        self.deleted_sessions: list[str] = []
        self.session_commands: list[
            tuple[str, Any, int | None]
        ] = []
        self.polls: list[tuple[str, str]] = []
        self.log_reads: list[tuple[str, str]] = []
        self.exec_commands: list[
            tuple[str, int | None]
        ] = []

    def create_session(
        self,
        session_id: str,
    ) -> None:
        self.created_sessions.append(
            session_id
        )

    def execute_session_command(
        self,
        session_id: str,
        request,
        timeout: int | None = None,
    ):
        self.session_commands.append(
            (
                session_id,
                request,
                timeout,
            )
        )

        return _SessionResponse()

    def get_session_command(
        self,
        session_id: str,
        command_id: str,
    ):
        self.polls.append(
            (
                session_id,
                command_id,
            )
        )

        return _Command(
            exit_code=self.exit_code
        )

    def get_session_command_logs(
        self,
        session_id: str,
        command_id: str,
    ):
        self.log_reads.append(
            (
                session_id,
                command_id,
            )
        )

        return _Logs(
            stdout=self.stdout,
            stderr=self.stderr,
        )

    def delete_session(
        self,
        session_id: str,
    ) -> None:
        self.deleted_sessions.append(
            session_id
        )

    def exec(
        self,
        command: str,
        timeout: int | None = None,
    ):
        self.exec_commands.append(
            (
                command,
                timeout,
            )
        )

        if command.startswith("cat "):
            return _ExecResult(
                exit_code=0,
                result="4321\n",
            )

        return _ExecResult(
            exit_code=0,
            result="",
        )


class _NativeSandbox:
    def __init__(
        self,
        process: _Process,
    ):
        self.process = process


class _Backend:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        stdout: str = "",
        stderr: str = "",
    ):
        self.process = _Process(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )

        self._sandbox = _NativeSandbox(
            self.process
        )


@pytest.mark.asyncio
async def test_non_daytona_provider_is_untouched() -> None:
    backend: Any = _Backend()

    configured = await configure_git_transport(
        backend,
        sandbox_type="langsmith",
    )

    assert configured is False
    assert backend.process.created_sessions == []
    assert backend.process.session_commands == []


@pytest.mark.asyncio
async def test_daytona_configures_proxy_ca_http11_and_secret_header() -> None:
    backend: Any = _Backend()

    configured = await configure_git_transport(
        backend,
        sandbox_type="daytona",
    )

    assert configured is True

    assert len(
        backend.process.created_sessions
    ) == 1

    assert len(
        backend.process.session_commands
    ) == 1

    (
        session_id,
        request,
        timeout,
    ) = backend.process.session_commands[0]

    assert session_id == (
        backend.process.created_sessions[0]
    )

    assert timeout == 10
    assert request.run_async is True

    command = request.command

    assert "SSL_CERT_FILE" in command
    assert (
        "/etc/daytona/netleash/ca.crt"
        in command
    )
    assert "http.version HTTP/1.1" in command
    assert "http.sslCAInfo" in command
    assert "GITHUB_BASIC_AUTH" in command
    assert (
        "Authorization: Basic "
        "$GITHUB_BASIC_AUTH"
        in command
    )
    assert (
        "--unset-all "
        "http.https://github.com/.extraHeader"
        in command
    )

    assert backend.process.polls == [
        (
            session_id,
            "git-command-1",
        )
    ]

    assert backend.process.deleted_sessions == [
        session_id
    ]


@pytest.mark.asyncio
async def test_daytona_configuration_failure_is_fatal() -> None:
    backend: Any = _Backend(
        exit_code=42,
        stderr=(
            "Daytona Git CA is unavailable"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Failed to configure "
            "Daytona Git transport"
        ),
    ):
        await configure_git_transport(
            backend,
            sandbox_type="daytona",
        )


@pytest.mark.asyncio
async def test_daytona_requires_native_process_apis() -> None:
    class _BackendWithoutNative:
        pass

    backend: Any = _BackendWithoutNative()

    with pytest.raises(
        RuntimeError,
        match=(
            "required native process APIs"
        ),
    ):
        await configure_git_transport(
            backend,
            sandbox_type="daytona",
        )


@pytest.mark.asyncio
async def test_daytona_accepts_open_swe_sandbox_proxy() -> None:
    backend: Any = _Backend()

    proxy = SandboxBackendProxy(
        backend,
        thread_id="thread-proxy",
    )

    configured = await configure_git_transport(
        proxy,
        sandbox_type="daytona",
    )

    assert configured is True
    assert len(
        backend.process.created_sessions
    ) == 1
    assert len(
        backend.process.deleted_sessions
    ) == 1
