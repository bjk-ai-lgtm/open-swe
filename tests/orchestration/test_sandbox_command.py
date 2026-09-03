import asyncio
import threading
from dataclasses import dataclass
from typing import Any

import pytest

import agent.orchestration.sandbox_command as sandbox_command
from agent.orchestration.sandbox_command import (
    execute_control_plane_command,
)
from agent.utils.sandbox_state import SandboxBackendProxy


@dataclass
class NativeExecResult:
    exit_code: int = 0
    result: str = ""


@dataclass
class BackendResult:
    exit_code: int = 0
    output: str = ""


@dataclass
class SessionResponse:
    cmd_id: str = "command-1"
    exit_code: int | None = None
    output: str | None = None


@dataclass
class RemoteCommand:
    exit_code: int | None



@dataclass
class CommandLogs:
    output: str = ""
    stdout: str = ""
    stderr: str = ""


class Process:
    def __init__(
        self,
        *,
        exit_code: int | None = 0,
        stdout: str = "native-ok",
        stderr: str = "",
        pgid: int = 4321,
        kill_exit_code: int = 0,
    ):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.pgid = pgid
        self.kill_exit_code = kill_exit_code

        self.create_calls: list[str] = []
        self.execute_calls: list[tuple[str, Any, int | None]] = []
        self.poll_calls: list[tuple[str, str]] = []
        self.log_calls: list[tuple[str, str]] = []
        self.delete_calls: list[str] = []
        self.exec_calls: list[tuple[str, int | None]] = []

        self.poll_event = threading.Event()

    def create_session(self, session_id: str) -> None:
        self.create_calls.append(session_id)

    def execute_session_command(
        self,
        session_id: str,
        request,
        timeout: int | None = None,
    ):
        self.execute_calls.append(
            (session_id, request, timeout)
        )
        return SessionResponse()

    def get_session_command(
        self,
        session_id: str,
        command_id: str,
    ):
        self.poll_calls.append(
            (session_id, command_id)
        )
        self.poll_event.set()

        return RemoteCommand(
            exit_code=self.exit_code
        )

    def get_session_command_logs(
        self,
        session_id: str,
        command_id: str,
    ):
        self.log_calls.append(
            (session_id, command_id)
        )

        return CommandLogs(
            output="\x01\x01framed-output",
            stdout=self.stdout,
            stderr=self.stderr,
        )

    def delete_session(
        self,
        session_id: str,
    ) -> None:
        self.delete_calls.append(session_id)

    def exec(
        self,
        command: str,
        timeout: int | None = None,
    ):
        self.exec_calls.append(
            (command, timeout)
        )

        if command.startswith("cat "):
            return NativeExecResult(
                exit_code=0,
                result=f"{self.pgid}\n",
            )

        if "/bin/kill -TERM" in command:
            return NativeExecResult(
                exit_code=self.kill_exit_code,
                result="",
            )

        return NativeExecResult(
            exit_code=0,
            result="",
        )


class NativeSandbox:
    def __init__(self, process):
        self.process = process


class DaytonaBackend:
    def __init__(self, process=None):
        self.process = process or Process()
        self._sandbox = NativeSandbox(
            self.process
        )

    async def aexecute(
        self,
        command,
        timeout=None,
    ):
        raise AssertionError(
            "Daytona control-plane commands "
            "must bypass aexecute"
        )


class StandardBackend:
    def __init__(self):
        self.calls = []

    async def aexecute(
        self,
        command,
        timeout=None,
    ):
        self.calls.append(
            (command, timeout)
        )

        return BackendResult(
            exit_code=0,
            output="standard-ok",
        )


@pytest.mark.asyncio
async def test_daytona_uses_isolated_session() -> None:
    backend: Any = DaytonaBackend()


    result = await execute_control_plane_command(
        backend,
        "echo test",
        timeout=45,
        sandbox_type="daytona",
    )

    assert result.exit_code == 0
    assert result.output == "native-ok"

    assert len(
        backend.process.create_calls
    ) == 1

    session_id = (
        backend.process.create_calls[0]
    )

    assert session_id.startswith(
        "open-swe-control-"
    )

    assert len(
        backend.process.execute_calls
    ) == 1

    execute_session_id, request, control_timeout = (
        backend.process.execute_calls[0]
    )

    assert execute_session_id == session_id
    assert control_timeout == (
        sandbox_command.DAYTONA_CONTROL_COMMAND_TIMEOUT_SECONDS
    )

    assert request.run_async is True
    assert "setsid sh -c" in request.command
    assert "echo test" in request.command
    assert ".pgid" in request.command

    assert backend.process.poll_calls == [
        (session_id, "command-1")
    ]

    assert backend.process.log_calls == [
        (session_id, "command-1")
    ]

    assert backend.process.delete_calls == [
        session_id
    ]

    kill_commands = [
        command
        for command, _ in backend.process.exec_calls
        if "/bin/kill -TERM" in command
    ]

    assert kill_commands == []


@pytest.mark.asyncio
async def test_daytona_combines_clean_stdout_and_stderr() -> None:
    process = Process(
        stdout="stdout-value\n",
        stderr="stderr-value\n",
    )

    backend: Any = DaytonaBackend(process)

    result = await execute_control_plane_command(
        backend,
        "echo test",
        sandbox_type="daytona",
    )

    assert result.exit_code == 0
    assert result.output == (
        "stdout-value\n"
        "stderr-value\n"
    )

    assert "\x01" not in result.output


@pytest.mark.asyncio
async def test_daytona_preserves_nonzero_exit_code() -> None:
    process = Process(
        exit_code=17,
        stdout="",
        stderr="command failed\n",
    )

    backend: Any = DaytonaBackend(process)

    result = await execute_control_plane_command(
        backend,
        "false",
        sandbox_type="daytona",
    )

    assert result.exit_code == 17
    assert result.output == "command failed\n"


@pytest.mark.asyncio
async def test_daytona_unwraps_open_swe_proxy() -> None:
    backend: Any = DaytonaBackend()

    proxy = SandboxBackendProxy(
        backend,
        thread_id="thread-control-plane",
    )

    result = await execute_control_plane_command(
        proxy,
        "echo proxy",
        sandbox_type="daytona",
    )

    assert result.exit_code == 0
    assert result.output == "native-ok"

    assert len(
        backend.process.create_calls
    ) == 1

    assert len(
        backend.process.delete_calls
    ) == 1


@pytest.mark.asyncio
async def test_daytona_timeout_terminates_process_group() -> None:
    process = Process(
        exit_code=None,
    )

    backend: Any = DaytonaBackend(process)

    with pytest.raises(
        TimeoutError,
        match="timed out",
    ):
        await execute_control_plane_command(
            backend,
            "sleep 999",
            timeout=1,
            sandbox_type="daytona",
        )

    kill_commands = [
        command
        for command, _ in process.exec_calls
        if "/bin/kill -TERM" in command
    ]

    assert len(kill_commands) == 1
    assert (
        "/bin/kill -TERM -- -4321"
        in kill_commands[0]
    )

    assert len(process.delete_calls) == 1


@pytest.mark.asyncio
async def test_daytona_cancellation_terminates_process_group() -> None:
    process = Process(
        exit_code=None,
    )

    backend: Any = DaytonaBackend(process)

    task = asyncio.create_task(
        execute_control_plane_command(
            backend,
            "sleep 999",
            timeout=60,
            sandbox_type="daytona",
        )
    )

    polled = await asyncio.to_thread(
        process.poll_event.wait,
        2,
    )

    assert polled is True

    task.cancel()

    with pytest.raises(
        asyncio.CancelledError
    ):
        await task

    kill_commands = [
        command
        for command, _ in process.exec_calls
        if "/bin/kill -TERM" in command
    ]

    assert len(kill_commands) == 1
    assert (
        "/bin/kill -TERM -- -4321"
        in kill_commands[0]
    )

    assert len(process.delete_calls) == 1


@pytest.mark.asyncio
async def test_timeout_preserves_original_error_when_cleanup_fails() -> None:
    process = Process(
        exit_code=None,
        kill_exit_code=70,
    )

    backend: Any = DaytonaBackend(process)

    with pytest.raises(
        TimeoutError,
        match="timed out",
    ) as exc_info:
        await execute_control_plane_command(
            backend,
            "sleep 999",
            timeout=1,
            sandbox_type="daytona",
        )

    assert len(process.delete_calls) == 1

    notes = getattr(
        exc_info.value,
        "__notes__",
        [],
    )

    assert any(
        "Daytona cleanup also failed" in note
        for note in notes
    )

    assert any(
        "process termination failed" in note
        for note in notes
    )


@pytest.mark.asyncio
async def test_cleanup_attempts_session_delete_after_termination_failure() -> None:
    process = Process(
        exit_code=None,
        kill_exit_code=70,
    )

    with pytest.raises(
        RuntimeError,
        match="control-plane cleanup failed",
    ):
        await sandbox_command._cleanup_session(
            process,
            session_id="cleanup-session",
            pid_file="/tmp/open-swe-cleanup-test.pgid",
            pgid=4321,
            terminate=True,
        )

    assert process.delete_calls == [
        "cleanup-session"
    ]


@pytest.mark.asyncio
async def test_daytona_termination_failure_is_not_silent() -> None:
    process = Process(
        exit_code=0,
        kill_exit_code=70,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Unable to terminate Daytona "
            "control-plane process group"
        ),
    ):
        await sandbox_command._terminate_process_group(
            process,
            4321,
            "/tmp/open-swe-test.pgid",
        )


@pytest.mark.asyncio
async def test_non_daytona_uses_backend_aexecute() -> None:
    backend: Any = StandardBackend()

    result = await execute_control_plane_command(
        backend,
        "echo normal",
        timeout=30,
        sandbox_type="langsmith",
    )

    assert result.exit_code == 0
    assert result.output == "standard-ok"

    assert backend.calls == [
        ("echo normal", 30)
    ]


@pytest.mark.asyncio
async def test_daytona_requires_native_process_apis() -> None:
    class MissingNativeBackend:
        pass

    backend: Any = MissingNativeBackend()

    with pytest.raises(
        RuntimeError,
        match="required native process APIs",
    ):
        await execute_control_plane_command(
            backend,
            "echo fail",
            sandbox_type="daytona",
        )


@pytest.mark.asyncio
async def test_timeout_must_be_positive() -> None:
    backend: Any = StandardBackend()

    with pytest.raises(
        ValueError,
        match="timeout must be positive",
    ):
        await execute_control_plane_command(
            backend,
            "echo fail",
            timeout=0,
            sandbox_type="langsmith",
        )
