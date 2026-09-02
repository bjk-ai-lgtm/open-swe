from dataclasses import dataclass

import pytest

from agent.orchestration.sandbox_command import (
    execute_control_plane_command,
)
from agent.utils.sandbox_state import SandboxBackendProxy


@dataclass
class NativeResult:
    exit_code: int = 0
    result: str = ""


@dataclass
class BackendResult:
    exit_code: int = 0
    output: str = ""


class Process:
    def __init__(self):
        self.calls = []

    def exec(self, command, timeout=None):
        self.calls.append((command, timeout))
        return NativeResult(
            exit_code=0,
            result="native-ok",
        )


class NativeSandbox:
    def __init__(self, process):
        self.process = process


class DaytonaBackend:
    def __init__(self):
        self.process = Process()
        self._sandbox = NativeSandbox(self.process)

    async def aexecute(self, command, timeout=None):
        raise AssertionError(
            "Daytona control-plane commands must bypass aexecute"
        )


class StandardBackend:
    def __init__(self):
        self.calls = []

    async def aexecute(self, command, timeout=None):
        self.calls.append((command, timeout))
        return BackendResult(
            exit_code=0,
            output="standard-ok",
        )


@pytest.mark.asyncio
async def test_daytona_uses_native_process_exec() -> None:
    backend = DaytonaBackend()

    result = await execute_control_plane_command(
        backend,
        "echo test",
        timeout=45,
        sandbox_type="daytona",
    )

    assert result.exit_code == 0
    assert result.output == "native-ok"
    assert backend.process.calls == [
        ("echo test", 45),
    ]


@pytest.mark.asyncio
async def test_daytona_unwraps_open_swe_proxy() -> None:
    backend = DaytonaBackend()
    proxy = SandboxBackendProxy(
        backend,
        thread_id="thread-control-plane",
    )

    result = await execute_control_plane_command(
        proxy,
        "echo proxy",
        sandbox_type="daytona",
    )

    assert result.output == "native-ok"
    assert backend.process.calls == [
        ("echo proxy", 60),
    ]


@pytest.mark.asyncio
async def test_non_daytona_uses_backend_aexecute() -> None:
    backend = StandardBackend()

    result = await execute_control_plane_command(
        backend,
        "echo normal",
        timeout=30,
        sandbox_type="langsmith",
    )

    assert result.exit_code == 0
    assert result.output == "standard-ok"
    assert backend.calls == [
        ("echo normal", 30),
    ]


@pytest.mark.asyncio
async def test_daytona_requires_native_process_exec() -> None:
    class MissingNativeBackend:
        pass

    with pytest.raises(
        RuntimeError,
        match="does not expose native process.exec",
    ):
        await execute_control_plane_command(
            MissingNativeBackend(),
            "echo fail",
            sandbox_type="daytona",
        )


@pytest.mark.asyncio
async def test_timeout_must_be_positive() -> None:
    with pytest.raises(
        ValueError,
        match="timeout must be positive",
    ):
        await execute_control_plane_command(
            StandardBackend(),
            "echo fail",
            timeout=0,
            sandbox_type="langsmith",
        )
