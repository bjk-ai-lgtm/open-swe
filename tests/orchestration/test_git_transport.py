from dataclasses import dataclass

import pytest

from agent.orchestration.git_transport import configure_git_transport
from agent.utils.sandbox_state import SandboxBackendProxy


@dataclass
class _Result:
    exit_code: int = 0
    result: str = ""


class _Process:
    def __init__(self, result: _Result | None = None):
        self.result = result or _Result()
        self.commands: list[tuple[str, int | None]] = []

    def exec(self, command: str, timeout: int | None = None):
        self.commands.append((command, timeout))
        return self.result


class _NativeSandbox:
    def __init__(self, process: _Process):
        self.process = process


class _Backend:
    def __init__(self, result: _Result | None = None):
        self.process = _Process(result)
        self._sandbox = _NativeSandbox(self.process)


@pytest.mark.asyncio
async def test_non_daytona_provider_is_untouched() -> None:
    backend = _Backend()

    configured = await configure_git_transport(
        backend,
        sandbox_type="langsmith",
    )

    assert configured is False
    assert backend.process.commands == []


@pytest.mark.asyncio
async def test_daytona_configures_proxy_ca_http11_and_secret_header() -> None:
    backend = _Backend()

    configured = await configure_git_transport(
        backend,
        sandbox_type="daytona",
    )

    assert configured is True
    assert len(backend.process.commands) == 1

    command, timeout = backend.process.commands[0]

    assert timeout == 60
    assert "SSL_CERT_FILE" in command
    assert "/etc/daytona/netleash/ca.crt" in command
    assert "http.version HTTP/1.1" in command
    assert "http.sslCAInfo" in command
    assert "GITHUB_BASIC_AUTH" in command
    assert "Authorization: Basic $GITHUB_BASIC_AUTH" in command
    assert "--unset-all http.https://github.com/.extraHeader" in command


@pytest.mark.asyncio
async def test_daytona_configuration_failure_is_fatal() -> None:
    backend = _Backend(
        _Result(
            exit_code=42,
            result="Daytona Git CA is unavailable",
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to configure Daytona Git transport",
    ):
        await configure_git_transport(
            backend,
            sandbox_type="daytona",
        )


@pytest.mark.asyncio
async def test_daytona_requires_native_process_exec() -> None:
    class _BackendWithoutNative:
        pass

    with pytest.raises(
        RuntimeError,
        match="does not expose native process.exec",
    ):
        await configure_git_transport(
            _BackendWithoutNative(),
            sandbox_type="daytona",
        )

@pytest.mark.asyncio
async def test_daytona_accepts_open_swe_sandbox_proxy() -> None:
    backend = _Backend()
    proxy = SandboxBackendProxy(
        backend,
        thread_id="thread-proxy",
    )

    configured = await configure_git_transport(
        proxy,
        sandbox_type="daytona",
    )

    assert configured is True
    assert len(backend.process.commands) == 1
