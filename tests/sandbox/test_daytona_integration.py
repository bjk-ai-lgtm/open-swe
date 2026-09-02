import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


class _FakeSandboxState(Enum):
    STARTED = "started"
    STOPPED = "stopped"
    STARTING = "starting"
    ERROR = "error"


class _FakeCreateSandboxFromSnapshotParams:
    def __init__(self, *, snapshot: str):
        self.snapshot = snapshot


class _FakeDaytonaConfig:
    def __init__(self, *, api_key: str):
        self.api_key = api_key


class _FakeDaytonaSandbox:
    def __init__(self, *, sandbox):
        self.sandbox = sandbox


def _load_daytona_module(monkeypatch):
    fake_daytona = types.ModuleType("daytona")
    fake_daytona.__dict__["CreateSandboxFromSnapshotParams"] = _FakeCreateSandboxFromSnapshotParams
    fake_daytona.__dict__["DaytonaConfig"] = _FakeDaytonaConfig
    fake_daytona.__dict__["Daytona"] = object
    fake_daytona.__dict__["SandboxState"] = _FakeSandboxState

    fake_langchain_daytona = types.ModuleType("langchain_daytona")
    fake_langchain_daytona.__dict__["DaytonaSandbox"] = _FakeDaytonaSandbox

    monkeypatch.setitem(sys.modules, "daytona", fake_daytona)
    monkeypatch.setitem(sys.modules, "langchain_daytona", fake_langchain_daytona)
    module_path = ROOT / "agent" / "integrations" / "daytona.py"
    spec = importlib.util.spec_from_file_location("daytona_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_daytona_params_default_to_existing_snapshot(monkeypatch):
    monkeypatch.delenv("DAYTONA_SANDBOX_SNAPSHOT", raising=False)
    module = _load_daytona_module(monkeypatch)

    params = module._get_daytona_sandbox_params()

    assert params.snapshot == "daytonaio/sandbox:0.6.0"


def test_daytona_params_use_env_snapshot(monkeypatch):
    monkeypatch.setenv("DAYTONA_SANDBOX_SNAPSHOT", "custom/snapshot:1.0")
    module = _load_daytona_module(monkeypatch)

    params = module._get_daytona_sandbox_params()

    assert params.snapshot == "custom/snapshot:1.0"


def test_daytona_params_reject_empty_snapshot(monkeypatch):
    monkeypatch.setenv("DAYTONA_SANDBOX_SNAPSHOT", "  ")
    module = _load_daytona_module(monkeypatch)

    try:
        module._get_daytona_sandbox_params()
    except ValueError as exc:
        assert "DAYTONA_SANDBOX_SNAPSHOT must not be empty" in str(exc)
    else:
        raise AssertionError("expected empty Daytona snapshot to fail")



class _FakeSandbox:
    def __init__(self, state: _FakeSandboxState):
        self.state = state


class _FakeDaytonaClient:
    def __init__(
        self,
        sandbox: _FakeSandbox,
        *,
        start_error: Exception | None = None,
    ):
        self.sandbox = sandbox
        self.start_error = start_error
        self.get_calls: list[str] = []
        self.start_calls: list[tuple[object, int]] = []
        self.create_calls: list[object] = []

    def get(self, sandbox_id: str):
        self.get_calls.append(sandbox_id)
        return self.sandbox

    def start(self, sandbox, timeout: int = 60) -> None:
        self.start_calls.append((sandbox, timeout))

        if self.start_error is not None:
            raise self.start_error

        sandbox.state = _FakeSandboxState.STARTED

    def create(self, *, params):
        self.create_calls.append(params)
        return self.sandbox


def _patch_daytona_client(
    monkeypatch,
    module,
    client: _FakeDaytonaClient,
) -> None:
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")
    monkeypatch.setattr(
        module,
        "Daytona",
        lambda *, config: client,
    )


def test_existing_stopped_sandbox_is_started_without_create(
    monkeypatch,
) -> None:
    module = _load_daytona_module(monkeypatch)
    sandbox = _FakeSandbox(_FakeSandboxState.STOPPED)
    client = _FakeDaytonaClient(sandbox)
    _patch_daytona_client(monkeypatch, module, client)

    result = module.create_daytona_sandbox("sandbox-1")

    assert result.sandbox is sandbox
    assert client.get_calls == ["sandbox-1"]
    assert client.start_calls == [
        (
            sandbox,
            module.DAYTONA_SANDBOX_START_TIMEOUT_SECONDS,
        )
    ]
    assert client.create_calls == []


def test_existing_started_sandbox_is_reused_without_start(
    monkeypatch,
) -> None:
    module = _load_daytona_module(monkeypatch)
    sandbox = _FakeSandbox(_FakeSandboxState.STARTED)
    client = _FakeDaytonaClient(sandbox)
    _patch_daytona_client(monkeypatch, module, client)

    result = module.create_daytona_sandbox("sandbox-1")

    assert result.sandbox is sandbox
    assert client.get_calls == ["sandbox-1"]
    assert client.start_calls == []
    assert client.create_calls == []


def test_existing_sandbox_start_failure_never_creates_replacement(
    monkeypatch,
) -> None:
    module = _load_daytona_module(monkeypatch)
    sandbox = _FakeSandbox(_FakeSandboxState.STOPPED)
    client = _FakeDaytonaClient(
        sandbox,
        start_error=RuntimeError("start failed"),
    )
    _patch_daytona_client(monkeypatch, module, client)

    with pytest.raises(RuntimeError, match="start failed"):
        module.create_daytona_sandbox("sandbox-1")

    assert client.get_calls == ["sandbox-1"]
    assert len(client.start_calls) == 1
    assert client.create_calls == []


def test_existing_sandbox_transitional_state_fails_closed(
    monkeypatch,
) -> None:
    module = _load_daytona_module(monkeypatch)
    sandbox = _FakeSandbox(_FakeSandboxState.STARTING)
    client = _FakeDaytonaClient(sandbox)
    _patch_daytona_client(monkeypatch, module, client)

    with pytest.raises(
        RuntimeError,
        match="state=starting",
    ):
        module.create_daytona_sandbox("sandbox-1")

    assert client.get_calls == ["sandbox-1"]
    assert client.start_calls == []
    assert client.create_calls == []


def test_new_daytona_sandbox_uses_create_without_reconnect(
    monkeypatch,
) -> None:
    module = _load_daytona_module(monkeypatch)
    sandbox = _FakeSandbox(_FakeSandboxState.STARTED)
    client = _FakeDaytonaClient(sandbox)
    _patch_daytona_client(monkeypatch, module, client)

    result = module.create_daytona_sandbox()

    assert result.sandbox is sandbox
    assert client.get_calls == []
    assert client.start_calls == []
    assert len(client.create_calls) == 1
