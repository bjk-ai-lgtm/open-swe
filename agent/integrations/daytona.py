import os

from daytona import CreateSandboxFromSnapshotParams, Daytona, DaytonaConfig, SandboxState
from langchain_daytona import DaytonaSandbox

DEFAULT_DAYTONA_SANDBOX_SNAPSHOT = "daytonaio/sandbox:0.6.0"
DAYTONA_SANDBOX_SNAPSHOT_ENV = "DAYTONA_SANDBOX_SNAPSHOT"
DAYTONA_SANDBOX_START_TIMEOUT_SECONDS = 120


def _get_daytona_sandbox_params() -> CreateSandboxFromSnapshotParams:
    snapshot = os.getenv(DAYTONA_SANDBOX_SNAPSHOT_ENV, DEFAULT_DAYTONA_SANDBOX_SNAPSHOT).strip()
    if not snapshot:
        raise ValueError(f"{DAYTONA_SANDBOX_SNAPSHOT_ENV} must not be empty")
    return CreateSandboxFromSnapshotParams(snapshot=snapshot)


def _ensure_existing_daytona_sandbox_started(
    daytona: Daytona,
    sandbox,
    *,
    sandbox_id: str,
):
    state = sandbox.state

    if state == SandboxState.STARTED:
        return sandbox

    if state == SandboxState.STOPPED:
        daytona.start(
            sandbox,
            timeout=DAYTONA_SANDBOX_START_TIMEOUT_SECONDS,
        )
        return sandbox

    state_value = getattr(state, "value", str(state))
    raise RuntimeError(
        f"Daytona sandbox {sandbox_id} cannot be reconnected safely "
        f"while state={state_value}"
    )


def create_daytona_sandbox(sandbox_id: str | None = None):
    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        raise ValueError("DAYTONA_API_KEY environment variable is required")

    daytona = Daytona(config=DaytonaConfig(api_key=api_key))

    if sandbox_id:
        sandbox = daytona.get(sandbox_id)
        sandbox = _ensure_existing_daytona_sandbox_started(
            daytona,
            sandbox,
            sandbox_id=sandbox_id,
        )
    else:
        sandbox = daytona.create(params=_get_daytona_sandbox_params())

    return DaytonaSandbox(sandbox=sandbox)
