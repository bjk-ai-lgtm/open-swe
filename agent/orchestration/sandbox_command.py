"""Provider-aware execution for orchestration control-plane commands."""

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from deepagents.backends.protocol import SandboxBackendProtocol

from agent.utils.sandbox_state import unwrap_sandbox_backend


@dataclass(frozen=True)
class SandboxCommandResult:
    """Normalized result for a control-plane sandbox command."""

    exit_code: int
    output: str


async def execute_control_plane_command(
    sandbox_backend: SandboxBackendProtocol,
    command: str,
    *,
    timeout: int = 60,
    sandbox_type: str | None = None,
) -> SandboxCommandResult:
    """Execute one deterministic runtime-owned sandbox command.

    Daytona bypasses ``langchain-daytona`` session execution because that
    wrapper can raise during session cleanup after the command side effect has
    already completed.
    """
    if timeout <= 0:
        raise ValueError("Control-plane command timeout must be positive")

    provider = (
        sandbox_type or os.getenv("SANDBOX_TYPE", "langsmith")
    ).strip().lower()

    if provider == "daytona":
        backend = unwrap_sandbox_backend(sandbox_backend)
        native_sandbox = getattr(backend, "_sandbox", None)
        process = getattr(native_sandbox, "process", None)
        exec_fn = getattr(process, "exec", None)

        if not callable(exec_fn):
            raise RuntimeError(
                "Daytona backend does not expose native process.exec"
            )

        result = await asyncio.to_thread(
            exec_fn,
            command,
            timeout=timeout,
        )

        exit_code = getattr(result, "exit_code", None)
        output: Any = getattr(result, "result", "")
    else:
        result = await sandbox_backend.aexecute(
            command,
            timeout=timeout,
        )

        exit_code = getattr(result, "exit_code", None)
        output = getattr(result, "output", "")

    if not isinstance(exit_code, int):
        raise RuntimeError(
            "Sandbox command returned an invalid exit code"
        )

    return SandboxCommandResult(
        exit_code=exit_code,
        output=(
            output
            if isinstance(output, str)
            else str(output or "")
        ),
    )
