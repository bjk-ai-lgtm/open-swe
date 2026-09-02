"""Provider-aware Git transport configuration for orchestration sandboxes."""

import asyncio
import os
from typing import Any

from deepagents.backends.protocol import SandboxBackendProtocol

from agent.utils.sandbox_state import unwrap_sandbox_backend

_DAYTONA_DEFAULT_CA = "/etc/daytona/netleash/ca.crt"
_DAYTONA_GITHUB_HEADER_KEY = "http.https://github.com/.extraHeader"
_DAYTONA_COMMAND_TIMEOUT_SECONDS = 60


async def _execute_daytona_native(
    sandbox_backend: SandboxBackendProtocol,
    command: str,
) -> tuple[int, str]:
    """Execute a command through Daytona's native SDK path.

    ``langchain-daytona`` can surface a session-cleanup exception after a command
    has already completed. Transport bootstrap must have unambiguous execution
    semantics, so Daytona bypasses that wrapper for this control-plane step.
    """
    sandbox_backend = unwrap_sandbox_backend(sandbox_backend)
    native_sandbox = getattr(sandbox_backend, "_sandbox", None)
    process = getattr(native_sandbox, "process", None)
    exec_fn = getattr(process, "exec", None)

    if not callable(exec_fn):
        raise RuntimeError("Daytona backend does not expose native process.exec")

    result = await asyncio.to_thread(
        exec_fn,
        command,
        timeout=_DAYTONA_COMMAND_TIMEOUT_SECONDS,
    )

    exit_code = getattr(result, "exit_code", None)
    if not isinstance(exit_code, int):
        raise RuntimeError("Daytona native process.exec returned an invalid exit code")

    output: Any = getattr(result, "result", "")
    return exit_code, output if isinstance(output, str) else str(output or "")


async def configure_git_transport(
    sandbox_backend: SandboxBackendProtocol,
    *,
    sandbox_type: str | None = None,
) -> bool:
    """Configure provider-specific Git transport behavior.

    Returns True when provider-specific configuration was applied. Non-Daytona
    providers are intentionally left untouched.

    Daytona routes outbound HTTPS through its egress proxy. Git needs the
    Daytona CA explicitly, and authenticated Git smart-HTTP needs the opaque
    ``GITHUB_BASIC_AUTH`` secret placeholder to remain literal in the header so
    Daytona can substitute the real credential at egress.
    """
    provider = (sandbox_type or os.getenv("SANDBOX_TYPE", "langsmith")).strip().lower()

    if provider != "daytona":
        return False

    command = f"""
set -eu

CA="${{SSL_CERT_FILE:-{_DAYTONA_DEFAULT_CA}}}"

if [ ! -r "$CA" ]; then
    echo "Daytona Git CA is unavailable: $CA" >&2
    exit 42
fi

git config --global --replace-all http.version HTTP/1.1
git config --global --replace-all http.sslCAInfo "$CA"

if [ -n "${{GITHUB_BASIC_AUTH:-}}" ]; then
    git config --global --replace-all {_DAYTONA_GITHUB_HEADER_KEY} \
        "Authorization: Basic $GITHUB_BASIC_AUTH"
else
    git config --global --unset-all {_DAYTONA_GITHUB_HEADER_KEY} || true
fi
""".strip()

    exit_code, output = await _execute_daytona_native(sandbox_backend, command)

    if exit_code != 0:
        detail = output.strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Failed to configure Daytona Git transport{suffix}")

    return True
