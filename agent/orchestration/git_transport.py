"""Provider-aware Git transport configuration for orchestration sandboxes."""

import os

from deepagents.backends.protocol import SandboxBackendProtocol

from .sandbox_command import execute_control_plane_command

_DAYTONA_DEFAULT_CA = "/etc/daytona/netleash/ca.crt"
_DAYTONA_GITHUB_HEADER_KEY = "http.https://github.com/.extraHeader"
_DAYTONA_COMMAND_TIMEOUT_SECONDS = 60


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

    result = await execute_control_plane_command(
        sandbox_backend,
        command,
        timeout=_DAYTONA_COMMAND_TIMEOUT_SECONDS,
        sandbox_type=provider,
    )

    if result.exit_code != 0:
        detail = result.output.strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Failed to configure Daytona Git transport{suffix}")

    return True
