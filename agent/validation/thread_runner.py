"""Resolve validation runners against an Open SWE thread sandbox."""

import os

from agent.utils.sandbox_state import get_sandbox_backend

from .sandbox_runner import (
    DEFAULT_VALIDATION_TIMEOUT_SECONDS,
    SandboxCommandRunner,
)


async def sandbox_runner_for_thread(
    thread_id: str,
    *,
    timeout_seconds: int = DEFAULT_VALIDATION_TIMEOUT_SECONDS,
    allow_local: bool = False,
) -> SandboxCommandRunner:
    """Return a validation runner bound to the thread's existing sandbox."""
    if not thread_id.strip():
        raise ValueError("Thread ID cannot be empty")

    sandbox_type = os.getenv("SANDBOX_TYPE", "langsmith")

    if sandbox_type == "local" and not allow_local:
        raise RuntimeError(
            "Deterministic validation refuses SANDBOX_TYPE=local because "
            "local execution is not isolated"
        )

    backend = await get_sandbox_backend(thread_id)

    return SandboxCommandRunner(
        backend,
        timeout_seconds=timeout_seconds,
    )
