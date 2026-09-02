"""Bootstrap real Open SWE runtime resources for orchestration."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from deepagents.backends.protocol import SandboxBackendProtocol
from langgraph.graph.state import RunnableConfig

from agent.runtime import (
    ensure_sandbox_for_thread,
    graph_loaded_for_execution,
)
from agent.utils.sandbox_paths import aresolve_sandbox_work_dir

from .git_transport import configure_git_transport

EnsureSandbox = Callable[
    [str],
    Awaitable[SandboxBackendProtocol],
]

ResolveWorkDir = Callable[
    [SandboxBackendProtocol],
    Awaitable[str],
]

ExecutionPredicate = Callable[
    [RunnableConfig],
    bool,
]

ConfigureGitTransport = Callable[
    [SandboxBackendProtocol],
    Awaitable[bool],
]


@dataclass(frozen=True)
class OrchestratorRuntimeContext:
    """Thread-bound Open SWE resources used by our runtime."""

    thread_id: str
    sandbox_backend: SandboxBackendProtocol
    work_dir: str


async def bootstrap_orchestrator_runtime(
    config: RunnableConfig,
    *,
    ensure_sandbox: EnsureSandbox = ensure_sandbox_for_thread,
    resolve_work_dir: ResolveWorkDir = aresolve_sandbox_work_dir,
    configure_transport: ConfigureGitTransport = configure_git_transport,
    execution_predicate: ExecutionPredicate = (graph_loaded_for_execution),
) -> OrchestratorRuntimeContext | None:
    """Resolve the real thread sandbox and writable workspace.

    Returns None during schema/introspection graph loads where no actual
    execution should create or reconnect a sandbox.
    """
    configurable = config.get("configurable") or {}

    thread_id = configurable.get("thread_id")

    if not isinstance(thread_id, str) or not thread_id.strip():
        return None

    if not execution_predicate(config):
        return None

    thread_id = thread_id.strip()

    backend = await ensure_sandbox(thread_id)

    await configure_transport(backend)

    work_dir = await resolve_work_dir(backend)

    if not isinstance(work_dir, str) or not work_dir.strip():
        raise RuntimeError("Open SWE returned an invalid sandbox working directory")

    return OrchestratorRuntimeContext(
        thread_id=thread_id,
        sandbox_backend=backend,
        work_dir=work_dir.strip(),
    )
