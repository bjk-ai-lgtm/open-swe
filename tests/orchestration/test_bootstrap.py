import pytest

from agent.orchestration.bootstrap import (
    bootstrap_orchestrator_runtime,
)


async def test_bootstrap_resolves_thread_sandbox_and_work_dir() -> None:
    backend = object()
    calls = []

    async def ensure_sandbox(thread_id):
        calls.append(("sandbox", thread_id))
        return backend

    async def resolve_work_dir(received_backend):
        calls.append(("work_dir", received_backend))
        return "/workspace/project"

    context = await bootstrap_orchestrator_runtime(
        {
            "configurable": {
                "thread_id": "thread-123",
            }
        },
        ensure_sandbox=ensure_sandbox,
        resolve_work_dir=resolve_work_dir,
        execution_predicate=lambda config: True,
    )

    assert context is not None
    assert context.thread_id == "thread-123"
    assert context.sandbox_backend is backend
    assert context.work_dir == "/workspace/project"

    assert calls == [
        ("sandbox", "thread-123"),
        ("work_dir", backend),
    ]


async def test_bootstrap_skips_non_execution_graph_load() -> None:
    calls = []

    async def ensure_sandbox(thread_id):
        calls.append(thread_id)
        return object()

    context = await bootstrap_orchestrator_runtime(
        {
            "configurable": {
                "thread_id": "thread-123",
            }
        },
        ensure_sandbox=ensure_sandbox,
        execution_predicate=lambda config: False,
    )

    assert context is None
    assert calls == []


async def test_bootstrap_skips_missing_thread_id() -> None:
    calls = []

    async def ensure_sandbox(thread_id):
        calls.append(thread_id)
        return object()

    context = await bootstrap_orchestrator_runtime(
        {
            "configurable": {},
        },
        ensure_sandbox=ensure_sandbox,
        execution_predicate=lambda config: True,
    )

    assert context is None
    assert calls == []


async def test_bootstrap_rejects_invalid_work_dir() -> None:
    backend = object()

    async def ensure_sandbox(thread_id):
        return backend

    async def resolve_work_dir(received_backend):
        assert received_backend is backend
        return "   "

    with pytest.raises(
        RuntimeError,
        match="invalid sandbox working directory",
    ):
        await bootstrap_orchestrator_runtime(
            {
                "configurable": {
                    "thread_id": "thread-123",
                }
            },
            ensure_sandbox=ensure_sandbox,
            resolve_work_dir=resolve_work_dir,
            execution_predicate=lambda config: True,
        )
