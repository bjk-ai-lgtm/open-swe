from agent.orchestration.bootstrap import OrchestratorRuntimeContext
from agent.orchestration.validation_profiles import ValidationProfile
from agent.orchestration.validation_resolution import resolve_validation_checks


async def test_auto_profile_inspects_runtime_repository() -> None:
    context = OrchestratorRuntimeContext(
        thread_id="thread-auto",
        sandbox_backend=object(),
        work_dir="/workspace/project",
    )

    calls = []

    async def inspector(backend, *, work_dir):
        calls.append((backend, work_dir))
        return {
            "pyproject.toml": """
[project]
name = "demo"

[tool.pytest.ini_options]
""",
            "uv.lock": "",
        }

    checks = await resolve_validation_checks(
        ValidationProfile.AUTO,
        context,
        repository_inspector=inspector,
    )

    assert calls == [
        (
            context.sandbox_backend,
            context.work_dir,
        )
    ]
    assert [check.name for check in checks] == ["pytest"]
    assert checks[0].command == (
        "uv",
        "run",
        "pytest",
        "-q",
    )


async def test_explicit_profile_does_not_inspect_repository() -> None:
    context = OrchestratorRuntimeContext(
        thread_id="thread-explicit",
        sandbox_backend=object(),
        work_dir="/workspace/project",
    )

    async def inspector(*args, **kwargs):
        raise AssertionError("explicit validation profile must not inspect repository")

    checks = await resolve_validation_checks(
        ValidationProfile.OPEN_SWE_PYTHON,
        context,
        repository_inspector=inspector,
    )

    assert checks


async def test_auto_unknown_repository_is_fail_closed() -> None:
    context = OrchestratorRuntimeContext(
        thread_id="thread-unknown",
        sandbox_backend=object(),
        work_dir="/workspace/project",
    )

    async def inspector(backend, *, work_dir):
        return {
            "README.md": "# Demo",
        }

    checks = await resolve_validation_checks(
        ValidationProfile.AUTO,
        context,
        repository_inspector=inspector,
    )

    assert checks == ()
