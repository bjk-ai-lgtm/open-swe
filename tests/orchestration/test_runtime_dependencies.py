from agent.orchestration.bootstrap import (
    OrchestratorRuntimeContext,
)
from agent.orchestration.runtime_dependencies import (
    build_specialist_runtime_dependencies,
)


def _tool_name(tool) -> str:
    return getattr(tool, "name", None) or getattr(tool, "__name__", "")


def test_default_dependencies_include_open_swe_local_tools():
    context = OrchestratorRuntimeContext(
        thread_id="thread-dependencies",
        sandbox_backend=object(),
        work_dir="/workspace/project",
    )

    dependencies = build_specialist_runtime_dependencies(
        {
            "configurable": {
                "thread_id": "thread-dependencies",
            }
        },
        context,
    )

    names = {_tool_name(tool) for tool in dependencies.tools}

    assert {
        "http_request",
        "fetch_url",
        "web_search",
    } <= names


def test_default_dependencies_do_not_enable_extra_capabilities():
    context = OrchestratorRuntimeContext(
        thread_id="thread-minimal",
        sandbox_backend=object(),
        work_dir="/workspace/project",
    )

    dependencies = build_specialist_runtime_dependencies(
        {"configurable": {}},
        context,
    )

    assert dependencies.skills == ()
    assert dependencies.middleware == ()
    assert dependencies.use_gateway is None
    assert dependencies.model_effort is None
