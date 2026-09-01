from agent.orchestration.capability_policy import (
    filter_tools_for_role,
    registered_tool_name,
)
from agent.routing import SpecialistRole


def http_request():
    pass


def fetch_url():
    pass


def web_search():
    pass


def dangerous_admin_tool():
    pass


TOOLS = (
    http_request,
    fetch_url,
    web_search,
    dangerous_admin_tool,
)


def names(tools) -> set[str]:
    return {name for tool in tools if (name := registered_tool_name(tool))}


def test_backend_receives_only_backend_capabilities():
    filtered = filter_tools_for_role(
        SpecialistRole.BACKEND,
        TOOLS,
    )

    assert names(filtered) == {
        "http_request",
        "fetch_url",
    }


def test_research_receives_web_search():
    filtered = filter_tools_for_role(
        SpecialistRole.RESEARCH,
        TOOLS,
    )

    assert names(filtered) == {
        "http_request",
        "fetch_url",
        "web_search",
    }


def test_qa_does_not_receive_web_search():
    filtered = filter_tools_for_role(
        SpecialistRole.QA,
        TOOLS,
    )

    assert "web_search" not in names(filtered)


def test_unknown_capability_is_fail_closed():
    for role in SpecialistRole:
        filtered = filter_tools_for_role(
            role,
            TOOLS,
        )

        assert "dangerous_admin_tool" not in names(filtered)
