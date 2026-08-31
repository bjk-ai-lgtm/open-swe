from agent.routing import SpecialistRole, classify_task


def test_routes_backend_task() -> None:
    result = classify_task("Implement a new API endpoint and persist the result in the database.")

    assert result.role is SpecialistRole.BACKEND
    assert result.confidence > 0.5
    assert "api" in result.matched_signals
    assert "database" in result.matched_signals


def test_routes_research_task() -> None:
    result = classify_task("Research the official documentation and compare dependency versions.")

    assert result.role is SpecialistRole.RESEARCH
    assert result.confidence > 0.5


def test_routes_qa_task() -> None:
    result = classify_task("Run the tests, reproduce the bug, and verify the regression is fixed.")

    assert result.role is SpecialistRole.QA
    assert result.confidence > 0.5


def test_unknown_task_falls_back_to_general() -> None:
    result = classify_task("Help with this project.")

    assert result.role is SpecialistRole.GENERAL


def test_empty_task_falls_back_to_general() -> None:
    result = classify_task("   ")

    assert result.role is SpecialistRole.GENERAL
    assert result.confidence == 0.0
