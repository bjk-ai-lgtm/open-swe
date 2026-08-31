from agent.routing import (
    SpecialistRole,
    build_execution_plan,
    validation_policy_for,
)


def test_backend_plan_requires_qa_gate() -> None:
    plan = build_execution_plan(
        "Implement a backend API endpoint that stores records in the database."
    )

    assert plan.primary_role is SpecialistRole.BACKEND
    assert plan.execution_roles == (
        SpecialistRole.BACKEND,
        SpecialistRole.QA,
    )

    assert plan.validation.required is True
    assert plan.validation.validator is SpecialistRole.QA
    assert plan.validation.max_retries == 2
    assert plan.validation.block_completion_on_failure is True


def test_research_plan_does_not_use_code_qa_gate() -> None:
    plan = build_execution_plan(
        "Research the official documentation and compare dependency versions."
    )

    assert plan.execution_roles == (SpecialistRole.RESEARCH,)
    assert plan.validation.required is False


def test_qa_task_does_not_recursively_add_another_qa_step() -> None:
    plan = build_execution_plan("Run the tests and verify the regression is fixed.")

    assert plan.execution_roles == (SpecialistRole.QA,)
    assert plan.validation.required is False


def test_general_task_uses_general_purpose_fallback() -> None:
    plan = build_execution_plan("Help with this project.")

    assert plan.execution_roles == (SpecialistRole.GENERAL,)
    assert plan.validation.required is False


def test_backend_failure_policy_blocks_completion() -> None:
    policy = validation_policy_for(SpecialistRole.BACKEND)

    assert policy.required is True
    assert policy.block_completion_on_failure is True
    assert policy.max_retries > 0
