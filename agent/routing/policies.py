"""Deterministic execution and validation policies."""

from dataclasses import dataclass

from .types import SpecialistRole


@dataclass(frozen=True)
class ValidationPolicy:
    """Rules controlling whether a task may be considered complete."""

    required: bool
    validator: SpecialistRole | None
    max_retries: int
    block_completion_on_failure: bool


VALIDATION_POLICIES: dict[SpecialistRole, ValidationPolicy] = {
    SpecialistRole.BACKEND: ValidationPolicy(
        required=True,
        validator=SpecialistRole.QA,
        max_retries=2,
        block_completion_on_failure=True,
    ),
    SpecialistRole.RESEARCH: ValidationPolicy(
        required=False,
        validator=None,
        max_retries=0,
        block_completion_on_failure=False,
    ),
    SpecialistRole.QA: ValidationPolicy(
        required=False,
        validator=None,
        max_retries=0,
        block_completion_on_failure=False,
    ),
    SpecialistRole.GENERAL: ValidationPolicy(
        required=False,
        validator=None,
        max_retries=0,
        block_completion_on_failure=False,
    ),
}


def validation_policy_for(role: SpecialistRole) -> ValidationPolicy:
    """Return the deterministic validation policy for a specialist role."""
    return VALIDATION_POLICIES[role]
