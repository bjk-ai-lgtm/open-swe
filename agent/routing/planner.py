"""Execution-plan builder for the custom Open SWE orchestrator."""

from dataclasses import dataclass

from .policies import ValidationPolicy, validation_policy_for
from .task_classifier import classify_task
from .types import RoutingDecision, SpecialistRole


@dataclass(frozen=True)
class ExecutionPlan:
    """Runtime-owned plan for dispatching and validating one task."""

    task: str
    routing: RoutingDecision
    execution_roles: tuple[SpecialistRole, ...]
    validation: ValidationPolicy

    @property
    def primary_role(self) -> SpecialistRole:
        return self.execution_roles[0]


def build_execution_plan(task: str) -> ExecutionPlan:
    """Build a deterministic execution plan without calling an LLM."""
    routing = classify_task(task)
    validation = validation_policy_for(routing.role)

    roles: list[SpecialistRole] = [routing.role]

    if validation.required and validation.validator is not None:
        roles.append(validation.validator)

    return ExecutionPlan(
        task=task,
        routing=routing,
        execution_roles=tuple(roles),
        validation=validation,
    )
