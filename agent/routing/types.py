"""Core routing types for the custom Open SWE orchestrator."""

from dataclasses import dataclass
from enum import StrEnum


class SpecialistRole(StrEnum):
    BACKEND = "backend-engineer"
    RESEARCH = "research-specialist"
    QA = "qa-engineer"
    GENERAL = "general-purpose"


@dataclass(frozen=True)
class RoutingDecision:
    """Deterministic result produced by the task classifier."""

    role: SpecialistRole
    confidence: float
    matched_signals: tuple[str, ...]
