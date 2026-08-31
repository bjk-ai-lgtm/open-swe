"""Deterministic task classifier for specialist routing."""

import re

from .types import RoutingDecision, SpecialistRole

ROLE_SIGNALS: dict[SpecialistRole, tuple[str, ...]] = {
    SpecialistRole.BACKEND: (
        "api",
        "endpoint",
        "backend",
        "server",
        "database",
        "migration",
        "sql",
        "model",
        "schema",
        "service",
        "repository",
        "business logic",
        "authentication",
        "authorization",
    ),
    SpecialistRole.RESEARCH: (
        "research",
        "investigate",
        "documentation",
        "docs",
        "compare",
        "find out",
        "look up",
        "compatibility",
        "dependency",
        "version",
        "analyze library",
    ),
    SpecialistRole.QA: (
        "test",
        "tests",
        "pytest",
        "verify",
        "validate",
        "regression",
        "reproduce",
        "bug",
        "quality",
        "acceptance criteria",
        "lint",
        "type check",
    ),
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def classify_task(task: str) -> RoutingDecision:
    """Classify a task without making an LLM call."""
    normalized = _normalize(task)

    if not normalized:
        return RoutingDecision(
            role=SpecialistRole.GENERAL,
            confidence=0.0,
            matched_signals=(),
        )

    scores: dict[SpecialistRole, list[str]] = {}

    for role, signals in ROLE_SIGNALS.items():
        matches = [signal for signal in signals if signal in normalized]
        scores[role] = matches

    ranked = sorted(
        scores.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    best_role, best_matches = ranked[0]

    if not best_matches:
        return RoutingDecision(
            role=SpecialistRole.GENERAL,
            confidence=0.25,
            matched_signals=(),
        )

    best_score = len(best_matches)
    second_score = len(ranked[1][1])

    if best_score == second_score:
        return RoutingDecision(
            role=SpecialistRole.GENERAL,
            confidence=0.4,
            matched_signals=tuple(best_matches),
        )

    confidence = min(0.55 + (0.1 * best_score), 0.95)

    return RoutingDecision(
        role=best_role,
        confidence=confidence,
        matched_signals=tuple(best_matches),
    )
