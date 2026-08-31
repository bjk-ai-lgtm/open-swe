"""Command runner abstraction for deterministic validation."""

from typing import Protocol

from .types import CommandResult, ValidationCheck


class CommandRunner(Protocol):
    """Runs validation commands inside an execution environment."""

    async def run(
        self,
        check: ValidationCheck,
        *,
        work_dir: str,
    ) -> CommandResult:
        """Execute one validation check."""
        ...
