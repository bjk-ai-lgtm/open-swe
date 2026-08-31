"""Deep Agents sandbox adapter for deterministic validation."""

import shlex

from deepagents.backends.protocol import SandboxBackendProtocol

from .types import CommandResult, ValidationCheck

DEFAULT_VALIDATION_TIMEOUT_SECONDS = 300


class SandboxCommandRunner:
    """Run deterministic validation commands through an Open SWE sandbox."""

    def __init__(
        self,
        backend: SandboxBackendProtocol,
        *,
        timeout_seconds: int = DEFAULT_VALIDATION_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Validation timeout must be positive")

        self._backend = backend
        self._timeout_seconds = timeout_seconds

    async def run(
        self,
        check: ValidationCheck,
        *,
        work_dir: str,
    ) -> CommandResult:
        """Execute one validation check inside the supplied sandbox."""
        if not work_dir.strip():
            raise ValueError("Validation work directory cannot be empty")

        if not check.command:
            raise ValueError("Validation command cannot be empty")

        command = self._build_command(
            check.command,
            work_dir=work_dir,
        )

        response = await self._backend.aexecute(
            command,
            timeout=self._timeout_seconds,
        )

        output = response.output or ""

        timed_out = response.exit_code == 124 and "timed out" in output.lower()

        return CommandResult(
            exit_code=response.exit_code,
            stdout=output,
            stderr="",
            timed_out=timed_out,
        )

    @staticmethod
    def _build_command(
        command: tuple[str, ...],
        *,
        work_dir: str,
    ) -> str:
        """Build a shell-safe command for execution in the target directory."""
        rendered_command = shlex.join(command)
        rendered_work_dir = shlex.quote(work_dir)

        return f"cd {rendered_work_dir} && {rendered_command}"
