"""Read deterministic repository metadata from a live sandbox."""

import shlex
from collections.abc import Sequence
from dataclasses import dataclass

from deepagents.backends.protocol import SandboxBackendProtocol

CONTENT_METADATA_FILES = (
    "pyproject.toml",
    "package.json",
    "go.mod",
)

PRESENCE_METADATA_FILES = (
    "uv.lock",
    "pnpm-lock.yaml",
    "yarn.lock",
    "package-lock.json",
)

REPOSITORY_METADATA_FILES = (
    *CONTENT_METADATA_FILES,
    *PRESENCE_METADATA_FILES,
)

DEFAULT_INSPECTION_TIMEOUT_SECONDS = 30


class RepositoryInspectionError(RuntimeError):
    """Raised when deterministic repository inspection cannot complete safely."""


@dataclass(frozen=True)
class _InspectionTarget:
    filename: str
    include_content: bool


def _inspection_targets() -> tuple[_InspectionTarget, ...]:
    return (
        *(
            _InspectionTarget(filename=name, include_content=True)
            for name in CONTENT_METADATA_FILES
        ),
        *(
            _InspectionTarget(filename=name, include_content=False)
            for name in PRESENCE_METADATA_FILES
        ),
    )


def _inspection_command(
    target: _InspectionTarget,
    *,
    work_dir: str,
) -> str:
    rendered_work_dir = shlex.quote(work_dir)
    rendered_filename = shlex.quote(target.filename)

    if target.include_content:
        body = (
            f"if [ -f {rendered_filename} ]; then "
            f"printf '1\\n'; cat -- {rendered_filename}; "
            "else printf '0\\n'; fi"
        )
    else:
        body = f"if [ -f {rendered_filename} ]; then printf '1\\n'; else printf '0\\n'; fi"

    return f"cd {rendered_work_dir} && {body}"


def _parse_inspection_output(
    target: _InspectionTarget,
    output: str,
) -> tuple[bool, str]:
    marker, separator, content = output.partition("\n")

    if not separator or marker not in {"0", "1"}:
        raise RepositoryInspectionError(
            f"Malformed repository inspection output for {target.filename}"
        )

    if marker == "0":
        return False, ""

    return True, content if target.include_content else ""


async def read_repository_metadata(
    backend: SandboxBackendProtocol,
    *,
    work_dir: str,
    timeout_seconds: int = DEFAULT_INSPECTION_TIMEOUT_SECONDS,
    targets: Sequence[_InspectionTarget] | None = None,
) -> dict[str, str]:
    """Read only known validation metadata files from the target repository."""
    if not work_dir.strip():
        raise ValueError("Repository inspection work directory cannot be empty")

    if timeout_seconds <= 0:
        raise ValueError("Repository inspection timeout must be positive")

    selected_targets = tuple(targets) if targets is not None else _inspection_targets()
    files: dict[str, str] = {}

    for target in selected_targets:
        response = await backend.aexecute(
            _inspection_command(
                target,
                work_dir=work_dir,
            ),
            timeout=timeout_seconds,
        )

        if response.exit_code != 0:
            raise RepositoryInspectionError(
                "Repository inspection command failed for "
                f"{target.filename} with exit code {response.exit_code}"
            )

        exists, content = _parse_inspection_output(
            target,
            response.output or "",
        )

        if exists:
            files[target.filename] = content

    return files
