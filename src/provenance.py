"""Best-effort source identity; absence of Git is never a verification success."""

import subprocess
from pathlib import Path


def source_state(root: Path) -> tuple[str | None, bool | None]:
    values = []
    for arguments in (["rev-parse", "HEAD"], ["status", "--porcelain"]):
        try:
            result = subprocess.run(
                ["git", *arguments], cwd=root, capture_output=True, text=True, timeout=3
            )
            values.append(result.stdout.strip() if result.returncode == 0 else None)
        except (OSError, subprocess.TimeoutExpired):
            values.append(None)
    revision, status = values
    # Unknown cleanliness is distinct from clean: missing Git must not upgrade evidence.
    return revision, None if status is None else bool(status)
