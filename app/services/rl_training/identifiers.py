from __future__ import annotations

import re
from pathlib import Path


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_identifier(value: str, *, field: str) -> str:
    """Validate an opaque local identifier before it is used in a path.

    Identifiers are intentionally not normalised: silently rewriting caller
    input can create collisions between distinct external IDs.
    """
    text = str(value or "").strip()
    if text in {".", ".."} or not _IDENTIFIER.fullmatch(text):
        raise ValueError(
            f"{field} must be 1-128 ASCII letters, numbers, dots, underscores, or hyphens"
        )
    return text


def resolve_child_dir(root: Path, value: str, *, field: str) -> Path:
    """Resolve a validated child directory and reject symlink/path escapes."""
    identifier = validate_identifier(value, field=field)
    resolved_root = Path(root).resolve()
    candidate = (resolved_root / identifier).resolve(strict=False)
    if candidate.parent != resolved_root:
        raise ValueError(f"{field} resolves outside the configured root")
    return candidate
