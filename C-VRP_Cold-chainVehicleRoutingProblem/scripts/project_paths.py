"""Workspace layout contract for the DynMaskCO extension.

The upstream MaskCO repository is kept read-only under ``MASKCO_code/`` while
all extension code lives under ``C-VRP_Cold-chainVehicleRoutingProblem/``.
Entry points may be launched from any current working directory, so paths must
be derived from this file rather than from ``cwd`` or a fixed number of parent
directories in each caller.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


SCRIPTS_ROOT = Path(__file__).resolve().parent
EXTENSION_ROOT = SCRIPTS_ROOT.parent
WORKSPACE_ROOT = EXTENSION_ROOT.parent
MASKCO_ROOT = WORKSPACE_ROOT / "MASKCO_code"


class ProjectLayoutError(RuntimeError):
    """Raised when the workspace no longer satisfies the frozen layout."""


def _resolved_path(path: os.PathLike[str] | str) -> Path:
    return Path(path).expanduser().resolve()


def validate_layout() -> dict[str, str]:
    """Validate the frozen workspace layout and return serializable roots."""

    required = {
        "workspace_root": WORKSPACE_ROOT,
        "extension_root": EXTENSION_ROOT,
        "scripts_root": SCRIPTS_ROOT,
        "maskco_root": MASKCO_ROOT,
        "maskco_models": MASKCO_ROOT / "models",
        "maskco_training": MASKCO_ROOT / "training",
        "maskco_helpers": MASKCO_ROOT / "helpers",
        "maskco_modules": MASKCO_ROOT / "modules",
        "maskco_decoding": MASKCO_ROOT / "decoding",
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.is_dir()]
    if missing:
        raise ProjectLayoutError(
            "DynMaskCO workspace layout is incomplete; missing directories: "
            + ", ".join(missing)
        )
    if MASKCO_ROOT.parent != WORKSPACE_ROOT:
        raise ProjectLayoutError(f"MASKCO_ROOT escaped workspace: {MASKCO_ROOT}")
    if EXTENSION_ROOT.parent != WORKSPACE_ROOT:
        raise ProjectLayoutError(f"EXTENSION_ROOT escaped workspace: {EXTENSION_ROOT}")
    if MASKCO_ROOT == EXTENSION_ROOT:
        raise ProjectLayoutError("Upstream and extension roots must be isolated")
    return {name: str(path) for name, path in required.items()}


def prepend_sys_paths(*paths: os.PathLike[str] | str) -> tuple[str, ...]:
    """Prepend existing paths while preserving caller-declared priority.

    The first argument receives the highest import priority. Existing copies
    are removed first so repeated bootstrapping remains deterministic.
    """

    normalized: list[str] = []
    for path in paths:
        value = str(_resolved_path(path))
        if not Path(value).is_dir():
            raise ProjectLayoutError(f"Import path does not exist: {value}")
        if value not in normalized:
            normalized.append(value)

    normalized_keys = {os.path.normcase(os.path.normpath(value)) for value in normalized}
    sys.path[:] = [
        value
        for value in sys.path
        if os.path.normcase(os.path.normpath(value or os.curdir)) not in normalized_keys
    ]
    sys.path[0:0] = normalized
    return tuple(normalized)


def configure_imports(
    extension_subdirs: Iterable[str] = (),
    *,
    include_extension_root: bool = False,
    include_upstream_models: bool = False,
) -> tuple[str, ...]:
    """Activate upstream and requested extension import roots.

    ``extension_subdirs`` are ordered from highest to lowest priority. The
    upstream package root is always last so extension-local direct imports keep
    their historical precedence without shadowing ``models.*`` or
    ``training.*`` package imports.
    """

    validate_layout()
    paths: list[Path] = [SCRIPTS_ROOT / name for name in extension_subdirs]
    if include_extension_root:
        paths.append(EXTENSION_ROOT)
    if include_upstream_models:
        paths.append(MASKCO_ROOT / "models")
    paths.append(MASKCO_ROOT)
    return prepend_sys_paths(*paths)


__all__ = [
    "EXTENSION_ROOT",
    "MASKCO_ROOT",
    "ProjectLayoutError",
    "SCRIPTS_ROOT",
    "WORKSPACE_ROOT",
    "configure_imports",
    "prepend_sys_paths",
    "validate_layout",
]
