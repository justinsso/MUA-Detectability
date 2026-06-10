"""Portable path helpers for the MUA detectability project.

The active reference script resolves paths relative to its own location. These
helpers keep that behavior available to newer modules without hardcoding a
machine-specific checkout path.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike[str]]

FIGURE_DIR_ENV = "MUA_FIGURE_DIR"
OUTPUT_DIR_ENV = "MUA_OUTPUT_DIR"
DEFAULT_MORPHOLOGY_NAME = "L5_Mainen96_LFPy.hoc"


def repo_root(anchor: Optional[PathLike] = None) -> Path:
    """Return the repository root.

    By default this resolves from ``code/mua_config.py`` to ``<repo>``. Passing
    an anchor is useful from tests or scripts that need the same convention,
    including scripts nested under ``code/`` subdirectories.
    """
    if anchor is None:
        return Path(__file__).resolve().parent.parent

    path = Path(anchor).resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if candidate.name == "code":
            return candidate.parent
    return path


def repo_path(*parts: PathLike, root: Optional[PathLike] = None) -> Path:
    """Build a path under the repository root."""
    base = repo_root(root)
    return base.joinpath(*(Path(part) for part in parts))


def morphology_dir(root: Optional[PathLike] = None) -> Path:
    """Return the bundled LFPy morphology directory."""
    return repo_path("LFPy-2.3.6", "examples", "morphologies", root=root)


def default_morphology_path(
    root: Optional[PathLike] = None,
    name: str = DEFAULT_MORPHOLOGY_NAME,
) -> Path:
    """Return the default Mainen L5 morphology path."""
    return morphology_dir(root).joinpath(name)


def default_figure_dir(
    root: Optional[PathLike] = None,
    env: str = FIGURE_DIR_ENV,
    create: bool = False,
) -> Path:
    """Return the figure directory, honoring ``MUA_FIGURE_DIR`` if set."""
    path = Path(os.environ[env]).expanduser() if os.environ.get(env) else repo_path("figures", root=root)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir(
    root: Optional[PathLike] = None,
    env: str = OUTPUT_DIR_ENV,
    create: bool = False,
) -> Path:
    """Return the sweep output directory, honoring ``MUA_OUTPUT_DIR`` if set."""
    path = Path(os.environ[env]).expanduser() if os.environ.get(env) else repo_path("outputs", root=root)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def neuron_path(path: PathLike) -> str:
    """Return a path string with forward slashes for NEURON/HOC loading."""
    return Path(path).expanduser().resolve().as_posix()
