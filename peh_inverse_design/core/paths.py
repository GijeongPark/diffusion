"""Shared filesystem helpers for the peh_inverse_design package."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (the directory that contains ``peh_inverse_design/``).

    This module lives at ``peh_inverse_design/core/paths.py``, so the repository
    root is two directory levels up.
    """
    return Path(__file__).resolve().parents[2]
