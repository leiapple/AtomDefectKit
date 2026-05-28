"""Shared working-directory helpers."""

from __future__ import annotations

import os


def ensure_working_dir(working_dir=".", *parts) -> str:
    """Create and return a workflow output directory.

    Args:
        working_dir: Base output directory.
        *parts: Optional subdirectory components appended to ``working_dir``.

    Returns:
        str: Created directory path.
    """
    path = os.path.join(working_dir, *parts) if parts else working_dir
    os.makedirs(path, exist_ok=True)
    return path


def working_path(working_dir, *parts) -> str:
    """Join one or more path components under a working directory.

    Args:
        working_dir: Base output directory.
        *parts: Path components below ``working_dir``.

    Returns:
        str: Joined output path.
    """
    return os.path.join(working_dir, *parts)


class WorkingDirectoryMixin:
    """Mixin providing consistent working-directory setup and path helpers."""

    def init_working_dir(self, working_dir=".", *parts) -> str:
        """Create and store the workflow output directory on ``self``.

        Args:
            working_dir: Base output directory.
            *parts: Optional subdirectory components appended to ``working_dir``.

        Returns:
            str: Created directory path.
        """
        self.working_dir = ensure_working_dir(working_dir, *parts)
        return self.working_dir

    def path(self, *parts) -> str:
        """Join one or more path components under ``self.working_dir``.

        Args:
            *parts: Path components below ``self.working_dir``.

        Returns:
            str: Joined output path.
        """
        return working_path(self.working_dir, *parts)
