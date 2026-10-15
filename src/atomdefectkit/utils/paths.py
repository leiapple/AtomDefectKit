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

    Parent directories of the returned path are created, so the path can be
    passed directly to optimizers, ``savefig`` or ``open``.

    Args:
        working_dir: Base output directory.
        *parts: Path components below ``working_dir``.

    Returns:
        str: Joined output path.
    """
    path = os.path.join(working_dir, *parts)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


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


class CalculatorWorkflow(WorkingDirectoryMixin):
    """Base class for workflows that hold an ASE calculator and an output directory.

    Subclasses store the calculator on ``self.calculator``. ``self.calc`` is kept
    as an alias for backward compatibility.
    """

    calculator = None

    @property
    def calc(self):
        """Alias of ``self.calculator``."""
        return self.calculator

    @calc.setter
    def calc(self, value):
        self.calculator = value
