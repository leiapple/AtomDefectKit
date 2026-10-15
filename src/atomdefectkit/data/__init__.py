"""Packaged reference data for AtomDefectKit."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

#: Packaged reference-data folders, keyed by the data kind used in the workflows.
REFERENCE_PACKAGES = {
    "basic_properties": "atomdefectkit.data.basic_properties",
    "peierls_barrier": "atomdefectkit.data.PeierlsPotential",
    "stacking_fault": "atomdefectkit.data.StackingFaults",
    "traction_separation": "atomdefectkit.data.TractionSeparation",
}


def miller_label(miller) -> str:
    """Return the filename label for a Miller index, e.g. ``(1, -1, 0) -> "110"``."""
    return "".join(str(abs(int(index))) for index in miller)


def reference_path(kind: str, filename: str) -> Path | None:
    """Return the path of a packaged reference-data file, or ``None`` if it is missing.

    Args:
        kind: Reference-data kind, one of ``REFERENCE_PACKAGES``.
        filename: File name inside that data folder, e.g. ``"V_110.csv"``.

    Returns:
        Path | None: Path to the packaged file when it exists.

    Raises:
        KeyError: If ``kind`` is not a known reference-data kind.
    """
    packaged = resources.files(REFERENCE_PACKAGES[kind]).joinpath(filename)
    if packaged.is_file():
        return Path(str(packaged))
    return None
