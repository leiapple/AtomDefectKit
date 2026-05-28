"""Shared slab-construction helpers."""

from __future__ import annotations

from ase.build import surface


def build_surface_slab(
    structure,
    miller_indices,
    layers,
    repeat=(1, 1, 1),
    vacuum=0.0,
    center_axis=None,
):
    """Build a repeated surface slab from a bulk structure.

    Args:
        structure: Bulk structure used to generate the slab.
        miller_indices: Surface Miller index passed to ``ase.build.surface``.
        layers: Number of atomic layers in the slab.
        repeat: Repetition counts along the slab cell vectors.
        vacuum: Vacuum spacing added by ``ase.build.surface`` in Angstrom.
        center_axis: Optional cell axis passed to ``Atoms.center`` after building.

    Returns:
        ase.Atoms: Generated slab.
    """
    slab = surface(
        structure,
        miller_indices,
        layers=layers,
        periodic=True,
        vacuum=vacuum,
    )
    slab = slab.repeat(tuple(repeat))
    if center_axis is not None:
        slab.center(vacuum=0.0, axis=center_axis)
    return slab


def build_repeated_slab(structure, repeat=(1, 1, 1), vacuum=0.0, center_axis=2):
    """Repeat an existing slab or unit cell and center it along one axis.

    Args:
        structure: Structure to repeat.
        repeat: Repetition counts along the three cell vectors.
        vacuum: Vacuum spacing used when centering.
        center_axis: Cell axis passed to ``Atoms.center``.

    Returns:
        ase.Atoms: Repeated and centered structure.
    """
    slab = structure.repeat(tuple(repeat))
    slab.center(vacuum=vacuum, axis=center_axis)
    return slab
