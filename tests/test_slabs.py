from ase.build import bulk

from atomdefectkit.utils.slabs import build_repeated_slab, build_surface_slab


def test_build_repeated_slab_repeats_structure_and_centers_axis():
    atoms = bulk("W", "bcc", a=3.2, cubic=True)

    slab = build_repeated_slab(atoms, repeat=(2, 1, 1), vacuum=8.0, center_axis=2)

    assert len(slab) == 2 * len(atoms)
    assert slab.cell[2, 2] > atoms.cell[2, 2]


def test_build_surface_slab_repeats_surface_cell():
    atoms = bulk("W", "bcc", a=3.2, cubic=True)

    slab = build_surface_slab(
        atoms,
        miller_indices=(1, 0, 0),
        layers=4,
        repeat=(2, 2, 1),
        vacuum=6.0,
        center_axis=2,
    )

    assert len(slab) > len(atoms)
    assert slab.cell[2, 2] > 0.0
    assert slab.pbc[2]
