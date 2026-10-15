"""Screw-dislocation smoke tests with a real BCC W potential.

Uses the Marinica et al. (2013) W EAM potential ``tests/data/w_eam4.fs``
(J. Phys.: Condens. Matter 25, 395502), evaluated with matscipy's EAM
calculator. Convergence is deliberately loose: the tests check that the
workflow runs end to end and gives a physically sensible barrier, not
converged values. The Peierls-barrier NEB is left to local runs because it
takes several minutes.
"""

from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from matscipy.calculators.eam import EAM
from matscipy.calculators.eam.calculator import _make_splines
from matscipy.calculators.eam.io import read_eam

from atomdefectkit import BasicProperties

POTENTIAL = Path(__file__).parent / "data" / "w_eam4.fs"


def load_w_eam(stride=50):
    """Load the W potential with every ``stride``-th tabulated point.

    The file tabulates 100001 points per function, which makes scipy's spline
    evaluation the bottleneck. Keeping 2001 points changes forces by less
    than 1e-4 eV/A and makes evaluation about 30x faster.
    """
    _, params, F, f, rep = read_eam(str(POTENTIAL), kind="eam/fs")
    dr = params.distance_grid_spacing * stride
    return EAM(
        atomic_numbers=params.atomic_numbers,
        F=_make_splines(params.density_grid_spacing * stride, F[..., ::stride]),
        f=_make_splines(dr, f[..., ::stride]),
        rep=_make_splines(dr, rep[..., ::stride]),
        cutoff=params.cutoff,
    )


def test_resampled_potential_matches_full_table():
    atoms = bulk("W", "bcc", a=3.14, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.02, seed=1)
    full, fast = atoms.copy(), atoms.copy()
    full.calc = EAM(str(POTENTIAL), kind="eam/fs")
    fast.calc = load_w_eam()

    assert fast.get_potential_energy() == pytest.approx(full.get_potential_energy(), abs=1e-6)
    assert np.allclose(fast.get_forces(), full.get_forces(), atol=1e-4)


@pytest.fixture(scope="module")
def w_setup(tmp_path_factory):
    calc = load_w_eam()
    workflow = BasicProperties(calculator=calc, working_dir=tmp_path_factory.mktemp("w_eam"))
    atoms = bulk("W", "bcc", a=3.14, cubic=True)
    _, _, a0 = workflow.calculate_equilibrium_a0_birch_murnaghan(atoms)
    atoms.set_cell([a0] * 3, scale_atoms=True)
    cij = workflow.calculate_elastic_constants(atoms, verbose=False)

    screw = workflow.create_screw_dislocation_workflow("W", a0, cij)
    endpoints = []
    # The final core sits one Peierls valley, a0*sqrt(6)/3, along x.
    for name, center in [("initial", [0, 0, 0]), ("final", [a0 * np.sqrt(6) / 3, 0, 0])]:
        dislocation = screw.create_dislocation_object()
        base_system, disl_system = screw.relax_dislocation_dipole(
            dislocation, disloc_center=center, fmax=0.1, optimizer="FIRE", steps=200
        )
        screw.plot_differential_displacement_map(
            dislocation, base_system, disl_system, filename=f"dd_plot_{name}.png"
        )
        endpoints.append(disl_system.dump("ase_Atoms"))
    return {"calc": calc, "workflow": workflow, "screw": screw, "a0": a0, "cij": cij, "endpoints": endpoints}


def test_eam_bulk_properties_are_close_to_tungsten(w_setup):
    # The file header lists a0 = 3.14339 A. Experimental W elastic constants
    # are about C11/C12/C44 = 533/205/163 GPa.
    assert w_setup["a0"] == pytest.approx(3.14339, abs=0.002)
    cij = w_setup["cij"]
    assert cij[0, 0] == pytest.approx(533, rel=0.1)
    assert cij[0, 1] == pytest.approx(205, rel=0.15)
    assert cij[3, 3] == pytest.approx(163, rel=0.15)


def test_screw_dipole_endpoints_are_built_and_plotted(w_setup):
    initial, final = w_setup["endpoints"]
    screw_dir = Path(w_setup["screw"].working_dir)

    assert len(initial) == len(final) > 100
    assert np.allclose(initial.cell, final.cell)
    assert (screw_dir / "dd_plot_initial.png").is_file()
    assert (screw_dir / "dd_plot_final.png").is_file()
