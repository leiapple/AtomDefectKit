"""End-to-end smoke tests that run the workflows with ASE's fast EMT calculator.

EMT only covers FCC metals, so these tests check that the workflows run and
return sensible values, not that BCC physics is reproduced.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

from atomdefectkit import BasicProperties, StackingFaultWorkflow, TractionSeparationWorkflow
from atomdefectkit.benchmarks import run_bcc_benchmark
from atomdefectkit.neb import BCCScrewDislocPeierlsBarrier


@pytest.fixture
def cu_workflow(tmp_path):
    return BasicProperties(calculator=EMT(), working_dir=tmp_path)


def test_lattice_constant_from_relax_and_eos_agree(cu_workflow):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)

    a0_relax = cu_workflow.calculate_equilibrium_a0_relax(atoms.copy(), fmax=1e-4)
    _, _, a0_eos = cu_workflow.calculate_equilibrium_a0_birch_murnaghan(
        atoms.copy(), vol_range=np.linspace(0.97, 1.03, 15)
    )

    assert a0_relax == pytest.approx(3.59, abs=0.02)
    assert a0_eos == pytest.approx(a0_relax, abs=2e-3)


def test_eos_lattice_constant_does_not_depend_on_supercell_size(cu_workflow):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    scales = np.linspace(0.97, 1.03, 15)

    _, _, a0_unit = cu_workflow.calculate_equilibrium_a0_birch_murnaghan(atoms, vol_range=scales)
    _, _, a0_super = cu_workflow.calculate_equilibrium_a0_birch_murnaghan(
        atoms.repeat((2, 2, 2)), vol_range=scales
    )

    assert a0_super == pytest.approx(2 * a0_unit, rel=1e-6)


def test_surface_and_vacancy_energies_are_physical(cu_workflow):
    atoms = bulk("Cu", "fcc", a=3.59, cubic=True)

    gamma_100 = cu_workflow.calculate_surface_energy(atoms.copy(), (1, 0, 0), vacuum=8, n_layers=4)
    gamma_111 = cu_workflow.calculate_surface_energy(atoms.copy(), (1, 1, 1), vacuum=8, n_layers=4)
    e_vac = cu_workflow.calculate_vacancy_formation_energy(atoms, supercell=(2, 2, 2))

    # FCC close-packed (111) is the lowest-energy surface.
    assert 0.5 < gamma_111 < gamma_100 < 3.0
    assert 0.5 < e_vac < 2.0


def test_stacking_fault_writes_curve_and_summary_separately(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.59, cubic=True)
    workflow = StackingFaultWorkflow(atoms, EMT(), formula="Cu", info="EMT", working_dir=tmp_path)

    energies = workflow.stacking_fault(
        a=(1, 0, 0), b=(0, 1, 0), miller=(0, 0, 1), distance=0.5, layers=6, num_steps=2, write_xyz=False
    )

    curve_lines = (tmp_path / "Cu_0-0-1_stacking_fault.out").read_text().splitlines()
    summary = (tmp_path / "Cu_stacking_fault_summary.txt").read_text()
    assert len(energies) == 3
    assert energies[0] == 0.0
    assert curve_lines[0] == "Reaction_Coordinate   Energy(mJ/m^2)"
    assert len(curve_lines) == 4
    assert "Stacking_Fault" in summary and "mJ/m^2" in summary


def test_pure_traction_separation_runs(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.59, cubic=True)
    workflow = TractionSeparationWorkflow(
        atoms, EMT(), surface_index=(1, 0, 0), repeat=(1, 1, 8), working_dir=tmp_path
    )

    results = workflow.run_pure_separation(
        vacuum_values=np.linspace(0.0, 4.0, 9), write_xyz=False, cell_optimizer="FIRE"
    )

    assert results["sigma_max_GPa"] > 0.0
    assert results["surface_energy_J_m2"] > 0.0
    assert len(results["traction"]) == 9


def test_odd_hydrogen_count_is_rejected(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.59, cubic=True)
    workflow = TractionSeparationWorkflow(atoms, EMT(), surface_index=(1, 0, 0), working_dir=tmp_path)

    with pytest.raises(ValueError, match="nH must be even"):
        workflow.add_hydrogen_sites(atoms, 3, [], [])


def test_neb_pbc_unwrap_matches_brute_force_search():
    rng = np.random.default_rng(0)
    cell = np.array([[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [0.0, 0.0, 2.8]])
    initial = Atoms("Nb20", scaled_positions=rng.random((20, 3)), cell=cell, pbc=True)
    final = initial.copy()
    final.positions += rng.normal(scale=0.3, size=(20, 3))
    final.wrap()

    unwrapped = BCCScrewDislocPeierlsBarrier._fix_pbc_issues(initial, final)

    expected = final.get_positions()
    for i, vec in enumerate(final.positions - initial.positions):
        shifts = [np.array([a, b, c]) @ cell for a, b, c in np.ndindex(3, 3, 3)]
        shifts = [shift - cell.sum(axis=0) for shift in shifts]
        expected[i] += min(shifts, key=lambda shift: np.linalg.norm(vec + shift))
    assert np.allclose(unwrapped, expected)


def test_benchmark_rejects_unknown_steps(cu_workflow):
    with pytest.raises(ValueError, match="Unknown benchmark steps"):
        run_bcc_benchmark(cu_workflow, "Nb", steps=["basic", "not-a-step"])


def test_benchmark_basic_step_runs_end_to_end(tmp_path):
    # EMT has no BCC parameters, so this only checks that the pipeline runs.
    workflow = BasicProperties(calculator=EMT(), working_dir=tmp_path)

    results = run_bcc_benchmark(workflow, "Cu", initial_a0=2.87, label="EMT", steps=["basic"])

    saved = (tmp_path / "Cu_basicProp.json").read_text()
    assert results["a0"] > 0.0
    assert "inter_100_formation_energy" in saved
    assert (tmp_path / "comparison.pdf").is_file()
