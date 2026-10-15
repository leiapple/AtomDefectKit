import numpy as np
import pytest
from ase.calculators.calculator import Calculator

import atomdefectkit
from atomdefectkit.b2 import (
    B2Crystal,
    B2SlipSystem,
    B2Workflow,
    EV_PER_ANGSTROM3_TO_GPA,
    estimate_sinusoidal_peierls_stress,
)


def test_b2_workflow_is_public_api():
    assert atomdefectkit.B2Workflow is B2Workflow


def test_b2_crystal_builds_ordered_sublattices():
    crystal = B2Crystal(("Ni", "Al"), lattice_constant=2.88)
    atoms = crystal.build((2, 1, 1))

    assert atoms.get_chemical_symbols() == ["Ni", "Al", "Ni", "Al"]
    assert np.array_equal(atoms.arrays["b2_sublattice"], [0, 1, 0, 1])
    assert np.allclose(atoms.cell.lengths(), [5.76, 2.88, 2.88])
    assert atoms.info["crystal_structure"] == "B2"


def test_b2_crystal_rejects_identical_species():
    with pytest.raises(ValueError, match="different species"):
        B2Crystal(("Ni", "Ni"), lattice_constant=2.88)


def test_slip_system_derives_second_in_plane_vector_and_burgers_length():
    slip = B2SlipSystem(
        plane=(0, 1, 1),
        direction=(1, 0, 0),
        translation_fraction=1.0,
    )

    assert slip.transverse_direction == (0, 1, -1)
    assert slip.burgers_magnitude(2.88) == pytest.approx(2.88)


def test_slip_direction_must_lie_in_plane():
    with pytest.raises(ValueError, match="must lie"):
        B2SlipSystem(plane=(1, 1, 0), direction=(1, 0, 0))


def test_peierls_stress_distinguishes_amplitude_from_full_barrier():
    amplitude_stress = estimate_sinusoidal_peierls_stress(
        1.0,
        line_length_angstrom=1.0,
        burgers_angstrom=1.0,
        energy_definition="amplitude",
    )
    barrier_stress = estimate_sinusoidal_peierls_stress(
        1.0,
        line_length_angstrom=1.0,
        burgers_angstrom=1.0,
        energy_definition="peak_to_valley",
    )

    assert amplitude_stress == pytest.approx(2.0 * np.pi * EV_PER_ANGSTROM3_TO_GPA)
    assert barrier_stress == pytest.approx(0.5 * amplitude_stress)


def test_b2_workflow_passes_crystallographic_gamma_line(monkeypatch, tmp_path):
    captured = {}

    class DummyStackingFault:
        def stacking_fault(self, **kwargs):
            captured.update(kwargs)
            return [0.0, 0.1, 0.0]

    workflow = B2Workflow(
        B2Crystal(("Ni", "Al"), lattice_constant=2.88),
        calculator=Calculator(),
        working_dir=tmp_path,
    )
    monkeypatch.setattr(
        workflow.properties,
        "create_stacking_fault_workflow",
        lambda **kwargs: DummyStackingFault(),
    )
    slip = B2SlipSystem(
        plane=(0, 1, 1),
        direction=(1, 0, 0),
        name="011_100",
    )

    energies = workflow.calculate_gamma_line(slip, write_xyz=False)

    assert energies == [0.0, 0.1, 0.0]
    assert captured["a"] == (1, 0, 0)
    assert captured["b"] == (0, 1, -1)
    assert captured["miller"] == (0, 1, 1)
    assert captured["distance"] == pytest.approx(2.88)
    assert captured["fix_normal"] is False
    assert np.allclose(captured["initial_shift"], 0.0)


def test_combined_gamma_line_plot_writes_csv_and_figure(tmp_path):
    workflow = B2Workflow(
        B2Crystal(("Ni", "Al"), lattice_constant=2.88),
        calculator=Calculator(),
        working_dir=tmp_path,
    )

    fig, _, data = workflow.plot_gamma_lines(
        {
            "{011}<100>": [0.0, 0.10, 0.0],
            "{1-10}<111>": [0.0, 0.08, 0.04, 0.08, 0.0],
        }
    )

    assert len(data) == 8
    assert set(data["slip_system"]) == {"{011}<100>", "{1-10}<111>"}
    assert (tmp_path / "gamma_lines" / "b2_gamma_lines.csv").is_file()
    assert (tmp_path / "gamma_lines" / "b2_gamma_lines.png").is_file()
    fig.clear()


def test_offset_gamma_path_retains_absolute_reference(monkeypatch, tmp_path):
    captured = {}

    class DummyStackingFault:
        def stacking_fault(self, **kwargs):
            captured.update(kwargs)
            return [0.08, 0.04, 0.08]

    workflow = B2Workflow(B2Crystal(("Al", "Ni"), 2.88),
                          calculator=Calculator(), working_dir=tmp_path)
    monkeypatch.setattr(workflow.properties, "create_stacking_fault_workflow",
                        lambda **kwargs: DummyStackingFault())
    energies = workflow.calculate_gamma_line(
        B2SlipSystem((1, 1, 0), (-1, 1, 0), name="110_transverse"),
        initial_translation=(0, 0, 0.5),
    )
    assert np.allclose(captured["initial_shift"], [0, 1.44, 0])
    fig, _, data = workflow.plot_gamma_lines({"transverse": energies})
    assert np.allclose(data.energy_meV_per_A2, [80, 40, 80])
    fig.clear()
