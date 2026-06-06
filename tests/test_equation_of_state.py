import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from atomdefectkit.BasicProperties import BasicProperties


class VolumeWellCalculator(Calculator):
    implemented_properties = ["energy"]

    def __init__(self, target_volume):
        super().__init__()
        self.target_volume = target_volume

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        volume = atoms.get_volume()
        self.results["energy"] = (volume - self.target_volume) ** 2


def test_birch_murnaghan_scan_scales_lattice_not_volume_factor():
    atoms = Atoms("V", positions=[[0.0, 0.0, 0.0]], cell=[2.0, 2.0, 2.0], pbc=True)
    workflow = BasicProperties(calculator=VolumeWellCalculator(target_volume=8.0))

    scales = np.array([0.95, 1.0, 1.05])
    volumes, _, a0_fit = workflow.calculate_equilibrium_a0_birch_murnaghan(
        atoms,
        vol_range=scales,
    )

    assert np.allclose(volumes, atoms.get_volume() * scales**3)
    assert np.isfinite(a0_fit)
    assert a0_fit > 0.0


def test_plot_comparison_falls_back_to_packaged_dft_data(tmp_path):
    workflow = BasicProperties(calculator=VolumeWellCalculator(target_volume=8.0), working_dir=tmp_path)
    calculated_data = {
        "volumes": [8.0, 9.0],
        "energies": [0.0, 1.0],
        "C11": 1.0,
        "C12": 2.0,
        "C44": 3.0,
        "surface_energies": {"(1, 0, 0)": 1.0, "(1, 1, 0)": 2.0, "(1, 1, 1)": 3.0, "(1, 1, 2)": 4.0},
        "vacancy_formation_energy": 1.0,
        "octahedral_formation_energy": 2.0,
        "tetrahedral_formation_energy": 3.0,
        "inter_100_formation_energy": 4.0,
        "inter_110_formation_energy": 5.0,
        "inter_111_formation_energy": 6.0,
    }

    output_path = workflow.plot_comparison(
        calculated_data,
        "missing/path/dft_V.json",
        save_name="comparison_test.pdf",
    )

    assert output_path.endswith("comparison_test.pdf")
