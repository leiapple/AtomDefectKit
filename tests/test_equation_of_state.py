import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from atomdefectkit.basic_properties import BasicProperties


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
