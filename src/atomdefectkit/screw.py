"""BCC screw dislocation setup and analysis tools."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE, LBFGS
import atomman as am
import atomman.unitconvert as uc
from matscipy.elasticity import fit_elastic_constants


class BCCScrewDislocation:
    """Create, relax, and analyze screw dislocation dipoles in BCC metals."""

    def __init__(self, element, lattice_constant, elastic_constant, calculator):
        self.element = element
        self.a0 = lattice_constant
        self.cij = elastic_constant
        self.calculator = calculator
        self.structure = None
        self.relaxed_structure = None
        self.dd_map = None
        self.dd_map_relaxed = None

    def create_dislocation_object(self):
        alat = uc.set_in_units(self.a0, "angstrom")
        C11 = uc.set_in_units(self.cij[0,0], "GPa")
        C12 = uc.set_in_units(self.cij[0,1], "GPa")
        C44 = uc.set_in_units(self.cij[3,3], "GPa")

        unit_cell = am.load("prototype", "A2--W--bcc", a=alat, symbols=self.element)
        elastic_constants = am.ElasticConstants(C11=C11, C12=C12, C44=C44)

        burgers_vector = np.array([0.5, 0.5, 0.5])
        slip_plane = np.array([1, -1, 0])
        line_direction = np.array([0.5, 0.5, 0.5])
        shift_vector = np.array([0.0, 0.66666666666667, 0.0])

        return am.defect.Dislocation(
            unit_cell,
            elastic_constants,
            burgers_vector,
            line_direction,
            slip_plane,
            m="x",
            n="y",
            conventional_setting="i",
            shift=shift_vector,
            shiftscale=True,
        )

    def relax_dislocation_dipole(self, dislocation, disloc_center=(0, 0, 0), fmax=0.01, optimizer="BFGS"):
        base_system, dislocation_system = dislocation.dipole(
            sizemults=[7, 5.5, 1],
            center=disloc_center,
            centerscale=False,
            boxtilt=True,
            return_base_system=True,
        )

        dislocation_dipole_ase, properties = dislocation_system.dump("ase_Atoms", return_prop=True)
        dislocation_dipole_ase.calc = self.calculator

        if optimizer == "BFGS":
            opt = BFGS(dislocation_dipole_ase)
        elif optimizer == "LBFGS":
            opt = LBFGS(dislocation_dipole_ase)
        elif optimizer == "FIRE":
            opt = FIRE(dislocation_dipole_ase)
        else:
            raise ValueError("Optimizer must be 'BFGS', 'LBFGS', or 'FIRE'")

        opt.run(fmax=fmax)
        relaxed_system = am.load("ase_Atoms", dislocation_dipole_ase, prop=properties)
        return base_system, relaxed_system

    def plot_differential_displacement_map(self, dislocation, base_system, dislocation_system, filename="dislocation.png"):
        lattice_constant = dislocation.ucell.box.a
        burgers_vector = dislocation.dislsol.burgers
        big_base_system = base_system.supersize(1, 1, 3)
        big_dislocation_system = dislocation_system.supersize(1, 1, 3)
        neighbor_cutoff = 0.9 * lattice_constant
        neighbors = big_dislocation_system.neighborlist(cutoff=neighbor_cutoff)

        dd = am.defect.DifferentialDisplacement(
            big_base_system,
            big_dislocation_system,
            neighbors=neighbors,
            reference=1,
        )
        plot_params = {
            "ddmax": np.linalg.norm(burgers_vector) / 2,
            "plotxaxis": "x",
            "plotyaxis": "y",
            "xlim": (
                0,
                dislocation_system.box.avect[0]
                + dislocation_system.box.bvect[1]
                + self.a0,
            ),
            "ylim": (0, dislocation_system.box.bvect[1] + 1.0),
            "zlim": (
                lattice_constant * 3**0.5 / 2 - 0.01,
                2 * lattice_constant * 3**0.5 / 2 + 0.01,
            ),
            "figsize": 14,
            "arrowwidth": 1 / 100,
            "arrowscale": 2.5,
        }
        dd.plot("z", use0z=True, atomcmap="rainbow", **plot_params)
        plt.title(f"DD map: {self.element}")
        plt.savefig(filename, dpi=300)
        return plt.gcf()

