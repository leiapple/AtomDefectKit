"""BCC screw dislocation setup and analysis tools."""

from __future__ import annotations
import os

import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk
from ase.filters import FrechetCellFilter
import atomman as am
import atomman.unitconvert as uc
from matscipy.elasticity import fit_elastic_constants
from atomdefectkit.utils.optimizers import build_optimizer, normalize_optimizer_name
from atomdefectkit.utils.paths import WorkingDirectoryMixin


class BCCScrewDislocation(WorkingDirectoryMixin):
    """Create, relax, and analyze screw dislocation dipoles in BCC metals."""

    def __init__(self, element, lattice_constant, elastic_constant, calculator, working_dir='.'):
        self.element = element
        self.a0 = lattice_constant
        self.cij = elastic_constant
        self.calculator = calculator
        self.structure = None
        self.relaxed_structure = None
        self.dd_map = None
        self.dd_map_relaxed = None
        self.init_working_dir(working_dir, "screw_disloc")

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

    def relax_dislocation_dipole(
        self,
        dislocation,
        disloc_center=(0, 0, 0),
        fmax=0.01,
        optimizer="BFGS",
        logfile=None,
    ):
        """Build and relax a screw-dislocation dipole.

        Args:
            dislocation: Atomman dislocation object used to construct the dipole.
            disloc_center: Dislocation center passed to ``dislocation.dipole``.
            fmax: Force convergence threshold.
            optimizer: ASE optimizer label.
            logfile: Optional optimizer logfile saved under ``working_dir``.

        Returns:
            tuple: Base atomman system and relaxed dislocation atomman system.
        """
        base_system, dislocation_system = dislocation.dipole(
            sizemults=[7, 5.5, 1],
            center=disloc_center,
            centerscale=False,
            boxtilt=True,
            return_base_system=True,
        )

        dislocation_dipole_ase, properties = dislocation_system.dump("ase_Atoms", return_prop=True)
        dislocation_dipole_ase.calc = self.calculator

        optimizer_name = normalize_optimizer_name(optimizer)
        if logfile is None:
            logfile = f"{optimizer_name.lower()}_dislocation_dipole_relax.log"
        logfile = self.path(logfile)
        os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)
        opt = build_optimizer(dislocation_dipole_ase, optimizer_name, logfile=logfile)
        opt.run(fmax=fmax)
        relaxed_system = am.load("ase_Atoms", dislocation_dipole_ase, prop=properties)
        return base_system, relaxed_system

    def plot_differential_displacement_map(self, dislocation, base_system, dislocation_system):

        lattice_constant = dislocation.ucell.box.a
        burgers_vector = dislocation.dislsol.burgers
        big_base_system = base_system.supersize(1, 1, 3)
        big_dislocation_system = dislocation_system.supersize(1, 1, 3)
        neighbor_cutoff = 0.9 * lattice_constant
        neighbors = big_dislocation_system.neighborlist(cutoff=neighbor_cutoff)
        filename = self.path("dd_plot.png")

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
