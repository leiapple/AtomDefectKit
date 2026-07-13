"""BCC screw dislocation setup and analysis tools."""

from __future__ import annotations
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import atomman as am
import atomman.unitconvert as uc
from ase.data import atomic_numbers
from ase.geometry import find_mic, wrap_positions
from atomdefectkit.utils.optimizers import build_optimizer, normalize_optimizer_name
from atomdefectkit.utils.paths import WorkingDirectoryMixin


if not hasattr(matplotlib.cm, "get_cmap"):
    def _get_cmap_compat(name=None, lut=None):
        """Restore the Matplotlib API used by older AtomMan releases."""
        cmap = matplotlib.colormaps.get_cmap(name)
        return cmap.resampled(lut) if lut is not None else cmap

    matplotlib.cm.get_cmap = _get_cmap_compat


def insert_light_element_at_dislocation_core(
    structure,
    light_element,
    core_position,
    search_radius=1.0,
    grid_spacing=0.25,
    min_host_distance=0.6,
    line_direction=None,
    inplace=False,
):
    """Insert a light element at the nearest open site around a screw core.

    The search is performed in the plane normal to the dislocation line. Among
    sites satisfying ``min_host_distance``, the candidate closest to the
    requested Cartesian core position is selected. The input structure is
    copied unless ``inplace=True``.

    Args:
        structure: ASE ``Atoms`` containing the dislocation.
        light_element: Chemical symbol to insert, for example ``"H"`` or ``"C"``.
        core_position: Requested Cartesian core position ``(x, y, z)`` in Angstrom.
        search_radius: Maximum transverse search radius in Angstrom.
        grid_spacing: Spacing of transverse candidate points in Angstrom.
        min_host_distance: Required minimum distance to every existing atom.
        line_direction: Cartesian screw-line vector. By default the third cell
            vector is used.
        inplace: Modify and return ``structure`` instead of a copy.

    Returns:
        ase.Atoms: Structure containing the inserted light atom as its last atom.
    """
    if light_element not in atomic_numbers:
        raise ValueError(f"Unknown light-element symbol: {light_element!r}.")

    target = np.asarray(core_position, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("core_position must contain three finite Cartesian values.")
    if not np.isfinite(search_radius) or search_radius < 0.0:
        raise ValueError("search_radius must be finite and non-negative.")
    if not np.isfinite(grid_spacing) or grid_spacing <= 0.0:
        raise ValueError("grid_spacing must be positive and finite.")
    if not np.isfinite(min_host_distance) or min_host_distance < 0.0:
        raise ValueError("min_host_distance must be finite and non-negative.")
    if len(structure) == 0:
        raise ValueError("structure must contain at least one host atom.")

    line = np.array(
        structure.cell[2] if line_direction is None else line_direction,
        dtype=float,
        copy=True,
    )
    if line.shape != (3,) or not np.all(np.isfinite(line)):
        raise ValueError("line_direction must contain three finite Cartesian values.")
    line_norm = np.linalg.norm(line)
    if line_norm == 0.0:
        raise ValueError("line_direction must have non-zero length.")
    line /= line_norm

    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, line)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    transverse_1 = np.cross(line, trial)
    transverse_1 /= np.linalg.norm(transverse_1)
    transverse_2 = np.cross(line, transverse_1)

    offsets = np.arange(-search_radius, search_radius + 0.5 * grid_spacing, grid_spacing)
    offsets = np.unique(np.append(offsets, 0.0))
    valid_candidates = []
    largest_clearance = 0.0
    host_positions = structure.get_positions()
    for offset_1 in offsets:
        for offset_2 in offsets:
            radial_distance = float(np.hypot(offset_1, offset_2))
            if radial_distance > search_radius + 1e-12:
                continue
            candidate = target + offset_1 * transverse_1 + offset_2 * transverse_2
            _, distances = find_mic(
                host_positions - candidate,
                cell=np.array(structure.cell, dtype=float, copy=True),
                pbc=structure.pbc,
            )
            clearance = float(np.min(distances))
            largest_clearance = max(largest_clearance, clearance)
            if clearance >= min_host_distance:
                valid_candidates.append((radial_distance, -clearance, candidate, clearance))

    if not valid_candidates:
        raise ValueError(
            f"No insertion site within {search_radius:.3f} A of the requested core "
            f"has the required {min_host_distance:.3f} A host clearance. "
            f"Largest sampled clearance: {largest_clearance:.3f} A."
        )

    _, _, position, clearance = min(valid_candidates, key=lambda item: item[:2])
    result = structure if inplace else structure.copy()
    realized_position = wrap_positions(
        np.asarray([position]),
        cell=np.array(result.cell, dtype=float, copy=True),
        pbc=result.pbc,
    )[0]
    result.append(light_element)
    result.positions[-1] = realized_position
    result.info.update(
        {
            "core_interstitial_symbol": light_element,
            "core_interstitial_position_A": realized_position.tolist(),
            "core_interstitial_target_A": target.tolist(),
            "core_interstitial_host_clearance_A": clearance,
        }
    )
    return result


class BCCScrewDislocation(WorkingDirectoryMixin):
    """Create, relax, and analyze screw dislocation dipoles in BCC metals."""

    def __init__(self, element, lattice_constant, elastic_constant, calculator, working_dir='.'):
        self.element = element
        self.a0 = lattice_constant
        self.cij = np.array(elastic_constant, dtype=float, copy=True)
        if self.cij[3, 3] < 0.0:
            warnings.warn(
                f"Calculated C_44={self.cij[3, 3]:.6g} GPa is negative; "
                "setting C_44 to 0.001 GPa for screw-dislocation setup.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.cij[3, 3] = 0.001
        self.calculator = calculator
        self.structure = None
        self.relaxed_structure = None
        self.dd_map = None
        self.dd_map_relaxed = None
        self.init_working_dir(working_dir, "screw_disloc")

    def validate_cubic_elastic_constants(self):
        """Validate cubic elastic constants before screw-dislocation setup."""
        c11 = float(self.cij[0, 0])
        c12 = float(self.cij[0, 1])
        c44 = float(self.cij[3, 3])

        if (c11 - c12) <= 0.0 or (c11 + 2.0 * c12) <= 0.0 or c44 <= 0.0:
            raise ValueError(
                "Elastic constants are mechanically unstable for cubic BCC "
                f"(C11={c11:.6g}, C12={c12:.6g}, C44={c44:.6g} GPa); "
                "skipping screw-dislocation workflow."
            )

    def create_dislocation_object(self):
        self.validate_cubic_elastic_constants()
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

    @staticmethod
    def insert_light_element(
        structure,
        light_element,
        core_position,
        **kwargs,
    ):
        """Insert a light element near a Cartesian dislocation-core position."""
        return insert_light_element_at_dislocation_core(
            structure,
            light_element,
            core_position,
            **kwargs,
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
        base_system, dislocation_dipole_ase, properties = (
            self.build_dislocation_dipole_ase(dislocation, disloc_center)
        )
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

    @staticmethod
    def build_dislocation_dipole_ase(dislocation, disloc_center=(0, 0, 0)):
        """Build an unrelaxed screw-dislocation dipole as an ASE structure.

        Keeping construction separate from relaxation allows alloy workflows to
        apply one identical chemical decoration to translated endpoint cells.
        """
        base_system, dislocation_system = dislocation.dipole(
            sizemults=[7, 5.5, 1],
            center=disloc_center,
            centerscale=False,
            boxtilt=True,
            return_base_system=True,
        )

        dislocation_dipole_ase, properties = dislocation_system.dump("ase_Atoms", return_prop=True)
        return base_system, dislocation_dipole_ase, properties

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
