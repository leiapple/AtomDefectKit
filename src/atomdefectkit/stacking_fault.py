"""Generalized stacking-fault workflows and helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ase.build.tools import cut, rotate
from ase.constraints import FixedLine
from ase.io import write
import ase.units as units

import numpy as np
from atomdefectkit.utils.optimizers import build_optimizer, normalize_optimizer_name
from atomdefectkit.utils.paths import WorkingDirectoryMixin
from atomdefectkit.utils.plotting import plot_xy_curves


@dataclass
class StackingFaultCurve:
    """Container and helpers for generalized stacking-fault calculations."""

    displacements: np.ndarray
    energies: np.ndarray
    area: float | None = None

    @classmethod
    def from_arrays(cls, displacements, energies, area=None):
        """Build a ``StackingFaultCurve`` from array-like inputs.

        Args:
            displacements: Reaction-coordinate values for the stacking-fault path.
            energies: Energy values associated with ``displacements``.
            area: Optional interfacial area used for normalization.

        Returns:
            StackingFaultCurve: Curve object with NumPy-backed arrays.
        """
        return cls(np.asarray(displacements, dtype=float), np.asarray(energies, dtype=float), area=area)

    def normalized_energies(self):
        """Shift energies so the minimum value is zero, optionally normalizing by area.

        Returns:
            np.ndarray: Normalized energy values.
        """
        values = self.energies - np.min(self.energies)
        if self.area is not None:
            return values / self.area
        return values

    def unstable_fault_energy(self):
        """Return the unstable stacking-fault energy along the path.

        Returns:
            float: Maximum normalized stacking-fault energy.
        """
        return float(np.max(self.normalized_energies()))

    def intrinsic_fault_energy(self):
        """Return the intrinsic stacking-fault energy estimate from the path.

        Returns:
            float: Minimum normalized energy away from the two endpoints.
        """
        energies = self.normalized_energies()
        if len(energies) < 3:
            return float(np.min(energies))
        return float(np.min(energies[1:-1]))


class StackingFaultWorkflow(WorkingDirectoryMixin):
    """Build, relax, and plot generalized stacking-fault energy curves."""

    def __init__(
        self,
        atoms,
        calculator,
        formula: str | None = None,
        info: str = "Calculated",
        optimizer: str = "FIRE",
        working_dir: str = ".",
    ) -> None:
        """Initialize the stacking-fault workflow.

        Args:
            atoms: Reference bulk or slab structure used to build the faulted slab.
            calculator: ASE-compatible calculator used for energies and relaxations.
            formula: Optional chemical formula label used in output filenames.
            info: Short label written into summary files and plot legends.
            optimizer: Geometry optimizer to use. Supported values are ``FIRE``,
                ``BFGS``, and ``LBFGS``.
            working_dir: Directory where logs, plots, and text outputs are saved.

        Raises:
            ValueError: If ``optimizer`` is not one of the supported names.
        """
        self.atoms = atoms.copy()
        self.calc = calculator
        self.formula = formula or self.atoms.get_chemical_formula()
        self.info = info
        self.optimizer = normalize_optimizer_name(optimizer)
        self.init_working_dir(working_dir)

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _reference_curve_path(self, miller) -> Path | None:
        miller_label = "".join(str(abs(int(index))) for index in miller)
        candidate = self._project_root() / "data" / "stacking_faults" / f"{self.formula}_{miller_label}.csv"
        return candidate if candidate.exists() else None

    @staticmethod
    def _format_reference_curve(reference_curve: np.ndarray, miller) -> tuple[np.ndarray, np.ndarray]:
        x_values = reference_curve[:, 0]
        y_values = reference_curve[:, 1] * 1000

        miller_abs = tuple(abs(int(index)) for index in miller)
        if miller_abs == (1, 1, 2):
            y_values = y_values[::-1]

        return x_values, y_values

    def _relax(self, atoms, fmax=0.01, steps=1000, logfile="stacking_fault_relax.log"):
        """Relax a structure with the workflow calculator and optimizer.

        Args:
            atoms: Structure to relax.
            fmax: Force convergence threshold in eV/Angstrom.
            steps: Maximum number of optimizer steps.
            logfile: Optimizer logfile name saved in ``working_dir``.

        Returns:
            ase.Atoms: Relaxed structure.
        """
        atoms.calc = self.calc
        optimizer = build_optimizer(
            atoms,
            self.optimizer,
            logfile=self.path(logfile),
        )
        optimizer.run(fmax=fmax, steps=steps)
        return atoms

    @staticmethod
    def _constrain_xy(atoms):
        """Constrain all atoms to move only along the z direction.

        Args:
            atoms: Structure to constrain in place.
        """
        atoms.set_constraint(FixedLine(indices=list(range(len(atoms))), direction=[0.0, 0.0, 1.0]))

    @staticmethod
    def _strip_for_xyz(atoms):
        """Return a clean copy suitable for XYZ or EXTXYZ trajectory output.

        Args:
            atoms: Structure to serialize.

        Returns:
            ase.Atoms: Copy without calculator or constraint state attached.
        """
        clean = atoms.copy()
        clean.calc = None
        clean.constraints = []
        return clean

    def stacking_fault(
        self,
        a,
        b,
        miller,
        distance,
        layers=18,
        num_steps=10,
        fmax=0.01,
        steps=1000,
        write_xyz=True,
    ) -> list[float]:
        """Calculate a generalized stacking-fault energy curve.

        Parameters
        ----------
        a, b : sequence[float]
            In-plane slab vectors in scaled coordinates passed to ``ase.build.cut``.
        miller : sequence[int]
            Miller-index label used in filenames and summary output.
        distance : float
            Fractional translation distance applied along vector ``a``.
        layers : int
            Number of layers in the generated slab.
        num_steps : int
            Number of sliding increments along the reaction path.
        fmax : float
            Force threshold used during each relaxation.
        steps : int
            Maximum optimizer steps for each relaxation.
        write_xyz : bool
            Whether to write each relaxed shifted slab to an XYZ trajectory.

        Returns
        -------
        list[float]
            Stacking-fault energies in eV/Angstrom^2 relative to the unshifted slab.
        """
        slab = cut(
            self.atoms.copy(),
            a,
            b,
            clength=None,
            origo=(0, 0, 0),
            nlayers=layers,
            extend=1.0,
            tolerance=0.01,
            maxatoms=None,
        )
        rotate(slab, a, (1, 0, 0), b, (0, 1, 0), center=(0, 0, 0))
        slab = self._relax(slab, fmax=fmax, steps=steps, logfile="stacking_fault_slab_relax.log")
        slab.center(vacuum=10.0, axis=2)

        box = slab.get_cell()
        area = np.linalg.norm(np.cross(box[0], box[1]))

        shift_distance = np.linalg.norm(np.asarray(a, dtype=float)) * distance
        shift_indices = [atom.index for atom in slab if atom.position[2] > 0.5 * slab.cell[2][2]]
        slide_step = shift_distance / num_steps

        miller_str = "-".join(map(str, miller))
        xyz_path = self.path(f"{self.formula}_{miller_str}_StackingFault.xyz")
        if write_xyz and os.path.exists(xyz_path):
            os.remove(xyz_path)

        energies = []
        for i in range(num_steps + 1):
            slab_shift = slab.copy()
            slab_shift.positions[shift_indices] += [slide_step * i, 0, 0]
            self._constrain_xy(slab_shift)
            slab_shift = self._relax(
                slab_shift,
                fmax=fmax,
                steps=steps,
                logfile=f"stacking_fault_step_{i}.log",
            )
            defect_energy = 1 / units.J / (1/units.m ** 2) * slab_shift.get_potential_energy() / area
            energies.append(defect_energy)
            if write_xyz:
                write(xyz_path, self._strip_for_xyz(slab_shift), append=True)

        energies = np.asarray(energies)
        energies -= energies[0]
        coords = np.linspace(0, 1, len(energies))

        out_path = self.path(f"{self.formula}_{miller_str}_stacking_fault.out")
        fig_path = self.path(f"{self.formula}_{miller_str}_stacking_fault.png")
        summary_path = self.path(f"{self.formula}_{miller_str}_stacking_fault.out")

        with open(out_path, "w", encoding="utf-8") as file:
            file.write("Reaction_Coordinate   Energy(meV/A^2)\n")
            for coord, energy in zip(coords, energies):
                file.write(f"{coord:.4f}   {energy * 1000:.4f}\n")

        with open(summary_path, "a", encoding="utf-8") as file:
            print(
                f"{self.info:<12}{miller} Stacking_Fault: {max(energies) * 1000:.4f} meV/A^2",
                file=file,
            )

        curves = [{"x": coords, "y": energies * 1000, "marker": "o", "label": self.info}]
        reference_path = self._reference_curve_path(miller)
        if reference_path is not None:
            reference_curve = np.loadtxt(reference_path, delimiter=",")
            reference_x, reference_y = self._format_reference_curve(reference_curve, miller)
            curves.append(
                {
                    "x": reference_x,
                    "y": reference_y,
                    "marker": "s",
                    "label": "DFT",
                }
            )

        plot_xy_curves(
            curves=curves,
            xlabel="Reaction Coordinate",
            ylabel="Energy (meV/A^2)",
            title="Stacking-Fault Curve",
            save_path=fig_path,
            figsize=(5, 4),
            show_legend=True,
        )

        return energies.tolist()
