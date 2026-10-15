"""Ordered B2 structures and defect-property workflow coordination."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read

from atomdefectkit.basic_properties import BasicProperties
from atomdefectkit.neb import BCCScrewDislocPeierlsBarrier
from atomdefectkit.units import EV_PER_A2_TO_J_PER_M2

EV_PER_ANGSTROM3_TO_GPA = 160.2176634


def _integer_triplet(values: Sequence[int], name: str) -> tuple[int, int, int]:
    vector = np.asarray(values, dtype=int)
    if vector.shape != (3,) or not np.any(vector):
        raise ValueError(f"{name} must be a non-zero integer triplet.")
    return tuple(int(value) for value in vector)


def _reduced_integer_vector(values: Sequence[int]) -> tuple[int, int, int]:
    vector = np.asarray(values, dtype=int)
    divisor = math.gcd(*[abs(int(value)) for value in vector if value])
    if divisor:
        vector //= divisor
    return tuple(int(value) for value in vector)


@dataclass(frozen=True)
class B2Crystal:
    """An ordered AB crystal in the cubic B2 (CsCl) structure."""

    species: tuple[str, str]
    lattice_constant: float

    def __post_init__(self) -> None:
        if len(self.species) != 2:
            raise ValueError("B2Crystal requires exactly two species.")
        if self.species[0] == self.species[1]:
            raise ValueError("The two ordered B2 sublattices must use different species.")
        unknown = [symbol for symbol in self.species if symbol not in atomic_numbers]
        if unknown:
            raise ValueError(f"Unknown chemical symbol(s): {', '.join(unknown)}")
        if not np.isfinite(self.lattice_constant) or self.lattice_constant <= 0.0:
            raise ValueError("lattice_constant must be a positive finite number.")

    @property
    def formula(self) -> str:
        return "".join(self.species)

    def build(self, repetitions: Sequence[int] = (1, 1, 1)) -> Atoms:
        """Build a conventional cubic B2 cell and preserve sublattice labels."""
        repeats = np.asarray(repetitions, dtype=int)
        if repeats.shape != (3,) or np.any(repeats <= 0):
            raise ValueError("repetitions must contain three positive integers.")

        atoms = Atoms(
            symbols=list(self.species),
            scaled_positions=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
            cell=np.eye(3) * self.lattice_constant,
            pbc=True,
        )
        atoms.new_array("b2_sublattice", np.array([0, 1], dtype=np.int8))
        atoms.info.update(
            {
                "crystal_structure": "B2",
                "b2_sublattice_0": self.species[0],
                "b2_sublattice_1": self.species[1],
            }
        )
        return atoms.repeat(tuple(int(value) for value in repeats))


@dataclass(frozen=True)
class B2SlipSystem:
    """A B2 slip translation expressed in cubic lattice coordinates.

    ``translation_fraction`` multiplies ``a * direction``. For example,
    ``direction=(1, 1, 1)`` and ``translation_fraction=0.5`` represents
    a 1/2<111> translation.
    """

    plane: tuple[int, int, int]
    direction: tuple[int, int, int]
    translation_fraction: float = 1.0
    name: str | None = None

    def __post_init__(self) -> None:
        plane = _integer_triplet(self.plane, "plane")
        direction = _integer_triplet(self.direction, "direction")
        if np.dot(plane, direction) != 0:
            raise ValueError("Slip direction must lie in the slip plane.")
        if not np.isfinite(self.translation_fraction) or self.translation_fraction <= 0.0:
            raise ValueError("translation_fraction must be a positive finite number.")
        object.__setattr__(self, "plane", plane)
        object.__setattr__(self, "direction", direction)

    @property
    def transverse_direction(self) -> tuple[int, int, int]:
        """Return a reduced second in-plane vector for slab construction."""
        return _reduced_integer_vector(np.cross(self.plane, self.direction))

    def translation_distance(self, lattice_constant: float) -> float:
        """Return the scalar used with the integer direction by the GSFE workflow."""
        return float(lattice_constant) * self.translation_fraction

    def burgers_magnitude(self, lattice_constant: float) -> float:
        direction_norm = np.linalg.norm(np.asarray(self.direction, dtype=float))
        return self.translation_distance(lattice_constant) * direction_norm


def estimate_sinusoidal_peierls_stress(
    peierls_energy_ev: float,
    *,
    line_length_angstrom: float,
    burgers_angstrom: float,
    migration_period_angstrom: float | None = None,
    dislocation_count: int = 1,
    energy_definition: str = "amplitude",
) -> float:
    """Estimate Peierls stress in GPa from a sinusoidal energy landscape.

    The force balance is ``tau * b = max(dU/dx)``, where ``U`` is energy per
    unit dislocation length. ``energy_definition='amplitude'`` corresponds to
    ``U(x) = E_P [1 - cos(2*pi*x/lambda)]`` and therefore reproduces
    ``tau_P = 2*pi*E_P/(b*lambda)``. Use ``'peak_to_valley'`` when the supplied
    value is the full NEB barrier, which is twice the sinusoidal amplitude.
    """
    positive_values = {
        "peierls_energy_ev": peierls_energy_ev,
        "line_length_angstrom": line_length_angstrom,
        "burgers_angstrom": burgers_angstrom,
    }
    for name, value in positive_values.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number.")
    if dislocation_count < 1:
        raise ValueError("dislocation_count must be a positive integer.")

    period = burgers_angstrom if migration_period_angstrom is None else migration_period_angstrom
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("migration_period_angstrom must be positive and finite.")
    if energy_definition == "amplitude":
        amplitude_ev = peierls_energy_ev
    elif energy_definition == "peak_to_valley":
        amplitude_ev = 0.5 * peierls_energy_ev
    else:
        raise ValueError("energy_definition must be 'amplitude' or 'peak_to_valley'.")

    amplitude_per_length = amplitude_ev / (dislocation_count * line_length_angstrom)
    stress_ev_per_angstrom3 = (
        2.0 * np.pi * amplitude_per_length / (burgers_angstrom * period)
    )
    return float(stress_ev_per_angstrom3 * EV_PER_ANGSTROM3_TO_GPA)


class B2PeierlsBarrier(BCCScrewDislocPeierlsBarrier):
    """Endpoint-based NEB with an explicit dislocation multiplicity."""

    def __init__(self, *args, dislocation_count: int, **kwargs) -> None:
        if not isinstance(dislocation_count, (int, np.integer)) or dislocation_count < 1:
            raise ValueError("dislocation_count must be a positive integer.")
        super().__init__(*args, **kwargs)
        self.dislocation_count = int(dislocation_count)

    def barrier_profile(self, trajectory: str = "neb.traj") -> pd.DataFrame:
        """Return total and per-dislocation NEB energies without hidden scaling."""
        images = read(f"{self.path(trajectory)}@-{self.Nreplica}:")
        total_energies = []
        for image in images:
            image.calc = self.calc
            total_energies.append(image.get_potential_energy())
        total_energies = np.asarray(total_energies, dtype=float)
        relative_total = total_energies - np.min(total_energies)
        return pd.DataFrame(
            {
                "reaction_coordinate": np.linspace(0.0, 1.0, len(images)),
                "total_energy_eV": total_energies,
                "relative_total_energy_eV": relative_total,
                "relative_energy_per_dislocation_eV": (
                    relative_total / self.dislocation_count
                ),
            }
        )

    def plot_barrier(self, compound: str, trajectory: str = "neb.traj"):
        """Save and plot the barrier per dislocation for a B2 endpoint cell."""
        data = self.barrier_profile(trajectory)
        data.to_csv(self.path("peierls_barrier_data.csv"), index=False)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(
            data["reaction_coordinate"],
            1000.0 * data["relative_energy_per_dislocation_eV"],
            "-o",
            label=self.model_name,
        )
        ax.set_xlabel("Reaction coordinate")
        ax.set_ylabel("Energy per dislocation (meV)")
        ax.set_title(f"Peierls barrier in B2 {compound}")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.path("peierls_barrier.png"), dpi=300)
        return fig, ax, data


class B2Workflow:
    """Coordinate bulk, gamma-line, and endpoint-based Peierls calculations."""

    def __init__(
        self,
        crystal: B2Crystal,
        *,
        model_name: str | None = None,
        model_parameters: dict | None = None,
        device: str = "cuda",
        calculator=None,
        working_dir: str | Path = ".",
    ) -> None:
        self.crystal = crystal
        self.working_dir = Path(working_dir)
        self.properties = BasicProperties(
            model_name=model_name,
            model_parameters=model_parameters,
            device=device,
            calculator=calculator,
            working_dir=self.working_dir,
        )

    @property
    def calculator(self):
        return self.properties.get_calculator()

    def build_bulk(self, repetitions: Sequence[int] = (1, 1, 1)) -> Atoms:
        return self.crystal.build(repetitions)

    def calculate_equilibrium_lattice_constant(
        self,
        volume_scales: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Fit a cubic Birch-Murnaghan EOS for the ordered two-atom cell."""
        if volume_scales is None:
            volume_scales = np.linspace(0.97, 1.03, 13)
        return self.properties.calculate_equilibrium_a0_birch_murnaghan(
            self.build_bulk(),
            vol_range=np.asarray(volume_scales, dtype=float),
        )

    def calculate_elastic_constants(self, lattice_constant: float | None = None, verbose=False):
        """Calculate the cubic elastic matrix in GPa."""
        bulk = self.build_bulk()
        if lattice_constant is not None:
            bulk.set_cell(np.eye(3) * lattice_constant, scale_atoms=True)
        return self.properties.calculate_elastic_constants(bulk, verbose=verbose)

    def calculate_gamma_line(
        self,
        slip_system: B2SlipSystem,
        *,
        lattice_constant: float | None = None,
        layers: int = 24,
        num_steps: int = 20,
        fmax: float = 0.01,
        steps: int = 1000,
        fix_normal: bool = False,
        write_xyz: bool = True,
        initial_translation: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> list[float]:
        """Return an ordered-B2 gamma line in eV/Angstrom^2.

        The shared stacking-fault engine returns J/m^2. Convert at this API
        boundary to retain the B2 result and combined-plot conventions.
        """
        a0 = self.crystal.lattice_constant if lattice_constant is None else lattice_constant
        bulk = self.build_bulk()
        bulk.set_cell(np.eye(3) * a0, scale_atoms=True)
        initial_translation = np.asarray(initial_translation, dtype=float)
        if initial_translation.shape != (3,) or not np.all(np.isfinite(initial_translation)):
            raise ValueError("initial_translation must contain three finite lattice components.")
        if not np.isclose(np.dot(initial_translation, slip_system.plane), 0.0):
            raise ValueError("initial_translation must lie in the slip plane.")
        direction = np.asarray(slip_system.direction, dtype=float)
        transverse = np.asarray(slip_system.transverse_direction, dtype=float)
        normal = np.asarray(slip_system.plane, dtype=float)
        basis = np.array([direction / np.linalg.norm(direction),
                          transverse / np.linalg.norm(transverse),
                          normal / np.linalg.norm(normal)])
        initial_shift = basis @ (a0 * initial_translation)
        label = slip_system.name or "B2"
        directory_label = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in label
        )
        workflow = self.properties.create_stacking_fault_workflow(
            atoms=bulk,
            formula=self.crystal.formula,
            info=label,
            working_dir=self.working_dir / "gamma_lines" / directory_label,
        )
        energies_j_per_m2 = workflow.stacking_fault(
            a=slip_system.direction,
            b=slip_system.transverse_direction,
            miller=slip_system.plane,
            distance=slip_system.translation_distance(a0),
            layers=layers,
            num_steps=num_steps,
            fmax=fmax,
            steps=steps,
            fix_normal=fix_normal,
            write_xyz=write_xyz,
            initial_shift=initial_shift,
        )
        return (np.asarray(energies_j_per_m2) / EV_PER_A2_TO_J_PER_M2).tolist()

    def plot_gamma_lines(
        self,
        gamma_lines: Mapping[str, Sequence[float]],
        *,
        save_name: str = "b2_gamma_lines.png",
    ):
        """Save a combined comparison of normalized B2 gamma-line paths."""
        if not gamma_lines:
            raise ValueError("gamma_lines must contain at least one curve.")

        rows = []
        fig, ax = plt.subplots(figsize=(5, 4))
        for label, values in gamma_lines.items():
            energies = np.asarray(values, dtype=float)
            if energies.ndim != 1 or len(energies) < 2 or not np.all(np.isfinite(energies)):
                raise ValueError(f"Gamma line '{label}' must contain finite 1D data.")
            coordinates = np.linspace(0.0, 1.0, len(energies))
            energies_mev = 1000.0 * energies
            ax.plot(coordinates, energies_mev, "-o", markersize=3, label=label)
            rows.extend(
                {
                    "slip_system": label,
                    "reaction_coordinate": coordinate,
                    "energy_meV_per_A2": energy,
                }
                for coordinate, energy in zip(coordinates, energies_mev)
            )

        output_dir = self.working_dir / "gamma_lines"
        output_dir.mkdir(parents=True, exist_ok=True)
        data = pd.DataFrame(rows)
        data.to_csv(output_dir / "b2_gamma_lines.csv", index=False)
        ax.set_xlabel("Reaction coordinate")
        ax.set_ylabel("Energy (meV/A$^2$)")
        ax.set_title(f"B2 {self.crystal.formula} gamma lines")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / save_name, dpi=300)
        return fig, ax, data

    def calculate_gamma_lines(
        self,
        slip_systems: Sequence[B2SlipSystem],
        **kwargs,
    ) -> dict[str, list[float]]:
        """Calculate several gamma lines and create one comparison figure."""
        if not slip_systems:
            raise ValueError("slip_systems must contain at least one slip system.")
        labels = [slip.name or f"path_{index}" for index, slip in enumerate(slip_systems)]
        if len(set(labels)) != len(labels):
            raise ValueError("Each gamma-line slip system must have a unique name.")

        results = {
            label: self.calculate_gamma_line(slip, **kwargs)
            for label, slip in zip(labels, slip_systems, strict=True)
        }
        self.plot_gamma_lines(results)
        return results

    def create_peierls_workflow(
        self,
        initial_config: Atoms,
        final_config: Atoms,
        *,
        model_label: str = "uMLIP",
        replicas: int = 11,
        optimizer: str = "FIRE",
        dislocation_count: int,
    ) -> B2PeierlsBarrier:
        """Create NEB from externally constructed, atom-matched B2 endpoints."""
        return B2PeierlsBarrier(
            initial_config=initial_config,
            final_config=final_config,
            calc=self.calculator,
            model_name=model_label,
            Nreplica=replicas,
            optimizer=optimizer,
            working_dir=self.working_dir / "peierls_barrier",
            dislocation_count=dislocation_count,
        )
