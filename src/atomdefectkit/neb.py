"""Peierls barrier workflows for BCC screw dislocations."""

from __future__ import annotations

from pathlib import Path

from ase.filters import FrechetCellFilter
from ase.io import read
from ase.mep import NEB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from atomdefectkit.utils.optimizers import build_optimizer, normalize_optimizer_name
from atomdefectkit.utils.paths import WorkingDirectoryMixin


class BCCScrewDislocPeierlsBarrier(WorkingDirectoryMixin):
    """Calculate Peierls barriers with a NEB workflow."""

    def __init__(
        self,
        initial_config,
        final_config,
        calc=None,
        model_name="uMLFFs",
        Nreplica=16,
        optimizer="FIRE",
        working_dir=".",
    ):
        self.initial_config = initial_config
        self.final_config = final_config
        self.model_name = model_name
        self.Nreplica = Nreplica
        self.calc = calc
        self.optimizer = normalize_optimizer_name(optimizer)
        self.init_working_dir(working_dir, "screw_disloc")

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _reference_barrier_path(self, element: str) -> Path | None:
        candidate = self._project_root() / "data" / "PeierlsPotential" / f"{element}_bcc_VASP.csv"
        return candidate if candidate.exists() else None

    def relax_initial_final(self, fmax=0.001, steps=10000):
        for label, config in [("initial", self.initial_config), ("final", self.final_config)]:
            config.calc = self.calc
            opt = build_optimizer(
                FrechetCellFilter(config),
                self.optimizer,
                logfile=self.path(f"{label}_{self.optimizer.lower()}_relax.log"),
            )
            opt.run(fmax=fmax, steps=steps)

        E_diff = self.final_config.get_potential_energy() - self.initial_config.get_potential_energy()
        if abs(E_diff) > 1e-3:
            print(f"WARNING: Initial and final energies differ by {E_diff * 1000:.2f} meV")
        return E_diff

    @staticmethod
    def _fix_pbc_issues(initial, final):
        cell = final.get_cell()
        pos_i = initial.get_positions()
        pos_f = final.get_positions()
        delta = pos_f - pos_i

        for i, vec in enumerate(delta):
            min_dist = float("inf")
            best_shift = np.zeros(3)
            for a, b, c in np.ndindex(3, 3, 3):
                shift = (a - 1) * cell[0] + (b - 1) * cell[1] + (c - 1) * cell[2]
                dist = np.linalg.norm(vec + shift)
                if dist < min_dist:
                    min_dist = dist
                    best_shift = shift
            pos_f[i] += best_shift
        return pos_f

    @staticmethod
    def _interpolate(initial, final, ratio):
        for config in [initial, final]:
            if config.constraints:
                del config.constraints
        pos_i = initial.get_positions()
        pos_f = final.get_positions()
        delta = pos_f - pos_i
        com = np.mean(delta, axis=0)
        interp_pos = pos_i + ratio * (delta - com)
        image = initial.copy()
        image.set_positions(interp_pos)
        return image

    def run_neb(self, fmax=0.001, steps=10000, spring_constant=0.5):
        initial = self.initial_config.copy()
        final = self.final_config.copy()
        final.set_cell(initial.get_cell(), scale_atoms=True)
        final.set_positions(self._fix_pbc_issues(initial, final))

        images = [self._interpolate(initial, final, r) for r in np.linspace(0, 1, self.Nreplica)]
        for image in images:
            image.calc = self.calc

        neb = NEB(images, k=spring_constant, allow_shared_calculator=True, climb=True)
        optimizer = build_optimizer(
            neb,
            self.optimizer,
            logfile=self.path(f"neb_{self.optimizer.lower()}_relax.log"),
            trajectory=self.path("neb.traj"),
        )
        optimizer.run(fmax=fmax, steps=steps)

    def plot_barrier(
        self,
        element,
        trajectory="neb.traj",
        write_poscar=True,
        save_csv=True,
        compare_vasp=True,
        vasp_data_file=None,
    ):
        images = read(f"{self.path(trajectory)}@-{self.Nreplica}:")
        reaction_coords = np.linspace(0, 1, self.Nreplica)
        energies = []

        for i, image in enumerate(images):
            image.calc = self.calc
            energies.append(image.get_potential_energy())
            if write_poscar:
                image.write(self.path(f"neb_{i}.poscar"))

        energies = np.array(energies)
        energies = 1000 * (energies - min(energies)) / 2

        if save_csv:
            df = pd.DataFrame({"Reaction_Coordinate": reaction_coords, "Energy_meV": energies})
            csv_path = self.path("peierls_barrier_data.csv")
            df.to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(reaction_coords, energies, "-o", label=self.model_name)
        if compare_vasp:
            reference_path = Path(vasp_data_file) if vasp_data_file else self._reference_barrier_path(element)
            if reference_path is not None and reference_path.exists():
                reference_data = np.loadtxt(reference_path)
                reference_reaction_coords = reference_data[:, 1] / max(reference_data[:, 1])
                reference_energies = reference_data[:, 2] * 1000 / 2
                ax.plot(reference_reaction_coords, reference_energies, "-s", label="DFT")
        ax.set_xlabel("Reaction coordinate")
        ax.set_ylabel("Energy (meV)")
        ax.set_title(f"Peierls Barrier in {element} predicted by {self.model_name} + {self.optimizer}")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(self.path("peierls_barrier.png"), dpi=300)
        return fig, ax
