"""Peierls barrier workflows for BCC screw dislocations."""

from __future__ import annotations

import os

from ase.filters import FrechetCellFilter
from ase.io import read
from ase.mep import NEB
from ase.optimize import BFGS, FIRE, LBFGS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BCCScrewDislocPeierlsBarrier:
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
        self.optimizer = optimizer.upper()
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        if self.optimizer not in ["FIRE", "BFGS", "LBFGS"]:
            raise ValueError(f"Optimizer must be 'FIRE', 'BFGS', or 'LBFGS', not {optimizer}")

    def _get_optimizer(self, atoms, trajectory=None):
        if trajectory:
            trajectory = f"{self.working_dir}/{trajectory}"
        if self.optimizer == "FIRE":
            return FIRE(atoms, trajectory=trajectory)
        if self.optimizer == "BFGS":
            return BFGS(atoms, trajectory=trajectory)
        return LBFGS(atoms, trajectory=trajectory)

    def relax_initial_final(self, fmax=0.001, steps=10000):
        for config in [self.initial_config, self.final_config]:
            config.calc = self.calc
            opt = self._get_optimizer(FrechetCellFilter(config))
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
        optimizer = self._get_optimizer(neb, trajectory="neb.traj")
        optimizer.run(fmax=fmax, steps=steps)

    def plot_barrier(
        self,
        element,
        trajectory="neb.traj",
        write_poscar=True,
        save_csv=True,
        compare_vasp=False,
        vasp_data_file=None,
    ):
        images = read(f"{self.working_dir}/{trajectory}@-{self.Nreplica}:")
        reaction_coords = np.linspace(0, 1, self.Nreplica)
        energies = []

        for i, image in enumerate(images):
            image.calc = self.calc
            energies.append(image.get_potential_energy())
            if write_poscar:
                image.write(f"{self.working_dir}/neb_{i}.poscar")

        energies = np.array(energies)
        energies = 1000 * (energies - min(energies)) / 2

        if save_csv:
            df = pd.DataFrame({"Reaction_Coordinate": reaction_coords, "Energy_meV": energies})
            csv_path = os.path.join(self.working_dir, "peierls_barrier_data.csv")
            df.to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(reaction_coords, energies, "-o", label=self.model_name)
        if compare_vasp and vasp_data_file:
            vasp_data = np.loadtxt(vasp_data_file)
            vasp_reaction_coords = vasp_data[:, 1] / max(vasp_data[:, 1])
            vasp_energies = vasp_data[:, 2] * 1000 / 2
            ax.plot(vasp_reaction_coords, vasp_energies, "-s", label="VASP")
        ax.set_xlabel("Reaction coordinate")
        ax.set_ylabel("Energy (meV)")
        ax.set_title(f"Peierls Barrier in {element} predicted by {self.model_name} + {self.optimizer}")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.working_dir, "peierls_barrier.png"), dpi=300)
        return fig, ax

