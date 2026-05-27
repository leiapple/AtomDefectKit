"""Traction-separation workflows and helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ase import Atom, Atoms
from ase.filters import UnitCellFilter
from ase.io import write
from ase.optimize import BFGS, FIRE, LBFGS
from ase.optimize.sciopt import SciPyFminCG
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np


@dataclass
class TractionSeparationCurve:
    """Container for cohesive traction-separation relations."""

    separation: np.ndarray
    energy: np.ndarray
    area: float | None = None

    @classmethod
    def from_arrays(cls, separation, energy, area=None):
        """Build a ``TractionSeparationCurve`` from array-like inputs.

        Args:
            separation: Separation distances along the opening path.
            energy: Energies associated with ``separation``.
            area: Optional interfacial area used for normalization.

        Returns:
            TractionSeparationCurve: Curve object backed by NumPy arrays.
        """
        return cls(np.asarray(separation, dtype=float), np.asarray(energy, dtype=float), area=area)

    def traction(self):
        """Differentiate the energy curve with respect to separation.

        Returns:
            np.ndarray: Traction values computed by finite differences.
        """
        return np.gradient(self.energy, self.separation)

    def peak_traction(self):
        """Return the maximum traction on the curve.

        Returns:
            float: Peak cohesive traction.
        """
        return float(np.max(self.traction()))

    def work_of_separation(self):
        """Return the work of separation derived from the energy curve.

        Returns:
            float: Maximum shifted energy, optionally normalized by area.
        """
        values = self.energy - np.min(self.energy)
        if self.area is not None:
            values = values / self.area
        return float(np.max(values))


class TractionSeparationWorkflow:
    """Build and analyze traction-separation curves for pure and H-decorated slabs."""

    def __init__(
        self,
        atoms,
        calculator,
        directions=None,
        repeat=(3, 3, 16),
        working_dir=".",
        optimizer="FIRE",
    ) -> None:
        """Initialize the traction-separation workflow.

        Args:
            calculator: ASE-compatible calculator used for all energy evaluations.
            lattice_constant: Lattice constant of the reference crystal in Angstrom.
            element: Host element symbol used to build the slab.
            directions: Crystal directions passed to ``BodyCenteredCubic``.
            repeat: Supercell repetition counts along the three slab directions.
            working_dir: Directory where trajectories, logs, text files, and plots are saved.
            optimizer: Geometry optimizer label. Supported values are ``FIRE``,
                ``BFGS``, and ``LBFGS``.

        Raises:
            ValueError: If ``optimizer`` is not one of the supported names.
        """
        self.atoms = atoms
        self.calc = calculator
        self.directions = directions or ([0, 0, -1], [1, 1, 0], [1, -1, 0])
        self.repeat = repeat
        self.working_dir = working_dir
        self.optimizer = optimizer.upper()
        os.makedirs(self.working_dir, exist_ok=True)
        if self.optimizer not in {"FIRE", "BFGS", "LBFGS"}:
            raise ValueError(f"Optimizer must be 'FIRE', 'BFGS', or 'LBFGS', not {optimizer}")

    def generate_tetra_sites(self, cell: np.ndarray, nx: int, ny: int, eps: float = 1e-3):
        """Tile the reference tetrahedral H sites across the in-plane supercell.

        Args:
            cell: Simulation cell used for the slab.
            nx: Number of repeats along the first in-plane direction.
            ny: Number of repeats along the second in-plane direction.
            eps: Small z offset passed to ``base_tetra_sites_frac``.

        Returns:
            list[dict]: Site records containing positions, layer labels, and tile indices.
        """
        base_sites = [
            (0.37, 0.5, 0.5 + eps, "top"),
            (0.37, 0.0, 0.5 + eps, "top"),
            (0.37, 0.5, 0.5 - eps, "bottom"),
            (0.87, 0.0, 0.5 - eps, "bottom"),
        ]
        sites = []
        for ix in range(nx):
            for iy in range(ny):
                for site_id, (u, v, w, layer) in enumerate(base_sites):
                    U = (ix + (u % 1.0)) / nx
                    V = (iy + (v % 1.0)) / ny
                    cart = U * cell[0] + V * cell[1] + w * cell[2]
                    sites.append(
                        {
                            "pos": cart,
                            "layer": layer,
                            "ix": ix,
                            "iy": iy,
                            "site_id": site_id,
                        }
                    )
        return sites

    def build_bcc_slab(self):
        """Build the oriented BCC slab used for traction-separation scans.

        Returns:
            ase.Atoms: Repeated and centered slab structure.
        """
        
        slab = self.atoms.repeat(self.repeat)
        slab.center(vacuum=0.0, axis=2)
        return slab

    @staticmethod
    def make_gap_from_base(slab, vac):
        """Open a cleavage gap by extending the cell and shifting the top half.

        Args:
            slab: Reference slab structure.
            vac: Gap opening distance in Angstrom.

        Returns:
            ase.Atoms: Copied structure with the imposed opening displacement.
        """
        shifted = slab.copy()
        z = shifted.positions[:, 2]
        zmid = shifted.get_cell()[2][2] / 2.0
        top_mask = z >= zmid

        cell = shifted.get_cell()
        cell[2, 2] += vac
        shifted.set_cell(cell, scale_atoms=False)
        shifted.positions[top_mask, 2] += vac
        return shifted

    @staticmethod
    def surface_area_from_atoms(atoms):
        """Compute the in-plane surface area from the first two cell vectors.

        Args:
            atoms: Structure providing the simulation cell.

        Returns:
            float: In-plane area in Angstrom squared.
        """
        cell = atoms.get_cell()
        return np.linalg.norm(np.cross(cell[0], cell[1]))

    @staticmethod
    def _compute_traction_curve(seps, energies, areas):
        """Convert sampled separation energies into midpoint tractions.

        Args:
            seps: Separation distances in Angstrom.
            energies: Total energies in eV.
            areas: Surface areas in Angstrom squared.

        Returns:
            tuple[np.ndarray, np.ndarray]: Midpoint separations and tractions in GPa.
        """
        n = len(seps) - 1
        sepas = np.empty(n)
        tracs = np.empty(n)
        for i in range(n):
            sep0, sep1 = seps[i], seps[i + 1]
            E0, E1 = energies[i], energies[i + 1]
            A0, A1 = areas[i], areas[i + 1]
            sepas[i] = 0.5 * (sep0 + sep1)
            area_avg = 0.5 * (A0 + A1)
            eVA32GPa = 160.2176621
            tracs[i] = eVA32GPa * (E1 - E0) / (area_avg * (sep1 - sep0))
        return sepas, tracs

    @staticmethod
    def count_surface_H(atoms, z_tol=1.6):
        """Count H atoms associated with the top and bottom free surfaces.

        Args:
            atoms: Structure containing host and hydrogen atoms.
            z_tol: Distance cutoff used to assign H atoms to a surface.

        Returns:
            tuple[int, int]: Number of H atoms on the top and bottom surfaces.
        """
        pos = atoms.get_positions()
        z = pos[:, 2]
        symbols = atoms.get_chemical_symbols()
        zmid = atoms.get_cell()[2, 2] / 2.0

        fe_idx = [i for i, symbol in enumerate(symbols) if symbol != "H"]
        z_fe = z[fe_idx]
        bottom_surface_z = z_fe[z_fe < zmid].max()
        top_surface_z = z_fe[z_fe > zmid].min()

        nH_bottom = 0
        nH_top = 0
        for i, symbol in enumerate(symbols):
            if symbol != "H":
                continue
            dz_bottom = z[i] - bottom_surface_z
            dz_top = top_surface_z - z[i]
            if 0.0 <= dz_bottom <= z_tol:
                nH_bottom += 1
            elif 0.0 <= dz_top <= z_tol:
                nH_top += 1
        return nH_top, nH_bottom

    def _get_optimizer(self, atoms, logfile):
        """Create the configured optimizer for a given structure.

        Args:
            atoms: Structure to optimize.
            logfile: Optimizer logfile name.

        Returns:
            FIRE | BFGS | LBFGS: Configured ASE optimizer instance.
        """
        logfile = os.path.join(self.working_dir, logfile)
        if self.optimizer == "FIRE":
            return FIRE(atoms, logfile=logfile)
        if self.optimizer == "BFGS":
            return BFGS(atoms, logfile=logfile)
        return LBFGS(atoms, logfile=logfile)

    def relax_base_structure(self, atoms, nH, fmax_cell=1e-3, fmax_atoms=1e-2, steps=200000):
        """Run the two-stage relaxation used before opening traction-separation gaps.

        Args:
            atoms: H-decorated slab to relax.
            nH: Initial number of H atoms, used in logfile names.
            fmax_cell: Force threshold for the cell relaxation stage.
            fmax_atoms: Force threshold for the atomic relaxation stage.
            steps: Maximum steps for each optimizer.

        Returns:
            ase.Atoms: Relaxed slab.
        """
        atoms.calc = self.calc
        ucf = UnitCellFilter(atoms)
        opt = SciPyFminCG(ucf, logfile=os.path.join(self.working_dir, f"box_relax_H{nH}.log"))
        opt.run(fmax=fmax_cell, steps=steps)
        opt2 = FIRE(atoms, logfile=os.path.join(self.working_dir, f"fire_relax_H{nH}.log"))
        opt2.run(fmax=fmax_atoms, steps=steps)
        atoms.set_constraint()
        return atoms

    def add_hydrogen_sites(self, slab, nH, top_sites, bot_sites):
        """Populate the slab with symmetric H coverages on the two free surfaces.

        Args:
            slab: Reference slab without H.
            nH: Total number of H atoms to add.
            top_sites: Candidate adsorption sites on the top surface.
            bot_sites: Candidate adsorption sites on the bottom surface.

        Returns:
            ase.Atoms: Copied slab decorated with H atoms.
        """
        struct = slab.copy()
        k_each = nH // 2
        chosen_top = top_sites[:k_each]
        chosen_bot = bot_sites[:k_each]
        for site in chosen_top + chosen_bot:
            struct.append(Atom("H", position=site["pos"]))
        return struct

    def _scan_separation_curve(self, base, label, vacuum_values, write_xyz=True):
        """Evaluate one traction-separation curve for a prepared base slab.

        Args:
            base: Relaxed reference slab before opening the separation gap.
            label: Short label used in trajectory and logfile names.
            vacuum_values: Separation distances used to open the cleavage gap.
            write_xyz: Whether to save an ``extxyz`` trajectory for the scan.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
            sampled separations, energies, and surface areas.
        """
        base_area = self.surface_area_from_atoms(base)
        energies = []
        seps = []
        areas = []
        frames = []

        for step_idx, vac in enumerate(vacuum_values):
            geom = self.make_gap_from_base(base, vac)
            geom.calc = self.calc
            energy = geom.get_potential_energy()

            energies.append(energy)
            seps.append(vac)
            areas.append(base_area)
            if write_xyz:
                frame = Atoms(
                    numbers=geom.get_atomic_numbers(),
                    positions=geom.get_positions(),
                    cell=geom.get_cell(),
                    pbc=geom.get_pbc(),
                )
                frame.info["vacuum_gap_Ang"] = float(vac)
                frame.info["frame_index"] = int(step_idx)
                frames.append(frame)

        if write_xyz:
            traj_name = os.path.join(self.working_dir, f"trajectory_{label}.extxyz")
            write(traj_name, frames, format="extxyz")

        return np.array(seps), np.array(energies), np.array(areas)

    def run_pure_separation(self, vacuum_values, write_xyz=True):
        """Run a traction-separation calculation for the pure metal slab.

        Args:
            vacuum_values: Separation distances used to open the cleavage gap.
            write_xyz: Whether to save an ``extxyz`` trajectory for the scan.

        Returns:
            dict: Separation distances, energies, areas, midpoint tractions,
            and derived cohesive metrics for the pure slab.
        """
        slab = self.build_bcc_slab()
        base = self.relax_base_structure(slab, nH=0)
        seps, energies, areas = self._scan_separation_curve(
            base,
            label="pure",
            vacuum_values=vacuum_values,
            write_xyz=write_xyz,
        )
        mid_seps, tractions = self._compute_traction_curve(seps, energies, areas)
        mid_seps = np.insert(mid_seps, 0, 0.0)
        tractions = np.insert(tractions, 0, 0.0)

        return {
            "separation": seps,
            "energy": energies,
            "area": areas,
            "midpoint_separation": mid_seps,
            "traction": tractions,
            "curve": TractionSeparationCurve.from_arrays(seps, energies, area=float(np.mean(areas))),
            "surface_energy_J_m2": float(np.trapz(tractions, mid_seps) * 0.1 / 2.0),
            "sigma_max_GPa": float(np.max(tractions)),
        }

    def run_h_separation(
        self,
        nH_list,
        vacuum_values,
        eps=1e-3,
        z_tol=1.6,
        write_xyz=True,
    ):
        """Run traction-separation scans across a list of H coverages.

        Args:
            nH_list: Iterable of H counts to scan.
            vacuum_values: Separation distances used to open the cleavage gap.
            eps: Small z offset used when generating adsorption sites.
            z_tol: Distance cutoff used when counting surface H atoms.
            write_xyz: Whether to save an ``extxyz`` trajectory for each coverage.

        Returns:
            dict: Raw TS curves, averaged metrics, effective coverage values,
            and output file paths for the completed scan.
        """
        slab = self.build_bcc_slab()
        nx, ny, _ = self.repeat
        all_sites = self.generate_tetra_sites(slab.get_cell(), nx=nx, ny=ny, eps=eps)
        top_sites = [site for site in all_sites if site["layer"] == "top"]
        bot_sites = [site for site in all_sites if site["layer"] == "bottom"]

        ts_data = {nH: {"x": None, "ys": []} for nH in nH_list}
        effective_cov = {}
        manifest = ["# nH\tvacuum_A\n"]

        for nH in nH_list:
            base = self.add_hydrogen_sites(slab, nH, top_sites, bot_sites)
            base = self.relax_base_structure(base, nH=nH)
            seps_arr, eners_arr, areas_arr = self._scan_separation_curve(
                base,
                label=f"H{nH}",
                vacuum_values=vacuum_values,
                write_xyz=write_xyz,
            )
            for vac in seps_arr:
                manifest.append(f"{nH}\t{vac:.6f}\n")
            nH_top, nH_bottom = self.count_surface_H(base, z_tol=z_tol)
            effective_cov[nH] = nH_top + nH_bottom
            mid_seps, tracs = self._compute_traction_curve(seps_arr, eners_arr, areas_arr)
            mid_seps = np.insert(mid_seps, 0, 0.0)
            tracs = np.insert(tracs, 0, 0.0)

            if ts_data[nH]["x"] is None:
                ts_data[nH]["x"] = mid_seps
            elif not np.allclose(ts_data[nH]["x"], mid_seps):
                raise RuntimeError("Midpoint separations differ between H coverages.")
            ts_data[nH]["ys"].append(tracs)

        manifest_path = os.path.join(self.working_dir, "manifest_all_seeds.txt")
        with open(manifest_path, "w", encoding="utf-8") as file:
            file.writelines(manifest)

        nH_sorted = sorted(nH_list)
        Nmax_surface = 2 * len(top_sites)
        theta_eff = {}
        gamma_data = {}
        sigma_max = {}
        mean_ts = {}
        std_ts = {}

        for nH in nH_sorted:
            x = ts_data[nH]["x"]
            ys = np.vstack(ts_data[nH]["ys"])
            mean_y = ys.mean(axis=0)
            std_y = ys.std(axis=0)
            mean_ts[nH] = mean_y
            std_ts[nH] = std_y

            nH_surf = effective_cov.get(nH, 0)
            theta = nH_surf / Nmax_surface
            theta_eff[nH] = theta

            area_GPa_A = np.trapz(mean_y, x)
            gamma_data[nH] = area_GPa_A * 0.1 / 2.0
            sigma_max[nH] = mean_y.max()

        results = {
            "ts_data": ts_data,
            "effective_cov": effective_cov,
            "theta_eff": theta_eff,
            "gamma_data": gamma_data,
            "sigma_max": sigma_max,
            "mean_ts": mean_ts,
            "std_ts": std_ts,
            "nH_sorted": nH_sorted,
            "manifest_path": manifest_path,
        }
        self._write_outputs(results)
        return results

    def run_coverage_scan(self, *args, **kwargs):
        """Backward-compatible alias for ``run_h_separation``.

        Returns:
            dict: Results dictionary returned by ``run_h_separation``.
        """
        return self.run_h_separation(*args, **kwargs)

    def _write_outputs(self, results):
        """Write text summaries and plots for a completed H-coverage scan.

        Args:
            results: Results dictionary returned by ``run_coverage_scan``.
        """
        ts_txt = os.path.join(self.working_dir, "TS_curves_with_effective_coverage.txt")
        with open(ts_txt, "w", encoding="utf-8") as file:
            file.write("# theta_eff  nH_init  nH_surface  sep_A  mean_stress_GPa  std_stress_GPa\n")
            for nH in results["nH_sorted"]:
                theta = results["theta_eff"][nH]
                nH_surf = results["effective_cov"].get(nH, 0)
                x = results["ts_data"][nH]["x"]
                mean_y = results["mean_ts"][nH]
                std_y = results["std_ts"][nH]
                for sep, mean_val, std_val in zip(x, mean_y, std_y):
                    file.write(
                        f"{theta:8.5f}  {nH:3d}  {nH_surf:3d}  "
                        f"{sep:10.5f}  {mean_val:12.6f}  {std_val:12.6f}\n"
                    )
                file.write("\n")

        gamma_txt = os.path.join(self.working_dir, "surface_energy_from_TS_all_seeds.txt")
        with open(gamma_txt, "w", encoding="utf-8") as file:
            file.write("# theta_eff  nH_init  nH_surface  gamma_J_per_m2\n")
            for nH in results["nH_sorted"]:
                theta = results["theta_eff"][nH]
                nH_surf = results["effective_cov"].get(nH, 0)
                gamma = results["gamma_data"][nH]
                file.write(f"{theta:8.5f}  {nH:3d}  {nH_surf:3d}  {gamma:.8f}\n")

        theta_sigma_txt = os.path.join(self.working_dir, "theta_sigma_max.txt")
        with open(theta_sigma_txt, "w", encoding="utf-8") as file:
            file.write("# theta_eff  nH_init  nH_surface  sigma_max_GPa\n")
            for nH in results["nH_sorted"]:
                theta = results["theta_eff"][nH]
                nH_surf = results["effective_cov"].get(nH, 0)
                sigma = results["sigma_max"][nH]
                file.write(f"{theta:8.5f}  {nH:3d}  {nH_surf:3d}  {sigma:12.6f}\n")

        nH_sorted = results["nH_sorted"]

        fig, ax = plt.subplots(figsize=(7, 5.5))
        cmap = plt.get_cmap("viridis")
        norm = Normalize(vmin=0.0, vmax=1.0)

        for nH in nH_sorted:
            x = results["ts_data"][nH]["x"]
            mean_y = results["mean_ts"][nH]
            std_y = results["std_ts"][nH]
            color = cmap(norm(results["theta_eff"][nH]))
            ax.plot(x, mean_y, lw=2.0, marker="v", color=color)
            ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=color, alpha=0.2)

        xmin = min(results["ts_data"][nH]["x"].min() for nH in nH_sorted)
        xmax = max(results["ts_data"][nH]["x"].max() for nH in nH_sorted)
        xx = np.linspace(xmin, xmax, 200)
        ax.plot(xx, np.zeros_like(xx), "k-", lw=1.0, alpha=0.5)
        ax.set_xlabel("Separation distance (Å)")
        ax.set_ylabel("Normal stress (GPa)")
        ax.set_title("TS curves with effective H coverage")
        ax.set_xlim(left=0.0)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r"Effective surface H coverage $\theta$")
        fig.tight_layout()
        fig.savefig(os.path.join(self.working_dir, "ts_curves_H_colorbar_effcov.png"), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5.0))
        thetas = np.array([results["theta_eff"][nH] for nH in nH_sorted])
        gammas = np.array([results["gamma_data"][nH] for nH in nH_sorted])
        colors = plt.get_cmap("viridis")(thetas)
        ax.bar(thetas, gammas, width=0.03, color=colors, edgecolor="black")
        for theta, gamma in zip(thetas, gammas):
            ax.text(theta, gamma, f"{gamma:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(thetas, [f"{theta:.2f}" for theta in thetas])
        ax.set_xlabel(r"Effective H coverage $\theta$")
        ax.set_ylabel("Surface energy (J/m²)")
        ax.set_title("Surface energy from TS integration vs effective coverage")
        fig.tight_layout()
        fig.savefig(os.path.join(self.working_dir, "surface_energy_bar_theta.png"), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sigma_vals = np.array([results["sigma_max"][nH] for nH in nH_sorted])
        ax.plot(thetas, sigma_vals, marker="o", linestyle="-")
        ax.set_xlabel(r"Effective H coverage $\theta$")
        ax.set_ylabel(r"Maximum normal stress $\sigma_\mathrm{max}$ (GPa)")
        ax.set_title(r"$\theta$–$\sigma_\mathrm{max}$ relation from TS curves")
        fig.tight_layout()
        fig.savefig(os.path.join(self.working_dir, "theta_vs_sigma_max.png"), dpi=300)
        plt.close(fig)

    def plot_pure_separation(self, results, save_name="pure_traction_separation.png"):
        """Plot stress versus separation for a pure-metal traction-separation run.

        Args:
            results: Results dictionary returned by ``run_pure_separation``.
            save_name: Output figure name saved inside ``working_dir``.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Created figure and axes.
        """
        x = results["midpoint_separation"]
        y = results["traction"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y, marker="o")
        ax.set_xlabel("Separation distance (Å)")
        ax.set_ylabel("Normal stress (GPa)")
        ax.set_title("Traction-Separation Curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.working_dir, save_name), dpi=300)
        return fig, ax

    def plot_h_separation(self, results, save_name="h_traction_separation.png"):
        """Plot stress versus separation for H-decorated traction-separation runs.

        Args:
            results: Results dictionary returned by ``run_h_separation``.
            save_name: Output figure name saved inside ``working_dir``.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Created figure and axes.
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        for nH in results["nH_sorted"]:
            x = results["ts_data"][nH]["x"]
            y = results["mean_ts"][nH]
            ax.plot(x, y, marker="o", label=f"nH={nH}")

        ax.set_xlabel("Separation distance (Å)")
        ax.set_ylabel("Normal stress (GPa)")
        ax.set_title("Traction-Separation Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.working_dir, save_name), dpi=300)
        return fig, ax
