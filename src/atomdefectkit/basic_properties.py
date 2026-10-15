"""Point-defect and basic-property workflows for BCC metals."""

from __future__ import annotations

import ast
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from ase import units
from matscipy.elasticity import fit_elastic_constants
from scipy.optimize import least_squares

from atomdefectkit.data import reference_path
from atomdefectkit.model_loader import load_model
from atomdefectkit.units import EV_PER_A2_TO_J_PER_M2
from atomdefectkit.utils.optimizers import relax
from atomdefectkit.utils.paths import CalculatorWorkflow
from atomdefectkit.utils.slabs import build_surface_slab

DEFAULT_SURFACES = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2)]
POINT_DEFECT_KEYS = [
    ("vacancy_formation_energy", "Vacancy"),
    ("octahedral_formation_energy", "Octahedral"),
    ("tetrahedral_formation_energy", "Tetrahedral"),
    ("inter_100_formation_energy", "Inter 100"),
    ("inter_110_formation_energy", "Inter 110"),
    ("inter_111_formation_energy", "Inter 111"),
]


class BasicProperties(CalculatorWorkflow):
    """Run basic bulk, surface, defect, and postprocessing workflows with an ASE calculator."""

    def __init__(
        self,
        model_name: str | None = None,
        model_parameters: dict | None = None,
        device: str = "cuda",
        calculator=None,
        working_dir=".",
    ):
        """Initialize the workflow with either a registered model or a direct calculator.

        Args:
            model_name (str | None): Registered MLIP name. Required if ``calculator`` is not provided.
            model_parameters (dict | None): Parameters for the MLIP. Defaults to None.
            device (str): Device for the calculations, either "cpu" or "cuda". Defaults to "cuda".
            calculator: Existing ASE-compatible calculator. If provided, model loading is skipped.
            working_dir (str): Directory used for logs and saved output files.

        Raises:
            ValueError: If neither or both of model_name and calculator are provided.
        """
        if calculator is not None and model_name is not None:
            raise ValueError("Please provide either 'calculator' or 'model_name', not both.")
        if calculator is None and model_name is None:
            raise ValueError("Please provide either 'calculator' or 'model_name'.")

        self.device = device
        self.init_working_dir(working_dir)
        self.calculator = (
            calculator if calculator is not None else load_model(model_name, model_parameters, device=device)
        )

    def get_calculator(self):
        """Return the ASE calculator attached to this workflow.

        Returns:
            ASE-compatible calculator: Calculator loaded directly or through a model
            registry entry.
        """
        return self.calculator

    def create_stacking_fault_workflow(
        self,
        atoms,
        formula: str | None = None,
        info: str = "Calculated",
        optimizer: str = "FIRE",
        working_dir: str | None = None,
    ):
        """Create a stacking-fault workflow that reuses this workflow calculator.

        Args:
            atoms: Reference structure used to build the faulted slab.
            formula: Optional chemical formula label used in output filenames.
            info: Short label written into summary files and plot legends.
            optimizer: Geometry optimizer label passed to the stacking-fault workflow.
            working_dir: Optional directory for stacking-fault outputs. When omitted,
                ``self.working_dir`` is reused.

        Returns:
            StackingFaultWorkflow: Workflow instance sharing this calculator.
        """
        from atomdefectkit.stacking_fault import StackingFaultWorkflow

        return StackingFaultWorkflow(
            atoms=atoms,
            calculator=self.calculator,
            formula=formula,
            info=info,
            optimizer=optimizer,
            working_dir=working_dir or self.working_dir,
        )

    def create_traction_separation_workflow(
        self,
        atoms,
        surface_index=None,
        repeat=(3, 3, 16),
        working_dir: str | None = None,
        optimizer: str = "FIRE",
    ):
        """Create a traction-separation workflow that reuses this workflow calculator.

        Args:
            atoms: Reference bulk or slab structure.
            surface_index: Optional surface Miller index such as ``(1, 0, 0)``.
            repeat: Supercell repetition counts along the slab directions.
            working_dir: Optional directory for traction-separation outputs. When
                omitted, ``self.working_dir`` is reused.
            optimizer: Geometry optimizer label passed to the TS workflow.

        Returns:
            TractionSeparationWorkflow: Workflow instance sharing this calculator.
        """
        from atomdefectkit.traction_separation import TractionSeparationWorkflow

        return TractionSeparationWorkflow(
            atoms=atoms,
            calculator=self.calculator,
            surface_index=surface_index,
            repeat=repeat,
            working_dir=working_dir or self.working_dir,
            optimizer=optimizer,
        )

    def create_screw_dislocation_workflow(
        self,
        element,
        lattice_constant,
        elastic_constant,
        working_dir: str | None = None,
    ):
        """Create a screw-dislocation workflow that reuses this workflow calculator.

        Args:
            element: Chemical symbol of the BCC metal.
            lattice_constant: Relaxed lattice constant in Angstrom.
            elastic_constant: Elastic constants array passed to the screw workflow.
            working_dir: Optional output directory. When omitted, ``self.working_dir``
                is reused.

        Returns:
            BCCScrewDislocation: Workflow instance sharing this calculator.
        """
        from atomdefectkit.screw import BCCScrewDislocation

        return BCCScrewDislocation(
            element=element,
            lattice_constant=lattice_constant,
            elastic_constant=elastic_constant,
            calculator=self.calculator,
            working_dir=working_dir or self.working_dir,
        )

    @staticmethod
    def birch_murnaghan_eos(params, vol):
        """Evaluate the third-order Birch-Murnaghan equation of state.

        Args:
            params: ``(E0, B0, B0', V0)`` in eV, eV/Angstrom^3, dimensionless and Angstrom^3.
            vol: Volume(s) in Angstrom^3.

        Returns:
            Energy (or energies) in eV.
        """
        E0, B0, Bp, V0 = params
        eta = (vol / V0) ** (1.0 / 3.0)
        return E0 + (9.0 * B0 * V0 / 16.0) * (eta**2 - 1.0) ** 2 * (
            6.0 + Bp * (eta**2 - 1.0) - 4.0 * eta**2
        )

    def fit_birch_murnaghan(self, volumes, energies):
        """Fit the Birch-Murnaghan EOS to energy-volume data.

        A quadratic fit provides the starting guess for a bounded least-squares fit.

        Args:
            volumes: Volumes in Angstrom^3.
            energies: Energies in eV.

        Returns:
            np.ndarray: Fitted ``(E0, B0, B0', V0)``.
        """
        a, b, c = np.polyfit(volumes, energies, 2)
        V0 = -b / (2 * a)
        if not np.isfinite(V0) or V0 <= 0:
            V0 = volumes[np.argmin(energies)]
        E0 = a * V0**2 + b * V0 + c
        B0 = max(abs(2 * a * V0), 1e-8)
        x0 = [E0, B0, 4.0, V0]
        lower = [-np.inf, 1e-12, 0.0, min(volumes) * 0.5]
        upper = [np.inf, np.inf, 20.0, max(volumes) * 1.5]

        def residual(params, vol, energy):
            return energy - self.birch_murnaghan_eos(params, vol)

        result = least_squares(residual, x0, bounds=(lower, upper), args=(volumes, energies))
        return result.x

    def calculate_equilibrium_a0_birch_murnaghan(self, atoms, vol_range=None):
        """Scan the cell size and fit the Birch-Murnaghan EOS.

        Args:
            atoms: Reference cell. Its first cell-vector length is taken as the
                lattice constant, e.g. a conventional cubic cell.
            vol_range: Linear scale factors applied to the cell vectors (not to
                the volume). Defaults to ``np.linspace(0.99, 1.01, 30)``.

        Returns:
            tuple[np.ndarray, np.ndarray, float]: Volumes, energies, and the
            equilibrium lattice constant in Angstrom.
        """
        if vol_range is None:
            vol_range = np.linspace(0.99, 1.01, 30)
        volumes = []
        energies = []
        for scale in vol_range:
            atoms_scaled = atoms.copy()
            atoms_scaled.set_cell(atoms.cell * scale, scale_atoms=True)
            atoms_scaled.calc = self.calculator
            volumes.append(atoms_scaled.get_volume())
            energies.append(atoms_scaled.get_potential_energy())

        volumes = np.array(volumes)
        energies = np.array(energies)
        _, _, _, V0 = self.fit_birch_murnaghan(volumes, energies)
        # Scale the input lattice constant isotropically to the fitted volume.
        equilibrium_lattice_constant = atoms.cell.cellpar()[0] * (V0 / atoms.get_volume()) ** (1.0 / 3.0)
        return volumes, energies, equilibrium_lattice_constant

    def calculate_equilibrium_a0_relax(self, atoms, fmax=0.01):
        """Relax the cell lengths with BFGS and return the first lattice parameter.

        Args:
            atoms: Structure to relax in place.
            fmax: Convergence threshold passed to the optimizer.

        Returns:
            float: Relaxed lattice parameter ``a`` in Angstrom.
        """
        relax(
            atoms,
            self.calculator,
            optimizer="BFGS",
            fmax=fmax,
            logfile=self.path("relax_box.log"),
            cell_filter="strain",
            mask=[True, True, True, False, False, False],
        )
        return atoms.cell.cellpar()[0]

    def calculate_elastic_constants(self, structure, verbose=True):
        """Fit cubic elastic constants with matscipy.

        Args:
            structure: Equilibrium structure.
            verbose: Passed to ``matscipy.elasticity.fit_elastic_constants``.

        Returns:
            np.ndarray: 6x6 elastic-constant matrix in GPa (Voigt notation).
        """
        structure.calc = self.calculator
        Cij, _ = fit_elastic_constants(structure, symmetry="cubic", verbose=verbose)
        return Cij / units.GPa

    def calculate_surface_energy(self, structure, miller_indices, vacuum=10.0, n_layers=5):
        """Calculate a relaxed surface energy.

        Args:
            structure: Bulk structure used to build the slab. It is relaxed in place.
            miller_indices: Surface Miller indices.
            vacuum: Vacuum thickness passed to ``ase.build.surface`` in Angstrom.
            n_layers: Number of atomic layers in the slab.

        Returns:
            float: Surface energy in J/m^2.
        """
        surf = build_surface_slab(
            structure,
            miller_indices,
            layers=n_layers,
            repeat=(1, 1, 1),
            vacuum=vacuum,
            center_axis=None,
        )
        surf_relaxed = self.relax_fire_bfgs(surf)
        bulk_relaxed = self.relax_fire_bfgs(structure)
        bulk_energy = bulk_relaxed.get_potential_energy() / len(bulk_relaxed)

        cell = surf_relaxed.get_cell()
        area = np.linalg.norm(np.cross(cell[0], cell[1]))
        surface_energy = (surf_relaxed.get_potential_energy() - bulk_energy * len(surf)) / (2 * area)
        return surface_energy * EV_PER_A2_TO_J_PER_M2

    def calculate_phonon_dispersion(
        self,
        structure,
        special_points: dict[str, tuple[float, float, float]] | None = None,
        labels_path=None,
        formula: str | None = None,
        info: dict | None = None,
        supercell_matrix=None,
        primitive_matrix="auto",
    ) -> str:
        """Calculate and plot the phonon dispersion relation.

        Args:
            structure: Structure used for the phonon calculation.
            special_points: Optional high-symmetry points in fractional reciprocal
                coordinates. When omitted, ASE derives them from the primitive cell.
            labels_path: Optional band path as a sequence of label paths, e.g.
                ``[['N', 'G', 'H', 'P', 'G']]``.
            formula: Optional label used in the output filename and plot title.
            info: Optional metadata dictionary; if it contains ``label`` or ``title``,
                that value is used in the plot title.
            supercell_matrix: Optional Phonopy supercell matrix. When omitted, a
                ``3x3x3`` diagonal supercell is used.
            primitive_matrix: Primitive-cell setting passed to Phonopy. The default
                ``"auto"`` keeps the standard primitive-cell phonon workflow for
                crystals like BCC while silencing the corresponding Phonopy default
                warning.

        Returns:
            str: Path to the saved phonon-dispersion figure.

        Raises:
            ImportError: If the required ``phonopy`` or ``spglib`` dependency is missing.
            RuntimeError: If the primitive cell cannot be determined from ``structure``.
        """
        try:
            import ase
            import spglib
            from ase import Atoms
            from ase.cell import Cell
            from phonopy import Phonopy
            from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
            from phonopy.structure.atoms import PhonopyAtoms
            from phonopy.structure.cells import PrimitiveMatrixAutoDefaultWarning
        except ImportError as exc:
            raise ImportError(
                "Phonon dispersion requires the optional dependencies 'phonopy' and 'spglib'."
            ) from exc

        def ase2phono(atoms):
            return PhonopyAtoms(
                symbols=atoms.get_chemical_symbols(),
                cell=atoms.cell.array,
                scaled_positions=atoms.get_scaled_positions(),
            )

        def phono2ase(cell):
            return Atoms(
                symbols=list(cell.symbols),
                cell=np.asarray(cell.cell),
                scaled_positions=np.asarray(cell.scaled_positions),
                pbc=True,
            )

        def format_k_label(label):
            return "Γ" if label == "G" else label

        atoms = structure.copy()
        atoms.calc = self.calculator
        if formula is None:
            formula = atoms.get_chemical_formula()
        if info is None:
            info = {}
        if supercell_matrix is None:
            supercell_matrix = np.diag([3, 3, 3])

        unitcell = ase2phono(atoms)
        with warnings.catch_warnings():
            if primitive_matrix == "auto":
                warnings.filterwarnings(
                    "ignore",
                    category=PrimitiveMatrixAutoDefaultWarning,
                )
            phonon = Phonopy(
                unitcell=unitcell,
                supercell_matrix=supercell_matrix,
                primitive_matrix=primitive_matrix,
            )
        phonon.generate_displacements(distance=0.01)

        set_of_forces = []
        for cell in phonon.supercells_with_displacements:
            displaced = phono2ase(cell)
            displaced.calc = self.calculator
            forces = displaced.get_forces()
            forces -= np.mean(forces, axis=0)
            set_of_forces.append(forces)
        phonon.produce_force_constants(forces=np.array(set_of_forces))

        lattice = atoms.get_cell().T
        positions = atoms.get_scaled_positions()
        numbers = atoms.get_atomic_numbers()
        atoms_cell = (lattice, positions, numbers)
        primitive = spglib.find_primitive(atoms_cell)
        if primitive is None:
            raise RuntimeError("Failed to determine the primitive cell for the phonon calculation.")
        cell = Cell(primitive[0])

        if special_points is None:
            special_points = cell.get_bravais_lattice().get_special_points()
        if labels_path is None:
            labels_path = ase.dft.kpoints.parse_path_string(cell.bandpath().path)

        labels = []
        path = []
        for label_path in labels_path:
            segment = []
            for label in label_path:
                labels.append(label)
                segment.append(list(special_points[label]))
            path.append(segment)

        qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
        phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
        bands_dict = phonon.get_band_structure_dict()
        bands_dict["labels_path"] = labels_path
        atoms.info["force_constants"] = phonon.force_constants
        atoms.info["band_dict"] = bands_dict

        fig, ax = plt.subplots(figsize=(6, 4))
        segment_distances = bands_dict["distances"]
        segment_frequencies = bands_dict["frequencies"]

        for distances, frequencies in zip(segment_distances, segment_frequencies):
            frequencies = np.asarray(frequencies)
            if frequencies.ndim == 1:
                frequencies = frequencies[:, None]
            for band in frequencies.T:
                ax.plot(distances, band, color="tab:blue", linewidth=1.5)
        tick_positions = [segment_distances[0][0]]
        tick_labels = [format_k_label(labels_path[0][0])]
        segment_index = 0

        for label_path in labels_path:
            for label in label_path[1:]:
                tick_positions.append(segment_distances[segment_index][-1])
                tick_labels.append(format_k_label(label))
                segment_index += 1

        for xpos in tick_positions:
            ax.axvline(x=xpos, color="0.8", linewidth=0.8)
        ax.axhline(y=0.0, color="0.4", linewidth=0.8, linestyle="--")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("Frequency (THz)")
        primitive_formula = "".join(phonon.primitive.symbols)
        title = info.get("title") or info.get("label") or f"{primitive_formula} phonon dispersion"
        ax.set_title(title, fontsize=12)
        ax.set_xlim(segment_distances[0][0], segment_distances[-1][-1])
        fig.tight_layout()

        fig_path = self.path(f"{formula}_phonon_dispersion.png")
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        return fig_path

    def calculate_vacancy_formation_energy(self, structure, supercell=(4, 4, 4)):
        """Calculate the relaxed mono-vacancy formation energy.

        Args:
            structure: Equilibrium unit cell.
            supercell: Repetitions used to build the supercell.

        Returns:
            float: Vacancy formation energy in eV.
        """
        structure = structure.repeat(supercell)
        structure.calc = self.calculator
        perfect_energy = structure.get_potential_energy()

        vacancy_structure = structure.copy()
        del vacancy_structure[0]
        vacancy_structure.calc = self.calculator
        vacancy_energy = self.relax_fire_bfgs(vacancy_structure).get_potential_energy()
        return vacancy_energy - (perfect_energy * (len(structure) - 1) / len(structure))

    def calculate_interstitial_formation_energy(
        self,
        structure,
        inter_atom,
        interstitial_position,
        dumbbell=False,
        relax_flag=True,
        supercell=(4, 4, 4),
    ):
        """Calculate a self- or foreign-interstitial formation energy.

        Args:
            structure: Equilibrium unit cell.
            inter_atom: Chemical symbol of the inserted atom.
            interstitial_position: Cartesian position in Angstrom. With
                ``dumbbell=True``, a pair of positions: atom 1 of the supercell is
                moved to the first one and the new atom is placed at the second.
            dumbbell: Whether to build a split (dumbbell) interstitial.
            relax_flag: Whether to relax the defect structure.
            supercell: Repetitions used to build the supercell.

        Returns:
            float: Interstitial formation energy in eV.
        """
        structure = structure.repeat(supercell)
        structure.calc = self.calculator
        perfect_energy = structure.get_potential_energy()

        interstitial_structure = structure.copy()
        if dumbbell:
            interstitial_structure.positions[1] = interstitial_position[0]
            interstitial_structure.append(inter_atom)
            interstitial_structure.positions[-1] = interstitial_position[1]
        else:
            interstitial_structure.append(inter_atom)
            interstitial_structure.positions[-1] = interstitial_position

        interstitial_structure.calc = self.calculator
        if relax_flag:
            interstitial_structure = self.relax_fire_bfgs(interstitial_structure)
        interstitial_energy = interstitial_structure.get_potential_energy()
        return interstitial_energy - (perfect_energy * (len(structure) + 1) / len(structure))

    def relax_fire_bfgs(self, structure, fmax_fire=0.01, fmax_bfgs=0.001, logfile="default"):
        """Relax atomic positions with FIRE, then refine with BFGS.

        Args:
            structure: Structure to relax in place.
            fmax_fire: Force threshold for the FIRE stage.
            fmax_bfgs: Force threshold for the BFGS stage.
            logfile: Prefix for the two optimizer logfiles in ``working_dir``.

        Returns:
            ase.Atoms: The relaxed ``structure``.
        """
        relax(structure, self.calculator, "FIRE", fmax=fmax_fire, logfile=self.path(f"{logfile}_fire_relax.log"))
        relax(structure, self.calculator, "BFGS", fmax=fmax_bfgs, logfile=self.path(f"{logfile}_bfgs_relax.log"))
        return structure

    def save_properties(
        self,
        volumes,
        energies,
        Cij,
        surface_energy,
        vacancy_formation_energy,
        octahedral_formation_energy,
        tetrahedral_formation_energy,
        inter_100_formation_energy,
        inter_110_formation_energy,
        inter_111_formation_energy,
        miller_indices_list=None,
        save_name="calculated_data.json",
        volume_scale=2.0,
        energy_scale=2.0,
    ) -> str:
        """Serialize calculated basic-property results to a JSON summary file.

        Parameters
        ----------
        volumes, energies : array-like
            Equation-of-state volumes and energies.
        Cij : array-like
            Elastic constants array.
        surface_energy : array-like
            Surface energies ordered by ``miller_indices_list``.
        vacancy_formation_energy, octahedral_formation_energy, tetrahedral_formation_energy : float
            Point-defect formation energies in eV.
        inter_100_formation_energy, inter_110_formation_energy, inter_111_formation_energy : float
            Dumbbell/interstitial formation energies in eV.
        miller_indices_list : list[tuple[int, int, int]] | None
            Miller indices associated with ``surface_energy``.
        save_name : str
            Output JSON filename saved inside ``self.working_dir``.
        volume_scale, energy_scale : float
            Divisors applied before serialization, matching the historical script output.

        Returns
        -------
        str
            Path to the saved JSON file.
        """
        if miller_indices_list is None:
            miller_indices_list = DEFAULT_SURFACES

        calculated_data = {
            "volumes": (np.asarray(volumes) / volume_scale).tolist(),
            "energies": (np.asarray(energies) / energy_scale).tolist(),
            "C11": float(Cij[0, 0]),
            "C12": float(Cij[0, 1]),
            "C44": float(Cij[3, 3]),
            "surface_energies": {
                str(miller): float(surface_energy[i]) for i, miller in enumerate(miller_indices_list)
            },
            "vacancy_formation_energy": float(vacancy_formation_energy),
            "octahedral_formation_energy": float(octahedral_formation_energy),
            "tetrahedral_formation_energy": float(tetrahedral_formation_energy),
            "inter_100_formation_energy": float(inter_100_formation_energy),
            "inter_110_formation_energy": float(inter_110_formation_energy),
            "inter_111_formation_energy": float(inter_111_formation_energy),
        }

        output_path = self.path(save_name)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(calculated_data, file, indent=4)

        return output_path

    def plot_comparison(self, calculated_data, dft_data=None, save_name="comparison.pdf"):
        """Plot calculated properties, optionally alongside DFT reference data.

        Parameters
        ----------
        calculated_data : dict | str | os.PathLike
            Calculated properties to visualize, or a path to a JSON file containing them.
        dft_data : dict | str | os.PathLike | None
            Optional DFT reference data, or a path to a JSON file containing it.
            When omitted, only calculated values are shown.
        save_name : str
            Output filename saved inside ``self.working_dir``.

        Returns
        -------
        str
            Path to the saved figure.
        """
        def _resolve_packaged_reference(path_like):
            """Resolve bundled DFT reference JSONs when a relative path is unavailable."""
            candidate = os.fspath(path_like)
            if os.path.exists(candidate):
                return candidate
            filename = os.path.basename(candidate)
            if filename.startswith("dft_") and filename.endswith(".json"):
                return reference_path("basic_properties", filename) or candidate
            return candidate

        def _load_data(data_or_path):
            """Normalize either a mapping or a JSON file into the plotting schema."""
            if isinstance(data_or_path, (str, os.PathLike)):
                with open(_resolve_packaged_reference(data_or_path), encoding="utf-8") as file:
                    data = json.load(file)
            else:
                data = dict(data_or_path)

            if "surface_energies" in data:
                data["surface_energies"] = {
                    ast.literal_eval(key) if isinstance(key, str) else key: value
                    for key, value in data["surface_energies"].items()
                }
            return data

        def _bar_compare(ax, labels, calculated, reference, xlabel, ylabel, title, rotate=False):
            """Draw calculated values, optionally side by side with DFT values."""
            x = np.arange(len(labels))
            if reference is None:
                ax.bar(x, calculated, 0.6, label="Calculated", color="blue", alpha=0.8)
            else:
                width = 0.35
                ax.bar(x - width / 2, calculated, width, label="Calculated", color="blue", alpha=0.8)
                ax.bar(x + width / 2, reference, width, label="DFT", color="orange", alpha=0.8)
            ax.set_xticks(x)
            if rotate:
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
            else:
                ax.set_xticklabels(labels, fontsize=14)
            _style(ax, xlabel, ylabel, title)

        def _style(ax, xlabel, ylabel, title):
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_ylabel(ylabel, fontsize=16)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.legend(fontsize=14)
            ax.grid(True, linestyle="--", alpha=0.7)

        calculated_data = _load_data(calculated_data)
        if dft_data is not None:
            dft_data = _load_data(dft_data)

        def _values(keys, data):
            return None if data is None else [data[key] for key in keys]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        title = "Calculated Data Summary" if dft_data is None else "Comparison of Calculated Data with DFT Data"
        fig.suptitle(title, fontsize=14, fontweight="bold")

        series = [("Calculated", calculated_data, "bo-")]
        if dft_data is not None:
            series.append(("DFT", dft_data, "ro-"))
        for label, data, style in series:
            energies = np.array(data["energies"])
            axes[0, 0].plot(
                data["volumes"], energies - energies.min(), style, label=label, linewidth=2, markersize=8
            )
        _style(axes[0, 0], "Volume (Å³)", "Energy (eV)", "Energy-Volume Curve")

        elastic_keys = ["C11", "C12", "C44"]
        _bar_compare(
            axes[0, 1],
            elastic_keys,
            _values(elastic_keys, calculated_data),
            _values(elastic_keys, dft_data),
            "Elastic Constants",
            "Value (GPa)",
            "Elastic Constants Comparison",
        )

        miller_indices = list(calculated_data["surface_energies"].keys())
        _bar_compare(
            axes[1, 0],
            [str(miller) for miller in miller_indices],
            _values(miller_indices, calculated_data["surface_energies"]),
            None if dft_data is None else _values(miller_indices, dft_data["surface_energies"]),
            "Miller Indices",
            "Surface Energy (J/m²)",
            "Surface Energy Comparison",
        )

        defect_keys = [key for key, _ in POINT_DEFECT_KEYS]
        _bar_compare(
            axes[1, 1],
            [label for _, label in POINT_DEFECT_KEYS],
            _values(defect_keys, calculated_data),
            _values(defect_keys, dft_data),
            "Defect Type",
            "Formation Energy (eV)",
            "Point Defects Formation Energy Comparison",
            rotate=True,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = self.path(save_name)
        fig.savefig(output_path, dpi=300, format="pdf")
        plt.close(fig)
        return output_path
