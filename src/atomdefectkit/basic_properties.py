"""Point-defect and basic-property workflows for BCC metals."""

from __future__ import annotations

import ast
import importlib
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

import ase.units as units
from ase.build import surface
from ase.filters import StrainFilter
from ase.optimize import BFGS, FIRE

from matscipy.elasticity import fit_elastic_constants
from scipy.optimize import leastsq

from atomdefectkit.model_discovery import discover_models
from atomdefectkit.model_errors import (
    format_model_error,
    format_unknown_model_error,
)
from atomdefectkit.registry import MODEL_REGISTRY

class BasicProperties:
    """Run basic bulk, surface, defect, and postprocessing workflows with an ASE calculator."""

    def __init__(
        self,
        model_name: str | None = None,
        model_parameters: dict | None = None,
        device: str = "cuda",
        calculator=None,
        working_dir='.',
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
            raise ValueError(
                "Please provide either 'calculator' or 'model_name', not both."
            )

        if calculator is None and model_name is None:
            raise ValueError(
                "Please provide either 'calculator' or 'model_name'."
            )

        self.device = device
        self.calculator = None
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)

        if calculator is not None:
            self.calculator = calculator
            return

        model_name = model_name.lower()
        self._set_calculator(model_name, model_parameters)

    def _lazy_import_model(self, model_name: str):
        """Import a model module so it can register itself with ``MODEL_REGISTRY``.

        Args:
            model_name (str): Input model name.

        Raises:
            ImportError: If the model name is not integrated in the package.
            ImportError: If the model could not be imported.
        """
        try:
            importlib.import_module(f"atomdefectkit.models.{model_name}")
        except ModuleNotFoundError as e:
            raise ImportError(f"Model module 'atomdefectkit.models.{model_name}' not found.\n") from e

        except ImportError as e:
            raise ImportError(
                f"Model '{model_name}' could not be imported.\nThis may be due to missing dependencies.\nOriginal error: {repr(e)}"
            ) from e

    def _set_calculator(self, model_name: str, model_parameters: dict | None):
        """Instantiate the requested MLIP calculator and attach it to the workflow.

        Args:
            model_name (str): Name of the model.
            model_parameters (dict | None): Parameters to be passed to the model.

        Raises:
            ValueError: If the model name provided is not available.
            ValueError: If the calculator could not be setup (e.g. unavailable GPU).
            RuntimeError: If the calculator doesn't contain the model (e.g. wrong parameters).
        """
        model_name = model_name.lower()
        model_parameters = dict(model_parameters or {})

        discoverable_models = discover_models()
        try:
            self._lazy_import_model(model_name)
        except ImportError as e:
            raise ValueError(format_unknown_model_error(model_name, discoverable_models)) from e

        # Confirm the import registered a builder before we try to instantiate it.
        if model_name not in MODEL_REGISTRY:
            raise RuntimeError(f"Model '{model_name}' was imported but did not register itself.")

        # Build the calculator from the registered factory.
        try:
            calculator = MODEL_REGISTRY[model_name](model_parameters, device=self.device)
        except ImportError as e:
            # Surface missing optional dependencies with a clearer workflow-level error.
            raise RuntimeError(
                f"Missing dependency for model '{model_name}'.\n\n{e}\n\n=> Please install the required package (version)."
            ) from e

        except ValueError as e:
            # Reformat parameter errors with model metadata when available.
            raise ValueError(format_model_error(model_name, model_parameters, e)) from e
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Fall back to CPU when the requested CUDA setup is unavailable.
            if "cuda" not in str(e).lower() and self.device.lower() == "cuda":
                raise ValueError(format_model_error(model_name, model_parameters, e)) from e
            warnings.warn("CUDA not available, falling back to CPU.", stacklevel=2)
            calculator = MODEL_REGISTRY[model_name](model_parameters, device="cpu")

        if calculator is None:
            raise RuntimeError(f"Builder for '{model_name}' returned None")

        self.calculator = calculator

    def birch_murnaghan_eos(self, params, vol):
        E0, B0, Bp, V0 = params
        eta = (vol / V0) ** (1.0 / 3.0)
        return E0 + (9.0 * B0 * V0 / 16.0) * (eta**2 - 1.0) ** 2 * (
            6.0 + Bp * (eta**2 - 1.0) - 4.0 * eta**2
        )

    def fit_birch_murnaghan(self, volumes, energies):
        a, b, c = np.polyfit(volumes, energies, 2)
        V0 = -b / (2 * a)
        E0 = a * V0**2 + b * V0 + c
        B0 = 2 * a * V0
        Bp = 4.0
        x0 = [E0, B0, Bp, V0]

        def residual(params, vol, energy):
            return energy - self.birch_murnaghan_eos(params, vol)

        params, _ = leastsq(residual, x0, args=(volumes, energies))
        return params

    def calculate_equilibrium_a0_birch_murnaghan(
        self, atoms, vol_range=np.linspace(0.99, 1.01, 30)
    ):
        initial_lattice_constant = atoms.cell.cellpar()[0]
        volumes = []
        energies = []
        for scale in vol_range:
            atoms_scaled = atoms.copy()
            atoms_scaled.set_cell(atoms.cell * scale**3, scale_atoms=True)
            atoms_scaled.calc = self.calculator
            volumes.append(atoms_scaled.get_volume())
            energies.append(atoms_scaled.get_potential_energy())

        _, _, _, V0 = self.fit_birch_murnaghan(np.array(volumes), np.array(energies))
        equilibrium_lattice_constant = V0 ** (1.0 / 3.0)
        return np.array(volumes), np.array(energies), equilibrium_lattice_constant

    def calculate_equilibrium_a0_relax(self, atoms, fmax=0.01):
        atoms.calc = self.calculator
        sf = StrainFilter(atoms, mask=[True, True, True, False, False, False])
        dyn = BFGS(sf, logfile=f"{self.working_dir}/relax_box.log")
        dyn.run(fmax=fmax)
        return atoms.cell.cellpar()[0]

    def calculate_elastic_constants(self, structure, verbose=True):
        structure.calc = self.calculator
        Cij, _ = fit_elastic_constants(structure, symmetry="cubic", verbose=verbose)
        return Cij / units.GPa

    def calculate_surface_energy(self, structure, miller_indices, vacuum=10.0, n_layers=5):
        surf = surface(structure, miller_indices, periodic=True, layers=n_layers, vacuum=vacuum)
        surf.calc = self.calculator
        structure.calc = self.calculator

        surf_relaxed = self.relax_fire_bfgs(surf)
        bulk_relaxed = self.relax_fire_bfgs(structure)
        bulk_energy = bulk_relaxed.get_potential_energy() / len(bulk_relaxed)

        surface_energy = (
            surf_relaxed.get_potential_energy() - bulk_energy * len(surf)
        ) / (2 * surf_relaxed.get_cell()[0, 0] * surf_relaxed.get_cell()[1, 1])
        return surface_energy * 16.0217662

    def calculate_phonon_dispersion(
        self,
        structure,
        special_points: dict[str, tuple[float, float, float]] | None = None,
        labels_path=None,
        formula: str | None = None,
        info: dict | None = None,
    ) -> str:
        """
        Calculate and plot the phonon dispersion relation using PhonoCalc.

        Parameters
        ----------
        structure : ase.Atoms
            Structure for the phonon calculation.
        special_points : dict[str, tuple[float, float, float]] | None
            High-symmetry points in fractional reciprocal coordinates.
            Example for bcc:
                {'G': (0, 0, 0),
                 'H': (0.5, -0.5, 0.5),
                 'N': (0, 0, 0.5),
                 'P': (0.25, 0.25, 0.25)}
        labels_path : list[list[str]] | None
            Band path as a list of label sequences.
            Example for bcc:
                [[['N', 'G', 'H', 'P', 'G']]]
        formula : str | None
            Label used when saving the phonon-dispersion figure. Defaults to the
            chemical formula from ``structure``.
        info : dict | None
            Extra metadata forwarded to ``plot_band_structure``.

        Returns
        -------
        str
            Path to the saved phonon-dispersion figure.
        """
        try:
            from phonocalc import PhonoCalc, plot_band_structure
        except ImportError as exc:
            raise ImportError(
                "Phonon dispersion requires the optional 'phonocalc' dependency."
            ) from exc

        atoms = structure.copy()
        atoms.calc = self.calculator

        if formula is None:
            formula = atoms.get_chemical_formula()
        if info is None:
            info = {}

        PhonoCalc(atoms, self.calculator).get_band_structure(
            special_points=special_points,
            labels_path=labels_path,
        )
        fig_path = plot_band_structure(atoms, formula, info)
        return fig_path

    def calculate_vacancy_formation_energy(self, structure):
        structure = structure.repeat((4, 4, 4))
        structure.calc = self.calculator
        perfect_energy = structure.get_potential_energy()

        vacancy_structure = structure.copy()
        del vacancy_structure[0]
        vacancy_structure.calc = self.calculator
        vacancy_structure_relaxed = self.relax_fire_bfgs(vacancy_structure)
        vacancy_energy = vacancy_structure_relaxed.get_potential_energy()
        return vacancy_energy - (perfect_energy * (len(structure) - 1) / len(structure))

    def calculate_interstitial_formation_energy(
        self, structure, inter_atom, interstitial_position, dumbbell=False, relax_flag=True
    ):
        structure = structure.repeat((4, 4, 4))
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
        dyn_fire = FIRE(structure, logfile=f"{self.working_dir}/{logfile}_fire_relax.log")
        dyn_fire.run(fmax=fmax_fire)
        dyn_bfgs = BFGS(structure, logfile=f"{self.working_dir}/{logfile}_bfgs_relax.log")
        dyn_bfgs.run(fmax=fmax_bfgs)
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
            miller_indices_list = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2)]

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

        output_path = os.path.join(self.working_dir, save_name)
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
        def _load_data(data_or_path):
            """Normalize either a mapping or a JSON file into the plotting schema."""
            if isinstance(data_or_path, (str, os.PathLike)):
                with open(data_or_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
            else:
                data = dict(data_or_path)

            if "surface_energies" in data:
                data["surface_energies"] = {
                    ast.literal_eval(key) if isinstance(key, str) else key: value
                    for key, value in data["surface_energies"].items()
                }
            return data

        calculated_data = _load_data(calculated_data)
        if dft_data is not None:
            dft_data = _load_data(dft_data)

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        title = "Calculated Data Summary" if dft_data is None else "Comparison of Calculated Data with DFT Data"
        fig.suptitle(title, fontsize=16, fontweight="bold")
        plt.rcParams.update({"font.size": 12})

        calculated_energies = np.array(calculated_data["energies"])
        axes[0, 0].plot(
            calculated_data["volumes"],
            calculated_energies - min(calculated_energies),
            "bo-",
            label="Calculated",
            linewidth=2,
            markersize=8,
        )
        if dft_data is not None:
            dft_energies = np.array(dft_data["energies"])
            axes[0, 0].plot(
                dft_data["volumes"],
                dft_energies - min(dft_energies),
                "ro-",
                label="DFT",
                linewidth=2,
                markersize=8,
            )
        axes[0, 0].set_xlabel("Volume (Å³)", fontsize=16)
        axes[0, 0].set_ylabel("Energy (eV)", fontsize=16)
        axes[0, 0].set_title("Energy-Volume Curve", fontsize=18, fontweight="bold")
        axes[0, 0].legend(fontsize=14)
        axes[0, 0].grid(True, linestyle="--", alpha=0.7)

        elastic_labels = ["C11", "C12", "C44"]
        calculated_cij = [calculated_data["C11"], calculated_data["C12"], calculated_data["C44"]]
        x = np.arange(len(elastic_labels))
        width = 0.35 if dft_data is not None else 0.6
        if dft_data is None:
            axes[0, 1].bar(x, calculated_cij, width, label="Calculated", color="blue", alpha=0.8)
        else:
            dft_cij = [dft_data["C11"], dft_data["C12"], dft_data["C44"]]
            axes[0, 1].bar(x - width / 2, calculated_cij, width, label="Calculated", color="blue", alpha=0.8)
            axes[0, 1].bar(x + width / 2, dft_cij, width, label="DFT", color="orange", alpha=0.8)
        axes[0, 1].set_xlabel("Elastic Constants", fontsize=16)
        axes[0, 1].set_ylabel("Value (GPa)", fontsize=16)
        axes[0, 1].set_title("Elastic Constants Comparison", fontsize=18, fontweight="bold")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(elastic_labels, fontsize=14)
        axes[0, 1].legend(fontsize=14)
        axes[0, 1].grid(True, linestyle="--", alpha=0.7)

        miller_indices = list(calculated_data["surface_energies"].keys())
        calculated_surface_energies = [
            calculated_data["surface_energies"][miller] for miller in miller_indices
        ]
        x = np.arange(len(miller_indices))
        width = 0.35 if dft_data is not None else 0.6
        if dft_data is None:
            axes[1, 0].bar(
                x,
                calculated_surface_energies,
                width,
                label="Calculated",
                color="blue",
                alpha=0.8,
            )
        else:
            dft_surface_energies = [dft_data["surface_energies"][miller] for miller in miller_indices]
            axes[1, 0].bar(
                x - width / 2,
                calculated_surface_energies,
                width,
                label="Calculated",
                color="blue",
                alpha=0.8,
            )
            axes[1, 0].bar(
                x + width / 2,
                dft_surface_energies,
                width,
                label="DFT",
                color="orange",
                alpha=0.8,
            )
        axes[1, 0].set_xlabel("Miller Indices", fontsize=16)
        axes[1, 0].set_ylabel("Surface Energy (J/m²)", fontsize=16)
        axes[1, 0].set_title("Surface Energy Comparison", fontsize=18, fontweight="bold")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([str(miller) for miller in miller_indices], fontsize=14)
        axes[1, 0].legend(fontsize=14)
        axes[1, 0].grid(True, linestyle="--", alpha=0.7)

        defect_labels = ["Vacancy", "Octahedral", "Tetrahedral", "Inter 100", "Inter 110", "Inter 111"]
        calculated_defects = [
            calculated_data["vacancy_formation_energy"],
            calculated_data["octahedral_formation_energy"],
            calculated_data["tetrahedral_formation_energy"],
            calculated_data["inter_100_formation_energy"],
            calculated_data["inter_110_formation_energy"],
            calculated_data["inter_111_formation_energy"],
        ]
        x = np.arange(len(defect_labels))
        width = 0.35 if dft_data is not None else 0.6
        if dft_data is None:
            axes[1, 1].bar(x, calculated_defects, width, label="Calculated", color="blue", alpha=0.8)
        else:
            dft_defects = [
                dft_data["vacancy_formation_energy"],
                dft_data["octahedral_formation_energy"],
                dft_data["tetrahedral_formation_energy"],
                dft_data["inter_100_formation_energy"],
                dft_data["inter_110_formation_energy"],
                dft_data["inter_111_formation_energy"],
            ]
            axes[1, 1].bar(
                x - width / 2,
                calculated_defects,
                width,
                label="Calculated",
                color="blue",
                alpha=0.8,
            )
            axes[1, 1].bar(
                x + width / 2,
                dft_defects,
                width,
                label="DFT",
                color="orange",
                alpha=0.8,
            )
        axes[1, 1].set_xlabel("Defect Type", fontsize=16)
        axes[1, 1].set_ylabel("Formation Energy (eV)", fontsize=16)
        axes[1, 1].set_title("Point Defects Formation Energy Comparison", fontsize=18, fontweight="bold")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(defect_labels, rotation=45, ha="right", fontsize=14)
        axes[1, 1].legend(fontsize=14)
        axes[1, 1].grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = os.path.join(self.working_dir, save_name)
        fig.savefig(output_path, dpi=300, format="pdf")
