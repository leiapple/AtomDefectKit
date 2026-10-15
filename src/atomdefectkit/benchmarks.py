"""End-to-end BCC benchmark: run every workflow for one element and one calculator.

This module replaces the per-model ``scripts/run_tests_<model>.py`` files.
Use :func:`run_bcc_benchmark` from Python, or ``scripts/run_benchmark.py``
from the command line.
"""

from __future__ import annotations

import logging

import numpy as np
from ase.build import bulk

from atomdefectkit.basic_properties import DEFAULT_SURFACES, BasicProperties
from atomdefectkit.data import reference_path
from atomdefectkit.utils.progress import progress

logger = logging.getLogger(__name__)

#: Benchmark steps, in the order they run.
ALL_STEPS = ("basic", "phonon", "stacking_fault", "traction_separation", "screw")

#: Initial lattice-parameter guesses (Angstrom) for the packaged BCC references.
BCC_INITIAL_A0 = {"V": 2.997, "Nb": 3.307, "Ta": 3.163, "Mo": 3.319, "W": 3.185}

#: BCC point defects in units of the lattice constant, for a conventional cubic
#: cell repeated 4x4x4. Dumbbells are split around supercell atom 1, which sits
#: at (0.5, 0.5, 0.5) a0: that atom moves to the first position and the extra
#: atom goes to the second.
BCC_INTERSTITIALS = {
    "octahedral": {"position": [0.5, 0.5, 0.0], "dumbbell": False},
    "tetrahedral": {"position": [0.25, 0.5, 0.0], "dumbbell": False},
    "inter_100": {"position": [[0.5, 0.5, 0.2], [0.5, 0.5, 0.8]], "dumbbell": True},
    "inter_110": {"position": [[0.3, 0.3, 0.5], [0.7, 0.7, 0.5]], "dumbbell": True},
    "inter_111": {"position": [[0.33, 0.33, 0.33], [0.7, 0.7, 0.7]], "dumbbell": True},
}

#: BCC high-symmetry path for the phonon dispersion.
BCC_SPECIAL_POINTS = {
    "G": (0.0, 0.0, 0.0),
    "H": (0.5, -0.5, 0.5),
    "N": (0.0, 0.0, 0.5),
    "P": (0.25, 0.25, 0.25),
}
BCC_PHONON_PATH = [["N", "G", "H", "P", "G"]]

#: Generalized stacking-fault planes: in-plane vectors ``a``/``b`` for
#: ``ase.build.cut`` and the number of layers.
BCC_STACKING_FAULTS = {
    "110": {"a": (1, 1, -1), "b": (1, 1, 2), "miller": (1, -1, 0), "layers": 30},
    "112": {"a": (1, 1, -1), "b": (-1, 1, 0), "miller": (1, 1, 2), "layers": 40},
}

#: Surfaces used for traction-separation curves.
BCC_TS_SURFACES = [(1, 0, 0), (1, 1, 0)]


def run_basic_properties(workflow: BasicProperties, element: str, initial_a0: float) -> dict:
    """Run the EOS, elastic, point-defect and surface-energy steps and save them.

    Args:
        workflow: Workflow holding the calculator and output directory.
        element: BCC element symbol.
        initial_a0: Initial lattice-parameter guess in Angstrom.

    Returns:
        dict: ``atoms`` (equilibrium cell), ``a0``, ``cij`` and the saved JSON path.
    """
    atoms = bulk(element, "bcc", a=initial_a0, cubic=True)

    progress("Relaxing lattice constant")
    a0_relax = workflow.calculate_equilibrium_a0_relax(atoms.copy(), fmax=0.001)
    progress(f"Relaxed lattice constant: {a0_relax:.6f} A")
    progress("Fitting Birch-Murnaghan EOS")
    volumes, energies, a0 = workflow.calculate_equilibrium_a0_birch_murnaghan(
        atoms.copy(), vol_range=np.linspace(0.95, 1.05, 40)
    )
    progress(f"EOS lattice constant: {a0:.6f} A")
    if abs(a0_relax - a0) > 1e-4:
        logger.warning("Relaxed and EOS lattice constants differ by %.2e A", abs(a0_relax - a0))

    atoms.set_cell([a0] * 3, scale_atoms=True)
    progress("Calculating elastic constants")
    cij = workflow.calculate_elastic_constants(atoms.copy(), verbose=False)
    progress("Calculating vacancy formation energy")
    defects = {"vacancy_formation_energy": workflow.calculate_vacancy_formation_energy(atoms)}

    progress("Calculating interstitial formation energies")
    for name, site in BCC_INTERSTITIALS.items():
        key = f"{name}_formation_energy"
        defects[key] = workflow.calculate_interstitial_formation_energy(
            atoms.copy(), element, a0 * np.asarray(site["position"]), dumbbell=site["dumbbell"]
        )

    progress("Calculating surface energies")
    surface_energy = np.zeros(len(DEFAULT_SURFACES))
    for i, miller in enumerate(DEFAULT_SURFACES):
        progress(f"Calculating surface energy for {miller}")
        surface_energy[i] = workflow.calculate_surface_energy(atoms, miller, vacuum=20, n_layers=8)

    progress("Saving basic properties")
    json_path = workflow.save_properties(
        volumes=volumes,
        energies=energies,
        Cij=cij,
        surface_energy=surface_energy,
        miller_indices_list=DEFAULT_SURFACES,
        save_name=f"{element}_basicProp.json",
        **defects,
    )
    progress("Plotting basic-property comparison")
    dft_reference = reference_path("basic_properties", f"dft_{element}.json")
    workflow.plot_comparison(json_path, dft_reference)
    return {"atoms": atoms, "a0": a0, "cij": cij, "json_path": json_path}


def run_screw_dislocation(
    workflow: BasicProperties,
    element: str,
    a0: float,
    cij,
    label: str,
    optimizer: str = "FIRE",
    n_images: int = 11,
    fmax: float = 0.005,
    spring_constant: float = 0.1,
):
    """Relax the screw dipole at two Peierls valleys, then run the NEB barrier.

    Args:
        workflow: Workflow holding the calculator and output directory.
        element: BCC element symbol.
        a0: Equilibrium lattice constant in Angstrom.
        cij: 6x6 elastic constants in GPa.
        label: Model label used in plot titles.
        optimizer: ASE optimizer label.
        n_images: Number of NEB images.
        fmax: Force threshold for relaxations and NEB.
        spring_constant: NEB spring constant.

    Returns:
        BCCScrewDislocPeierlsBarrier | None: The NEB workflow, or ``None`` when the
        elastic constants are unstable and the step is skipped.
    """
    from atomdefectkit.neb import BCCScrewDislocPeierlsBarrier

    screw = workflow.create_screw_dislocation_workflow(
        element=element, lattice_constant=a0, elastic_constant=cij
    )
    try:
        endpoints = []
        # The final core sits one Peierls valley, a0*sqrt(6)/3, along x.
        for name, center in [("initial", [0, 0, 0]), ("final", [a0 * np.sqrt(6) / 3, 0, 0])]:
            dislocation = screw.create_dislocation_object()
            progress(f"Relaxing {name} screw-dislocation dipole")
            base_system, disl_system = screw.relax_dislocation_dipole(
                dislocation,
                disloc_center=center,
                fmax=fmax,
                optimizer=optimizer,
                logfile=f"{name}_{optimizer.lower()}_dislocation_relax.log",
            )
            progress(f"Plotting {name} differential-displacement map")
            screw.plot_differential_displacement_map(
                dislocation, base_system, disl_system, filename=f"dd_plot_{name}.png"
            )
            endpoints.append(disl_system.dump("ase_Atoms"))
    except ValueError as exc:
        progress(f"Skipping screw-dislocation workflow: {exc}")
        return None

    progress("Running screw-dislocation NEB")
    neb = BCCScrewDislocPeierlsBarrier(
        endpoints[0],
        endpoints[1],
        workflow.calculator,
        model_name=label,
        Nreplica=n_images,
        optimizer=optimizer,
        working_dir=workflow.working_dir,
    )
    neb.relax_initial_final()
    neb.run_neb(fmax=fmax, spring_constant=spring_constant)
    neb.plot_barrier(element=element)
    neb.plot_core_trajectory(Cij=cij, a0_eq=a0)
    return neb


def run_bcc_benchmark(
    workflow: BasicProperties,
    element: str,
    initial_a0: float | None = None,
    label: str = "Calculated",
    steps=ALL_STEPS,
    sf_num_steps: dict | None = None,
    ts_cell_optimizer: str = "FIRE",
) -> dict:
    """Run the full BCC defect benchmark for one element.

    Args:
        workflow: ``BasicProperties`` workflow holding the calculator and output directory.
        element: BCC element symbol, e.g. ``"Nb"``.
        initial_a0: Initial lattice-parameter guess in Angstrom. Defaults to
            ``BCC_INITIAL_A0[element]``.
        label: Model label used in legends and titles.
        steps: Subset of ``ALL_STEPS`` to run. ``"basic"`` is always run because
            the other steps need its lattice and elastic constants.
        sf_num_steps: Sliding increments per stacking-fault plane, e.g.
            ``{"110": 20, "112": 40}`` (the default).
        ts_cell_optimizer: Optimizer for the traction-separation cell relaxation.

    Returns:
        dict: Results of the basic-property step plus the outputs of the other steps.
    """
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        raise ValueError(f"Unknown benchmark steps: {sorted(unknown)}. Choose from {ALL_STEPS}.")
    if initial_a0 is None:
        initial_a0 = BCC_INITIAL_A0[element]
    sf_num_steps = {"110": 20, "112": 40, **(sf_num_steps or {})}

    progress(f"Starting {label} benchmark for {element} with initial a0={initial_a0} A")
    results = run_basic_properties(workflow, element, initial_a0)
    atoms, a0, cij = results["atoms"], results["a0"], results["cij"]

    if "phonon" in steps:
        progress("Calculating phonon dispersion")
        results["phonon_figure"] = workflow.calculate_phonon_dispersion(
            structure=atoms,
            special_points=BCC_SPECIAL_POINTS,
            labels_path=BCC_PHONON_PATH,
            supercell_matrix=np.diag([4, 4, 4]),
        )

    if "stacking_fault" in steps:
        sf = workflow.create_stacking_fault_workflow(
            atoms=atoms.copy(),
            formula=element,
            info=label,
            optimizer="FIRE",
            working_dir=workflow.path("stacking_fault"),
        )
        results["stacking_fault"] = {}
        for plane, spec in BCC_STACKING_FAULTS.items():
            progress(f"Running stacking fault calculation for ({plane})")
            results["stacking_fault"][plane] = sf.stacking_fault(
                a=spec["a"],
                b=spec["b"],
                miller=spec["miller"],
                distance=a0 / 2,
                layers=spec["layers"],
                num_steps=sf_num_steps[plane],
                fmax=0.005,
                steps=1000,
                write_xyz=True,
            )

    if "traction_separation" in steps:
        results["traction_separation"] = {}
        for surface in BCC_TS_SURFACES:
            label_hkl = "".join(map(str, surface))
            progress(f"Running traction separation for ({label_hkl})")
            ts = workflow.create_traction_separation_workflow(
                atoms=atoms,
                surface_index=surface,
                repeat=(3, 3, 16),
                working_dir=workflow.path(f"ts_{label_hkl}"),
            )
            ts_results = ts.run_pure_separation(
                vacuum_values=np.linspace(0.0, 4.0, 40),
                write_xyz=True,
                cell_optimizer=ts_cell_optimizer,
            )
            ts.plot_pure_separation(ts_results)
            results["traction_separation"][label_hkl] = ts_results

    if "screw" in steps:
        progress("Creating screw-dislocation workflow")
        results["screw_neb"] = run_screw_dislocation(workflow, element, a0, cij, label)

    progress(f"{label} benchmark for {element} complete")
    return results
