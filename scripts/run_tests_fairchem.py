#

import argparse

from atomdefectkit.BasicProperties import BasicProperties
from atomdefectkit.neb import BCCScrewDislocPeierlsBarrier

import numpy as np

from ase.build import bulk

from atomdefectkit.utils.progress import progress


parser = argparse.ArgumentParser(description="Run the FairChem BCC defect workflow for one element.")
parser.add_argument("--element", default="Nb", help="Chemical element symbol.")
parser.add_argument("--initial-a0", type=float, default=3.3, help="Initial BCC lattice-parameter guess in Angstrom.")
parser.add_argument("--working-dir", default=None, help="Output directory. Defaults to Test_<element>_fairchem.")
args = parser.parse_args()

# Define the benchmark material and output location.
elem = args.element
crystal_structure = 'bcc'
initial_a0 = args.initial_a0
working_dir = args.working_dir or f'Test_{elem}_fairchem'

# Initialize the workflow with the FairChem calculator backend.
progress(f"Starting FairChem workflow for {elem} with initial a0={initial_a0} A")
progress("Loading FairChem calculator")
workflow = BasicProperties(
    model_name="fairchem",
    model_parameters={"model_size": "s", "model_version": "1p2", "model_task": "omat"},
    device="cuda",
    working_dir=working_dir,
)
calc = workflow.get_calculator()
progress("FairChem calculator loaded")

# Build the conventional cubic BCC unit cell.
progress("Building BCC unit cell")
atoms = bulk(elem, crystal_structure, a=initial_a0, cubic=True)

# Estimate the equilibrium lattice parameter from both relaxation and EOS fitting.
progress("Relaxing lattice constant")
a0_relax = workflow.calculate_equilibrium_a0_relax(atoms.copy(), fmax=0.001)
progress(f"Relaxed lattice constant: {a0_relax:.6f} A")
progress("Fitting Birch-Murnaghan EOS")
volumes, energies, a0_fit = workflow.calculate_equilibrium_a0_birch_murnaghan(atoms.copy(), vol_range=np.linspace(0.95, 1.05, 40))
progress(f"EOS lattice constant: {a0_fit:.6f} A")

# Sanity-check that the two lattice-parameter estimates agree.
if abs(a0_relax - a0_fit) > 1e-4:
    print('Lattice constant difference is', abs(a0_relax - a0_fit))
    print('Check the approach for getting lattice constant!')

# Update the bulk cell before computing defect and surface properties.
atoms.set_cell([a0_fit] * 3, scale_atoms=True)
progress("Calculating elastic constants")
Cij = workflow.calculate_elastic_constants(atoms.copy(), verbose=False)
progress("Calculating vacancy formation energy")
vacancy_formation_energy = workflow.calculate_vacancy_formation_energy(atoms)

# Evaluate representative interstitial and vacancy formation energies.
progress("Calculating interstitial formation energies")
octahedral_formation_energy = workflow.calculate_interstitial_formation_energy(atoms.copy(), 
                                                                               elem, 
                                                                               a0_fit * np.array([0.5, 0.5, 0]))
tetrahedral_formation_energy = workflow.calculate_interstitial_formation_energy(atoms.copy(), 
                                                                                elem, 
                                                                                a0_fit * np.array([0.25, 0.5, 0]))
inter_100_formation_energy = workflow.calculate_interstitial_formation_energy(atoms.copy(), 
                                                                              elem, 
                                                                              a0_fit * np.array([0.5, 0.5, 0]))
inter_110_formation_energy = workflow.calculate_interstitial_formation_energy(atoms.copy(), 
                                                                              elem, 
                                                                              a0_fit * np.array([0.1, 0.1, 0.5]))
inter_111_formation_energy = workflow.calculate_interstitial_formation_energy(atoms.copy(), 
                                                                              elem, 
                                                                              a0_fit * np.array([[0.33, 0.33, 0.33],[0.7, 0.7, 0.7]]), 
                                                                              dumbbell=True)

# Calculate surface energies for a small benchmark set of facets.
progress("Calculating surface energies")
surface_energy = np.zeros(4)
miller_indices_list = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2)]
for i, miller_indices in enumerate(miller_indices_list):
    progress(f"Calculating surface energy for {miller_indices}")
    surface_energy[i] = workflow.calculate_surface_energy(atoms, miller_indices, vacuum=20, n_layers=8)

progress("Saving basic properties")
workflow.save_properties(
    volumes=volumes,
    energies=energies,
    Cij=Cij,
    surface_energy=surface_energy,
    vacancy_formation_energy=vacancy_formation_energy,
    octahedral_formation_energy=octahedral_formation_energy,
    tetrahedral_formation_energy=tetrahedral_formation_energy,
    inter_100_formation_energy=inter_100_formation_energy,
    inter_110_formation_energy=inter_110_formation_energy,
    inter_111_formation_energy=inter_111_formation_energy,
    save_name=f'{elem}_basicProp.json'
)

# Plot the calculated properties against the stored DFT reference data.
progress("Plotting basic-property comparison")
workflow.plot_comparison(f"{working_dir}/{elem}_basicProp.json", f'../data/BasicProperties/dft_{elem}.json')

# calculate phonon dispersion
progress("Calculating phonon dispersion")
special_points = {
    "G": (0.0, 0.0, 0.0),
    "H": (0.5, -0.5, 0.5),
    "N": (0.0, 0.0, 0.5),
    "P": (0.25, 0.25, 0.25),
}

labels_path = [["N", "G", "H", "P", "G"]]

fig_path = workflow.calculate_phonon_dispersion(
    structure=atoms,
    special_points=special_points,
    labels_path=labels_path,
    supercell_matrix=np.diag([4, 4, 4]),
)

# Calculate stacking fault energy curve
progress("Creating stacking-fault workflow")
SF = workflow.create_stacking_fault_workflow(
    atoms=atoms.copy(),
    formula=elem,
    info="FairChem",
    optimizer="FIRE",
    working_dir=f'{working_dir}/stacking_fault',
)

# (110) plane
progress("Running stacking fault calculation for (110)")
SF.stacking_fault(
    a=(1, 1, -1),
    b=(1, 1, 2),
    miller=(1, -1, 0),
    distance=a0_fit/2,
    layers=30,
    num_steps=20,
    fmax=0.005,
    steps=1000,
    write_xyz=True,
)

# (112) plane
progress("Running stacking fault calculation for (112)")
SF.stacking_fault(
    a=(1, 1, -1),
    b=(-1, 1, 0),
    miller=(1, 1, 2),
    distance=a0_fit/2,
    layers=40,
    num_steps=40,
    fmax=0.005,
    steps=1000,
    write_xyz=True,
)

# Traction separation (100)
progress("Running traction separation for (100)")
TS_100 = workflow.create_traction_separation_workflow(
    atoms=atoms,
    surface_index=(1, 0, 0),
    repeat=(3, 3, 16),
    working_dir=f'{working_dir}/ts_100',
)
results = TS_100.run_pure_separation(
    vacuum_values=np.linspace(0.0, 4.0, 40),
    write_xyz=True,
    cell_optimizer="FIRE",
)
TS_100.plot_pure_separation(results)

# Traction separation (110)
progress("Running traction separation for (110)")
TS_110 = workflow.create_traction_separation_workflow(
    atoms=atoms,
    surface_index=(1, 1, 0),
    repeat=(3, 3, 16),
    working_dir=f'{working_dir}/ts_110',
)
results = TS_110.run_pure_separation(
    vacuum_values=np.linspace(0.0, 4.0, 40),
    write_xyz=True,
    cell_optimizer="FIRE",
)
TS_110.plot_pure_separation(results)

# screw dislocation
progress("Creating screw-dislocation workflow")
dislocation_system = workflow.create_screw_dislocation_workflow(
    element=elem,
    lattice_constant=a0_fit,
    elastic_constant=Cij,
    working_dir=working_dir,
)

try:
    bcc_disl_init = dislocation_system.create_dislocation_object()
    # Relax the initial dislocation dipole configuration.
    progress("Relaxing initial screw-dislocation dipole")
    base_system_init, disl_system_init = dislocation_system.relax_dislocation_dipole(
                                                bcc_disl_init,
                                                disloc_center=[0, 0, 0],
                                                fmax=0.005,
                                                optimizer='FIRE',
                                                logfile="initial_fire_dislocation_relax.log",
                                                )

    # Convert the relaxed atomman system to ASE for downstream workflows.

    dislocation_dipole_ase_initial, properties = disl_system_init.dump('ase_Atoms', return_prop=True)
    # Save the differential-displacement map for the initial core location.
    progress("Plotting initial differential-displacement map")
    dislocation_system.plot_differential_displacement_map(bcc_disl_init,
                                                          base_system_init,
                                                          disl_system_init
                                                          )

    # Repeat for the displaced final-state core configuration.
    bcc_disl_final = dislocation_system.create_dislocation_object()

    progress("Relaxing final screw-dislocation dipole")
    base_system_final, disl_system_final = dislocation_system.relax_dislocation_dipole(
                                                bcc_disl_final,
                                                disloc_center=[a0_fit*np.sqrt(6)/3, 0, 0],
                                                fmax=0.005,
                                                optimizer='FIRE',
                                                logfile="final_fire_dislocation_relax.log",
                                                )
    dislocation_dipole_ase_final, properties = disl_system_final.dump('ase_Atoms', return_prop=True)

    # Save the differential-displacement map for the final core location.
    progress("Plotting final differential-displacement map")
    dislocation_system.plot_differential_displacement_map(bcc_disl_final,
                                                          base_system_final,
                                                          disl_system_final,
                                                          )

    progress("Running screw-dislocation NEB")
    bcc_screw_neb = BCCScrewDislocPeierlsBarrier(dislocation_dipole_ase_initial,
                                                 dislocation_dipole_ase_final,
                                                 calc, model_name='FairChem-OMAT',
                                                 Nreplica=11,
                                                 optimizer='FIRE',
                                                 working_dir=f'{working_dir}/')
    bcc_screw_neb.relax_initial_final()
    bcc_screw_neb.run_neb(fmax=0.005, spring_constant=0.1)
    bcc_screw_neb.plot_barrier(element=f'{elem}')
except ValueError as exc:
    progress(f"Skipping screw-dislocation workflow: {exc}")
progress("FairChem workflow complete")
