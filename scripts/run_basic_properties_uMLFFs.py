from ase.build import bulk
from atomdefectkit.basic_properties import BasicProperties
import numpy as np

# Define the reference crystal used for the property workflow.
elem = 'Nb'
crystal_structure = 'bcc'
initial_a0 = 3.3  # Initial lattice-parameter guess in Angstrom.
eVA3_to_GPa = 160.21766208  # Conversion factor from eV/Å^3 to GPa

# Initialize the workflow with the GRACE calculator backend.
workflow = BasicProperties(
    model_name="GRACE",
    model_parameters={"model_size": "small", "num_layers": 1, "model_task": "OAM"},
)

# Build the conventional cubic BCC unit cell.
atoms = bulk(elem, crystal_structure, a=initial_a0, cubic=True)

# Estimate the equilibrium lattice parameter from both cell relaxation and EOS fitting.
a0_relax = workflow.calculate_equilibrium_a0_relax(atoms.copy(), fmax=0.001)
volumes, energies, a0_fit = workflow.calculate_equilibrium_a0_birch_murnaghan(
    atoms.copy(), vol_range=np.linspace(0.95, 1.05, 40)
)

# Sanity-check that the two lattice-parameter estimates are consistent.
if abs(a0_relax - a0_fit) > 1e-4:
    print('Lattice constant difference is', abs(a0_relax - a0_fit))
    print('Check the approach for getting lattice constant!')

# Update the bulk cell before evaluating the remaining properties.
atoms.set_cell([a0_fit] * 3, scale_atoms=True)
Cij = workflow.calculate_elastic_constants(atoms, verbose=False)
vacancy_formation_energy = workflow.calculate_vacancy_formation_energy(atoms)

# Evaluate interstitial and vacancy formation energies for representative sites.
octahedral_position = a0_fit * np.array([0.5, 0.5, 0])
octahedral_formation_energy = workflow.calculate_interstitial_formation_energy(atoms, elem, octahedral_position)
tetrahedral_position = a0_fit * np.array([0.25, 0.5, 0])
tetrahedral_formation_energy = workflow.calculate_interstitial_formation_energy(atoms, elem, tetrahedral_position)
inter_100_position = a0_fit * np.array([0.5, 0.5, 0])
inter_100_formation_energy = workflow.calculate_interstitial_formation_energy(atoms, elem, inter_100_position)
inter_110_position = a0_fit * np.array([0.1, 0.1, 0.5])
inter_110_formation_energy = workflow.calculate_interstitial_formation_energy(atoms, elem, inter_110_position)
inter_111_position = a0_fit * np.array([[0.33, 0.33, 0.33], [0.7, 0.7, 0.7]])
inter_111_formation_energy = workflow.calculate_interstitial_formation_energy(atoms, elem, inter_111_position, dumbbell=True)

# Calculate surface energies for a small benchmark set of facets.
surface_energy = np.zeros(4)
miller_indices_list = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2)]
for i, miller_indices in enumerate(miller_indices_list):
    surface_energy[i] = workflow.calculate_surface_energy(atoms, miller_indices, vacuum=20, n_layers=8)

# Highlight the key numerical results in the console output.
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"
print(RED + BOLD + f"Equilibrium lattice constant: {a0_fit:.5f} Å")
print(RED + BOLD + f"Equilibrium lattice constant: {a0_relax:.5f} Å")
# Print the elastic constants
print(RED + BOLD + "Elastic constants (GPa):" + RESET)
print(RED + BOLD + f"C11 = {Cij[0, 0] * eVA3_to_GPa:.2f}" + RESET)
print(RED + BOLD + f"C12 = {Cij[0, 1] * eVA3_to_GPa:.2f}" + RESET)
print(RED + BOLD + f"C44 = {Cij[3, 3] * eVA3_to_GPa:.2f}" + RESET)
print(f"Vacancy formation energy: {vacancy_formation_energy:.3f} eV")
print(f"Octa Interstitial formation energy: {octahedral_formation_energy:.3f} eV")
print(f"Tetra Interstitial formation energy: {tetrahedral_formation_energy:.3f} eV")
print(f"(100) Interstitial formation energy: {inter_100_formation_energy:.3f} eV")
print(f"(110) Interstitial formation energy: {inter_110_formation_energy:.3f} eV")
print(f"(111) Interstitial formation energy: {inter_111_formation_energy:.3f} eV")

# Report the surface-energy summary for each facet.
for i, miller_indices in enumerate(miller_indices_list):
    print(RED + BOLD +f"Surface energy for {miller_indices}: {surface_energy[i]:.3f} J/m^2")

# Save the JSON summary using the reusable workflow helper.
json_path = workflow.save_properties(
    volumes=volumes,
    energies=energies,
    Cij=Cij * eVA3_to_GPa,
    surface_energy=surface_energy,
    vacancy_formation_energy=vacancy_formation_energy,
    octahedral_formation_energy=octahedral_formation_energy,
    tetrahedral_formation_energy=tetrahedral_formation_energy,
    inter_100_formation_energy=inter_100_formation_energy,
    inter_110_formation_energy=inter_110_formation_energy,
    inter_111_formation_energy=inter_111_formation_energy,
    miller_indices_list=miller_indices_list,
)
print(f"Data saved to '{json_path}'")
