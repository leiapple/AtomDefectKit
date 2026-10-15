"""Named unit conversions shared by the workflows.

ASE works in eV and Angstrom. These constants convert those values into the
units reported by AtomDefectKit outputs.
"""

from __future__ import annotations

from ase import units

#: eV/Angstrom^2 -> J/m^2 (about 16.0218).
EV_PER_A2_TO_J_PER_M2 = units.m**2 / units.J

#: eV/Angstrom^3 -> GPa (about 160.2177).
EV_PER_A3_TO_GPA = 1.0 / units.GPa

#: GPa * Angstrom -> J/m^2 (1e9 Pa * 1e-10 m).
GPA_A_TO_J_PER_M2 = 0.1

#: eV -> meV.
EV_TO_MEV = 1000.0

#: J/m^2 -> mJ/m^2.
J_TO_MJ = 1000.0
