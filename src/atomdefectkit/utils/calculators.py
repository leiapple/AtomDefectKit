"""Compatibility helpers for third-party ASE calculators."""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class ScalarEnergyCalculator(Calculator):
    """Proxy an ASE calculator while enforcing scalar total energies.

    Some calculator versions return ``energy`` and ``free_energy`` as
    one-element arrays.  ASE optimizers require these total-energy properties
    to be scalar values.
    """

    def __init__(self, calculator):
        super().__init__()
        self.calculator = calculator
        self.implemented_properties = list(calculator.implemented_properties)

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        self.calculator.calculate(atoms, properties, system_changes)
        self.results = dict(self.calculator.results)

        for property_name in ("energy", "free_energy"):
            if property_name not in self.results:
                continue
            value = np.asarray(self.results[property_name])
            if value.size != 1:
                raise ValueError(
                    f"Expected scalar {property_name!r}, but calculator "
                    f"returned an array with shape {value.shape}."
                )
            self.results[property_name] = value.item()


def ensure_scalar_energy_calculator(calculator):
    """Return an ASE calculator that exposes scalar total energies."""
    if isinstance(calculator, ScalarEnergyCalculator):
        return calculator
    return ScalarEnergyCalculator(calculator)
