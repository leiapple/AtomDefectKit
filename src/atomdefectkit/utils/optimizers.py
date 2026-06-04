"""Shared ASE optimizer helpers."""

from __future__ import annotations

from ase.optimize import BFGS, FIRE, LBFGS
from ase.optimize.sciopt import SciPyFminCG


OPTIMIZER_REGISTRY = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "SCIPYFMINCG": SciPyFminCG,
}

OPTIMIZER_ALIASES = {
    "FIRE": "FIRE",
    "BFGS": "BFGS",
    "LBFGS": "LBFGS",
    "SCIPYFMINCG": "SCIPYFMINCG",
    "SCIPY_FMIN_CG": "SCIPYFMINCG",
    "SCIPY-FMIN-CG": "SCIPYFMINCG",
    "SCIPY FMIN CG": "SCIPYFMINCG",
}


def normalize_optimizer_name(optimizer: str) -> str:
    """Normalize and validate an ASE optimizer name.

    Args:
        optimizer: User-provided optimizer label.

    Returns:
        str: Canonical optimizer name accepted by ``OPTIMIZER_REGISTRY``.

    Raises:
        ValueError: If the requested optimizer is not supported.
    """
    name = optimizer.upper()
    canonical_name = OPTIMIZER_ALIASES.get(name)
    if canonical_name not in OPTIMIZER_REGISTRY:
        allowed = "', '".join(OPTIMIZER_REGISTRY)
        raise ValueError(f"Optimizer must be '{allowed}', not {optimizer}")
    return canonical_name


def build_optimizer(target, optimizer: str, **kwargs):
    """Create an ASE optimizer for a structure, filter, or NEB object.

    Args:
        target: ASE object passed to the optimizer constructor.
        optimizer: Optimizer label such as ``FIRE``, ``BFGS``, ``LBFGS``,
            or ``SciPyFminCG``.
        **kwargs: Extra keyword arguments forwarded to the optimizer constructor.

    Returns:
        FIRE | BFGS | LBFGS | SciPyFminCG: Configured optimizer instance.
    """
    name = normalize_optimizer_name(optimizer)
    return OPTIMIZER_REGISTRY[name](target, **kwargs)
