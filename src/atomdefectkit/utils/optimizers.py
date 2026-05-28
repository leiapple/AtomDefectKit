"""Shared ASE optimizer helpers."""

from __future__ import annotations

from ase.optimize import BFGS, FIRE, LBFGS


OPTIMIZER_REGISTRY = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
}


def normalize_optimizer_name(optimizer: str) -> str:
    """Normalize and validate an ASE optimizer name.

    Args:
        optimizer: User-provided optimizer label.

    Returns:
        str: Uppercase optimizer name accepted by ``OPTIMIZER_REGISTRY``.

    Raises:
        ValueError: If the requested optimizer is not supported.
    """
    name = optimizer.upper()
    if name not in OPTIMIZER_REGISTRY:
        allowed = "', '".join(OPTIMIZER_REGISTRY)
        raise ValueError(f"Optimizer must be '{allowed}', not {optimizer}")
    return name


def build_optimizer(target, optimizer: str, **kwargs):
    """Create an ASE optimizer for a structure, filter, or NEB object.

    Args:
        target: ASE object passed to the optimizer constructor.
        optimizer: Optimizer label such as ``FIRE``, ``BFGS``, or ``LBFGS``.
        **kwargs: Extra keyword arguments forwarded to the optimizer constructor.

    Returns:
        FIRE | BFGS | LBFGS: Configured optimizer instance.
    """
    name = normalize_optimizer_name(optimizer)
    return OPTIMIZER_REGISTRY[name](target, **kwargs)
