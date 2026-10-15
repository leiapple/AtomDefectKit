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


CELL_FILTERS = {
    "strain": "StrainFilter",
    "unitcell": "UnitCellFilter",
    "frechet": "FrechetCellFilter",
}


def relax(
    atoms,
    calculator=None,
    optimizer: str = "FIRE",
    fmax: float = 0.01,
    steps: int | None = None,
    logfile=None,
    cell_filter: str | None = None,
    **filter_kwargs,
):
    """Relax a structure in place with one ASE optimizer.

    Args:
        atoms: Structure to relax. It is modified in place and returned.
        calculator: Optional calculator attached before relaxing. When omitted,
            the calculator already attached to ``atoms`` is used.
        optimizer: Optimizer label accepted by ``build_optimizer``.
        fmax: Force convergence threshold in eV/Angstrom.
        steps: Optional maximum number of optimizer steps.
        logfile: Optional optimizer logfile path.
        cell_filter: Optional cell filter, one of ``"strain"``, ``"unitcell"`` or
            ``"frechet"``. When omitted, only atomic positions are relaxed.
        **filter_kwargs: Extra keyword arguments passed to the cell filter,
            for example ``mask``.

    Returns:
        ase.Atoms: The relaxed ``atoms``.
    """
    if calculator is not None:
        atoms.calc = calculator

    target = atoms
    if cell_filter is not None:
        import ase.filters

        try:
            filter_name = CELL_FILTERS[cell_filter.lower()]
        except KeyError as exc:
            allowed = "', '".join(CELL_FILTERS)
            raise ValueError(f"cell_filter must be '{allowed}', not {cell_filter!r}") from exc
        target = getattr(ase.filters, filter_name)(atoms, **filter_kwargs)

    run_kwargs = {"fmax": fmax}
    if steps is not None:
        run_kwargs["steps"] = steps
    build_optimizer(target, optimizer, logfile=logfile).run(**run_kwargs)
    return atoms
