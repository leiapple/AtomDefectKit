"""Single entry point for building ASE calculators from registered MLIP backends."""

from __future__ import annotations

import importlib
import logging

from atomdefectkit.model_discovery import discover_models
from atomdefectkit.model_errors import format_model_error, format_unknown_model_error
from atomdefectkit.registry import MODEL_METADATA, MODEL_REGISTRY

logger = logging.getLogger(__name__)


def import_model_module(model_name: str) -> None:
    """Import a backend module so that it registers itself in ``MODEL_REGISTRY``.

    Args:
        model_name: Lower-case backend name, e.g. ``"mace"``.

    Raises:
        ValueError: If no backend module with that name exists.
        ImportError: If the backend module exists but its dependencies are missing.
    """
    try:
        importlib.import_module(f"atomdefectkit.models.{model_name}")
    except ModuleNotFoundError as exc:
        if exc.name == f"atomdefectkit.models.{model_name}":
            raise ValueError(format_unknown_model_error(model_name, discover_models())) from exc
        raise ImportError(
            f"Model '{model_name}' could not be imported.\n"
            f"This may be due to missing dependencies.\nOriginal error: {exc!r}"
        ) from exc


def _check_required_parameters(model_name: str, params: dict) -> None:
    missing = [
        key
        for key, spec in MODEL_METADATA.get(model_name, {}).items()
        if isinstance(spec, dict) and spec.get("required") and key not in params
    ]
    if missing:
        raise ValueError(
            format_model_error(
                model_name,
                params,
                ValueError(f"Missing required parameter(s): {', '.join(missing)}"),
            )
        )


def load_model(model_name: str, model_parameters: dict | None = None, device: str = "cuda"):
    """Build an ASE-compatible calculator from a registered model backend.

    If building on CUDA fails with a CUDA-related error, the calculator is built
    on CPU instead and a warning is logged.

    Args:
        model_name: Registered backend name (case-insensitive), e.g. ``"mace"``.
        model_parameters: Backend-specific parameters. See
            ``atomdefectkit.available_model_metadata()`` for the options.
        device: ``"cuda"``, ``"cuda:<index>"`` or ``"cpu"``.

    Returns:
        ASE calculator built by the backend.

    Raises:
        ValueError: If the backend is unknown or its parameters are invalid.
        RuntimeError: If a dependency is missing or the backend returns nothing.
    """
    model_name = model_name.lower()
    params = dict(model_parameters or {})

    try:
        import_model_module(model_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency for model '{model_name}'.\n\n{exc}\n\n"
            f"=> Install the backend extra, e.g. `uv sync --extra {model_name}`."
        ) from exc

    if model_name not in MODEL_REGISTRY:
        raise RuntimeError(f"Model '{model_name}' was imported but did not register itself.")
    _check_required_parameters(model_name, params)
    builder = MODEL_REGISTRY[model_name]

    try:
        calculator = builder(params, device=device)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency for model '{model_name}'.\n\n{exc}\n\n"
            f"=> Install the backend extra, e.g. `uv sync --extra {model_name}`."
        ) from exc
    except ValueError as exc:
        raise ValueError(format_model_error(model_name, params, exc)) from exc
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if not (device.lower().startswith("cuda") and "cuda" in str(exc).lower()):
            raise ValueError(format_model_error(model_name, params, exc)) from exc
        logger.warning("CUDA not available for '%s' (%s); falling back to CPU.", model_name, exc)
        calculator = builder(params, device="cpu")

    if calculator is None:
        raise RuntimeError(f"Builder for '{model_name}' returned None")
    return calculator
