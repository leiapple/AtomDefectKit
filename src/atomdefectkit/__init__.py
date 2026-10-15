"""Atomistic defect workflows and model-loader utilities."""

from atomdefectkit.basic_properties import BasicProperties
from atomdefectkit.model_discovery import discover_models
from atomdefectkit.model_loader import import_model_module, load_model
from atomdefectkit.registry import MODEL_METADATA, MODEL_REGISTRY
from atomdefectkit.stacking_fault import StackingFaultCurve, StackingFaultWorkflow
from atomdefectkit.traction_separation import (
    TractionSeparationCurve,
    TractionSeparationWorkflow,
)

__all__ = [
    "MODEL_METADATA",
    "MODEL_REGISTRY",
    "BasicProperties",
    "StackingFaultCurve",
    "StackingFaultWorkflow",
    "TractionSeparationCurve",
    "TractionSeparationWorkflow",
    "available_model_metadata",
    "available_models",
    "load_model",
]


def available_models() -> list[str]:
    """Return the names of model loaders bundled with AtomDefectKit."""
    return discover_models()


def available_model_metadata() -> dict:
    """Return metadata for bundled model loaders that can be imported."""
    for model_name in available_models():
        try:
            import_model_module(model_name)
        except ImportError:
            continue
    return dict(MODEL_METADATA)


def main() -> None:
    """Command-line entry point; see ``scripts/run_benchmark.py`` for the benchmark driver."""
    print("AtomDefectKit: see `python scripts/run_benchmark.py --help` for the benchmark driver.")
