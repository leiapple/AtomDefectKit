import importlib

from atomdefectkit.model_discovery import discover_models
from atomdefectkit.registry import MODEL_METADATA, MODEL_REGISTRY


def _import_model(model_name: str) -> None:
    importlib.import_module(f"atomdefectkit.models.{model_name.lower()}")


def available_models() -> list[str]:
    """Return the names of model loaders bundled with AtomDefectKit."""
    return discover_models()


def available_model_metadata() -> dict:
    """Return metadata for bundled model loaders that can be imported."""
    for model_name in available_models():
        try:
            _import_model(model_name)
        except ImportError:
            continue
    return dict(MODEL_METADATA)


def load_model(model_name: str, model_parameters: dict | None = None, device: str = "cuda"):
    """Build an ASE-compatible calculator from a registered model loader."""
    model_name = model_name.lower()
    _import_model(model_name)
    return MODEL_REGISTRY[model_name](dict(model_parameters or {}), device=device)


def main() -> None:
    print("Hello from atomdefectkit!")
