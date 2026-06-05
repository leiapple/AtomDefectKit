from __future__ import annotations

from atomdefectkit.registry import register_model


GRACE_MODEL_CHOICES = [
    "GRACE-1L-SMAX-large",
    "GRACE-1L-SMAX-OMAT-large",
    "GRACE-2L-SMAX-medium",
    "GRACE-2L-SMAX-large",
    "GRACE-2L-SMAX-OMAT-medium",
    "GRACE-2L-SMAX-OMAT-large",
    "GRACE-1L-OMAT",
    "GRACE-1L-OMAT-medium-base",
    "GRACE-1L-OMAT-medium-ft-E",
    "GRACE-1L-OMAT-large-base",
    "GRACE-1L-OMAT-large-ft-E",
    "GRACE-2L-OMAT",
    "GRACE-2L-OMAT-medium-base",
    "GRACE-2L-OMAT-medium-ft-E",
    "GRACE-2L-OMAT-large-base",
    "GRACE-2L-OMAT-large-ft-E",
    "GRACE-1L-OAM",
    "GRACE-1L-OMAT-medium-ft-AM",
    "GRACE-1L-OMAT-large-ft-AM",
    "GRACE-2L-OAM",
    "GRACE-2L-OMAT-medium-ft-AM",
    "GRACE-2L-OMAT-large-ft-AM",
]

_LEGACY_MODEL_MAPPING = {
    ("omat", "small", 1): "GRACE-1L-OMAT",
    ("omat", "small", 2): "GRACE-2L-OMAT",
    ("omat", "medium", 1): "GRACE-1L-OMAT-medium-base",
    ("omat", "medium", 2): "GRACE-2L-OMAT-medium-base",
    ("omat", "large", 1): "GRACE-1L-OMAT-large-base",
    ("omat", "large", 2): "GRACE-2L-OMAT-large-base",
    ("oam", "small", 1): "GRACE-1L-OAM",
    ("oam", "small", 2): "GRACE-2L-OAM",
    ("oam", "medium", 1): "GRACE-1L-OMAT-medium-ft-AM",
    ("oam", "medium", 2): "GRACE-2L-OMAT-medium-ft-AM",
    ("oam", "large", 1): "GRACE-1L-OMAT-large-ft-AM",
    ("oam", "large", 2): "GRACE-2L-OMAT-large-ft-AM",
}


def resolve_grace_model_name(params: dict) -> str:
    """Resolve a GRACE foundation model from explicit or legacy parameters."""
    explicit_name = params.get("model_name")
    if explicit_name is not None:
        if explicit_name not in GRACE_MODEL_CHOICES:
            raise ValueError(
                f"Unknown GRACE model_name: {explicit_name!r}. "
                f"Known model_name values: {GRACE_MODEL_CHOICES}"
            )
        return explicit_name

    size = params.get("model_size", "small").lower()
    layers = int(params.get("num_layers", 1))
    task = params.get("model_task", "omat").lower()

    try:
        return _LEGACY_MODEL_MAPPING[(task, size, layers)]
    except KeyError as e:
        raise ValueError(
            f"Unknown GRACE model parameters: {params}. "
            f"Known legacy parameters: {list(_LEGACY_MODEL_MAPPING.keys())}"
        ) from e


@register_model(
    "grace",
    metadata={
        "model_name": {
            "type": "str",
            "choices": GRACE_MODEL_CHOICES,
            "description": (
                "Exact GRACE foundation-model name accepted by grace_fm(). "
                "This is the recommended parameter because it exposes all "
                "published GRACE foundation models."
            ),
            "default": "GRACE-1L-OMAT",
        },
        "model_size": {
            "type": "str",
            "choices": ["small", "medium", "large"],
            "description": (
                "Legacy compatibility parameter for selecting a GRACE model by size. "
                "Use model_name for full access to all published models."
            ),
            "default": "small",
        },
        "num_layers": {
            "type": "int",
            "choices": [1, 2],
            "description": (
                "Legacy compatibility parameter for selecting one- or two-layer GRACE models. "
                "Use model_name for full access to all published models."
            ),
            "default": 1,
        },
        "model_task": {
            "type": "str",
            "choices": ["oam", "omat"],
            "description": (
                "Legacy compatibility parameter for task family selection. "
                "Use model_name for full access to all published models."
            ),
            "default": "omat",
        },
    },
)
def _build(params, device):
    """Import and build a GRACE foundation model."""
    from tensorpotential.calculator.foundation_models import grace_fm

    return grace_fm(resolve_grace_model_name(params), device=device)
