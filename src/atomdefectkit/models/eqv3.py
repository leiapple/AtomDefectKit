"""Implementation of EqV3 and compatible pretrained checkpoints via OCPCalculator."""

from __future__ import annotations

import copy
from pathlib import Path
from urllib.request import urlretrieve

from atomdefectkit.registry import register_model


LEGACY_MODEL_CHOICES = [
    "EquiformerV2-83M-S2EF-OC20-2M",
    "EquiformerV2-31M-S2EF-OC20-All+MD",
    "EquiformerV2-153M-S2EF-OC20-All+MD",
    "EquiformerV2-lE4-lF100-S2EFS-OC22",
    "EquiformerV2-S2EF-ODAC",
    "EquiformerV2-Large-S2EF-ODAC",
    "EquiformerV2-IS2RE-ODAC",
]
EQV3_MODEL_URLS = {
    "eqV3-omat24-direct": "https://huggingface.co/mirror-physics/equiformer_v3/resolve/main/checkpoint/omat24_direct.pt",
    "eqV3-omat24-gradient": "https://huggingface.co/mirror-physics/equiformer_v3/resolve/main/checkpoint/omat24_gradient.pt",
    "eqV3-omat24-mptrj-salex_gradient": "https://huggingface.co/mirror-physics/equiformer_v3/resolve/main/checkpoint/omat24-mptrj-salex_gradient.pt",
}
EQV3_MODEL_CHOICES = LEGACY_MODEL_CHOICES + sorted(EQV3_MODEL_URLS)


def _download_direct_checkpoint(model_name: str, local_cache: str) -> str:
    """Download a known direct checkpoint to the requested cache folder."""
    try:
        url = EQV3_MODEL_URLS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown direct-checkpoint model name: {model_name!r}. "
            f"Known values: {sorted(EQV3_MODEL_URLS)}"
        ) from exc

    cache_dir = Path(local_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / Path(url).name
    if not destination.exists():
        urlretrieve(url, destination)
    return str(destination)


def _resolve_checkpoint_path(model_name: str, local_cache: str) -> str:
    """Resolve a model name to a local checkpoint path."""
    if model_name in EQV3_MODEL_URLS:
        return _download_direct_checkpoint(model_name, local_cache)

    try:
        from fairchem.core.models.model_registry import model_name_to_local_file
    except ImportError as exc:
        raise ImportError(
            "Could not import model_name_to_local_file from fairchem. "
            "Install the EqV3 backend dependencies with the EqV3 extra via `uv sync --extra eqv3`."
        ) from exc

    return model_name_to_local_file(model_name, local_cache=local_cache)


def _prepare_ocp_config_from_checkpoint(checkpoint_path: str) -> dict:
    """Prepare an OCPCalculator config dictionary from a downloaded checkpoint."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(checkpoint["config"])

    config.setdefault("trainer", config.get("trainer", "ocp"))
    config.setdefault("task", {})
    config["task"].setdefault(
        "dataset",
        config.get("dataset", {}).get("format", "ase_db"),
    )

    if "loss_fns" not in config and "loss_functions" in config:
        config["loss_fns"] = copy.deepcopy(config["loss_functions"])
    if "eval_metrics" not in config and "evaluation_metrics" in config:
        config["eval_metrics"] = copy.deepcopy(config["evaluation_metrics"])

    return config


EQV3_METADATA = {
    "model_name": {
        "type": "str",
        "choices": EQV3_MODEL_CHOICES,
        "description": (
            "EqV3 pretrained model name, or one of the direct eqV3 checkpoint "
            "aliases that downloads from Hugging Face into local_cache before loading "
            "with OCPCalculator."
        ),
        "default": "EquiformerV2-31M-S2EF-OC20-All+MD",
    },
    "local_cache": {
        "type": "str",
        "description": "Directory used to cache downloaded EqV3 checkpoints.",
        "default": "pretrained_models",
    },
    "cpu": {
        "type": "bool",
        "description": "Force CPU execution in the underlying OCPCalculator.",
        "default": False,
    },
    "seed": {
        "type": "int",
        "description": "Random seed passed to OCPCalculator for reproducible setup.",
        "default": 42,
    },
}


def _build_impl(params, device):
    """Import and build an OCPCalculator-backed ASE calculator."""
    try:
        from fairchem.core import OCPCalculator
    except ImportError as exc:
        try:
            from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        except ImportError:
            raise ImportError(
                "The 'eqV3' backend requires fairchem-core with OCPCalculator support. "
                "Install the EqV3 environment with `uv sync --extra eqv3` on Python 3.11."
            ) from exc

    model_name = params.get("model_name", "EquiformerV2-31M-S2EF-OC20-All+MD")
    local_cache = params.get("local_cache", "pretrained_models")
    checkpoint_path = _resolve_checkpoint_path(model_name, local_cache)

    use_cpu = params.get("cpu")
    if use_cpu is None:
        use_cpu = device.lower() == "cpu"
    seed = params.get("seed", 42)

    if model_name in EQV3_MODEL_URLS:
        return OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=bool(use_cpu),
            seed=seed,
        )

    return OCPCalculator(
        checkpoint_path=checkpoint_path,
        cpu=bool(use_cpu),
        seed=seed,
    )


@register_model("eqv3", metadata=EQV3_METADATA)
def _build_eqv3(params, device):
    return _build_impl(params, device)

