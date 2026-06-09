"""Discovery of available MLIPs by querying the models directory."""

import pkgutil

import atomdefectkit.models as models_pkg


def discover_models() -> list[str]:
    """Finds all available models in the models package.

    Returns:
        list[str]: List of available model names
    """

    models = []
    aliases_to_hide = {"ocp"}

    for _, module_name, _ in pkgutil.iter_modules(models_pkg.__path__):
        if module_name in aliases_to_hide:
            continue
        models.append(module_name)

    return sorted(models)
