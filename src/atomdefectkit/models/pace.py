from atomdefectkit.registry import register_model


@register_model(
    "pace",
    metadata={
        "potential_file": {
            "type": "str",
            "description": "Path to an ACE/PACE potential YAML file.",
            "required": True,
        },
    },
)
def _build(params, device=None):
    """Import and build a PyACE calculator."""
    from pyace import PyACECalculator

    try:
        potential_file = params["potential_file"]
    except KeyError as e:
        raise ValueError("Missing required parameter 'potential_file'.") from e

    return PyACECalculator(potential_file)
