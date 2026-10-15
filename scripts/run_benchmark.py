"""Run the AtomDefectKit BCC benchmark for one model and one or more elements.

Examples::

    uv run --extra mace python scripts/run_benchmark.py --model mace --element Nb
    uv run --extra grace python scripts/run_benchmark.py --model grace --element V,Nb,Ta,Mo,W
    uv run --extra pace python scripts/run_benchmark.py --model pace \\
        --model-params '{"potential_file": "Nb/output_potential.yaml"}' --label ACE2025 \\
        --element Nb --steps basic,screw
"""

from __future__ import annotations

import argparse
import json
import logging

from atomdefectkit import BasicProperties
from atomdefectkit.benchmarks import ALL_STEPS, BCC_INITIAL_A0, run_bcc_benchmark

#: Parameters, device and label used for each backend in the published benchmarks.
MODEL_PRESETS = {
    "mace": {
        "params": {"model_task": "omat_pbe", "default_dtype": "float64"},
        "label": "MACE-OMAT",
    },
    "fairchem": {
        "params": {"model_size": "s", "model_version": "1p2", "model_task": "omat"},
        "label": "FairChem-OMAT",
    },
    "grace": {
        "params": {"model_size": "small", "num_layers": 1, "model_task": "OMAT"},
        "device": "cpu",
        "label": "GRACE-OAM",
    },
    "7net": {
        "params": {"model_name": "7net-omni", "model_task": "omat24"},
        "label": "7net-omni",
    },
    "upet": {
        "params": {"model_task": "omat", "model_size": "s", "model_version": "latest"},
        "label": "UPET-OMAT-S",
    },
    "eqv3": {
        "params": {"model_name": "eqV3-omat24-gradient", "local_cache": "pretrained_models", "seed": 42},
        "label": "EqV3",
    },
    "pace": {
        "params": {},
        "label": "ACE",
        "sf_num_steps": {"110": 50, "112": 50},
        "ts_cell_optimizer": "SciPyFminCG",
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the BCC defect benchmark for one model.")
    parser.add_argument("--model", required=True, help="Registered model backend, e.g. mace, grace, pace.")
    parser.add_argument(
        "--model-params",
        default=None,
        help="JSON object merged over the preset parameters, e.g. '{\"model_task\": \"omat\"}'.",
    )
    parser.add_argument("--device", default=None, help="cuda, cuda:<index> or cpu. Defaults to the preset or cuda.")
    parser.add_argument("--label", default=None, help="Model label for plots. Defaults to the preset label.")
    parser.add_argument(
        "--element",
        default="Nb",
        help=f"Element symbol or comma-separated list. Known: {','.join(BCC_INITIAL_A0)}.",
    )
    parser.add_argument(
        "--initial-a0",
        type=float,
        default=None,
        help="Initial lattice-parameter guess in Angstrom (single element only).",
    )
    parser.add_argument(
        "--steps",
        default=",".join(ALL_STEPS),
        help=f"Comma-separated subset of: {','.join(ALL_STEPS)}.",
    )
    parser.add_argument(
        "--working-dir",
        default=None,
        help="Output directory. Defaults to Test_<element>_<model>.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    model = args.model.lower()
    preset = MODEL_PRESETS.get(model, {})
    params = {**preset.get("params", {}), **json.loads(args.model_params or "{}")}
    device = args.device or preset.get("device", "cuda")
    label = args.label or preset.get("label", model)
    elements = [element.strip() for element in args.element.split(",") if element.strip()]
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    if args.initial_a0 is not None and len(elements) > 1:
        raise SystemExit("--initial-a0 can only be used with a single element.")
    if args.working_dir is not None and len(elements) > 1:
        raise SystemExit("--working-dir can only be used with a single element.")

    calculator = None
    for element in elements:
        workflow = BasicProperties(
            model_name=None if calculator else model,
            model_parameters=params,
            device=device,
            calculator=calculator,
            working_dir=args.working_dir or f"Test_{element}_{model}",
        )
        calculator = workflow.calculator  # Load the model once and reuse it.
        run_bcc_benchmark(
            workflow,
            element,
            initial_a0=args.initial_a0,
            label=label,
            steps=steps,
            sf_num_steps=preset.get("sf_num_steps"),
            ts_cell_optimizer=preset.get("ts_cell_optimizer", "FIRE"),
        )


if __name__ == "__main__":
    main()
