"""Run the initial ordered-B2 GRACE benchmark workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

from atomdefectkit.b2 import B2Crystal, B2SlipSystem, B2Workflow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("species_a", help="Corner-sublattice species, e.g. Ni")
    parser.add_argument("species_b", help="Body-center-sublattice species, e.g. Al")
    parser.add_argument("--a0", type=float, required=True, help="Initial lattice constant in A")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--working-dir", default=None)
    parser.add_argument("--skip-gamma", action="store_true")
    parser.add_argument("--model-name", default="GRACE-1L-OMAT", help="Exact GRACE foundation-model name")
    parser.add_argument("--peierls-initial", help="Relaxed initial dislocation endpoint")
    parser.add_argument("--peierls-final", help="Relaxed final dislocation endpoint")
    parser.add_argument("--neb-replicas", type=int, default=11)
    parser.add_argument(
        "--dislocation-count",
        type=int,
        help="Number of equivalent dislocations represented by the NEB cell",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    formula = f"{args.species_a}{args.species_b}"
    working_dir = Path(args.working_dir or f"B2_{formula}_grace")
    workflow = B2Workflow(
        B2Crystal((args.species_a, args.species_b), args.a0),
        model_name="grace",
        model_parameters={"model_name": args.model_name},
        device=args.device,
        working_dir=working_dir,
    )

    volumes, energies, a0 = workflow.calculate_equilibrium_lattice_constant()
    cij = workflow.calculate_elastic_constants(a0)
    working_dir.mkdir(parents=True, exist_ok=True)
    with (working_dir / "bulk_results.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "formula": formula,
                "model_name": args.model_name,
                "lattice_constant_A": a0,
                "volumes_A3": volumes.tolist(),
                "energies_eV": energies.tolist(),
                "elastic_constants_GPa": np.asarray(cij).tolist(),
            },
            stream,
            indent=2,
        )

    if not args.skip_gamma:
        workflow.calculate_gamma_lines(
            [
                B2SlipSystem(
                    plane=(0, 1, 1),
                    direction=(1, 0, 0),
                    translation_fraction=1.0,
                    name="{011}<100>",
                ),
                B2SlipSystem(
                    plane=(1, -1, 0),
                    direction=(1, 1, 1),
                    translation_fraction=1.0,
                    name="{1-10}<111>",
                ),
            ],
            lattice_constant=a0,
        )

    if bool(args.peierls_initial) != bool(args.peierls_final):
        raise ValueError("Provide both --peierls-initial and --peierls-final.")
    if args.peierls_initial:
        if args.dislocation_count is None:
            raise ValueError("Provide --dislocation-count with Peierls endpoints.")
        peierls = workflow.create_peierls_workflow(
            read(args.peierls_initial),
            read(args.peierls_final),
            model_label="GRACE",
            replicas=args.neb_replicas,
            dislocation_count=args.dislocation_count,
        )
        peierls.run_neb()
        peierls.plot_barrier(compound=formula)


if __name__ == "__main__":
    main()
