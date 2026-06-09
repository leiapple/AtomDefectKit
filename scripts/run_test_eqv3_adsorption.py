import argparse

from ase.build import add_adsorbate, fcc100, molecule
from ase.optimize import LBFGS

from atomdefectkit import load_model
from atomdefectkit.utils.progress import progress


parser = argparse.ArgumentParser(description="Run a simple EqV3/OCPCalculator adsorption example.")
parser.add_argument(
    "--model-name",
    default="eqV3-omat24-gradient",
    help="EqV3 pretrained model name, or one of the EqV3 direct-checkpoint aliases.",
)
parser.add_argument(
    "--local-cache",
    default="pretrained_models",
    help="Directory used to cache the downloaded checkpoint.",
)
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Force CPU execution.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed passed through to OCPCalculator.",
)
args = parser.parse_args()

progress(f"Loading EqV3 model {args.model_name}")
calc = load_model(
    "eqV3",
    {
        "model_name": args.model_name,
        "local_cache": args.local_cache,
        "cpu": args.cpu,
        "seed": args.seed,
    },
    device="cpu" if args.cpu else "cuda",
)

progress("Building Cu(100)+CO adsorption system")
slab = fcc100("Cu", (3, 3, 3), vacuum=8)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")
slab.calc = calc

progress("Running LBFGS relaxation")
opt = LBFGS(slab, logfile="eqv3_adsorption_lbfgs.log")
opt.run(fmax=0.05, steps=200)

progress(f"Final energy: {slab.get_potential_energy():.6f} eV")
