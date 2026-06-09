import argparse

from ase.build import add_adsorbate, fcc100, molecule
from ase.optimize import LBFGS

from atomdefectkit import load_model
from atomdefectkit.utils.progress import progress


parser = argparse.ArgumentParser(description="Run a simple OCP/OCPCalculator adsorption example.")
parser.add_argument(
    "--model-name",
    default="EquiformerV2-31M-S2EF-OC20-All+MD",
    help="OCP pretrained model name, or one of the EqV3 direct-checkpoint aliases.",
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
args = parser.parse_args()

progress(f"Loading OCP model {args.model_name}")
calc = load_model(
    "ocp",
    {
        "model_name": args.model_name,
        "local_cache": args.local_cache,
        "cpu": args.cpu,
    },
    device="cpu" if args.cpu else "cuda",
)

progress("Building Cu(100)+CO adsorption system")
slab = fcc100("Cu", (3, 3, 3), vacuum=8)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")
slab.calc = calc

progress("Running LBFGS relaxation")
opt = LBFGS(slab, logfile="ocp_adsorption_lbfgs.log")
opt.run(fmax=0.05, steps=200)

progress(f"Final energy: {slab.get_potential_energy():.6f} eV")
