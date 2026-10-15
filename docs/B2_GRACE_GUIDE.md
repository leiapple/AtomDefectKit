# AtomDefectKit: B2 Materials with GRACE Foundation Models

This guide describes the B2 extension in AtomDefectKit. It covers installation, ordered bulk calculations, generalized stacking-fault lines, plotting, and Peierls calculations from supplied dislocation endpoints.

Local checkout:

```text
/Users/leizhang/Nextcloud/5_Coding/G_github/ASDbcc
```

Use a checkout containing `src/atomdefectkit/b2.py` and `scripts/run_b2_grace.py`. The name `ASDbcc` is the local directory name in these examples; the Python package is `atomdefectkit`.

## 1. Scope and Current Status

The B2 workflow constructs an ordered two-atom CsCl-type cell with species A at `(0, 0, 0)` and species B at `(1/2, 1/2, 1/2)`. It retains an integer `b2_sublattice` array when repeating the cell.

The command-line driver computes:

| Calculation | Current behavior |
| --- | --- |
| Bulk EOS | 13 energy evaluations using lattice scale factors from 0.97 to 1.03; Birch-Murnaghan fit |
| Equilibrium lattice constant | Cubic cell edge obtained from fitted equilibrium volume |
| Elastic constants | Cubic 6 by 6 stiffness matrix in GPa |
| Gamma lines | `{011}<100>` and `{1-10}<111>` paths with individual and combined figures |
| Peierls barrier | Optional NEB using supplied initial/final dislocation structures |
| Peierls stress | Python helper using a sinusoidal approximation; not automatically invoked by the driver |

The current driver does not build B2 dislocation endpoints or calculate quadrupole core energies, elastic corrections, CRSS, finite-temperature properties, antisites, vacancies, or non-stoichiometric configurations. Gamma lines are one-dimensional paths, not complete two-dimensional gamma surfaces.

**Gamma-energy units:** the shared `StackingFaultWorkflow` returns J/m² and writes individual-path outputs in mJ/m². `B2Workflow.calculate_gamma_line()` converts to eV/Å²; its combined plots and CSV report meV/Å². The conversion is explicit at the B2 API boundary. Earlier versions mislabeled converted values as eV/Å²; recalculate or correct those historical outputs.

## 2. Install in the Existing Local Checkout

Use a dedicated environment for GRACE. The package declares Python 3.11 through 3.13 support; Python 3.11 is the starting choice for these commands.

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Install the GRACE extra from the local project:

```bash
cd /Users/leizhang/Nextcloud/5_Coding/G_github/ASDbcc
uv python install 3.11
uv sync --python 3.11 --extra grace
```

This installs AtomDefectKit and the `tensorpotential` backend together with the resolved dependencies. The project treats model extras as mutually exclusive, so synchronize `grace` alone in this environment.

Confirm imports and the script interface:

```bash
uv run --extra grace python -c "from atomdefectkit import B2Crystal, B2Workflow; import tensorpotential; print('B2 and GRACE imports OK')"
uv run --extra grace python scripts/run_b2_grace.py --help
```

Always retain `--extra grace` when using `uv run`; otherwise an automatic environment synchronization may remove optional backend packages.

For another machine, first obtain a checkout containing the B2 additions:

```bash
git clone https://github.com/leiapple/AtomDefectKit.git
cd AtomDefectKit
uv sync --python 3.11 --extra grace
```

Check that `src/atomdefectkit/b2.py` exists and record the Git revision used for each run.

## 3. Download and Select GRACE Models

List models offered by the installed TensorPotential version:

```bash
uv run --extra grace grace_models list
uv run --extra grace grace_models download GRACE-1L-OMAT
```

The first model load can require network access and a download. GRACE stores foundation models under `~/.cache/grace` by default. Current upstream documentation supports `GRACE_CACHE` for choosing a different directory; verify this against your installed TensorPotential version before using it on compute nodes.

The B2 command-line driver selects `GRACE-1L-OMAT` by default through its exact model name:

```python
model_parameters={"model_name": "GRACE-1L-OMAT"}
```

The Python API also accepts an exact name:

```python
model_parameters={"model_name": "GRACE-1L-OMAT-medium-ft-AM"}
```

Examples accepted by the local wrapper include `GRACE-1L-OMAT`, `GRACE-2L-OMAT`, `GRACE-1L-OMAT-medium-ft-AM`, and `GRACE-2L-OMAT-large-ft-AM`. Inspect the actual wrapper list:

```bash
uv run --extra grace python -c "from atomdefectkit.models.grace import GRACE_MODEL_CHOICES; print('\n'.join(GRACE_MODEL_CHOICES))"
```

Availability in the wrapper and availability in the installed backend must both agree. The driver accepts `--model-name`, for example `--model-name GRACE-2L-OMAT`. Record the exact model name, backend version, precision, and downloaded model identity for each benchmark.

## 4. First Calculation: Bulk NiAl

Start with the bulk-only calculation to verify that model loading, energies, and stresses work:

```bash
uv run --extra grace python scripts/run_b2_grace.py Ni Al \
    --a0 2.88 \
    --device cpu \
    --skip-gamma \
    --working-dir B2_NiAl_grace_bulk
```

Here `2.88` Å is an initial lattice-constant guess, not a calculated or validated result. The two positional arguments define the ordered sublattices.

The EOS scan rescales the lattice vectors. Thus a scale factor `s` changes the volume by `s**3`; it is not a fractional volume change. Inspect the energy-volume data and confirm the minimum lies inside the sampled interval. Widen the scan through the Python API if necessary.

The bulk output is `B2_NiAl_grace_bulk/bulk_results.json`:

```text
formula
lattice_constant_A
volumes_A3
energies_eV
elastic_constants_GPa
```

Energies are total energies of the two-atom cell, and volumes are cell volumes. The elastic matrix is in Voigt notation `(xx, yy, zz, yz, xz, xy)`. Extract `C11=C[0,0]`, `C12=C[0,1]`, and `C44=C[3,3]`.

For a cubic material, check `C11-C12 > 0`, `C11+2*C12 > 0`, and `C44 > 0`. These are mechanical-stability checks, not proof of phase stability. The bulk modulus is `(C11+2*C12)/3`, and the Zener anisotropy is `2*C44/(C11-C12)`.

## 5. Calculate and Plot Gamma Lines

Run the default workflow without `--skip-gamma`:

```bash
uv run --extra grace python scripts/run_b2_grace.py Ni Al \
    --a0 2.88 \
    --device cpu \
    --working-dir B2_NiAl_grace
```

This first computes bulk properties, then calculates two slip paths:

- `(011)[100]`: full translation `a[100]`.
- `(1-10)[111]`: full translation `a[111]`.

For the second path, reaction coordinate 0.5 corresponds to `a/2[111]`, which exchanges the two chemical sublattices across the fault. Its energy is an APB-related quantity for this specific fault construction; verify the plane termination and fault location before treating it as an APB benchmark.

Defaults are 24 atomic layers, 20 sliding increments (21 sampled structures), a force tolerance of 0.01 eV/Å, and up to 1000 optimization steps per shifted structure. Atomic relaxation is restricted to the fault-plane normal. The inherited method adds 10 Å vacuum on each side along the slab normal.

Outputs include individual path plots and trajectories plus:

```text
B2_NiAl_grace/
    bulk_results.json
    gamma_lines/
        b2_gamma_lines.png
        b2_gamma_lines.csv
        <sanitized-path-name>/
            <formula>_<plane>_stacking_fault.png
            <formula>_<plane>_stacking_fault.out
            <formula>_<plane>_StackingFault.xyz
            stacking_fault_step_<index>.log
```

The combined CSV stores `slip_system`, `reaction_coordinate`, and `energy_meV_per_A2`. These gamma energies are in meV/Å² with the October 15 unit correction. The horizontal coordinate is the fraction of the specified translation, so different paths have different physical translation lengths.

The maximum sampled gamma value gives a sampled unstable-fault energy. Individual path plots, data and summary text use mJ/m²; the combined B2 plot and CSV use meV/Å². The driver does not write a separately named APB summary field. Confirm convergence with respect to layers, vacuum, sliding resolution, force tolerance, and model choice.

## 6. Use the Python API for Custom Paths and Models

Execute this example in a Python script or notebook using the GRACE environment:

```python
import numpy as np
from atomdefectkit import B2Crystal, B2SlipSystem, B2Workflow

workflow = B2Workflow(
    B2Crystal(("Ni", "Al"), lattice_constant=2.88),
    model_name="grace",
    model_parameters={"model_name": "GRACE-1L-OMAT"},
    device="cpu",
    working_dir="B2_NiAl_custom",
)

bulk = workflow.build_bulk()
supercell = workflow.build_bulk(repetitions=(4, 4, 4))
volumes, energies, a0 = workflow.calculate_equilibrium_lattice_constant(
    volume_scales=np.linspace(0.95, 1.05, 21),
)
cij = workflow.calculate_elastic_constants(a0, verbose=False)

paths = [
    B2SlipSystem((0, 1, 1), (1, 0, 0), name="011_100"),
    B2SlipSystem((1, -1, 0), (1, 1, 1), name="1-10_111"),
]
gamma_lines = workflow.calculate_gamma_lines(
    paths,
    lattice_constant=a0,
    layers=32,
    num_steps=40,
    fmax=0.005,
    steps=2000,
    fix_normal=False,
    write_xyz=True,
)
```

`fix_normal=False` permits normal-only relaxation. `fix_normal=True` instead fixes normal motion and permits in-plane relaxation; it is a different constraint and can relax the imposed sliding displacement. Use normal-only relaxation for the initial gamma benchmark unless a different constraint is deliberately required.

The API returns arrays and a gamma-line dictionary. Bulk JSON serialization is performed by the command-line driver, not automatically by each API method. `calculate_gamma_lines()` saves the combined plot and CSV. To replot already available arrays, call `workflow.plot_gamma_lines(gamma_lines)`.

`translation_fraction` multiplies `a * direction`. Integer direction `(1, 1, 1)` with fraction 1 gives `a[111]`; fraction 0.5 gives `a/2[111]`. The slip direction must lie in the specified plane. Use a unique, filesystem-friendly name for every path.

To calculate the transverse path through the APB used in Fig. 2c of La Rosa et al., specify its nonzero initial translation:

```python
transverse = workflow.calculate_gamma_line(
    B2SlipSystem((1, 1, 0), (-1, 1, 0), name="110_transverse"),
    lattice_constant=a0,
    initial_translation=(0.0, 0.0, 0.5),
    layers=24,
    num_steps=20,
)
```

`initial_translation` is in cubic lattice coordinates, in units of the lattice constant. This samples `0.5[001] + xi[-110]`, with the APB at `xi=0.5`. Its energies remain relative to a separately relaxed unfaulted slab. The plotting method preserves that reference instead of subtracting the nonzero first point. This path is available through the Python API; the default command-line driver still runs only its original two paths.

## 7. Run CoTi and FeAl

The same driver supports another ordered binary pair:

```bash
uv run --extra grace python scripts/run_b2_grace.py Co Ti \
    --a0 3.00 --device cpu --working-dir B2_CoTi_grace

uv run --extra grace python scripts/run_b2_grace.py Fe Al \
    --a0 2.90 --device cpu --working-dir B2_FeAl_grace
```

These lattice values are starting guesses. Use the same GRACE model, calculation constraints, convergence criteria, and energy conventions for all three materials. Benchmark against Lorenzo's reference structures and calculation settings. For FeAl, assess how the model's reference magnetic states affect the comparison; this workflow does not set or solve magnetic order explicitly.

## 8. Peierls Barrier from Supplied Endpoints

Supply relaxed initial and final configurations with identical atom counts, chemical symbols in identical index order, cells, and boundary conditions. They must describe the same defect configuration translated between the chosen core positions while preserving the chemical decoration.

For a cell containing two equivalent dislocations migrating together:

```bash
uv run --extra grace python scripts/run_b2_grace.py Ni Al \
    --a0 2.88 --device cpu --skip-gamma \
    --working-dir B2_NiAl_peierls \
    --peierls-initial initial.xyz \
    --peierls-final final.xyz \
    --neb-replicas 11 \
    --dislocation-count 2
```

Use a count of four only if the cell actually contains four equivalent migrating dislocations. The phrase "quadrupole configuration" alone does not determine the count. Dividing the total energy by the count assumes equivalent contributions; inspect the individual core trajectories when this assumption is uncertain.

The driver expects relaxed endpoints and does not relax them before NEB. It uses climbing-image NEB with FIRE, a default force threshold of 0.001 eV/Å, a maximum of 10000 steps, and a spring constant of 0.5 in ASE units.

For controlled fixed-cell calculations through the API:

```python
from ase.io import read

peierls = workflow.create_peierls_workflow(
    read("initial.xyz"),
    read("final.xyz"),
    model_label="GRACE-1L-OMAT",
    replicas=11,
    optimizer="FIRE",
    dislocation_count=2,
)
images = peierls.run_neb(
    fmax=0.005,
    steps=5000,
    spring_constant=0.1,
    require_same_cell=True,
    preserve_constraints=True,
)
fig, ax, data = peierls.plot_barrier(compound="NiAl")
```

The inherited general NEB interface can rescale a final cell to match the initial one when `require_same_cell=False`. Its default interpolation also removes mean translation and drops atomic constraints. Use the explicit settings above when fixed boundary atoms and identical cells are part of the intended experiment, and inspect the interpolated images.

The B2 barrier CSV contains total cell energies and relative energies per dislocation. These are eV, not eV per unit dislocation length. CSV, plot, NEB trajectory, and optimizer log are written under `peierls_barrier/screw_disloc/` because the inherited workflow appends that subdirectory.

Verify optimizer convergence, endpoint energy differences, image resolution, retained core type, and displacement of each dislocation. A barrier calculation does not establish CRSS under a finite-temperature loading protocol.

## 9. Estimate Peierls Stress

For an energy landscape per unit dislocation length `U(x)=A[1-cos(2*pi*x/lambda)]`, the sinusoidal estimate is:

```text
tau_P = 2*pi*A/(b*lambda)
```

`A` is the sinusoidal amplitude, `b` the Burgers-vector magnitude, and `lambda` the spatial period of core migration. The full peak-to-valley barrier is `2*A`. The migration period need not equal `b`.

Example using illustrative values, not a NiAl prediction:

```python
from atomdefectkit import estimate_sinusoidal_peierls_stress

tau_gpa = estimate_sinusoidal_peierls_stress(
    peierls_energy_ev=0.10,
    line_length_angstrom=10.0,
    burgers_angstrom=2.88,
    migration_period_angstrom=2.88,
    dislocation_count=2,
    energy_definition="peak_to_valley",
)
print(tau_gpa)
```

Pass the total cell barrier with its dislocation count. If passing the already normalized per-dislocation barrier from the CSV, use `dislocation_count=1` to avoid dividing twice. The line length is the physical length of each dislocation represented in the cell. The helper returns GPa.

This estimate assumes a sinusoidal landscape. Compare it against the derivative of a resolved energy-versus-core-position curve and selected explicit zero-temperature stress calculations before using it to rank competing dislocations. A generic NEB reaction coordinate does not directly provide physical core position.

## 10. Snellius or Another Linux GPU System

Create a persistent GRACE environment from the updated project checkout:

```bash
cd "$HOME/AtomDefectKit"
export ADKIT_ENV_DIR="/scratch-shared/$USER/adkit-envs/grace"
export UV_CACHE_DIR="/scratch-shared/$USER/uv-cache"
export UV_LINK_MODE=copy
uv python install 3.11
UV_PROJECT_ENVIRONMENT="$ADKIT_ENV_DIR" uv sync --python 3.11 --extra grace --no-dev
```

Download the selected model on a node with network access and retain access to its cache during the calculation. Verify the installed backend's `GRACE_CACHE` support if moving the model cache to scratch.

Check TensorFlow GPU visibility within a GPU allocation:

```bash
"$ADKIT_ENV_DIR/bin/python" -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

A GPU run requires a TensorFlow/CUDA environment compatible with the node and its drivers. `--device cuda` selects the requested backend device; it does not install CUDA dependencies. Consult the upstream installation guide and the cluster's current module instructions. AtomDefectKit may fall back to CPU, so check the calculation log as well as GPU visibility.

Example SLURM template; replace the partition and account placeholders with values for your allocation:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=b2-nial
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=b2-nial-%j.out
set -euo pipefail

export ADKIT_ENV_DIR="/scratch-shared/$USER/adkit-envs/grace"
export MPLCONFIGDIR="/scratch-shared/$USER/matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$HOME/AtomDefectKit"

"$ADKIT_ENV_DIR/bin/python" scripts/run_b2_grace.py Ni Al \
    --a0 2.88 --device cuda \
    --working-dir "/scratch-shared/$USER/B2_NiAl_grace_${SLURM_JOB_ID}"
```

Start with `--skip-gamma` for an environment check. Measure actual runtime and memory before setting production resource requests. The template is not a verified Snellius partition configuration.

## 11. Troubleshooting and Reproducibility

- **Cannot import B2 classes:** confirm that this is the updated checkout and that `atomdefectkit.__file__` points to it.
- **Cannot import TensorPotential:** synchronize and run with `--extra grace`.
- **Unknown model:** compare the local wrapper list with `grace_models list` for the installed backend.
- **Model download fails:** download before submitting a job, then check cache access from the compute node.
- **GPU list is empty:** inspect the TensorFlow installation and cluster CUDA environment; use CPU for the initial functional check.
- **Keras compatibility error:** upstream documents `TF_USE_LEGACY_KERAS=1` for configurations that require the legacy Keras API; apply it before importing TensorFlow if this issue occurs.
- **EOS minimum is outside the scan:** widen the lattice scale range and inspect the fitted curve.
- **Negative elastic stability combinations:** investigate phase/model validity before constructing dislocations.
- **Gamma endpoint is not near its initial energy:** check the ordered-lattice translation, chemistry, fault-plane construction, relaxation convergence, and units.
- **NEB endpoints differ in chemistry or atom order:** repair the structure correspondence before interpolation.
- **Optimizer reaches its step limit:** inspect the log and final forces; the current driver does not issue a separate convergence certificate.

Archive the input structures, model identity, `uv.lock`, package versions, parameters, optimizer logs, and raw outputs. In a dedicated GRACE environment, record installed dependencies with:

```bash
uv pip freeze
git rev-parse HEAD
git status --short
```

The Git revision alone does not capture uncommitted B2 changes. Preserve the patch or publish the exact tested version before transferring the workflow to a collaborator.

## 12. Sources

- [AtomDefectKit repository](https://github.com/leiapple/AtomDefectKit)
- [AtomDefectKit documentation](https://atomdefectkit.readthedocs.io/en/latest/)
- [GRACE installation instructions](https://github.com/ICAMS/grace-tensorpotential/blob/master/docs/gracemaker/install.md)
- [GRACE foundation-model tutorial](https://github.com/ICAMS/grace-tensorpotential/blob/master/docs/gracemaker/tutorials.md)
- [GRACE foundation models and cache configuration](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/blob/main/README.md)

The AtomDefectKit command descriptions above were checked against the local B2 implementation. Upstream GRACE installation and model-cache details were checked against its official documentation. Full GRACE material calculations and GPU installation were not executed while preparing this guide.
