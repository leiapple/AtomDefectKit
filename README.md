## AtomDefectKit

![AtomDefectKit logo](assets/atomdefectkit_logo_primary.png)

`AtomDefectKit` is a Python package for atomistic defect simulations and analysis, with the current workflows focused on BCC metals, screw dislocations, and related defect physics.

Current migrated components:

- basic PACE-backed property and point-defect calculations
- model loading through a registry-based `models/` package
- screw dislocation setup and DD-map workflows
- NEB-based Peierls barrier workflow
- PACE training-log analysis and benchmark plotting
- stacking-fault and traction-separation curve containers

### Package layout

```text
AtomDefectKit/

```

### PACE calculator note

The atomistic PACE calculator used for BCC potentials is available through the optional `pace` dependency group. It is installed from the upstream `ICAMS/python-ace` repository instead of the unrelated package that is published on PyPI as `pyace`.

### Managing model environments with uv

This project uses `uv` for Python and dependency management. The package supports Python 3.11, 3.12, and 3.13. If the system Python is not suitable, let `uv` install and pin Python:

```bash
uv python install 3.11
uv python pin 3.11
```

The core package and most model backends can target Python 3.11-3.13. The
optional `pace` extra currently depends on `pyace`, which should be treated as
Python 3.11 only until its upstream packaging catches up.

Install the base environment with:

```bash
uv sync
```

When installed from PyPI, the equivalent base install will be:

```bash
pip install atomdefectkit
```

Each MLIP backend is exposed as an optional dependency extra. These extras are intentionally marked as mutually exclusive because the upstream model stacks often require incompatible versions of NumPy, PyTorch, TensorFlow, JAX, or CUDA helper packages. Sync one model backend at a time:

```bash
uv sync --extra fairchem
uv run --extra fairchem python scripts/run_tests_fairchem.py

uv sync --extra mace
uv run --extra mace python scripts/run_tests_mace.py

uv sync --extra 7net
uv run --extra 7net python scripts/run_tests_7net.py

uv sync --extra upet
uv run --extra upet python scripts/run_tests_upet.py

uv sync --extra grace
uv run --extra grace python scripts/run_tests_grace.py
```

From PyPI, install one backend extra in a fresh environment:

```bash
pip install "atomdefectkit[fairchem]"
pip install "atomdefectkit[mace]"
pip install "atomdefectkit[upet]"
```

For PACE-backed workflows, sync the PACE extra:

```bash
uv sync --extra pace
uv run --extra pace python scripts/run_tests_pace.py
```

If you run `uv` from outside the repository, `--extra` has no effect unless you point `uv` at this project. Use `--project` with the path to the cloned repository:

```bash
uv sync --project /path/to/AtomDefectKit --extra upet
uv run --project /path/to/AtomDefectKit --extra upet python /path/to/AtomDefectKit/scripts/run_tests_upet.py
uv run --project /path/to/AtomDefectKit --extra upet python /path/to/AtomDefectKit/scripts/run_tests_upet_bcc_elements.py
```

If you want to keep separate persistent virtual environments for different models, set `UV_PROJECT_ENVIRONMENT` per backend:

```bash
UV_PROJECT_ENVIRONMENT=.venv-fairchem uv sync --extra fairchem
UV_PROJECT_ENVIRONMENT=.venv-fairchem uv run --extra fairchem python scripts/run_tests_fairchem.py

UV_PROJECT_ENVIRONMENT=.venv-mace uv sync --extra mace
UV_PROJECT_ENVIRONMENT=.venv-mace uv run --extra mace python scripts/run_tests_mace.py
```

On HPC systems, it is usually helpful to put caches on scratch storage:

```bash
export UV_CACHE_DIR=$SCRATCH/uv-cache
export HF_HOME=$SCRATCH/huggingface
export TORCH_HOME=$SCRATCH/torch
export MPLCONFIGDIR=$SCRATCH/matplotlib
```

The repository includes a SLURM template for model test workflows:

```bash
# Five-element UPET run: V, Nb, Ta, Mo, W
sbatch --export=ALL,MODEL=upet,RUN_MODE=bcc_elements scripts/slurm_run_model_test.sh

# Single-element MACE run
sbatch --export=ALL,MODEL=mace,RUN_MODE=single,ELEMENT=V,INITIAL_A0=2.997 scripts/slurm_run_model_test.sh
```

Set `PROJECT_DIR=/path/to/AtomDefectKit` in `--export` if you submit the job
from outside the repository.

Load a model calculator:

```python
import atomdefectkit
from atomdefectkit.basic_properties import BasicProperties

calc = atomdefectkit.load_model(
    "pace",
    {"potential_file": "/path/to/potential.yaml"},
)
workflow = BasicProperties(calculator=calc)
```

Available bundled model loaders currently include:

- `7net`
- `chgnet`
- `fairchem`
- `grace`
- `mace`
- `mattersim`
- `nequip`
- `nequix`
- `pace`
- `upet`

Inspect available loaders programmatically:

```python
import atomdefectkit

print(atomdefectkit.available_models())
print(atomdefectkit.available_model_metadata())
```

### Release checklist

Before publishing to PyPI, run the lightweight checks:

```bash
uv lock --check
uv run --group dev pytest
uv build --no-sources
```

Publish releases from a clean tagged commit. PyPI Trusted Publishing through GitHub Actions is preferred; for a manual upload with uv, use:

```bash
uv publish
```

### Notes

This first migration pass keeps the code close to the existing source projects so functionality is preserved while the package structure becomes cleaner. A follow-up cleanup pass should standardize APIs, add tests, and separate reusable library code from workflow scripts more sharply.
