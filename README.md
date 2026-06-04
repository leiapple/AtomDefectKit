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

This project uses `uv` for Python and dependency management. The package requires Python 3.11. If the system Python is not suitable, let `uv` install and pin Python:

```bash
uv python install 3.11
uv python pin 3.11
```

Install the base environment with:

```bash
uv sync
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

For PACE-backed workflows, sync the PACE extra:

```bash
uv sync --extra pace
uv run --extra pace python scripts/run_tests_pace.py
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

### Notes

This first migration pass keeps the code close to the existing source projects so functionality is preserved while the package structure becomes cleaner. A follow-up cleanup pass should standardize APIs, add tests, and separate reusable library code from workflow scripts more sharply.
