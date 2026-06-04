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

If you want to run PACE-backed workflows, sync the optional dependency first:

```bash
uv sync --extra pace
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

Model backend extras are resolved as mutually exclusive environments because the upstream MLIP stacks pin incompatible low-level packages. Install one backend extra at a time, for example `uv sync --extra mace`.

Inspect available loaders programmatically:

```python
import atomdefectkit

print(atomdefectkit.available_models())
print(atomdefectkit.available_model_metadata())
```

### Notes

This first migration pass keeps the code close to the existing source projects so functionality is preserved while the package structure becomes cleaner. A follow-up cleanup pass should standardize APIs, add tests, and separate reusable library code from workflow scripts more sharply.
