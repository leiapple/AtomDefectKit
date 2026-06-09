Installation and environments
=============================

Python versions
---------------

The core package is tested on Python 3.11, 3.12, and 3.13.

The optional ``pace`` backend should currently be treated as Python 3.11 only,
because its upstream ``pyace`` dependency is more restrictive than the rest of
the project.

Why model extras are separated
------------------------------

AtomDefectKit supports several MLIP backends, but those backends often depend
on different versions of NumPy, PyTorch, TensorFlow, CUDA helper packages, or
other compiled scientific libraries. For that reason, the project is designed
so that you install one model backend at a time.

The main optional extras are:

* ``7net``
* ``chgnet``
* ``fairchem``
* ``grace``
* ``mace``
* ``mattersim``
* ``nequip``
* ``nequix``
* ``eqv3``
* ``pace``
* ``upet``

Using uv in a repository checkout
---------------------------------

For development in a cloned repository:

.. code-block:: bash

   uv sync

Then install one backend extra:

.. code-block:: bash

   uv sync --extra mace
   uv run --extra mace python scripts/run_tests_mace.py

   uv sync --extra upet
   uv run --extra upet python scripts/run_tests_upet.py

If you want separate persistent environments for different backends:

.. code-block:: bash

   UV_PROJECT_ENVIRONMENT=.venv-mace uv sync --extra mace
   UV_PROJECT_ENVIRONMENT=.venv-upet uv sync --extra upet

Using uv from outside the repository
------------------------------------

If you call ``uv`` from another directory, ``--extra`` only applies when the
project is explicitly provided:

.. code-block:: bash

   uv sync --project /path/to/AtomDefectKit --extra upet
   uv run --project /path/to/AtomDefectKit --extra upet \
     python /path/to/AtomDefectKit/scripts/run_tests_upet.py

Installing directly from GitHub
-------------------------------

For a clean install outside the repository:

.. code-block:: bash

   uv venv --python 3.13
   source .venv/bin/activate
   uv pip install "atomdefectkit[mace] @ git+https://github.com/leiapple/AtomDefectKit.git@main"

Swap the extra name to install another backend:

.. code-block:: bash

   uv pip install "atomdefectkit[7net] @ git+https://github.com/leiapple/AtomDefectKit.git@main"
   uv pip install "atomdefectkit[fairchem] @ git+https://github.com/leiapple/AtomDefectKit.git@main"
   uv pip install "atomdefectkit[grace] @ git+https://github.com/leiapple/AtomDefectKit.git@main"
   uv pip install "atomdefectkit[eqv3] @ git+https://github.com/leiapple/AtomDefectKit.git@main"
   uv pip install "atomdefectkit[upet] @ git+https://github.com/leiapple/AtomDefectKit.git@main"

The ``eqv3`` extra is aligned with the upstream Equiformer v3 environment
described by Atomic Architects and should currently be treated as Python 3.11
only, with
``torch==2.7.1``, ``torchvision==0.22.1``, ``torchaudio==2.7.1``,
``ase==3.25.0``, ``e3nn==0.5.6``, ``lmdb==1.7.3``, ``numba==0.61.2``,
``numpy==2.2.6``, ``orjson==3.11.1``, ``pandas==2.3.1``,
``pymatgen==2025.6.14``, ``pyyaml==6.0.2``, ``scipy==1.16.1``,
``submitit==1.5.3``, ``tensorboard==2.20.0``, ``timm==0.4.12``,
``torch-geometric``, ``tqdm==4.67.1``, ``wandb==0.21.0``, and the
``fairchem-core`` package from the Atomic Architects ``equiformer_v3``
repository. This extra is packaged in this project for Linux only.

For the closest match to the upstream EqV3 environment, install the
supplemental pinned requirements after syncing the extra:

.. code-block:: bash

   uv sync --extra eqv3
   uv pip install -r requirements/eqv3.txt

The lower-level PyG companion wheels such as ``pyg_lib``,
``torch-scatter``, ``torch-sparse``, ``torch-cluster``, and
``torch-spline-conv`` should still be installed with the upstream wheel-index
commands when required by your platform. If you are installing on a CUDA
system, prefer the exact PyTorch and PyG wheel commands recommended by the
upstream EqV3 environment guide so the binaries match your CUDA runtime:

.. code-block:: bash

   uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
   uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
   uv pip install torch_geometric
   uv pip install "atomdefectkit[eqv3] @ git+https://github.com/leiapple/AtomDefectKit.git@main"
   uv pip install -r requirements/eqv3.txt

If you want to install EqV3 directly from GitHub on HPC, use the following
order:

.. code-block:: bash

   uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
   uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
   uv pip install torch_geometric
   uv pip install "atomdefectkit[eqv3] @ git+https://github.com/leiapple/AtomDefectKit.git@main"
   uv pip install -r requirements/eqv3.txt

For ``pace``:

.. code-block:: bash

   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install "atomdefectkit[pace] @ git+https://github.com/leiapple/AtomDefectKit.git@main"

HPC notes
---------

On HPC systems, it is often useful to put caches on scratch storage:

.. code-block:: bash

   export UV_CACHE_DIR=$SCRATCH/uv-cache
   export HF_HOME=$SCRATCH/huggingface
   export TORCH_HOME=$SCRATCH/torch
   export MPLCONFIGDIR=$SCRATCH/matplotlib

If the package is installed into a clean scratch directory, the bundled
reference DFT JSON files are still available to
``BasicProperties.plot_comparison()`` because they are shipped with the
package.
