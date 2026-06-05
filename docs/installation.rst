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
   uv pip install "atomdefectkit[upet] @ git+https://github.com/leiapple/AtomDefectKit.git@main"

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
