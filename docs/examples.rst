Examples and scripts
====================

Benchmark driver
----------------

``scripts/run_benchmark.py`` runs the full BCC defect benchmark for any
registered model backend. It calls ``atomdefectkit.benchmarks.run_bcc_benchmark``,
which performs:

* lattice-constant relaxation and EOS fitting
* elastic constants
* vacancy and interstitial formation energies
* surface energies
* phonon dispersion
* stacking fault calculations
* traction-separation calculations
* screw-dislocation and NEB calculations, when the elastic tensor is stable

Each backend has a parameter preset (``MODEL_PRESETS`` in the script). Use
``--model-params`` to override it, ``--steps`` to run a subset, and a
comma-separated ``--element`` list to loop over the project BCC set
(``V``, ``Nb``, ``Ta``, ``Mo``, ``W``).

Running from a repository checkout
----------------------------------

.. code-block:: bash

   uv sync --extra mace
   uv run --extra mace python scripts/run_benchmark.py --model mace --element V --initial-a0 2.997

   # All five BCC elements with UPET
   uv run --extra upet python scripts/run_benchmark.py --model upet --element V,Nb,Ta,Mo,W

   # A local ACE potential, basic properties and screw dislocation only
   uv run --extra pace python scripts/run_benchmark.py --model pace \
     --model-params '{"potential_file": "/path/to/output_potential.yaml"}' \
     --label ACE2025 --element Nb --steps basic,screw

Running from outside the repository
-----------------------------------

.. code-block:: bash

   uv run --project /path/to/AtomDefectKit --extra upet \
     python /path/to/AtomDefectKit/scripts/run_benchmark.py --model upet --element V,Nb,Ta,Mo,W

SLURM usage
-----------

The repository includes ``scripts/slurm_run_model_test.sh`` as a starting point
for cluster submission. ``MODEL`` and ``ELEMENTS`` select the run:

.. code-block:: bash

   sbatch --export=ALL,MODEL=upet scripts/slurm_run_model_test.sh

   sbatch --export=ALL,MODEL=mace,ELEMENTS=V scripts/slurm_run_model_test.sh

Documentation-local example
---------------------------

A minimal programmatic usage pattern is:

.. code-block:: python

   import atomdefectkit
   from atomdefectkit import BasicProperties

   calc = atomdefectkit.load_model(
       "pace",
       {"potential_file": "/path/to/potential.yaml"},
   )

   workflow = BasicProperties(calculator=calc, working_dir="Test_Nb_pace")
