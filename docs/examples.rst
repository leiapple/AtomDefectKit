Examples and scripts
====================

Single-element test scripts
---------------------------

The repository includes model-specific example scripts under ``scripts/``:

* ``run_tests_7net.py``
* ``run_tests_fairchem.py``
* ``run_tests_grace.py``
* ``run_tests_mace.py``
* ``run_tests_pace.py``
* ``run_tests_upet.py``

These scripts perform a relatively broad workflow:

* lattice-constant relaxation and EOS fitting
* elastic constants
* vacancy and interstitial formation energies
* stacking fault calculations
* traction-separation calculations
* screw-dislocation and NEB calculations, when the elastic tensor is stable

Five-element BCC batch scripts
------------------------------

The repository also includes wrappers that loop over the project BCC set:

* ``V``
* ``Nb``
* ``Ta``
* ``Mo``
* ``W``

Available wrappers include:

* ``run_tests_7net_bcc_elements.py``
* ``run_tests_fairchem_bcc_elements.py``
* ``run_tests_grace_bcc_elements.py``
* ``run_tests_upet_bcc_elements.py``

Running a script from a repository checkout
-------------------------------------------

.. code-block:: bash

   uv sync --extra mace
   uv run --extra mace python scripts/run_tests_mace.py --element V --initial-a0 2.997

Running from outside the repository
-----------------------------------

.. code-block:: bash

   uv run --project /path/to/AtomDefectKit --extra upet \
     python /path/to/AtomDefectKit/scripts/run_tests_upet_bcc_elements.py

SLURM usage
-----------

The repository includes ``scripts/slurm_run_model_test.sh`` as a starting point
for cluster submission. For example:

.. code-block:: bash

   sbatch --export=ALL,MODEL=upet,RUN_MODE=bcc_elements \
     scripts/slurm_run_model_test.sh

   sbatch --export=ALL,MODEL=mace,RUN_MODE=single,ELEMENT=V,INITIAL_A0=2.997 \
     scripts/slurm_run_model_test.sh

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
