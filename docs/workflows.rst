Workflows
=========

BasicProperties
---------------

``BasicProperties`` is the main high-level workflow entry point. It can be
initialized either from a registered model name or from an existing
ASE-compatible calculator.

Typical sequence
----------------

A common single-element workflow is:

1. build a bulk BCC cell
2. relax or fit the equilibrium lattice constant
3. compute elastic constants
4. compute point-defect properties
5. compute surface or stacking-fault properties
6. run screw-dislocation or traction-separation workflows if needed

Example
-------

.. code-block:: python

   import numpy as np
   from ase.build import bulk
   from atomdefectkit import BasicProperties

   workflow = BasicProperties(
       model_name="mace",
       model_parameters={"model_task": "omat_pbe", "default_dtype": "float64"},
       device="cuda",
       working_dir="Test_V_mace",
   )

   atoms = bulk("V", "bcc", a=2.997, cubic=True)

   a0_relax = workflow.calculate_equilibrium_a0_relax(atoms.copy(), fmax=0.001)
   volumes, energies, a0_fit = workflow.calculate_equilibrium_a0_birch_murnaghan(
       atoms.copy(),
       vol_range=np.linspace(0.95, 1.05, 40),
   )

   atoms.set_cell([a0_fit] * 3, scale_atoms=True)
   cij = workflow.calculate_elastic_constants(atoms.copy(), verbose=False)
   vacancy_energy = workflow.calculate_vacancy_formation_energy(atoms.copy())

Important sub-workflows
-----------------------

Stacking fault
^^^^^^^^^^^^^^

Create from ``BasicProperties.create_stacking_fault_workflow()`` to reuse the
same calculator and working directory conventions.

Traction separation
^^^^^^^^^^^^^^^^^^^

Create from ``BasicProperties.create_traction_separation_workflow()``.

This workflow is intended for opening-mode slab separation calculations and
plotting traction-separation curves for surfaces such as ``(100)`` and
``(110)``.

Screw dislocation
^^^^^^^^^^^^^^^^^

Create from ``BasicProperties.create_screw_dislocation_workflow()``.

The screw-dislocation workflow expects a physically reasonable cubic elastic
tensor. The package now checks the cubic stability conditions before building
the dislocation object:

* ``C11 - C12 > 0``
* ``C11 + 2*C12 > 0``
* ``C44 > 0``

If these conditions fail, the workflow raises a clear error and the example
scripts skip the screw-dislocation section instead of crashing the entire run.

NEB
^^^

The NEB helpers are used for the screw-dislocation Peierls barrier workflow and
write optimizer progress to log files in the chosen working directory.

Working directories and outputs
-------------------------------

Most workflows inherit from a shared path helper so they can write logs, plots,
and JSON outputs under a consistent working directory. This matters especially
for long HPC runs, where you want output files to be grouped by model and
element.
