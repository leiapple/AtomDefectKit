Model loaders
=============

Overview
--------

AtomDefectKit discovers bundled model loaders from ``atomdefectkit.models`` and
registers them into a model registry at import time.

The public entry points are:

* ``atomdefectkit.available_models()``
* ``atomdefectkit.available_model_metadata()``
* ``atomdefectkit.load_model()``
* ``atomdefectkit.BasicProperties.BasicProperties(model_name=...)``

Listing available models
------------------------

.. code-block:: python

   import atomdefectkit

   print(atomdefectkit.available_models())
   print(atomdefectkit.available_model_metadata())

Creating a calculator directly
------------------------------

.. code-block:: python

   import atomdefectkit

   calc = atomdefectkit.load_model(
       "mace",
       {
           "model_task": "omat_pbe",
           "default_dtype": "float64",
       },
       device="cuda",
   )

Creating a workflow from a model name
-------------------------------------

.. code-block:: python

   from atomdefectkit.BasicProperties import BasicProperties

   workflow = BasicProperties(
       model_name="upet",
       model_parameters={},
       device="cuda",
       working_dir="Test_V_upet",
   )

   calc = workflow.get_calculator()

Using an existing ASE calculator
--------------------------------

If you already have an ASE-compatible calculator, pass it directly and skip the
model registry:

.. code-block:: python

   from atomdefectkit.BasicProperties import BasicProperties

   workflow = BasicProperties(
       calculator=existing_calc,
       working_dir="my_results",
   )

Error handling
--------------

The workflow layer validates model names and surfaces missing dependencies or
invalid model parameters as workflow-level errors. When a CUDA-backed model
fails to initialize because CUDA is unavailable, the package attempts a CPU
fallback and emits a warning.

Backend-specific notes
----------------------

``pace``
   Uses a local potential file and should currently be kept on Python 3.11.

``ocp``
   Wraps the ``OCPCalculator`` flow with automatic checkpoint download to a
   local cache. The ``ocp`` extra is aligned with the upstream Equiformer v3
   environment, should currently be treated as Python 3.11 only, and includes
   the newer PyTorch/PyG stack together with pinned upstream scientific
   packages and the ``fairchem-core`` package from the Atomic Architects
   ``equiformer_v3`` repository. The repository also ships
   ``requirements/ocp-eqv3.txt`` as a supplemental pinned environment file for
   closer reproduction of the upstream EqV3 setup. The backend can load both legacy
   OC20/OC22/ODAC pretrained checkpoints such as
   ``EquiformerV2-31M-S2EF-OC20-All+MD`` and the direct EqV3 aliases such as
   ``eqV3-omat24-gradient``.

``mace``
   Often benefits from ``default_dtype="float64"`` for geometry optimization
   and elastic-property calculations.

``7net``, ``fairchem``, ``grace``, ``upet``
   Have dedicated example scripts in ``scripts/`` and batch wrappers for the
   BCC element set used in the project.
