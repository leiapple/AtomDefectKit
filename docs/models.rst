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

``mace``
   Often benefits from ``default_dtype="float64"`` for geometry optimization
   and elastic-property calculations.

``7net``, ``fairchem``, ``grace``, ``upet``
   Have dedicated example scripts in ``scripts/`` and batch wrappers for the
   BCC element set used in the project.
