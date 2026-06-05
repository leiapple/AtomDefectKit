Getting started
===============

Installation
------------

AtomDefectKit targets Python 3.11, 3.12, and 3.13. The optional ``pace``
backend should currently be treated as Python 3.11 only.

For repository development with ``uv``:

.. code-block:: bash

   uv sync

Install a single backend extra at a time:

.. code-block:: bash

   uv sync --extra mace
   uv run --extra mace python scripts/run_tests_mace.py

   uv sync --extra upet
   uv run --extra upet python scripts/run_tests_upet.py

For a clean installation directly from GitHub:

.. code-block:: bash

   uv venv --python 3.13
   source .venv/bin/activate
   uv pip install "atomdefectkit[mace] @ git+https://github.com/leiapple/AtomDefectKit.git@main"

Quick example
-------------

.. code-block:: python

   import atomdefectkit
   from atomdefectkit.basic_properties import BasicProperties

   calc = atomdefectkit.load_model(
       "pace",
       {"potential_file": "/path/to/potential.yaml"},
   )
   workflow = BasicProperties(calculator=calc)

Available model loaders
-----------------------

Inspect bundled model loaders programmatically:

.. code-block:: python

   import atomdefectkit

   print(atomdefectkit.available_models())
   print(atomdefectkit.available_model_metadata())

Notes
-----

The package bundles reference DFT JSON files used by
``BasicProperties.plot_comparison()``, so comparison plots work when the
package is installed outside the source tree as well.
