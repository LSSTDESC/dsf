.. |dsf-logo| image:: _static/dsf_logo.svg
   :alt: DSF logo
   :height: 36px
   :class: inline-dsf-logo

|dsf-logo| Installation
=======================

This page describes how to install DSF from a local checkout,
configure a Python environment, and verify the installation.

Getting Started
---------------

Clone the repository
~~~~~~~~~~~~~~~~~~~~

Clone the DSF repository and navigate to its root directory:

.. code-block:: console

   git clone https://github.com/LSSTDESC/dsf.git
   cd dsf

Set up your environment
~~~~~~~~~~~~~~~~~~~~~~~

Activate the Python environment used for your cosmology and forecasting
work. For example, with conda:

.. code-block:: console

   conda activate <your-env-name>

Install DSF
~~~~~~~~~~~

Install DSF in editable mode:

.. code-block:: console

   python -m pip install -e .

Editable installation allows changes made in ``src/dsf`` to take effect
without reinstalling the package.

Verify the installation
~~~~~~~~~~~~~~~~~~~~~~~

Verify that Python can import DSF:

.. code-block:: console

   python -c "import dsf; print(dsf.__file__)"

For an editable installation, the printed path should point to
``src/dsf`` inside your local repository.

Development
-----------

Install development dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development work, install DSF together with its development
dependencies from the repository root:

.. code-block:: console

   python -m pip install --group dev -e .

Development workflow
~~~~~~~~~~~~~~~~~~~~

Development should take place on a dedicated Git branch rather
than directly on ``main``.

Before starting substantial changes, open or find a GitHub issue
describing the proposed work.

For example, create a documentation branch with:

.. code-block:: console

   git switch -c docs/installation-page

Prefer small, focused pull requests that are easy to review.

Testing and documentation
~~~~~~~~~~~~~~~~~~~~~~~~~

DSF uses ``tox`` to manage testing, linting, and documentation
workflows.

List the available environments:

.. code-block:: console

   tox list

Run the test suite:

.. code-block:: console

   tox -m test

Run linting checks:

.. code-block:: console

   tox -e lint

Build the documentation:

.. code-block:: console

   tox -e docs

For detailed development guidelines, including code style,
testing, documentation, and pull requests, see
:doc:`contributing`.
