
.. |dsf-logo| image:: _static/dsf_logo.svg
   :alt: DSF logo
   :height: 36px
   :class: inline-dsf-logo

|dsf-logo| Installation
=======================

This page describes how to install DSF, configure a Python environment,
and verify the installation.

Getting Started
---------------

Set up your environment
~~~~~~~~~~~~~~~~~~~~~~~

Activate the Python environment used for your cosmology and forecasting
work. For example, with conda:

.. code-block:: console

   conda activate <your-env-name>

Install DSF
~~~~~~~~~~~

Install the latest released version of DSF from
`PyPI <https://pypi.org/project/lsst-desc-dsf/>`_:

.. code-block:: console

   python -m pip install lsst-desc-dsf

DSF requires Python 3.10 or newer.

Verify the installation
~~~~~~~~~~~~~~~~~~~~~~~

Verify that Python can import DSF:

.. code-block:: console

   python -c "import dsf; print(dsf.__file__)"

The PyPI distribution is named ``lsst-desc-dsf``, while the Python
package is imported as ``dsf``.

Development
-----------

Clone the repository
~~~~~~~~~~~~~~~~~~~~

To work on DSF, clone the repository and navigate to its root directory:

.. code-block:: console

   git clone https://github.com/LSSTDESC/dsf.git
   cd dsf

Install in editable mode
~~~~~~~~~~~~~~~~~~~~~~~~

Install DSF in editable mode:

.. code-block:: console

   python -m pip install -e .

Editable installation allows changes made in ``src/dsf`` to take effect
without reinstalling the package.

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
