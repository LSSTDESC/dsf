Installation
============

This page shows how to install DSF from a local checkout.

Clone the repository
--------------------

.. code-block:: bash

   git clone https://github.com/LSSTDESC/dsf.git
   cd dsf

Set up your environment
-----------------------

Activate the Python environment you use for DESC, CCL, and forecasting work.
For example, with conda:

.. code-block:: bash

   conda activate <your-env-name>

Install DSF locally
-------------------

For local development, install DSF in editable mode:

.. code-block:: bash

   python -m pip install -e .

If you want the optional development dependencies, use:

.. code-block:: bash

   python -m pip install -e ".[dev]"

Check the installation
----------------------

Verify that Python imports DSF from your local checkout:

.. code-block:: bash

   python -c "import dsf; print(dsf.__file__)"

The printed path should point to the ``src/dsf`` directory inside your local
repository.

Install DSF together with analysis scripts
------------------------------------------

If you are also working with the companion analysis repository, clone both
repositories side by side:

.. code-block:: bash

   mkdir dsf-project
   cd dsf-project

   git clone https://github.com/LSSTDESC/dsf.git dsf
   git clone https://github.com/LSSTDESC/dsf-analysis.git dsf-analysis

Then install both into the same active Python environment:

.. code-block:: bash

   python -m pip install -e ./dsf
   python -m pip install -e ./dsf-analysis

This setup keeps the modelling package and analysis scripts editable while
sharing one consistent environment.

Development setup
-----------------

For development work, make sure DSF is installed with the optional development
dependencies:

.. code-block:: bash

   python -m pip install -e ".[dev]"

This installs DSF in editable mode, so changes made in ``src/dsf`` are picked up
without reinstalling the package.

Before starting development work, open or find a GitHub issue describing the
change. Development should happen on a branch, not directly on ``main``. Use a
short branch name that describes the work, for example:

.. code-block:: bash

   git checkout -b docs/installation-page

Prefer small, focused pull requests. A single issue can be addressed by several
pull requests if that keeps the review easier.

When opening a pull request, link it to the relevant issue. For example, include
one of the following in the pull request description:

.. code-block:: text

   Closes #12
   Fixes #12
   Related to #12

Run the test suite before opening a pull request:

.. code-block:: bash

   pytest

Run linting checks with:

.. code-block:: bash

   ruff check .

For documentation work, install the documentation dependencies if they are
defined by the package, then build the docs locally from the repository root:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

The local HTML documentation will be written to:

.. code-block:: text

   docs/_build/html/index.html