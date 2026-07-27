Contributing
============

Contributions to DSF are welcome. We appreciate code contributions, bug
reports, feature requests, documentation improvements, tests, and scientific
feedback.

Development takes place in the
`DSF GitHub repository <https://github.com/LSSTDESC/dsf>`__.

Before starting substantial development, please open an issue or contact the
project maintainers so that the proposed work can be discussed and coordinated.

Submitting issues and pull requests
-----------------------------------

Bug reports and feature requests can be submitted through the
`GitHub issue tracker <https://github.com/LSSTDESC/dsf/issues>`__.

Bug reports should include:

- the expected behaviour,
- the actual behaviour,
- the DSF version or commit used,
- any relevant environment details,
- a minimal example where possible.

Code contributions can be submitted through a
`GitHub pull request <https://github.com/LSSTDESC/dsf/compare>`__.

Pull requests should explain what was changed, why the change is needed, and,
where applicable, reference the corresponding issue. New functionality must
include appropriate tests and documentation.

Development workflows
---------------------

DSF uses `tox <https://tox.wiki>`__ to manage its development workflows.

Install DSF together with the development dependencies from the repository
root:

.. code-block:: console

   pip install --group dev -e .

Run all configured workflows with:

.. code-block:: console

   tox

Testing
-------

Tests are located in the ``tests`` directory.
New functionality and bug fixes must include tests covering the added or
changed behaviour.

Run the test suite for all supported Python versions with:

.. code-block:: console

   tox -m test

Run the tests for a specific Python version with, for example:

.. code-block:: console

   tox -e py312

Arguments can be passed to ``pytest`` after ``--``. For example:

.. code-block:: console

   tox -e py312 -- tests/test_example.py

The coverage workflow can be run with:

.. code-block:: console

   tox -e cov

Linting
-------

DSF uses `Ruff <https://docs.astral.sh/ruff/>`__ for linting and code-quality
checks.

Run the linting workflow with:

.. code-block:: console

   tox -e lint

Automatically fix supported linting errors with:

.. code-block:: console

   tox -e lint -- --fix

Code style
----------

Code must follow PEP 8 and the general docstring conventions in PEP 257.

All Python modules must include a module-level docstring describing their
purpose. Public classes, functions, and methods must include clear, concise
Google-style docstrings.

Docstrings should describe the purpose and public interface of a module,
class, or function, rather than its implementation details.
They should document parameters, return values, raised exceptions, and any
important scientific or numerical assumptions where applicable.

Documentation
-------------

The documentation is written in reStructuredText and built with Sphinx.

Build the documentation with:

.. code-block:: console

   tox -e docs

On macOS, build the documentation, run doctests, and open the resulting pages
in a browser with:

.. code-block:: console

   tox -e do

Build the documentation for the main branch and release versions with:

.. code-block:: console

   tox -e docs-releases

Generated documentation is written to ``docs/_build``. New documentation pages
should also be added to the appropriate Sphinx ``toctree``.

Development team
----------------

DSF is developed collaboratively within the LSST Dark Energy Science
Collaboration.

Project maintainers
~~~~~~~~~~~~~~~~~~~

- Niko Šarčević
- Ben Levine

Contributors
~~~~~~~~~~~~

- Dani Leonard
- Navin Chaurasiya
- Carlos Garcia-Garcia

This list reflects the current development team and may evolve as DSF
continues to grow. Contributions are also recorded through the repository
history and acknowledged in software releases.