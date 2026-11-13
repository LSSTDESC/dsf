.. |dsf-logo| image:: _static/dsf_logo.svg
   :alt: DSF logo
   :height: 36px
   :class: inline-dsf-logo

|dsf-logo| Contributing
=======================

Contributions to DSF are welcome, including code, bug reports, feature
requests, documentation improvements, tests, and scientific feedback.

DSF is developed within the
`LSST Dark Energy Science Collaboration <https://lsstdesc.org/>`_,
with contributions from the wider scientific community.

Development takes place in the
`DSF GitHub repository <https://github.com/LSSTDESC/dsf>`__.

Before starting substantial development, please open an issue or contact
the project maintainers to discuss and coordinate the proposed work.

Getting Involved
----------------

Reporting issues
~~~~~~~~~~~~~~~~

Bug reports and feature requests can be submitted through the
`GitHub issue tracker <https://github.com/LSSTDESC/dsf/issues>`__.

Bug reports should include:

- The expected behaviour.
- The actual behaviour.
- The DSF version or commit used.
- Relevant environment details.
- A minimal reproducible example, where possible.

Submitting pull requests
~~~~~~~~~~~~~~~~~~~~~~~~

Code contributions can be submitted through a
`GitHub pull request <https://github.com/LSSTDESC/dsf/compare>`__.

Pull requests should explain what was changed, why the change is needed,
and, where applicable, reference the corresponding issue.

New functionality should include appropriate tests and documentation.

Development
-----------

Setting up the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install DSF together with its development dependencies from the
repository root:

.. code-block:: console

   pip install --group dev -e .

DSF uses `tox <https://tox.wiki>`__ to manage development workflows.

List the available environments:

.. code-block:: console

   tox list

Run a specific environment:

.. code-block:: console

   tox -e <environment>

Testing
~~~~~~~

Tests are located in the ``tests`` directory.
New functionality and bug fixes should include tests covering the
added or changed behaviour.

Run tests for all supported Python versions:

.. code-block:: console

   tox -m test

Run tests for a specific Python version:

.. code-block:: console

   tox -e py312

Pass arguments directly to ``pytest`` after ``--``:

.. code-block:: console

   tox -e py312 -- tests/test_example.py

Run the coverage workflow:

.. code-block:: console

   tox -e cov

Linting
~~~~~~~

DSF uses `Ruff <https://docs.astral.sh/ruff/>`__ for linting and
code-quality checks.

Run the linting workflow:

.. code-block:: console

   tox -e lint

Automatically fix supported linting errors:

.. code-block:: console

   tox -e lint -- --fix

Code style
~~~~~~~~~~

Code should follow `PEP 8 <https://peps.python.org/pep-0008/>`_
and the general docstring conventions in
`PEP 257 <https://peps.python.org/pep-0257/>`_.

All Python modules should include a module-level docstring describing
their purpose.

Public classes, functions, and methods should include clear, concise
`Google-style docstrings <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.

Docstrings should describe the purpose and public interface rather
than implementation details.

They should document parameters, return values, raised exceptions,
and relevant scientific or numerical assumptions.

Documentation
-------------

Writing documentation
~~~~~~~~~~~~~~~~~~~~~

DSF documentation is written in
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
and built with
`Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_.

New documentation pages should be added to the appropriate Sphinx
``toctree``.

Building documentation
~~~~~~~~~~~~~~~~~~~~~~

Build the documentation with:

.. code-block:: console

   tox -e docs

On macOS, build the documentation, run doctests, and open the
resulting pages in a browser:

.. code-block:: console

   tox -e do

Build the documentation for the main branch and release versions:

.. code-block:: console

   tox -e docs-releases

Generated documentation is written to ``docs/_build``.
