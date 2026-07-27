DSF
===

Delta Sigma forecasting tools for LSST DESC galaxy-galaxy lensing analyses.

DSF is a modelling package for projected galaxy-galaxy lensing observables,
with a focus on :math:`\Delta\Sigma` data vectors, covariance ingredients,
tomography choices, and LSST DESC forecasting workflows.

The primary observable modelled by DSF is the excess surface density,
defined as

.. math::

   \Delta\Sigma(R)
   =
   \overline{\Sigma}(<R)
   -
   \Sigma(R)
   =
   \Sigma_{\rm crit}\,\gamma_t(R),

where :math:`\Sigma(R)` is the projected surface mass density,
:math:`\overline{\Sigma}(<R)` is the mean projected surface density enclosed
within projected radius :math:`R`, :math:`\Sigma_{\rm crit}` is the critical
surface density, and :math:`\gamma_t(R)` is the tangential shear
:cite:p:`Mandelbaum_2005`.

The package is currently in early development. Interfaces, examples, and
recommended workflows may change as the codebase is expanded and validated.

What DSF does
-------------

DSF is designed to connect the main ingredients needed for
:math:`\Delta\Sigma` forecasting:

- tomographic lens and source samples,
- :math:`\Delta\Sigma` data-vector modelling,
- galaxy-matter covariance construction,
- forecast-ready inputs for Fisher or DALI analyses.

The intended workflow is:

.. code-block:: text

   tomography → data vector → covariance → forecast inputs

In practice, this means DSF helps users build matched lens-source bin
combinations, compute stacked :math:`\Delta\Sigma` observables, construct
covariance ingredients, and prepare forecast inputs for downstream inference
or forecasting tools.

Where to go next
----------------

The API reference documents the main DSF modules currently available. More
installation notes and worked examples will be added as the package matures.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   installation
   examples/index
   unit_conventions
   contributing
   api/index

References
----------

.. bibliography::

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`