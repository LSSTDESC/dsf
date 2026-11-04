
.. |dsf-logo| image:: _static/dsf_logo.svg
   :alt: DSF logo
   :height: 36px
   :class: inline-dsf-logo

|dsf-logo| Ecosystem
====================

DSF is developed within the
`LSST Dark Energy Science Collaboration (DESC) <https://lsstdesc.org/>`_
and is part of a broader ecosystem of scientific software for
cosmological modelling, galaxy-galaxy lensing, and forecasting.

This page highlights selected packages that DSF depends on, as well
as complementary tools for observational measurements, galaxy
population modelling, covariance estimation, and cosmological analysis.

Dependencies
------------

DSF builds on several scientific Python packages for cosmological
modelling, survey tomography, and statistical inference.

The complete list of dependencies, including version requirements,
is maintained in the
`pyproject.toml <https://github.com/LSSTDESC/dsf/blob/main/pyproject.toml>`_.

Cosmological modelling
~~~~~~~~~~~~~~~~~~~~~~

`Core Cosmology Library (CCL) <https://github.com/LSSTDESC/CCL>`_
   A cosmological modelling library developed within LSST DESC.
   CCL provides matter power spectra, correlation functions,
   halo-model calculations, and other cosmological quantities
   used in DSF predictions.

Survey tomography
~~~~~~~~~~~~~~~~~

`Binny <https://github.com/binny-org/binny>`_
   A Python package for constructing and manipulating survey
   redshift distributions, tomographic bins, and bin combinations.
   DSF uses Binny for survey and tomography modelling.

Statistical inference
~~~~~~~~~~~~~~~~~~~~~

`DerivKit <https://github.com/derivkit-org/derivkit>`_
   A Python package for numerical differentiation and
   derivative-based statistical inference, including Fisher
   information matrices and higher-order DALI approximations.
   DerivKit provides statistical forecasting capabilities
   used by DSF.

`GetDist <https://getdist.readthedocs.io/>`_
   A Python package for analysing and visualizing samples from
   probability distributions, particularly cosmological parameter
   posteriors. It provides tools for marginalized parameter
   constraints, credible regions, and posterior contour plots.

Related Software
----------------

The following packages address complementary aspects of cosmological
modelling, observational measurements, galaxy populations, covariance
estimation, and data analysis.

These packages are not required dependencies of DSF.

Galaxy-galaxy lensing measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`dsigma <https://dsigma.readthedocs.io/en/stable/>`_
   A Python package for measuring galaxy-galaxy lensing from
   observational catalogs. It estimates excess surface density
   around lens populations and supports jackknife covariance
   estimation and related summary statistics.

   While DSF models and forecasts excess surface density,
   ``dsigma`` measures the observable from survey data.

Galaxy luminosity functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`LFKit <https://cosmology-kit.github.io/lfkit/main/>`_
   A Python toolkit for modelling galaxy luminosity functions,
   photometric corrections, magnitude conversions, and
   magnitude-limited catalog selection.

   LFKit provides tools for computing galaxy number densities,
   completeness fractions, luminosity-function-weighted redshift
   trends, and observable or missing catalog populations.

   Its applications include photometric redshift modelling,
   intrinsic alignments, galaxy cluster science, gravitational-wave
   cosmology, and survey forecasting.

Galaxy cluster modelling
~~~~~~~~~~~~~~~~~~~~~~~~

`CROW <https://lsstdesc.org/crow/>`_
   The LSST-DESC Cluster Reconstruction of Observables Workbench
   is a Python library for predicting galaxy cluster observables.
   It provides complementary modelling capabilities for
   cluster-based cosmological analyses.

Covariance estimation
~~~~~~~~~~~~~~~~~~~~~

`TJPCov <https://github.com/LSSTDESC/TJPCov>`_
   A Python package for calculating covariance matrices of
   cosmological two-point statistics. It provides complementary
   covariance modelling capabilities for large-scale structure
   and weak-lensing analyses.

Cosmological data formats
~~~~~~~~~~~~~~~~~~~~~~~~~

`SACC <https://github.com/LSSTDESC/sacc>`_
   A Python library and standardized data format for storing
   cosmological summary statistics, covariance matrices,
   and associated metadata.

   SACC supports the exchange of cosmological measurements
   and predictions between analysis tools.
