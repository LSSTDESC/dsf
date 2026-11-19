.. |dsflogo| image:: ../_static/dsf_logo.svg
   :alt: DSF logo
   :height: 40px
   :class: inline-dsf-logo

|dsflogo| End-to-end forecast
=============================

This example demonstrates an end-to-end :math:`\Delta\Sigma(R)`
forecast using ``DeltaSigmaForecastBuilder``.

DSF constructs the tomographic samples, data vector, and covariance,
while DerivKit computes the Fisher matrix and third-order DALI forecast.

We vary two halo occupation distribution (HOD) parameters,

.. math::

   \boldsymbol{\theta}
   =
   (\log_{10}M_{\min}, \log_{10}M_1),

and compare forecasts with and without independent Gaussian priors
on the logarithmic HOD mass parameters.

The fiducial values and prior standard deviations are

.. math::

   \boldsymbol{\theta}_0 = (12.5, 13.5),
   \qquad
   \boldsymbol{\sigma}_{\mathrm{prior}} = (0.1, 0.2).


Forecast configuration
----------------------

The example uses DESI LRG lens galaxies and LSST Year 1 source
galaxies. The lens sample has three redshift bins, while the source
sample uses the default Binny survey tomography.

Lens-source pairs are selected by requiring the mean source redshift
to exceed the mean lens redshift and applying an overlap threshold
of 0.10.

The forecast uses five projected radial bins between
:math:`2` and :math:`25\,h^{-1}\mathrm{Mpc}`, a survey area of
1000 square degrees, and an intrinsic ellipticity dispersion of 0.26.

The HOD model is evaluated using ``pk2d_hod``.
DSF computes the fiducial excess surface density data vector
and its Gaussian covariance from the galaxy-matter and
galaxy-galaxy power spectra.


Forecast workflow
-----------------

The complete forecast is constructed using
``DeltaSigmaForecastBuilder``.

First, DSF prepares the tomographic samples, fiducial data
vector, and covariance:

.. code-block:: python

   forecast = builder.prepare()
   model, theta0, cov = builder.forecast_inputs()
   fk = builder.forecastkit()

The Fisher matrix and third-order DALI tensors are then computed
using DerivKit:

.. code-block:: python

   fisher = fk.fisher(
       method="finite",
       stepsize=1e-2,
       num_points=3,
   )

   dali = fk.dali(forecast_order=3)

DerivKit provides GetDist-compatible forecast distributions:

.. code-block:: python

   fisher_gaussian = fk.getdist_fisher_gaussian(
       fisher=fisher,
       names=names,
       labels=labels,
       label="Fisher",
   )

   dali_samples = fk.getdist_dali_emcee(
       dali=dali,
       names=names,
       labels=labels,
       label="DALI",
   )


Running the example
-------------------

The complete executable example is available in
:download:`forecast_hod.py <forecast_hod.py>`.

Run it from the DSF repository root:

.. code-block:: bash

   python -u docs/examples/forecast_hod.py

The script constructs the DSF data vector and covariance,
computes the Fisher and DALI forecasts, samples the DALI
distributions, and generates three comparison plots.

The Fisher matrix and DALI tensors are cached in
``forecast_hod_output/forecast_cache.npz``.

When the cache exists, the script reuses the stored tensors.
Otherwise, it computes and saves them.

The full source code is included below.

.. literalinclude:: forecast_hod.py
   :language: python


Notes
-----

This example illustrates the DSF forecasting interface rather
than a production LSST DESC analysis.

The covariance is evaluated at the fiducial parameters and
held fixed throughout the forecast.
Covariance derivatives are not included.

The Gaussian HOD parameter priors are illustrative and are not
derived from external HOD measurements.

Production configurations and survey-specific analyses are
maintained in the companion
`DSF analysis repository <https://github.com/LSSTDESC/dsf-analysis>`_.


Forecast comparison
-------------------

The figures below compare Fisher and third-order DALI forecasts
with and without Gaussian parameter priors.

The first two figures show the effect of the priors on Fisher
and DALI separately.
The third compares Fisher and DALI without external priors.

The Fisher matrix and DALI tensors are loaded from cached
DSF results.
Only GetDist sampling and plotting are performed during the
documentation build.

.. plot::
   :context: close-figs

   from pathlib import Path

   import dsf
   import numpy as np
   from derivkit import ForecastKit
   from getdist import plots

   from dsf import DSF_COLORS

   cache = (
       Path(dsf.__file__).resolve().parents[2]
       / "docs"
       / "examples"
       / "forecast_hod_output"
       / "forecast_cache.npz"
   )

   with np.load(cache, allow_pickle=True) as data:
       theta0 = data["theta0"]
       cov = data["cov"]
       fisher_like = data["fisher_like"]
       dali = data["dali"].item()

   names = ["log10Mmin_0", "log10M1_0"]
   labels = [
       r"\log_{10}(M_{\min}/[h^{-1}M_\odot])",
       r"\log_{10}(M_1/[h^{-1}M_\odot])",
   ]

   sigma_prior = np.array([0.1, 0.2])
   prior_cov = np.diag(sigma_prior**2)
   fisher_prior = np.diag(1.0 / sigma_prior**2)
   fisher_post = fisher_like + fisher_prior

   # GetDist utilities use the precomputed DSF forecast tensors.
   fk = ForecastKit(
       function=lambda theta: np.zeros(cov.shape[0]),
       theta0=theta0,
       cov=cov,
   )

   g_fisher = fk.getdist_fisher_gaussian(
       fisher=fisher_like,
       names=names,
       labels=labels,
       label="Fisher",
   )

   g_fisher_prior = fk.getdist_fisher_gaussian(
       fisher=fisher_post,
       names=names,
       labels=labels,
       label="Fisher + prior",
   )

   g_dali = fk.getdist_dali_emcee(
       dali=dali,
       names=names,
       labels=labels,
       label="DALI",
   )

   g_dali_prior = fk.getdist_dali_emcee(
       dali=dali,
       names=names,
       labels=labels,
       label="DALI + prior",
       prior_terms=[
           ("gaussian", {"mean": theta0, "cov": prior_cov}),
       ],
   )

   def make_plot(samples, colors, filled, legend):
       plotter = plots.get_subplot_plotter(width_inch=5)
       plotter.settings.linewidth_contour = 1.5
       plotter.settings.linewidth = 1.5
       plotter.settings.figure_legend_frame = False
       plotter.settings.legend_rect_border = False

       plotter.triangle_plot(
           samples,
           params=names,
           filled=filled,
           contour_colors=colors,
           contour_lws=[1.5] * len(samples),
           contour_ls=["-"] * len(samples),
           legend_labels=legend,
       )

       return plotter

   make_plot(
       [g_fisher, g_fisher_prior],
       [DSF_COLORS[1], DSF_COLORS[0]],
       [False, True],
       ["Fisher", "Fisher + prior"],
   )

   make_plot(
       [g_dali, g_dali_prior],
       [DSF_COLORS[4], DSF_COLORS[3]],
       [False, True],
       ["DALI", "DALI + prior"],
   )

   make_plot(
       [g_fisher, g_dali],
       [DSF_COLORS[1], DSF_COLORS[4]],
       [False, False],
       ["Fisher", "DALI"],
   )
