End-to-end forecast
===================

This example demonstrates an end-to-end :math:`\Delta\Sigma(R)` forecast
using ``DeltaSigmaForecastBuilder``.

DSF constructs the tomographic samples, data vector, and covariance,
while DerivKit computes the Fisher matrix and DALI tensors.

We vary two halo occupation distribution (HOD) parameters,

.. math::

   \boldsymbol{\theta}
   =
   (\log_{10}M_{\min}, \log_{10}M_1),

and compare Fisher and third-order DALI forecasts with and without
Gaussian priors corresponding to approximately 10% uncertainty in
the physical halo masses.


Forecast-builder example
------------------------

The forecast results are precomputed and stored as numerical arrays.
The Fisher forecasts are represented by analytic Gaussian distributions,
while the DALI forecasts are represented by posterior samples.

The numerical results were generated using DerivKit's
``ForecastKit`` interface, including its GetDist utilities for
Fisher and DALI forecasts.

The cached results allow the documentation to display the contours
without recomputing model derivatives or running MCMC.

The first two figures illustrate the effect of Gaussian mass priors
on Fisher and DALI-3 forecasts separately.
The third figure compares the two forecasting methods without
external priors.

.. plot::
   :include-source:
   :context: close-figs

   from pathlib import Path

   import dsf
   import numpy as np
   from getdist import MCSamples, plots as getdist_plots
   from getdist.gaussian_mixtures import GaussianND

   from dsf import DSF_COLORS

   data_dir = Path(dsf.__file__).resolve().parents[2] / "docs" / "examples" / "data"

   parameter_names = ["log10Mmin_0", "log10M1_0"]
   parameter_labels = [
       r"\log_{10}(M_{\min}/[h^{-1}M_\odot])",
       r"\log_{10}(M_1/[h^{-1}M_\odot])",
   ]

   results = {}

   for label in ["No prior", "10% mass prior"]:
       name = label.lower().replace(" ", "_").replace("%", "pct")

       with np.load(data_dir / f"{name}_fisher.npz") as data:
           fisher_mean = data["mean"]
           fisher_cov = data["cov"]

       gaussian = GaussianND(
           mean=fisher_mean,
           cov=fisher_cov,
           names=parameter_names,
           labels=parameter_labels,
           label=f"Fisher: {label}",
       )

       with np.load(data_dir / f"{name}_dali.npz") as data:
           samples = data["samples"]
           weights = data["weights"]
           loglikes = data["loglikes"]

       dali_posterior = MCSamples(
           samples=samples,
           weights=weights,
           loglikes=loglikes if len(loglikes) else None,
           names=parameter_names,
           labels=parameter_labels,
           label=f"DALI-3: {label}",
       )

       results[label] = {
           "fisher": gaussian,
           "dali": dali_posterior,
       }

   def make_plotter():
       plotter = getdist_plots.get_subplot_plotter(width_inch=5)
       plotter.settings.linewidth_contour = 1.5
       plotter.settings.linewidth = 1.5
       plotter.settings.figure_legend_frame = False
       plotter.settings.legend_rect_border = False
       return plotter

   def plot_prior_comparison(method, color):
       plotter = make_plotter()

       plotter.triangle_plot(
           [
               results["No prior"][method],
               results["10% mass prior"][method],
           ],
           params=parameter_names,
           filled=False,
           contour_colors=[color, color],
           contour_lws=[1.5, 1.5],
           contour_ls=["--", "-"],
           legend_labels=[
               f"{method.upper()}: No prior",
               f"{method.upper()}: 10% mass prior",
           ],
       )

       return plotter

   plot_prior_comparison("fisher", DSF_COLORS[1])

   plot_prior_comparison("dali", DSF_COLORS[4])

   plotter = make_plotter()

   plotter.triangle_plot(
       [
           results["No prior"]["fisher"],
           results["No prior"]["dali"],
       ],
       params=parameter_names,
       filled=False,
       contour_colors=[
           DSF_COLORS[1],
           DSF_COLORS[4],
       ],
       contour_lws=[1.5, 1.5],
       legend_labels=[
           "Fisher: No prior",
           "DALI-3: No prior",
       ],
   )


Fisher and DALI
---------------

For a Gaussian likelihood with parameter-independent covariance,
the Fisher information matrix is

.. math::

   F_{\alpha\beta}
   =
   \frac{\partial\boldsymbol{\mu}^{\mathsf T}}
        {\partial\theta_\alpha}
   C^{-1}
   \frac{\partial\boldsymbol{\mu}}
        {\partial\theta_\beta},

where :math:`\boldsymbol{\mu}` is the stacked
:math:`\Delta\Sigma` data vector.

DALI retains higher-order model derivatives and can describe
non-Gaussian posterior features.

Both forecasts are computed through DerivKit's ``ForecastKit``:

.. code-block:: python

   fk = builder.forecastkit()

   fisher = fk.fisher()
   dali = fk.dali(forecast_order=3)

DerivKit also provides GetDist-compatible representations of the
forecast distributions.

For Fisher, ``getdist_fisher_gaussian`` constructs the analytic
Gaussian approximation:

.. code-block:: python

   fisher_gaussian = fk.getdist_fisher_gaussian(
       fisher=fisher,
       names=parameter_names,
       labels=parameter_labels,
       label="Fisher",
   )

For DALI, ``getdist_dali_emcee`` samples the higher-order posterior
using ``emcee`` and returns a GetDist ``MCSamples`` object:

.. code-block:: python

   dali_samples = fk.getdist_dali_emcee(
       dali=dali,
       names=parameter_names,
       labels=parameter_labels,
       label="DALI-3",
   )

These operations were performed when generating the cached
forecast results and are not repeated during documentation builds.


Gaussian mass priors
--------------------

We consider independent Gaussian priors on the two logarithmic
HOD mass parameters.

The prior standard deviation is

.. math::

   \sigma_{\log_{10}M}
   =
   \log_{10}(1.10)
   \simeq 0.0414.

For Fisher, the prior precision is added to the information matrix:

.. math::

   F_{\mathrm{total}}
   =
   F_{\mathrm{data}} + C_{\mathrm{prior}}^{-1}.

For DALI, the same Gaussian prior is included in the sampled posterior.

In the figures, green denotes Fisher and cyan denotes DALI-3.
Dashed contours correspond to forecasts without external priors,
while solid contours include the 10% mass priors.

The final figure compares the two methods without external priors.


Forecast products
-----------------

The prepared data vector, covariance, and tomography products
are accessible through

.. code-block:: python

   forecast = builder.prepare()

The model and covariance can also be passed to external
forecasting tools:

.. code-block:: python

   model, theta0, cov = builder.forecast_inputs()

The covariance is evaluated at the fiducial parameters and
held fixed throughout the forecast.


Notes
-----

This example illustrates the DSF forecasting interface rather
than a production LSST DESC analysis.

The covariance is the ``gm`` block-diagonal covariance associated
with the :math:`\Delta\Sigma` data vector.
Covariance derivatives are not included.

The Gaussian priors are illustrative and are not derived from
external HOD measurements.

Production configurations and survey-specific analyses are
maintained in the companion
`DSF analysis repository <https://github.com/LSSTDESC/dsf-analysis>`_.
