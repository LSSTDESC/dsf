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

.. plot::
   :include-source:
   :context: close-figs

   import numpy as np

   from getdist import plots as getdist_plots

   from dsf.delta_sigma_forecast_builder import DeltaSigmaForecastBuilder
   from dsf.modelling.cosmology import make_ccl_cosmology
   from dsf.modelling.hod import pk2d_hod


   theta0 = np.array([12.5, 13.5])
   parameter_names = ["log10Mmin_0", "log10M1_0"]
   parameter_labels = [
       r"\log_{10}(M_{\min}/[h^{-1}M_\odot])",
       r"\log_{10}(M_1/[h^{-1}M_\odot])",
   ]

   prior_sigma = np.full(2, np.log10(1.10))

   prior_configs = {
       "No prior": None,
       "10% mass prior": prior_sigma,
   }

   cosmo = make_ccl_cosmology(
       transfer_function="eisenstein_hu",
       matter_power_spectrum="halofit",
   )

   builder = DeltaSigmaForecastBuilder(
       cosmo=cosmo,
       pk2d_func=pk2d_hod,
       pk2d_kwargs={
           "k_array": np.geomspace(1.0e-3, 20.0, 40),
           "a_array": np.linspace(0.35, 1.0, 10),
       },
       theta0=theta0,
       parameter_names=parameter_names,
       r=np.geomspace(2.0, 25.0, 5),
       rp_bin_edges=np.geomspace(2.0, 25.0, 6),
       area_deg2=1000.0,
       sigma_e=0.26,
       tomography_kwargs={
           "lens": {
               "survey": "lsst",
               "role": "lens",
               "year": "1",
               "overrides": {"bins": {"n_bins": 2}},
           },
           "source": {
               "survey": "lsst",
               "role": "source",
               "year": "1",
               "overrides": {"bins": {"n_bins": 2}},
           },
           "overlap_threshold": 1.0,
           "source_behind_lens": True,
       },
       verbose=False,
   )

   builder.prepare()

   fk = builder.forecastkit()

   fisher = fk.fisher()
   dali = fk.dali(forecast_order=3)

   results = {}

   for label, sigma in prior_configs.items():
       if sigma is None:
           fisher_total = fisher
           prior_terms = None
       else:
           prior_cov = np.diag(sigma**2)
           fisher_total = fisher + np.linalg.inv(prior_cov)
           prior_terms = [
               ("gaussian", {"mean": theta0, "cov": prior_cov}),
           ]

       gaussian = fk.getdist_fisher_gaussian(
           fisher=fisher_total,
           names=parameter_names,
           labels=parameter_labels,
           label=f"Fisher: {label}",
       )

       dali_kwargs = {
           "dali": dali,
           "names": parameter_names,
           "labels": parameter_labels,
           "label": f"DALI-3: {label}",
       }

       if prior_terms is not None:
           dali_kwargs["prior_terms"] = prior_terms

       dali_posterior = fk.getdist_dali_emcee(**dali_kwargs)

       results[label] = {
           "fisher": gaussian,
           "dali": dali_posterior,
       }

   plotter = getdist_plots.get_subplot_plotter(width_inch=5)

   plotter.settings.linewidth_contour = 1.5
   plotter.settings.linewidth = 1.5
   plotter.settings.figure_legend_frame = False
   plotter.settings.legend_rect_border = False

   plotter.triangle_plot(
       [
           results["No prior"]["fisher"],
           results["No prior"]["dali"],
           results["10% mass prior"]["fisher"],
           results["10% mass prior"]["dali"],
       ],
       params=parameter_names,
       filled=False,
       contour_colors=[
           "tab:blue",
           "tab:orange",
           "tab:blue",
           "tab:orange",
       ],
       contour_lws=[1.5, 1.5, 1.5, 1.5],
       contour_ls=["--", "--", "-", "-"],
       legend_labels=[
           "Fisher: No prior",
           "DALI: No prior",
           "Fisher: 10% mass prior",
           "DALI: 10% mass prior",
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
Here, third-order DALI is computed directly through DerivKit:

.. code-block:: python

   fk = builder.forecastkit()

   fisher = fk.fisher()
   dali = fk.dali(forecast_order=3)

The independent Gaussian mass priors have logarithmic standard
deviations

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

In the figure, blue and orange denote Fisher and DALI-3,
respectively.
Dashed contours have no external priors, while solid contours
include the 10% mass priors.


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
