Fisher and DALI forecasts
=========================

DSF provides the model predictions and covariance needed for derivative-based
forecasts of the excess surface density, :math:`\Delta\Sigma(R)`.
Fisher and DALI calculations are performed using
`DerivKit <https://docs.derivkit.org/>`_.

Fisher forecasts
----------------

Let :math:`\boldsymbol{\theta}` denote the model parameters,
:math:`\boldsymbol{\nu}(\boldsymbol{\theta})` the stacked
:math:`\Delta\Sigma` data vector, and :math:`C` its covariance.
For a Gaussian likelihood with parameter-independent covariance, the Fisher
matrix is

.. math::

   F_{ab} = \frac{\partial \boldsymbol{\nu}^{\mathrm T}}{\partial \theta_a}
            C^{-1}
            \frac{\partial \boldsymbol{\nu}}{\partial \theta_b}.

The inverse Fisher matrix approximates the parameter covariance in the local
Gaussian forecast, with marginalized uncertainties

.. math::

   \sigma(\theta_a) = \sqrt{(F^{-1})_{aa}}.

Independent Gaussian prior information can be included by adding its precision
matrix to the likelihood Fisher matrix.

DALI forecasts
--------------

The Fisher approximation describes a Gaussian parameter distribution near a
fiducial point. DALI (Derivative Approximation for LIkelihoods) extends this
approach by retaining higher-order derivatives of the model predictions.
Writing :math:`\boldsymbol{\delta}=\boldsymbol{\theta}-\boldsymbol{\theta}_0`,
the model is expanded as

.. math::

   \boldsymbol{\nu}(\boldsymbol{\theta})
   \simeq \boldsymbol{\nu}(\boldsymbol{\theta}_0)
   + \boldsymbol{\nu}_{,a}\,\delta^a
   + \frac{1}{2}\boldsymbol{\nu}_{,ab}\,\delta^a\delta^b
   + \cdots.

Inserting this expansion into the Gaussian likelihood yields higher-order
corrections to the Fisher approximation, allowing curved and non-elliptical
parameter contours. Like Fisher, DALI is a local approximation whose accuracy
depends on the model and the region of parameter space considered.

Using DSF
---------

:class:`~dsf.delta_sigma_forecast_builder.DeltaSigmaForecastBuilder` prepares
the tomographic data vector and matching covariance. Once configured, the
builder provides the inputs for both methods:

.. code-block:: python

   forecast = builder.prepare()
   kit = builder.forecastkit()

   fisher = kit.fisher()
   dali = kit.dali(forecast_order=3)

The current DSF forecast builder uses a fiducial, fixed ``gm`` covariance;
these calculations therefore use derivatives of the data vector rather than
covariance derivatives.

For the full mathematical formalism, derivative options, priors, posterior
sampling, and worked examples, see the
`DerivKit forecasting documentation <https://docs.derivkit.org/>`_.
