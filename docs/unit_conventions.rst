Unit conventions
================

DeltaSigma Forecast uses ``Mpc/h`` for public-facing radial quantities and
saved outputs.

Internally, different parts of the pipeline follow different conventions:

* The covariance module uses ``Mpc/h``.
* The data-vector module follows the CCL convention and uses ``Mpc``.
* The main API converts these conventions so that final user-facing results are
  reported in ``Mpc/h``.

Unless otherwise stated, radial bins, projected radii, covariance outputs, and
forecast results should be interpreted in ``Mpc/h``.