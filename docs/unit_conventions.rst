Unit conventions
================

DeltaSigma Forecast uses ``Mpc/h`` for user-facing radial quantities,
including forecast inputs such as ``r`` and ``rp_bin_edges`` and saved
forecast outputs.

Internally, different parts of the pipeline follow different conventions:

* The covariance module uses ``Mpc/h``.
* The data-vector module follows the CCL convention and uses ``Mpc``.
* The forecast builder converts input radii from ``Mpc/h`` to ``Mpc`` before
  calling the CCL-backed data-vector calculation.

Unless otherwise stated, radial bins, projected radii, covariance outputs, and
forecast results should be interpreted in ``Mpc/h``.