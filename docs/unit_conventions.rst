Unit conventions
================

Delta Sigma Forecast uses ``Mpc/h`` for user-facing radial quantities,
including forecast inputs such as ``r`` and ``rp_bin_edges`` and saved
forecast outputs.

Internally, different parts of the pipeline follow different conventions:

* The covariance module uses ``Mpc/h``.
* The data-vector module follows the CCL convention and uses ``Mpc``.
* The forecast builder converts input radii from ``Mpc/h`` to ``Mpc`` before
  calling the CCL-backed data-vector calculation.

Delta Sigma units
----------------

The forecast data vector currently reports :math:`\Delta\Sigma` in
``Msun / pc^2``. The associated projected-radius coordinate is reported in
``Mpc/h``.

The covariance calculation uses a ``Sigma_crit`` prefactor intended for
distances supplied in ``Mpc/h`` and Delta Sigma-like quantities expressed in
``Msun h / pc^2``.

Unless otherwise stated, radial bins, projected radii, and forecast radius
grids should be interpreted in ``Mpc/h``.