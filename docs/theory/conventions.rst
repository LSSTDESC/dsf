.. |dsflogo| image:: ../_static/dsf_logo.svg
   :alt: DSF logo
   :height: 40px
   :class: inline-dsf-logo

|dsflogo| Unit conventions
==========================

DSF uses ``Mpc/h`` for user-facing radial quantities, including
forecast inputs and saved forecast outputs.

Different components of the pipeline follow different internal
conventions, with conversions handled by the forecast builder.


Radial distances
----------------

The radial conventions are:

- **Covariance:** ``Mpc/h``.
- **Data vector:** ``Mpc``, following the CCL convention.
- **Forecast builder:** converts input radii from ``Mpc/h`` to ``Mpc``
  before evaluating the CCL-backed data vector.

Unless otherwise stated, projected radii, radial bin edges, and
forecast radius grids are expressed in ``Mpc/h``.


Excess surface density
----------------------

The excess surface density is defined as

.. math::

   \Delta\Sigma(R)
   =
   \overline{\Sigma}(<R)-\Sigma(R),

where :math:`\overline{\Sigma}(<R)` is the mean projected surface
density enclosed within radius :math:`R`, and :math:`\Sigma(R)` is
the projected surface density at that radius.

The forecast data vector currently reports :math:`\Delta\Sigma`
in :math:`M_\odot/\mathrm{pc}^2`, with projected radii in ``Mpc/h``.

The covariance calculation uses a critical-surface-density
prefactor intended for distances in ``Mpc/h`` and
:math:`\Delta\Sigma`-like quantities in
:math:`M_\odot h/\mathrm{pc}^2`.

.. important::

   The data-vector and covariance modules use different internal
   conventions. Their relative normalization must be accounted for
   when combining predictions and covariance matrices.
