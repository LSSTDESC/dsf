
.. |dsflogo| image:: ../_static/dsf_logo.svg
   :alt: DSF logo
   :height: 40px
   :class: inline-dsf-logo

|dsflogo| Data vector and systematics
=====================================

DSF models the excess surface density :math:`\Delta\Sigma(R)`,
the primary observable of galaxy-galaxy lensing.
The prediction is constructed from a three-dimensional
galaxy-matter power spectrum and can include additional
physical contributions and observational systematics.


Excess surface density
----------------------

The excess surface density is defined as

.. math::

   \Delta\Sigma(R,z)
   =
   \bar{\Sigma}(<R,z)-\Sigma(R,z),

where :math:`\Sigma(R,z)` is the projected surface density and
:math:`\bar{\Sigma}(<R,z)` is its mean within projected radius
:math:`R`.

In Fourier space, the galaxy-matter contribution can be
expressed as a second-order Hankel transform,

.. math::

   \Delta\Sigma(R,z)
   =
   \bar{\rho}_{m,0}
   \int_0^\infty
   \frac{k\,\mathrm{d}k}{2\pi}
   P_{gm}(k,z)J_2(kR),

where :math:`P_{gm}(k,z)` is the galaxy-matter power spectrum,
:math:`J_2` is the second-order Bessel function, and
:math:`\bar{\rho}_{m,0}` is the mean comoving matter density.

DSF evaluates this projection using a CCL ``Pk2D`` object,
allowing different physical models to be supplied through
the same interface.


Tomographic data vector
-----------------------

For a lens tomographic bin :math:`i` with redshift distribution
:math:`n_l^i(z)`, DSF computes the lens-averaged profile,

.. math::

   \Delta\Sigma^i(R)
   =
   \frac{
   \int \mathrm{d}z\,
   n_l^i(z)\Delta\Sigma(R,z)
   }{
   \int \mathrm{d}z\,n_l^i(z)
   }.

The tomographic data vector is formed by concatenating the
profiles for the selected lens-source bin pairs,

.. math::

   \boldsymbol{\mu}
   =
   \left(
   \Delta\Sigma^{i_1}(R_1),\ldots,
   \Delta\Sigma^{i_1}(R_{N_R}),
   \Delta\Sigma^{i_2}(R_1),\ldots
   \right)^{\mathsf T}.

The ordering follows the selected tomographic bin pairs.
For the baseline prediction, pairs sharing the same lens bin
have identical :math:`\Delta\Sigma` profiles; source-bin
dependence enters through the lensing geometry and covariance.

DSF uses projected comoving radii and reports
:math:`\Delta\Sigma` in :math:`M_\odot/\mathrm{pc}^2`.


Physical modelling and systematics
----------------------------------

The galaxy-matter power spectrum can be constructed using
different physical prescriptions without changing the
projection or data-vector interface.

**Halo occupation distribution (HOD)**

The HOD model describes the occupation of dark matter halos
by galaxies and determines the galaxy-matter clustering
signal through the adopted halo-model prescription.

**Intrinsic alignments**

Intrinsic alignments correlate galaxy shapes with the
surrounding tidal field.
DSF provides nonlinear alignment (NLA) combinations,
controlled by the intrinsic-alignment amplitude
:math:`A_{\mathrm{IA}}`.

**Baryonic effects**

Baryonic processes redistribute matter within halos,
modifying the small-scale density profile.
DSF provides a baryonified HOD prescription in which
the halo concentration-mass relation is rescaled
through the parameter :math:`f_c`.

**Lens magnification**

Magnification changes the observed lens-galaxy number
density through flux magnification and area distortion.
DSF provides an additive lens-magnification correction,

.. math::

   \Delta\Sigma_{\mathrm{obs}}(R)
   =
   \Delta\Sigma_{\mathrm{base}}(R)
   +
   \Delta\Sigma_{\mathrm{mag}}(R),

whose amplitude is proportional to
:math:`\alpha_{\mathrm{lens}}-1`, where
:math:`\alpha_{\mathrm{lens}}` characterizes the
magnification response of the lens sample.

**Stellar point mass**

The stellar mass of the central lens galaxy contributes

.. math::

   \Delta\Sigma_\star(R)
   =
   \frac{M_\star}{\pi R^2}.

This contribution is most relevant at small projected
separations.

These ingredients can be used to compare alternative
physical predictions or construct systematic corrections
to the baseline data vector.


Further examples
----------------

See :doc:`../examples/data_vector` for the basic
:math:`\Delta\Sigma` calculation and
:doc:`../examples/systematics` for comparisons of HOD,
intrinsic-alignment, baryonic, and magnification models.
