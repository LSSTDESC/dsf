
.. |dsflogo| image:: ../_static/dsf_logo.svg
   :alt: DSF logo
   :height: 40px
   :class: inline-dsf-logo

|dsflogo| Covariance modelling
==============================

DSF constructs Gaussian covariance matrices for projected
galaxy-galaxy lensing and clustering observables.

The covariance combines cosmological power spectra,
survey geometry, tomographic redshift distributions,
shape noise, and galaxy shot noise.
Fourier-space covariance contributions are projected into
radial bins using Hankel transforms.


Gaussian covariance
-------------------

For a data vector :math:`\boldsymbol{d}`, the covariance is

.. math::

   C_{ab}
   =
   \left\langle
   (d_a-\langle d_a\rangle)
   (d_b-\langle d_b\rangle)
   \right\rangle.

Under the Gaussian approximation, the covariance of two
power-spectrum estimators follows from Wick contractions,

.. math::

   \operatorname{Cov}
   \left[
   \hat P_{ab}(k),\hat P_{cd}(k)
   \right]
   \propto
   \left(P_{ac}+N_{ac}\right)
   \left(P_{bd}+N_{bd}\right)
   +
   \left(P_{ad}+N_{ad}\right)
   \left(P_{bc}+N_{bc}\right),

where :math:`P_{ab}` denotes a signal power spectrum and
:math:`N_{ab}` denotes the corresponding noise contribution.
The proportionality includes the survey-volume and
mode-counting normalization.

DSF uses the galaxy-galaxy, galaxy-matter, and matter-matter
power spectra,

.. math::

   P_{gg}(k,z), \qquad
   P_{gm}(k,z), \qquad
   P_{mm}(k,z),

together with the appropriate geometric and noise factors
to construct the projected covariance.


Covariance components
---------------------

DSF supports three covariance contributions:

- ``gm x gm``: covariance of the galaxy-matter
  excess surface density.
- ``gg x gg``: covariance of the projected
  galaxy-clustering contribution.
- ``gm x gg``: cross-covariance between the two observables.

For a joint data vector, these form the block matrix

.. math::

   \mathbf{C}_{\mathrm{joint}}
   =
   \begin{pmatrix}
   \mathbf{C}_{gm,gm} & \mathbf{C}_{gm,gg} \\
   \mathbf{C}_{gm,gg}^{\mathsf T} & \mathbf{C}_{gg,gg}
   \end{pmatrix}.

The covariance includes the relevant combinations of
power spectra, with galaxy shot noise and source shape
noise entering the corresponding terms.


Survey geometry and noise
-------------------------

The covariance depends on the survey area and the
comoving volume of each lens tomographic bin.

Galaxy shot noise is determined by the three-dimensional
lens number density,

.. math::

   N_g = \frac{1}{\bar n_g}.

Source shape noise depends on the intrinsic ellipticity
dispersion :math:`\sigma_e` and effective source number
density :math:`n_{\mathrm{eff}}`.

For galaxy-galaxy lensing, the conversion between shear
and excess surface density introduces critical surface
density factors,

.. math::

   \Sigma_{\mathrm{crit}}(z_l,z_s)
   =
   \frac{c^2}{4\pi G}
   \frac{D_s}{D_l D_{ls}},

in the proper-distance convention.
DSF evaluates the corresponding effective factors using
the lens and source redshift distributions, with the
appropriate conversion to its comoving convention.

Line-of-sight projection windows and lens-bin geometry
also enter the projected covariance.


Tomographic covariance
----------------------

Covariance ingredients are evaluated for each selected
lens-source tomographic bin pair.

DSF averages the relevant power spectra over the lens
redshift distribution and evaluates source-dependent
quantities, including shape noise and critical surface
density factors, for each source bin.

The current block-diagonal construction assumes that
different selected tomographic bin pairs are independent,

.. math::

   \mathbf{C}
   =
   \operatorname{diag}
   \left(
   \mathbf{C}^{(1)},
   \mathbf{C}^{(2)},
   \ldots,
   \mathbf{C}^{(N_{\mathrm{pairs}})}
   \right).

Within each bin-pair block, correlations between radial
bins are retained.
Cross-covariances between different tomographic bin
pairs are not included in this approximation.


Radial projection
-----------------

DSF projects Fourier-space covariance contributions into
real-space radial bins using Hankel transforms.

Schematically,

.. math::

   C(R,R')
   =
   \int
   \frac{k\,\mathrm{d}k}{2\pi}
   \int
   \frac{k'\,\mathrm{d}k'}{2\pi}
   J_n(kR)\,
   C(k,k')\,
   J_m(k'R').

The implementation supports radial bin averaging,
configurable Bessel orders, and numerical tapering.

For transform conventions, numerical backends, and
implementation details, see :doc:`hankel`.


Implementation
--------------

The ``DeltaSigmaCovarianceBuilder`` assembles the covariance
from the tomographic samples and cosmological inputs.

By default, the galaxy spectra are constructed using a
linear galaxy-bias prescription,

.. math::

   P_{gm}(k,z) = b_g(z)P_{mm}(k,z),

.. math::

   P_{gg}(k,z) = b_g^2(z)P_{mm}(k,z).

Alternatively, custom ``Pk2D`` objects can be supplied
for both :math:`P_{gm}` and :math:`P_{gg}`.

The builder provides individual covariance blocks,
joint covariance matrices, and block-diagonal
tomographic covariances.

For a practical calculation, see
:doc:`../examples/covariance`.
