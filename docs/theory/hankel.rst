
.. |dsflogo| image:: ../_static/dsf_logo.svg
   :alt: DSF logo
   :height: 40px
   :class: inline-dsf-logo

|dsflogo| Hankel transforms
===========================

Hankel transforms relate Fourier-space power spectra to real-space
correlation functions under radial symmetry. They arise naturally in
cosmology because statistically isotropic fields can be described
using power spectra that depend only on the magnitude of the
wavevector.

DSF uses Hankel transforms to compute projected correlation functions,
spherical correlation functions, and real-space covariance matrices.


Projected Hankel transforms
---------------------------

For an isotropic two-dimensional field, the projected correlation
function of Bessel order :math:`n` is

.. math::

   \xi_n(R)
   =
   \int_0^\infty
   \frac{k\,\mathrm{d}k}{2\pi}
   P(k) J_n(kR),

where :math:`P(k)` is the Fourier-space spectrum, :math:`R` is the
projected separation, and :math:`J_n` is the Bessel function of the
first kind.

The Bessel order depends on the observable. For example,
:math:`J_0` appears in isotropic projected correlations, while
:math:`J_2` arises in tangential-shear calculations.


Spherical Hankel transforms
---------------------------

For statistically isotropic three-dimensional fields, the
real-space correlation function is related to the power spectrum by

.. math::

   \xi(r)
   =
   \int_0^\infty
   \frac{k^2\,\mathrm{d}k}{2\pi^2}
   P(k) j_0(kr),

where :math:`j_0` is the spherical Bessel function of order zero.

More generally, spherical Hankel transforms involve
:math:`j_n(kr)` for higher orders.


Covariance transformations
--------------------------

Hankel transforms also convert Fourier-space covariance kernels
into real-space covariance matrices.

For a projected covariance kernel :math:`C(k,k')`, the corresponding
real-space covariance can be written schematically as

.. math::

   \operatorname{Cov}(R,R')
   =
   \int_0^\infty
   \frac{k\,\mathrm{d}k}{2\pi}
   \int_0^\infty
   \frac{k'\,\mathrm{d}k'}{2\pi}
   J_n(kR)\,
   C(k,k')\,
   J_m(k'R').

The Bessel orders :math:`n` and :math:`m` depend on the observables
being correlated. In DSF, these transformations are used to construct
covariance matrices for projected statistics.


Numerical implementation
------------------------

DSF provides three numerical backends through the
``HankelTransform`` interface:

- ``fftlog`` uses logarithmic sampling and FFT-based methods.
- ``matrix_zeros`` constructs transform operators using Bessel zeros.
- ``matrix_direct`` constructs transform operators on specified grids.

The appropriate backend depends on the dimensionality of the
transform, numerical sampling, and required accuracy.

For practical examples and backend selection, see
:doc:`../examples/hankel`.
