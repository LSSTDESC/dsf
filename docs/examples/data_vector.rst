Data vector
===========

DSF computes the excess surface density :math:`\Delta\Sigma(R)` from a
Fourier-space ``Pk2D`` model. This page demonstrates a minimal calculation
at a single lens redshift and an average over a lens redshift distribution.
These examples compute data vectors, not covariances.

For intrinsic alignments, baryonic effects, and magnification bias, see
:doc:`systematics`.

Minimal example
---------------

.. plot::
   :include-source:
   :context: close-figs

   import matplotlib.pyplot as plt
   import numpy as np

   from dsf import DSF_COLORS
   from dsf.data_vector.delta_sigma_builder import DeltaSigmaCalculator
   from dsf.modelling.cosmology import make_ccl_cosmology
   from dsf.modelling.hod import pk2d_hod


   cosmo = make_ccl_cosmology(
       transfer_function="eisenstein_hu",
       matter_power_spectrum="halofit",
   )

   r = np.geomspace(0.2, 30.0, 24)

   z_lens = 0.55
   a_lens = 1.0 / (1.0 + z_lens)

   k_array = np.geomspace(1.0e-3, 30.0, 64)
   a_array = np.linspace(0.3, 1.0, 16)


   def pk2d_func(*, cosmo):
       return pk2d_hod(
           cosmo,
           k_array=k_array,
           a_array=a_array,
       )["pk_deltasigma"]


   calculator = DeltaSigmaCalculator(pk2d_func=pk2d_func)

   delta_sigma = calculator.delta_sigma(
       r=r,
       a=a_lens,
       cosmo=cosmo,
   )

   z = np.linspace(0.45, 0.65, 9)
   n_z = np.exp(-0.5 * ((z - z_lens) / 0.05) ** 2)

   delta_sigma_bin = calculator.delta_sigma_lens_bin(
       r=r,
       lens_dndz=(z, n_z),
       cosmo=cosmo,
       z_min=0.45,
       z_max=0.65,
   )

   ratio = delta_sigma_bin / delta_sigma

   colors = DSF_COLORS[:3]  # First DSF color is the baseline.

   fig, (ax, ax_res) = plt.subplots(
       2,
       1,
       sharex=True,
       gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.06},
   )

   ax.plot(
       r,
       delta_sigma,
       color=colors[0],
       marker="o",
       markersize=6,
       label=rf"single redshift, $z_l={z_lens:.2f}$",
   )

   ax.scatter(
       r,
       delta_sigma_bin,
       color=colors[2],
       s=50,
       label="lens-bin averaged",
       zorder=3,
   )

   ax.set_xscale("log")
   ax.set_yscale("log")

   ax.set_ylabel(r"$\Delta\Sigma(R)\ [M_\odot / {\rm pc}^2]$")
   ax.legend(frameon=False)

   ax_res.axhline(1.0, color="lightgray", linestyle="--")
   ax_res.plot(
       r,
       ratio,
       color=colors[1],
   )

   ax_res.set_xscale("log")
   ax_res.set_ylabel("ratio")
   ax_res.set_xlabel(r"$R\ [{\rm Mpc}]$")

   fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.94)

   plt.show()

Using modelling functions directly
----------------------------------

The functions in ``dsf.modelling`` can also be called directly to construct
``Pk2D`` objects. Their returned ``"pk_deltasigma"`` entry is the power
spectrum passed to ``DeltaSigmaCalculator``; it is not itself a projected
:math:`\Delta\Sigma(R)` data vector.

.. code-block:: python

   from dsf.modelling.hod import pk2d_hod

   spectra = pk2d_hod(
       cosmo,
       k_array=k_array,
       a_array=a_array,
   )
   pk2d = spectra["pk_deltasigma"]

   calculator = DeltaSigmaCalculator(
       pk2d_func=lambda *, cosmo: pk2d_hod(
           cosmo,
           k_array=k_array,
           a_array=a_array,
       )["pk_deltasigma"],
   )
   delta_sigma = calculator.delta_sigma(r=r, a=a_lens, cosmo=cosmo)

Excess surface density
----------------------

The single-redshift calculation evaluates

.. math::

   \Delta\Sigma(R) = \bar{\Sigma}(<R) - \Sigma(R)

at one lens scale factor. The lens-bin-averaged calculation instead integrates
the signal over a toy lens redshift distribution,

.. math::

   \langle \Delta\Sigma(R) \rangle =
   \frac{\int \mathrm{d}z\, n_l(z)\,\Delta\Sigma(R, z)}
        {\int \mathrm{d}z\, n_l(z)}.

The lower panel shows the ratio between the lens-bin-averaged signal and the
single-redshift signal. This is useful for checking how much redshift averaging
changes the projected signal relative to evaluating the model at one effective
lens redshift.

All data-vector outputs are arrays with the same length as the input projected
radius grid.
