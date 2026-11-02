Tomography
==========

DSF uses `Binny <https://binny-org.github.io/binny/main/index.html>`_ to
construct lens and source tomography for Delta Sigma forecasts. Binny provides
survey presets, redshift distributions, tomographic binning, population
statistics, and lens-source pair filtering.

The :class:`~dsf.tomography.tomo_builder.TomographyBuilder` interface accepts
separate lens and source configuration dictionaries. Each configuration can
specify a survey preset, a custom parent redshift distribution, or predefined
tomographic bins.

The ``prepare_bins()`` method returns the Binny tomography products, bin
centers, population statistics, and selected lens-source bin pairs required
for covariance calculations.


Survey tomography
-----------------

The following example constructs DESI LRG lens tomography and LSST Y1 source
tomography.

The lens configuration specifies three tomographic bins with redshift edges
at 0.4, 0.6, 0.8, and 1.0. The source configuration uses the LSST Y1 survey
preset.

Lens-source pairs are selected using two criteria:

* ``source_behind_lens=True`` requires the mean source-bin redshift to exceed
  the mean lens-bin redshift.
* ``overlap_threshold=0.10`` requires the fractional overlap between the
  lens and source redshift distributions to be at most ten percent.

The example prints the selected lens-source bin pairs and their mean redshift
centers, then plots the tomographic redshift distributions.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   from dsf.tomography.tomo_builder import TomographyBuilder


   def plot_lens_and_source_bins(lens_result, source_result, figsize=(8, 5)):
       """Plot lens and source tomographic redshift distributions."""
       fig, ax = plt.subplots(figsize=figsize)

       z_lens = np.asarray(lens_result.z).reshape(-1)
       z_source = np.asarray(source_result.z).reshape(-1)

       lens_keys = sorted(lens_result.bins)
       source_keys = sorted(source_result.bins)

       lens_colors = plt.cm.viridis(np.linspace(0.10, 0.75, len(lens_keys)))
       source_colors = plt.cm.viridis(np.linspace(0.25, 1.00, len(source_keys)))

       for i, (key, color) in enumerate(zip(lens_keys, lens_colors, strict=True)):
           nz = np.asarray(lens_result.bins[key]).reshape(-1)

           ax.fill_between(
               z_lens,
               0.0,
               nz,
               facecolor=color,
               edgecolor=color,
               hatch="///",
               alpha=0.45,
               linewidth=0.0,
               zorder=10 + i,
               label=f"lens bin {key + 1}",
           )

           ax.plot(
               z_lens,
               nz,
               color="k",
               linewidth=1.8,
               linestyle="--",
               zorder=20 + i,
           )

       for i, (key, color) in enumerate(zip(source_keys, source_colors, strict=True)):
           nz = np.asarray(source_result.bins[key]).reshape(-1)

           ax.plot(
               z_source,
               nz,
               color=color,
               linewidth=2.4,
               zorder=40 + i,
               label=f"source bin {key + 1}",
           )

           ax.plot(
               z_source,
               nz,
               color="k",
               linewidth=0.8,
               alpha=0.55,
               zorder=39 + i,
           )

       ax.axhline(0.0, color="k", linewidth=1.5)
       ax.set_title("DESI LRG and LSST Y1 tomography")
       ax.set_xlabel(r"Redshift $z$")
       ax.set_ylabel(r"Normalized $n_i(z)$")
       ax.legend(frameon=False, ncol=2)

       fig.tight_layout()
       return fig, ax


   tomo = TomographyBuilder(
       lens={
           "survey": "desi",
           "sample": "lrg",
           "role": "lens",
           "overrides": {
               "bins": {
                   "edges": [0.4, 0.6, 0.8, 1.0],
               },
           },
       },
       source={
           "survey": "lsst",
           "year": "1",
           "role": "source",
       },
       overlap_threshold=0.10,
       source_behind_lens=True,
   )

   inputs = tomo.prepare_bins()

   lens_result = inputs["lens_result"]
   source_result = inputs["source_result"]
   lens_centers = inputs["lens_bin_centers"]
   source_centers = inputs["source_bin_centers"]
   bin_pairs = inputs["bin_pairs"]

   print("Selected lens-source bin pairs:")
   for lens_bin, source_bin in bin_pairs:
       print(f"  lens {lens_bin + 1} - source {source_bin + 1}")

   print("\nLens bin centers:")
   for bin_index, center in lens_centers.items():
       print(f"  lens {bin_index + 1}: z = {center:.3f}")

   print("\nSource bin centers:")
   for bin_index, center in source_centers.items():
       print(f"  source {bin_index + 1}: z = {center:.3f}")

   plot_lens_and_source_bins(lens_result, source_result)
   plt.show()


Synthetic tomography
--------------------

In addition to survey presets, ``TomographyBuilder`` accepts externally
constructed tomographic redshift distributions.

This is useful for testing covariance calculations, exploring alternative
binning schemes, and comparing survey configurations without relying on
a particular survey preset.

The following example constructs three synthetic lens bins and four synthetic
source bins using Gaussian redshift distributions.

The distributions are supplied directly to ``TomographyBuilder`` together
with population fractions and number-density metadata.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   from dsf.tomography.tomo_builder import TomographyBuilder


   def gaussian(z, mean, sigma):
       """Return a normalized Gaussian redshift distribution."""
       nz = np.exp(-0.5 * ((z - mean) / sigma) ** 2)
       return nz / np.trapezoid(nz, z)


   z = np.linspace(0.0, 3.0, 1000)

   lens_bins = {
       0: gaussian(z, 0.35, 0.10),
       1: gaussian(z, 0.55, 0.10),
       2: gaussian(z, 0.75, 0.10),
   }

   source_bins = {
       0: gaussian(z, 0.90, 0.15),
       1: gaussian(z, 1.20, 0.15),
       2: gaussian(z, 1.50, 0.15),
       3: gaussian(z, 1.80, 0.15),
   }

   lens_config = {
       "sample_properties": {
           "number_density": {
               "n_gal_comoving_h3_mpc3": 3e-4,
           },
       },
       "population_fractions": {
           0: 1 / 3,
           1: 1 / 3,
           2: 1 / 3,
       },
   }

   source_config = {
       "sample_properties": {
           "number_density": {
               "n_gal_arcmin2": 20.0,
           },
       },
       "population_fractions": {
           0: 0.25,
           1: 0.25,
           2: 0.25,
           3: 0.25,
       },
   }

   tomo = TomographyBuilder(
       lens={
           "z": z,
           "bins": lens_bins,
           "config": lens_config,
       },
       source={
           "z": z,
           "bins": source_bins,
           "config": source_config,
       },
       overlap_threshold=0.10,
       source_behind_lens=True,
   )

   inputs = tomo.prepare_bins()

   lens_result = inputs["lens_result"]
   source_result = inputs["source_result"]

   fig, ax = plt.subplots(figsize=(8, 5))

   lens_colors = plt.cm.viridis(np.linspace(0.10, 0.55, len(lens_bins)))
   source_colors = plt.cm.viridis(np.linspace(0.60, 0.95, len(source_bins)))

   for (index, nz), color in zip(
       sorted(lens_result.bins.items()), lens_colors, strict=True
   ):
       ax.fill_between(
           z,
           0.0,
           nz,
           color=color,
           alpha=0.35,
           label=f"lens bin {index + 1}",
       )

       ax.plot(z, nz, color=color, linewidth=1.5)

   for (index, nz), color in zip(
       sorted(source_result.bins.items()), source_colors, strict=True
   ):
       ax.plot(
           z,
           nz,
           color=color,
           linewidth=2.4,
           label=f"source bin {index + 1}",
       )

   ax.axhline(0.0, color="k", linewidth=1.5)
   ax.set_title("Synthetic lens and source tomography")
   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel(r"Normalized $n_i(z)$")
   ax.legend(frameon=False, ncol=2)

   fig.tight_layout()
   plt.show()

   print("Selected lens-source pairs:")
   for lens_bin, source_bin in inputs["bin_pairs"]:
       print(f"  lens {lens_bin + 1} - source {source_bin + 1}")


Configuration options
---------------------

The ``TomographyBuilder`` constructor accepts two optional dictionaries,
``lens`` and ``source``, together with the lens-source pair selection criteria.

For survey presets, the configuration dictionaries support the following
entries:

* ``survey``: Survey name, such as ``"desi"`` or ``"lsst"``.
* ``sample``: Galaxy sample, such as ``"lrg"``.
* ``year``: Survey observing year or data-release configuration.
* ``role``: Sample role, such as ``"lens"`` or ``"source"``.
* ``scenario``: Optional survey scenario.
* ``overrides``: Dictionary of modifications to the Binny survey preset.

These entries are passed to Binny when constructing the tomographic bins.

For example, the following configuration uses the DESI LRG lens sample
and LSST Y1 source sample:

.. code-block:: python

   tomo = TomographyBuilder(
       lens={
           "survey": "desi",
           "sample": "lrg",
       },
       source={
           "survey": "lsst",
           "year": "1",
       },
   )

   inputs = tomo.prepare_bins()

When ``lens`` or ``source`` is omitted, DSF uses its default configuration:
DESI LRG for lenses and LSST for sources.

The pair selection parameters are:

* ``overlap_threshold``: Maximum permitted fractional overlap between the
  lens and source redshift distributions. The default is ``0.10``.
* ``source_behind_lens``: Require the mean source-bin redshift to exceed the
  mean lens-bin redshift. The default is ``True``.

Bin centers are calculated using the mean-redshift method.


Tomography products
-------------------

The ``prepare_bins()`` method returns a dictionary containing the Binny
results and the quantities used by DSF.

The principal entries are:

* ``lens_result`` and ``source_result``: Binny tomography results containing
  the redshift grids, distributions, and tomographic bins.
* ``lens_bin_centers`` and ``source_bin_centers``: Mean-redshift centers of
  the tomographic bins.
* ``lens_population_stats`` and ``source_population_stats``: Population
  statistics, including per-bin number densities when available.
* ``lens_shape_stats`` and ``source_shape_stats``: Binny shape statistics.
* ``lens_metadata`` and ``source_metadata``: Tomographic metadata.
* ``bin_pairs``: Selected lens-source bin-index pairs.
* ``lens_builder`` and ``source_builder``: Binny ``NZTomography`` instances.

For example, the selected pairs and lens number densities can be accessed
using:

.. code-block:: python

   inputs = tomo.prepare_bins()

   bin_pairs = inputs["bin_pairs"]
   lens_density = inputs["lens_population_stats"]["density_per_bin"]

The bin indices are zero-based and correspond to the keys in the Binny
tomographic-bin dictionaries.


Custom parent distributions
---------------------------

A custom parent redshift distribution can be supplied instead of a survey
preset or predefined tomographic bins.

This requires a redshift grid ``z``, a parent distribution ``nz``, and a
Binny tomography specification supplied through ``config``.

The ``config`` entry accepts either a Python dictionary or a path to a YAML
configuration file.

For example:

.. code-block:: python

   lens = {
       "z": z_lens,
       "nz": nz_lens,
       "config": lens_config,
   }

   tomo = TomographyBuilder(
       lens=lens,
       source={
           "survey": "lsst",
           "year": "1",
       },
   )

   inputs = tomo.prepare_bins()

Here, ``lens_config`` must contain a valid Binny tomography specification
defining how the parent distribution is divided into tomographic bins.

Unlike predefined tomography, Binny constructs the individual tomographic
distributions from the supplied parent distribution and binning specification.


Predefined tomographic bins
---------------------------

Precomputed tomographic distributions can be supplied directly, as
demonstrated in the synthetic tomography example above.

For predefined bins, the lens or source configuration must contain:

* ``z``: Redshift grid.
* ``bins``: Dictionary mapping bin indices to their redshift distributions.
* ``config``: Binny metadata containing ``sample_properties`` and
  ``population_fractions``.

The ``sample_properties`` must include number-density information, and the
``population_fractions`` must specify a nonnegative fraction for each bin.
These fractions must sum to one.

This interface allows externally constructed tomographic distributions to
be used while retaining the population metadata required by DSF.

For additional information about survey presets, tomography specifications,
and redshift-distribution construction, see the
`Binny documentation <https://binny-org.github.io/binny/main/index.html>`_.
